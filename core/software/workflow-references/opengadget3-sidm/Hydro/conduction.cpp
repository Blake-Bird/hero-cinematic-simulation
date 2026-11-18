/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Computes conduction bases on an implicit diffusion solver.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
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

#ifdef CONDUCTION

#ifndef ACC_INCLUDE_CONDUCTION

#define LOCK_NEXPORT
#define UNLOCK_NEXPORT

#define MAX_COND_ITER 150
#define COND_ITER_ACCURACY 1.0e-4

#include "../CodeBase/greentree.hpp"
/**
 * @brief Struct containing data related to conduction.
 */
static struct conductiondata_in
{
    MyLongDouble Pos[3];
    MyFloat Hsml;
    MyFloat Density;
    MyFloat Kappa;
#if defined(LT_STELLAREVOLUTION)
    int Type;
#endif // defined(LT_STELLAREVOLUTION)
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    MyFloat KappaB;
    MyFloat B[3];
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)

#ifdef ACC_CONDUCTION
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    MyFloat BPred[3];
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
#endif // ACC_CONDUCTION
} *ConductionDataIn, *ConductionDataGet;

/**
 * @brief Struct containing output data related to conduction.
 */
static struct conductiondata_out
{
    double Out; // should have the same type as in thread_params
    double Sum;
} *ConductionDataResult, *ConductionDataOut;

/**
 * @brief Struct containing parameters for a thread.
 */
struct thread_params
{
    int tid; // thread id
    /// pointers to the matrix multiplication vectors
    double *in;
    double *out;
    double *sum;
} *threadparams;

static double *Energy, *EnergyOld;
static double *Residual, *DVec, *QVec;

#if defined(ACC_CONDUCTION) && defined(ACC_OPENMP)
#pragma omp declare target
#endif // defined(ACC_CONDUCTION) && defined(ACC_OPENMP)
MyFloat *Kappa;

#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
// extern  MyFloat *KappaB;
MyFloat *KappaB;
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)

#if defined(ACC_CONDUCTION) && defined(ACC_OPENMP)
#pragma omp end declare target
#endif // defined(ACC_CONDUCTION) && defined(ACC_OPENMP)

#ifdef ACC_CONDUCTION
#ifdef ACC_OPENACC
#pragma acc declare create(Kappa)
#endif // ACC_OPENACC
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
#define KAPPAB_POINTER KappaB
#define COPY_KAPPAB KappaB [0:N_gas],
#define KAPPAB_ZERO_NGAS KappaB [0:N_gas],
#ifdef ACC_OPENACC
#pragma acc declare create(KappaB)
#endif // ACC_OPENACC
#else //  defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
#define KAPPAB_POINTER NULL
#define COPY_KAPPAB
#define KAPPAB_ZERO_NGAS
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
#endif // ACC_CONDUCTION

#ifdef CONDUCTION_BICGSTAB
static double *PVec, *SVec, *TVec;
#endif // CONDUCTION_BICGSTAB

#ifdef ACC_CONDUCTION
ACC_DECLARE_ROUTINE_BEFORE
int ngb_treefind_pairs_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *aP, struct sph_walkdata *SphP, int *ngblist, int maxnumgb,
                                     int i_node);
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_CONDUCTION

// We should really implement such routines for conduction!
// So we make automatic copy in GPU and CPU if we change data structure! AR.
// static inline void out2particle_condiction(struct conductiondata_out *out, int i);
#define out2particle_conduction(fr, place, mode)                                                                       \
    if (mode == 1)                                                                                                     \
    {                                                                                                                  \
        (out)[place] += (fr)->Out;                                                                                     \
        (sum)[place] += (fr)->Sum;                                                                                     \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        (out)[place] = (fr)->Out;                                                                                      \
        (sum)[place] = (fr)->Sum;                                                                                      \
    }

#ifdef ACC_CONDUCTION
/**
 * @brief Copies particle data to a structure for conduction calculation.
 *
 * This function copies data from the particle at the specified location to a structure
 * for conduction calculation. It includes position, smoothing length, density, conductivity,
 * and optionally magnetic field data if defined.
 *
 * @param in Pointer to the structure to store the particle data.
 * @param place Index of the particle whose data is being copied.
 */
inline void particle2in_conduction(struct conductiondata_in *in, int place)
#else // ACC_CONDUCTION
/**
 * @brief Copies particle data to a structure for conduction calculation.
 *
 * This function copies data from the particle at the specified location to a structure
 * for conduction calculation. It includes position, smoothing length, density, conductivity,
 * and optionally magnetic field data if defined.
 *
 * @param in Pointer to the structure to store the particle data.
 * @param place Index of the particle whose data is being copied.
 */
static inline void particle2in_conduction(struct conductiondata_in *in, int place)
#endif // ACC_CONDUCTION
{
    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[place].Pos[k];
    }

    in->Hsml = P[place].Hsml;
#if defined(LT_STELLAREVOLUTION)
    in->Type = P[place].Type;
#endif // defined(LT_STELLAREVOLUTION)
    in->Density = SphP[place].Density;
    in->Kappa = Kappa[place];

#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    in->KappaB = KappaB[place];

    for (int k = 0; k < 3; k++)
    {
        in->B[k] = SphP[place].BPred[k];
    }
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
#ifdef ACC_CONDUCTION
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    for (int k = 0; k < 3;
         k++) // I know it is a duplicate of B, but I am unsure about its use in the evalue function, so better be safe
        in->BPred[k] = SphP[place].BPred[k];
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
#endif // ACC_CONDUCTION
}

/* we will use the conjugate gradient method to compute a solution
   of the implicitly formulate diffusion equation */
/* Note: the conduction equation we solve is really formulated with u instead of T, i.e.
   the factor (gamma-1)*mu*mp/k_B that converts from T to u is implicitely absorbed in a
   redefinition of kappa */
#if !(defined ACC_INCLUDE_CONDUCTION) && defined(ACC_CONDUCTION) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#include "../OpenACC/conduction.hpp"
#endif // !(defined ACC_INCLUDE_CONDUCTION) && defined(ACC_CONDUCTION)

/**
 * @brief Performs thermal conduction calculations.
 *
 * This function computes thermal conduction using either the Conjugate Gradient algorithm
 * or the BiCGSTAB algorithm, depending on compilation flags. It initializes necessary variables
 * and parameters, computes conductivities, and iterates until convergence to obtain the solution vector.
 * Finally, it updates the particle entropies based on the computed energy values.
 */
void conduction(void)
{
    TIMER_PUSH(1, "CONDUCTION");
    int iter;
    double delta0, delta1, alpha, beta, a3inv, dt, rel_change, loc_max_rel_change, glob_max_rel_change;
    double sumnew, sumold, sumtransfer, sumnew_tot, sumold_tot, sumtransfer_tot, atime;
#ifdef CONDUCTION_BICGSTAB
    double gamma;
#endif // CONDUCTION_BICGSTAB

    TIMER_PUSH(2, "Initializing");

    VERBOSE(3, "Start thermal conduction...\n");

    if (All.ComovingIntegrationOn)
    {
        atime = All.Time;
        a3inv = 1 / (All.Time * All.Time * All.Time);
    }
    else
    {
        atime = a3inv = 1.0;
    }

    dt = (All.Conduction_Ti_endstep - All.Conduction_Ti_begstep) * All.Timebase_interval;

    if (All.ComovingIntegrationOn)
    {
        dt *= All.Time / hubble_function(All.Time);
    }

    VERBOSE(2, "dt=%g\n", dt);

    if (dt == 0.0)
    {
        VERBOSE(1, "Nothing to do for zero timestep\n");
        TIMER_POP(2, "Initializing");
        TIMER_POP(1, "CONDUCTION");
        return;
    }

    Energy = (double *)mymalloc("Energy", N_gas * sizeof(double));
    EnergyOld = (double *)mymalloc("EnergyOld", N_gas * sizeof(double));
    Residual = (double *)mymalloc("Residual", N_gas * sizeof(double));
    DVec = (double *)mymalloc("DVec", N_gas * sizeof(double));
    QVec = (double *)mymalloc("QVec", N_gas * sizeof(double));
#ifdef CONDUCTION_BICGSTAB
    PVec = (double *)mymalloc("PVec", N_gas * sizeof(double));
    SVec = (double *)mymalloc("SVec", N_gas * sizeof(double));
    TVec = (double *)mymalloc("TVec", N_gas * sizeof(double));
#endif // CONDUCTION_BICGSTAB

    Kappa = (MyFloat *)mymalloc("Kappa", N_gas * sizeof(MyFloat));
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    KappaB = (MyFloat *)mymalloc("KappaB", N_gas * sizeof(MyFloat));
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)

    TIMER_PUSH(1, "COND_PRECOMP");

    /* First, let's compute the thermal energies per unit mass and
       conductivities for all particles */
#ifdef _OPENMP
#pragma omp parallel
    {
#endif // _OPENMP

        // these variables are private for all OMP threads
#ifdef CONDUCTION_SATURATION
        double electron_free_path, temp_scale_length;
#endif // CONDUCTION_SATURATION

#ifdef CONDUCTION_INCLUDEMAGNETIC
        double BVal2, mag_factor;
        double gradEnergyVal2, mag_skp, cosAngle;
#endif // CONDUCTION_INCLUDEMAGNETIC

#ifdef _OPENMP
#pragma omp for
#endif // _OPENMP
        for (int i = 0; i < N_gas; i++)
        {
            if (P[i].Type == type_gas && P[i].Mass > 0)
            {
                /* this gives the thermal energy per unit mass for particle i */
                Energy[i] = EnergyOld[i] = SphP[i].Entropy * pow(SphP[i].Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;

#ifdef CONDUCTION_CONSTANT
                Kappa[i] = All.ConductionEfficiency;
#else // CONDUCTION_CONSTANT
            Kappa[i] = All.ConductionEfficiency * pow(Energy[i], 2.5);
#ifdef CONDUCTION_SATURATION
            electron_free_path = All.ElectronFreePathFactor * Energy[i] * Energy[i] / (SphP[i].Density * a3inv);
            temp_scale_length =
                atime * abs(SphP[i].Entropy) /
                sqrt(SphP[i].GradEntr[0] * SphP[i].GradEntr[0] + SphP[i].GradEntr[1] * SphP[i].GradEntr[1] +
                     SphP[i].GradEntr[2] * SphP[i].GradEntr[2]);
            Kappa[i] /= (1 + 4.2 * electron_free_path / temp_scale_length);
#endif // CONDUCTION_SATURATION
#endif // CONDUCTION_CONSTANT

#ifdef CONDUCTION_INCLUDEMAGNETIC
                BVal2 = SphP[i].BPred[0] * SphP[i].BPred[0] + SphP[i].BPred[1] * SphP[i].BPred[1] +
                        SphP[i].BPred[2] * SphP[i].BPred[2];

                gradEnergyVal2 = SphP[i].GradEnergy[0] * SphP[i].GradEnergy[0] +
                                 SphP[i].GradEnergy[1] * SphP[i].GradEnergy[1] +
                                 SphP[i].GradEnergy[2] * SphP[i].GradEnergy[2];

                if (BVal2 != 0.0 && gradEnergyVal2 != 0.0)
                {
                    mag_skp = SphP[i].BPred[0] * SphP[i].GradEnergy[0] + SphP[i].BPred[1] * SphP[i].GradEnergy[1] +
                              SphP[i].BPred[2] * SphP[i].GradEnergy[2];

                    cosAngle = mag_skp / sqrt(BVal2 * gradEnergyVal2);
                }
                else
                {
                    cosAngle = 1.0;
                }

#ifdef CONDUCTION_INCLUDEMAGNETIC_EASY
                Kappa[i] *= abs(cosAngle);
#else // CONDUCTION_INCLUDEMAGNETIC_EASY

#ifdef CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE
                if (cosAngle * cosAngle > 0.2)
                {
#endif // CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE

#ifdef CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR
                    mag_factor = All.ConductionPerpendicularConstants * (sqrt(BVal2) * pow(Energy[i], 1.5)) /
                                 SphP[i].Density * atime;

#ifdef CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR_QUADRATIC
                    mag_factor = 1.0 / (1.0 + mag_factor * mag_factor);
#else // CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR_QUADRATIC
                    mag_factor = 1.0 / (1.0 + mag_factor);
#endif // CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR_QUADRATIC

#else // CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR
                mag_factor = 0.0;
#endif // CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR

                    if (BVal2 == 0.0)
                    {
                        KappaB[i] = 0.0;
                    }
                    else
                    {
                        KappaB[i] = Kappa[i] * (1.0 - mag_factor) / BVal2;
                    }
                    Kappa[i] *= mag_factor;

#ifdef CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE
                }
                else
                {
#ifdef CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE_NOCOND
                    Kappa[i] = 0.0;
#endif // CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE_NOCOND
                    KappaB[i] = 0.0;
                }
#endif // CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE

#endif // CONDUCTION_INCLUDEMAGNETIC_EASY
#endif // CONDUCTION_INCLUDEMAGNETIC

#ifdef SFR
                double myPhysDensThresh;
#ifdef LT_STELLAREVOLUTION
                myPhysDensThresh = get_PhysDensThresh(i);
#else // LT_STELLAREVOLUTION
                myPhysDensThresh = All.PhysDensThresh;
#endif // LT_STELLAREVOLUTION

                if (SphP[i].Density * a3inv >= myPhysDensThresh)
                {
                    Kappa[i] = 0.0;
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
                    KappaB[i] = 0.0;
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
                }
#ifdef WINDS
                /* if we have winds, we decouple particles briefly if delaytime>0 */
                if (SphP[i].DelayTime > 0)
                {
                    Kappa[i] = 0.0;
                }
#endif // WINDS
#endif // SFR

                /* we'll factor the timestep into the conductivities, for simplicity */
                Kappa[i] *= dt;
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
                KappaB[i] *= dt;
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
            }
            else
            {
                Kappa[i] = 0.0;
            }
        }
#ifdef _OPENMP
    }
#endif // _OPENMP

    /* Let's start the Conjugate Gradient Algorithm */

    /* Initialization */

    TIMER_POPPUSH(1, "COND_PRECOMP", "COND_MAT_MULT");
    conduction_matrix_multiply(EnergyOld, Residual);
    TIMER_POPPUSH(1, "COND_MAT_MULT", "COND_SUM_UP");

    VERBOSE(2, "Conduction: Finished first matrix multiplication ...\n");

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < N_gas; i++)
    {
        if (P[i].Type == type_gas && P[i].Mass > 0)
        {
            Residual[i] = EnergyOld[i] - Residual[i];
            DVec[i] = Residual[i];

#ifdef CONDUCTION_BICGSTAB
            PVec[i] = QVec[i] = 0.0;
#endif // CONDUCTION_BICGSTAB
        }
        else
        {
            Residual[i] = DVec[i] = QVec[i] = 0.0;
#ifdef CONDUCTION_BICGSTAB
            PVec[i] = 0.0;
            // KD: I think S and T be also zeroed !
            SVec[i] = TVec[i] = 0.0;
#endif // CONDUCTION_BICGSTAB
        }
    }

    TIMER_POP(1, "COND_SUM_UP");

    TIMER_POPPUSH(2, "Initializing", "Itteration");

#ifdef CONDUCTION_BICGSTAB
    delta1 = alpha = gamma = 1.0;
    iter = 0; /* iteration counter */
    glob_max_rel_change = 1 + COND_ITER_ACCURACY; /* to make sure that we enter the iteration */

    while (iter < MAX_COND_ITER && glob_max_rel_change > COND_ITER_ACCURACY)
    {
        TIMER_PUSH(1, "COND_VEC_MULT");
        delta0 = delta1;
        delta1 = conduction_vector_multiply(DVec, Residual);
        beta = delta1 / delta0 * alpha / gamma;
        TIMER_POPPUSH(1, "COND_VEC_MULT", "COND_SUM_UP");
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < N_gas; i++)
        {
            if (P[i].Type == type_gas && P[i].Mass > 0)
            {
                PVec[i] = Residual[i] + beta * (PVec[i] - gamma * QVec[i]);
            }
        }

        TIMER_POPPUSH(1, "COND_SUM_UP", "COND_MAT_MULT");
        conduction_matrix_multiply(PVec, QVec);
        TIMER_POPPUSH(1, "COND_MAT_MULT", "COND_VEC_MULT");
        alpha = delta1 / conduction_vector_multiply(DVec, QVec);
        TIMER_POPPUSH(1, "COND_VEC_MULT", "COND_SUM_UP");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < N_gas; i++)
        {
            if (P[i].Type == type_gas && P[i].Mass > 0)
            {
                SVec[i] = Residual[i] - alpha * QVec[i];
            }
        }

        //! @todo check if SVec is small then set result to Energy + alpha * PVec and quit
        TIMER_POPPUSH(1, "COND_SUM_UP", "COND_MAT_MULT");
        conduction_matrix_multiply(SVec, TVec);
        TIMER_POPPUSH(1, "COND_MAT_MULT", "COND_VEC_MULT");
        gamma = conduction_vector_multiply(TVec, SVec) / conduction_vector_multiply(TVec, TVec);
        TIMER_POPPUSH(1, "COND_VEC_MULT", "COND_SUM_UP");
        loc_max_rel_change = -MAX_REAL_NUMBER;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
            double my_loc_max_rel_change = 0;
#ifdef _OPENMP
#pragma omp for private(rel_change)
#endif // _OPENMP
            for (int i = 0; i < N_gas; i++)
            {
                if (P[i].Type == type_gas && P[i].Mass > 0)
                {
                    Energy[i] += alpha * PVec[i] + gamma * SVec[i];
                    Residual[i] = SVec[i] - gamma * TVec[i];

                    rel_change = (alpha * PVec[i] + gamma * SVec[i]) / Energy[i];

#ifdef CONDUCTION_NOSTOPNEGATIVE
                    if (Energy[i] <= 0.0)
                    {
                        rel_change = 1 + COND_ITER_ACCURACY;
                    }
#endif // CONDUCTION_NOSTOPNEGATIVE

                    if (my_loc_max_rel_change < rel_change)
                    {
                        my_loc_max_rel_change = rel_change;
                    }
                }
            }

            if (my_loc_max_rel_change > loc_max_rel_change)
#ifdef _OPENMP
#pragma omp critical
#endif // _OPENMP
                if (my_loc_max_rel_change > loc_max_rel_change)
                {
                    loc_max_rel_change = my_loc_max_rel_change;
                }
        }

        iter++;

        TIMER_POP(1, "COND_SUM_UP");

        MPI_Allreduce(&loc_max_rel_change, &glob_max_rel_change, 1, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);

        VERBOSE(2, "conduction iter=%d  max-rel-change=%g\n", iter, glob_max_rel_change);
    }
#else // CONDUCTION_BICGSTAB

    TIMER_PUSH(1, "COND_MAT_MULT");
    delta1 = conduction_vector_multiply(Residual, Residual);
    delta0 = delta1;
    TIMER_POP(1, "COND_MAT_MULT");

    VERBOSE(2, "Computed first vector_multiply, delta = %e \n", delta1);

    iter = 0; /* iteration counter */
    glob_max_rel_change = 1 + COND_ITER_ACCURACY; /* to make sure that we enter the iteration */

    while (iter < MAX_COND_ITER && glob_max_rel_change > COND_ITER_ACCURACY && delta1 > 0)
    {
        TIMER_PUSH(1, "COND_MAT_MULT");
        conduction_matrix_multiply(DVec, QVec);
        TIMER_POPPUSH(1, "COND_MAT_MULT", "COND_VEC_MULT");

        VERBOSE(2, "Conduction: Computed matrix_multiply \n");

        double delta_dummy = conduction_vector_multiply(DVec, QVec);
        TIMER_POPPUSH(1, "COND_VEC_MULT", "COND_SUM_UP");

        VERBOSE(2, "Computed vector_multiply, delta = %e \n", delta_dummy);

        alpha = delta1 / delta_dummy;

        loc_max_rel_change = -MAX_REAL_NUMBER;
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
            double my_loc_max_rel_change = 0;
#ifdef _OPENMP
#pragma omp for private(rel_change)
#endif // _OPENMP
            for (int i = 0; i < N_gas; i++)
            {
                if (P[i].Type == type_gas && P[i].Mass > 0)
                {
                    Energy[i] += alpha * DVec[i];
                    Residual[i] -= alpha * QVec[i];

                    rel_change = alpha * DVec[i] / Energy[i];

#ifdef CONDUCTION_NOSTOPNEGATIVE
                    if (Energy[i] <= 0.0)
                    {
                        rel_change = 1 + COND_ITER_ACCURACY;
                    }
#endif // CONDUCTION_NOSTOPNEGATIVE

                    if (my_loc_max_rel_change < rel_change)
                    {
                        my_loc_max_rel_change = rel_change;
                    }
                }
            }

            if (my_loc_max_rel_change > loc_max_rel_change)
#ifdef _OPENMP
#pragma omp critical
#endif // _OPENMP
                if (my_loc_max_rel_change > loc_max_rel_change)
                {
                    loc_max_rel_change = my_loc_max_rel_change;
                }
        }

        TIMER_POPPUSH(1, "COND_SUM_UP", "COND_VEC_MULT");
        delta0 = delta1;
        delta1 = conduction_vector_multiply(Residual, Residual);
        TIMER_POPPUSH(1, "COND_VEC_MULT", "COND_SUM_UP");

        beta = delta1 / delta0;

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < N_gas; i++)
        {
            if (P[i].Type == type_gas && P[i].Mass > 0)
            {
                DVec[i] = Residual[i] + beta * DVec[i];
            }
        }

        iter++;

        TIMER_POP(1, "COND_SUM_UP");

        MPI_Allreduce(&loc_max_rel_change, &glob_max_rel_change, 1, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);

        VERBOSE(2, "conduction iter=%d  delta1=%g delta1/delta0=%g  max-rel-change=%g\n", iter, delta1, delta1 / delta0,
                glob_max_rel_change);
    }
#endif // CONDUCTION_BICGSTAB

    TIMER_POPPUSH(2, "Itteration", "Finalizing");

    TIMER_PUSH(1, "COND_COMP_FINAL");

    /* Now we have the solution vector in Energy[] */
    /* assign it to the entropies, and update the pressure */
    sumnew = sumold = sumtransfer = 0;
    int warnings1 = 0, warnings2 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sumnew) reduction(+ : sumold)                                                   \
    reduction(+ : sumtransfer) private(warnings1, warnings2)
#endif // _OPENMP
    for (int i = 0; i < N_gas; i++)
    {
        if (P[i].Type == type_gas && P[i].Mass > 0)
        {
            sumnew += P[i].Mass * Energy[i];
            sumold += P[i].Mass * EnergyOld[i];
            sumtransfer += P[i].Mass * abs(Energy[i] - EnergyOld[i]);

#if defined(SPH_BND_PARTICLES) && !defined(SPH_BND_PARTICLES_ALLOW_CONDUCTION)
            if (P[i].ID != 0)
#endif // defined(SPH_BND_PARTICLES) && !defined(SPH_BND_PARTICLES_ALLOW_CONDUCTION)
            {
                if (Energy[i] > 0 && Energy[i] < 1e20)
                {
                    SphP[i].Entropy = Energy[i] / pow(SphP[i].Density * a3inv, GAMMA_MINUS1) * GAMMA_MINUS1;
#if GADGET_HYDRO == HYDRO_MFM
                    SphP[i].InternalEnergy = Energy[i];
#endif // GADGET_HYDRO == HYDRO_MFM
                }
                else
                {
                    if (warnings1 == 0)
                    {
                        VERBOSE_ALL(1,
                                    "WARNING ! Conduction: Particle %d with ID %llu got strange Energy, not updating "
                                    "it ! (old entropy %g and new energy %g)\n",
                                    i, (unsigned long long)P[i].ID, SphP[i].Entropy, Energy[i]);
#pragma omp atomic
                        warnings1++;
                    }
                }
                if (SphP[i].Entropy < 0.0)
                {
                    if (warnings2 == 0)
                    {
                        VERBOSE_ALL(1, "Particle %d with ID %llu got negative entropy %g and energy %g\n", i,
                                    (unsigned long long)P[i].ID, SphP[i].Entropy, Energy[i]);
#pragma omp atomic
                        warnings2++;
                    }
                }

#ifdef AA_CONDUCTION_POS
                SphP[i].Entropy = abs(SphP[i].Entropy);
#if GADGET_HYDRO == HYDRO_MFM
                SphP[i].InternalEnergy = fabs(SphP[i].InternalEnergy);
#endif // GADGET_HYDRO == HYDRO_MFM
#endif // AA_CONDUCTION_POS
            }
        }
    }

    TIMER_POP(1, "COND_COMP_FINAL");

    MPI_Allreduce(&sumnew, &sumnew_tot, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&sumold, &sumold_tot, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&sumtransfer, &sumtransfer_tot, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    VERBOSE(4, "\nconduction finished. energy_before=%g energy_after=%g  rel-change=%g rel-transfer=%g\n\n", sumold_tot,
            sumnew_tot, (sumnew_tot - sumold_tot) / sumold_tot, sumtransfer_tot / sumold_tot);

#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    myfree(KappaB);
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    myfree(Kappa);
#ifdef CONDUCTION_BICGSTAB
    myfree(TVec);
    myfree(SVec);
    myfree(PVec);
#endif // CONDUCTION_BICGSTAB
    myfree(QVec);
    myfree(DVec);
    myfree(Residual);
    myfree(EnergyOld);
    myfree(Energy);

    TIMER_POP(2, "Finalizing");
    TIMER_POP(1, "CONDUCTION");
}

/**
 * @brief Computes the dot product of two vectors for conduction calculations.
 *
 * This function computes the dot product of two vectors for conduction calculations.
 * It performs the computation in parallel if OpenMP is enabled and handles special cases
 * such as particles of type 0 with positive mass.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @return The dot product of the two vectors.
 */
double conduction_vector_multiply(double *a, double *b)
{
    double sumall;
    double sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif // _OPENMP
    for (int i = 0; i < N_gas; i++)
    {
        if (P[i].Type == type_gas && P[i].Mass > 0)
        {
            sum += a[i] * b[i];
        }
    }

    MPI_Allreduce(&sum, &sumall, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    return sumall;
}

/**
 * @brief Computes the product of a matrix and a vector for conduction calculations.
 *
 * This function computes the product of a matrix and a vector for conduction calculations.
 * It handles communication and parallel computation, including utilizing OpenACC if available.
 * The function also performs secondary phases of conduction calculations.
 *
 * @param in Pointer to the input vector.
 * @param out Pointer to the output vector.
 */
void conduction_matrix_multiply(double *in, double *out)
{
    int ndone;
    int NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    double *sum;

    /* allocate buffers to arrange communication */

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer;", NTaskTimesNumPart * sizeof(int));

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize = (int)((MyBufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct conductiondata_in) +
                           sizeof(struct conductiondata_out) +
                           sizemax(sizeof(struct conductiondata_in), sizeof(struct conductiondata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "Conduction: using %g MB for buffering, %g MB remaining for other buffers\n", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    sum = (double *)mymalloc("sum", N_gas * sizeof(double));

    NextParticle = 0;
    ////////////////
#ifdef ACC_CONDUCTION
    TIMER_PUSH(1, "ACC_INIT");
    struct conductiondata_in *LiteCIn;
    struct conductiondata_out *LiteCOut;
    int *ConductionActiveParticleList;
    int NConductionActivePart;
    int should_drift = 0;
    int DoPrimaryOnGPU = 0;
    NConductionActivePart = openacc_conduction_init(should_drift, DoPrimaryOnGPU, ConductionActiveParticleList,
                                                    LiteCOut, LiteCIn, out, sum);
    TIMER_POPPUSH(1, "ACC_INIT", "ACC_UPLOAD");
    if (DoPrimaryOnGPU)
    {
#if defined(ACC_OPENACC)
#pragma acc enter data create(ConductionActiveParticleList [0:NConductionActivePart],                                  \
                              LiteCIn [0:NConductionActivePart], LiteCOut [0:NConductionActivePart],                   \
                              COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas])
#pragma acc update device(ConductionActiveParticleList [0:NConductionActivePart], LiteCIn [0:NConductionActivePart],   \
                          LiteCOut [0:NConductionActivePart], COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas])               \
    ACC_ASYNC_COPY1
#elif defined(ACC_OPENMP)
#pragma omp target enter data map(alloc                                                                                \
                                  : ConductionActiveParticleList [0:NConductionActivePart],                            \
                                    LiteCIn [0:NConductionActivePart], LiteCOut [0:NConductionActivePart],             \
                                    COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas])

#pragma omp target update to(ConductionActiveParticleList [0:NConductionActivePart],                                   \
                             LiteCIn [0:NConductionActivePart], LiteCOut [0:NConductionActivePart],                    \
                             COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas])

#endif // ACC_OPENACC || ACC_OPENMP
    }
    {

        TIMER_POPPUSH(1, "ACC_UPLOAD", "ACC_I");
        openacc_conduction_primary_phase(DoPrimaryOnGPU, NConductionActivePart, ConductionActiveParticleList, LiteCOut,
                                         LiteCIn, in, Kappa, KAPPAB_POINTER);
        TIMER_POP(1, "ACC_I");
#else // ACC_CONDUCTION
    int DoPrimaryOnGPU = 0;
    int should_drift = 1;
#endif // ACC_CONDUCTION

        int group_particles = 1;
        do /* communication loop */
        {
            TIMER_PUSH(2, "Primary");
            TIMER_PUSH(1, "COND_COMPUTE");

            BufferFullFlag = 0;
            Nexport = 0;
            int save_NextParticle = NextParticle;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
                struct thread_params threadparams;
#ifdef _OPENMP
                threadparams.tid = omp_get_thread_num();
#else // _OPENMP
            threadparams.tid = 0;
#endif // _OPENMP
                threadparams.in = in;
                threadparams.out = out;
                threadparams.sum = sum;

                ar_green_evaluate_primary(conduction_isactive, conduction_evaluate, Module_Conduction,
                                          TreeWalkType_GreenTreeWalk, NextParticleLoop_ZeroToNgas,
                                          NeighbourSearchCriteria_Hydro, &threadparams, DoPrimaryOnGPU, should_drift,
                                          group_particles);
            }

#ifdef ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
            TIMER_POPPUSH(1, "COND_COMPUTE", "COND_WAIT");
            MPI_Barrier(MYMPI_COMM_WORLD); // Wait for all tasks to synchronize
            TIMER_POP(1, "COND_WAIT");
#else // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
        TIMER_POP(1, "COND_COMPUTE");
#endif // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
            TIMER_POPPUSH(2, "Primary", "Exchange");
            TIMER_PUSH(1, "COND_COMM_PREP");

            int panic_on_error = !group_particles; /* if we are not grouping particles and SendRecv fails, then we panic
                                                      as there is no memory to process even a single particles */
            int is_send_receive_ok = prepare_sendrec_offset_send_conduction(save_NextParticle, panic_on_error);

            TIMER_POP(1, "COND_COMM_PREP");

            if (!is_send_receive_ok)
            {
                TIMER_POP(2, "Exchange");
                group_particles = 0; /* if green tree fails, we retry without it */
                continue;
            }

            TIMER_PUSH(1, "COND_COPY_IN");

            ConductionDataGet =
                (struct conductiondata_in *)mymalloc("ConductionDataGet", Nimport * sizeof(struct conductiondata_in));
            DataNodeListGet =
                (struct data_nodelist *)mymalloc("DataNodeListGet", Nimport * sizeof(struct data_nodelist));
            ConductionDataIn =
                (struct conductiondata_in *)mymalloc("ConductionDataIn", Nexport * sizeof(struct conductiondata_in));
            DataNodeListIn = (struct data_nodelist *)mymalloc("DataNodeListIn", Nexport * sizeof(struct data_nodelist));
            /* prepare particle data for export */

            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                particle2in_conduction(&ConductionDataIn[j], place);
            }

            copy_data_nodelist();

            TIMER_POPPUSH(1, "COND_COPY_IN", "COND_COMM_EXC");
            templated_exchange_data_in_get(ConductionDataIn, ConductionDataGet, 1, TAG_HYDRO_A);
            TIMER_POP(1, "COND_COMM_EXC");

            TIMER_POPPUSH(2, "Exchange", "Secondary");

            TIMER_PUSH(1, "COND_COMPUTE");

            myfree(DataNodeListIn);
            myfree(ConductionDataIn);
            ConductionDataResult = (struct conductiondata_out *)mymalloc("ConductionDataResult",
                                                                         Nimport * sizeof(struct conductiondata_out));
            ConductionDataOut =
                (struct conductiondata_out *)mymalloc("ConductionDataOut", Nexport * sizeof(struct conductiondata_out));

            /* now do the particles that were sent to us */

            NextJ = 0;

#ifdef ACC_CONDUCTION
            openacc_conduction_secondary_phase(ConductionDataGet, ConductionDataResult, LiteCIn, DoPrimaryOnGPU,
                                               should_drift, in, Kappa, KAPPAB_POINTER, out, sum);
#else // ACC_CONDUCTION

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
            struct thread_params threadparams;
#ifdef _OPENMP
            threadparams.tid = omp_get_thread_num();
#else // _OPENMP
            threadparams.tid = 0;
#endif // _OPENMP

            threadparams.in = in;
            threadparams.out = out;
            threadparams.sum = sum;

            ar_green_evaluate_secondary(ConductionDataGet, conduction_evaluate, Module_Conduction,
                                        TreeWalkType_GreenTreeWalk, NeighbourSearchCriteria_Hydro, &threadparams,
                                        should_drift);
        }
#endif // ACC_CONDUCTION

            TIMER_POPPUSH(1, "COND_COMPUTE", "COND_WAIT");

            int ndone_flag;
            if (NextParticle < N_gas)
            {
                ndone_flag = 0;
            }
            else
            {
                ndone_flag = 1;
            }

            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

            TIMER_POP(1, "COND_WAIT");

            TIMER_POPPUSH(2, "Secondary", "Send_Results");

            TIMER_PUSH(1, "COND_COMM_EXC");
            templated_exchange_data_Sendrecv(ConductionDataResult, ConductionDataOut, TAG_HYDRO_B, Recv_offset,
                                             Recv_count, Send_offset, Send_count);
            TIMER_POPPUSH(1, "COND_COMM_EXC", "COND_COPY");

            /* add the result to the local particles */
            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                out2particle_conduction(&ConductionDataOut[j], place, 1);
            }

            TIMER_POP(1, "COND_COPY");
            TIMER_POP(2, "Send_Results");

            myfree(ConductionDataOut);
            myfree(ConductionDataResult);
            myfree(DataNodeListGet);
            myfree(ConductionDataGet);

        } while (ndone < NTask);
#ifdef ACC_CONDUCTION
        TIMER_PUSH(1, "ACC_WAIT");
        if (DoPrimaryOnGPU)
        {
#ifdef ACC_OPENACC
#pragma acc wait(1)
#elif defined(ACC_OPENMP)
#pragma omp taskwait
#endif
            TIMER_POPPUSH(1, "ACC_WAIT", "ACC_EXIT");

#if defined(ACC_OPENACC)
#pragma acc update host(LiteCOut [0:NConductionActivePart])

#pragma acc exit data delete (ConductionActiveParticleList [0:NConductionActivePart],                                  \
                              LiteCIn [0:NConductionActivePart], LiteCOut [0:NConductionActivePart],                   \
                              COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target update from(LiteCOut [0:NConductionActivePart])
#pragma omp target exit data map(delete                                                                                \
                                 : ConductionActiveParticleList [0:NConductionActivePart],                             \
                                   LiteCIn [0:NConductionActivePart], LiteCOut [0:NConductionActivePart],              \
                                   COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas])
#endif

            TIMER_POP(1, "ACC_EXIT");
        } // DoPrimaryOnGPU?
    } // and data

    TIMER_PUSH(1, "ACC_MERGE");

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int i = 0; i < N_gas; i++)
        {
            out2particle_conduction(&LiteCOut[i], i, 1);
        }
    }
    myfree(LiteCOut);
    myfree(ConductionActiveParticleList);
    myfree(LiteCIn);
    TIMER_POP(1, "ACC_MERGE");
#endif // ACC_CONDUCTION
    TIMER_PUSH(2, "Final_Operations");
    TIMER_PUSH(1, "COND_COMPUTE_FINAL");
    /* do final operations  */
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < N_gas; i++)
    {
        if (P[i].Type == type_gas && P[i].Mass > 0)
        {
            out[i] += in[i] * (1 + sum[i]);
        }
    }

    TIMER_POP(1, "COND_COMPUTE_FINAL");
    TIMER_POP(2, "Final_Operations");

    myfree(sum);
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(RecyclingBuffer);
    myfree(Ngblist);
}
#endif // ACC_INCLUDE_CONDUCTION

/* this function evaluates parts of the matrix A */
// int density_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node, int
// i_node)
#ifdef ACC_INCLUDE_CONDUCTION
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluate the contribution of neighboring particles to thermal conduction for a target particle (local
 * version).
 *
 * This function computes the contribution of neighboring particles to the thermal conduction for a given target
 * particle. It considers both thermal and magnetic conductivity if applicable. The function iterates over neighboring
 * particles, calculates their contribution based on their distance and properties, and accumulates the results for the
 * target particle.
 *
 * @param target Index of the target particle for which conduction is being evaluated.
 * @param p_target Index of the target particle (GPU processing).
 * @param mode Computation mode indicating the context of the computation.
 * @param in Pointer to an array containing input data.
 * @param Kappa Pointer to an array containing thermal conductivity values for particles.
 * @param KappaB Pointer to an array containing magnetic thermal conductivity values for particles.
 * @param ngblist List of neighboring particles.
 * @param extended_numngb_inbox Number of extended neighboring particles in the communication buffer.
 * @param ConductionDataGet Pointer to data structure containing input data related to conduction.
 * @param ConductionDataResult Pointer to data structure containing output data related to conduction.
 * @param P Pointer to data structure containing gravity-related data.
 * @param SphP Pointer to data structure containing properties of SPH particles.
 * @param node Index of the current node in the tree.
 * @param i_node Index of the current node in the tree (local to each thread).
 * @return Returns 0 upon successful completion.
 */
int conduction_evaluate_local(int target, int p_target, int mode, double *in, MyFloat *Kappa, MyFloat *KappaB,
                              int *ngblist, int extended_numngb_inbox, struct conductiondata_in *ConductionDataGet,
                              struct conductiondata_out *ConductionDataResult, struct gravity_walkdata *P,
                              struct conductiondata_in *SphP, int node, int i_node)
#else // ACC_INCLUDE_CONDUCTION
/**
 * @brief Evaluates the contribution of neighboring particles to conduction for a given target particle.
 *
 * This function evaluates the contribution of neighboring particles to conduction for a given target particle.
 * It computes the heat transfer between particles according to the conduction equation kernel.
 *
 * @param target Index of the target particle.
 * @param mode Computation mode.
 * @param ngblist List of neighboring particles.
 * @param extended_numngb_inbox Number of extended neighboring particles.
 * @param i_startnode Index of the starting node in the tree.
 * @param node Index of the current node in the tree.
 * @param i_node Index of the current node in the tree (local to each thread).
 * @param data Pointer to additional data.
 * @return Status code indicating success or failure.
 */
int conduction_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                        int i_node, void *data)
#endif // ACC_INCLUDE_CONDUCTION
{
#ifdef ACC_INCLUDE_CONDUCTION
    int startnode;
    // so far, conduction has all type0 particle, so ActiveParticleList = P[:N_gas] and p_target = active_target.
#else // ACC_INCLUDE_CONDUCTION
    double *in = ((struct thread_params *)data)->in;
    double *out = ((struct thread_params *)data)->out;
    double *sum = ((struct thread_params *)data)->sum;
    int startnode;
#endif // ACC_INCLUDE_CONDUCTION

    MyLongDouble *pos;
    MyFloat h_j, h_i, rho;
    MyAtLeastDouble hinv_i, hinv4_i, hinv_j, hinv3_i;
    MyAtLeastDouble dwk_i, dwk_j, dwk, wk_i;
    double Kappa_i, kappa_mean, w;

#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    double KappaB_i, b_skp_i, b_skp_j;
    MyFloat *B;
#ifndef CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED
    double b2;
#endif // CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)

    if (mode == 0 || mode == 2 || mode == 3)
    {
        pos = P[target].Pos;
        h_i = P[target].Hsml;
        rho = SphP[target].Density;
        Kappa_i = Kappa[target];
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
        KappaB_i = KappaB[target];
        B = SphP[target].BPred;
#endif //  defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
       // #endif //ACC_INCLUDE_CONDUCTION
    }
    else
    {
        struct conductiondata_in *in = &ConductionDataGet[target];
        pos = in->Pos;
        h_i = in->Hsml;
        rho = in->Density;
        Kappa_i = in->Kappa;
#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
        KappaB_i = in->KappaB;
        B = in->B;
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    }

    double h_i2 = h_i * h_i;

    kernel_hinv(h_i, &hinv_i, &hinv3_i, &hinv4_i);

    /* initialize variables before SPH loop is started */
    double out_sum = 0.0;
    double w_sum = 0.0;

    /* Now start the actual SPH computation for this particle */
    startnode = node;

    // this function in mode==1 can be called multiple times for the same particle (with different internal tree-nodes)
    if (i_node > 0)
    {
        out_sum = ConductionDataResult[target].Out;
        w_sum = ConductionDataResult[target].Sum;
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            int numngb = extended_numngb_inbox;
#ifndef ACC_INCLUDE_CONDUCTION
            startnode = *i_startnode;
#else // ACC_INCLUDE_CONDUCTION
            startnode = -1;
#endif // ACC_INCLUDE_CONDUCTION

            if (numngb < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb; n++)
            {
                int j = ngblist[n];

#ifndef ACC_INCLUDE_CONDUCTION
                if (P[j].Type != 0 || j >= N_gas)
                {
                    ENDRUN_MAIN("Error: Conduction got a non gas particle to evaluate. n=%d/%d j=%d/%d N_gas=%d", n,
                                numngb, j, NumPart, N_gas);
                }
#endif // ACC_INCLUDE_CONDUCTION

                double dx = pos[0] - P[j].Pos[0];
                double dy = pos[1] - P[j].Pos[1];
                double dz = pos[2] - P[j].Pos[2];

#ifdef PERIODIC /*  now find the closest image in the given box size  */
                dx = NEAREST_X(dx);
                dy = NEAREST_Y(dy);
                dz = NEAREST_Z(dz);
#endif // PERIODIC
                double r2 = (dx * dx) + (dy * dy) + (dz * dz);

                h_j = P[j].Hsml;

                if (r2 < h_i2 || r2 < h_j * h_j)
                {
                    MyAtLeastDouble r = sqrt(r2);
                    if (r > 0)
                    {
                        if (r2 < h_i2)
                        {
                            MyAtLeastDouble u = r * hinv_i;
                            kernel_main(u, hinv3_i, hinv4_i, &wk_i, &dwk_i, 1);
                        }
                        else
                        {
                            dwk_i = 0;
                        }

                        if (r2 < h_j * h_j)
                        {
                            MyAtLeastDouble hinv3_j, hinv4_j;
                            kernel_hinv(h_j, &hinv_j, &hinv3_j, &hinv4_j);
                            MyAtLeastDouble u = r * hinv_j;
                            MyAtLeastDouble wk_j;
                            kernel_main(u, hinv3_j, hinv4_j, &wk_j, &dwk_j, 1);
                        }
                        else
                        {
                            dwk_j = 0;
                        }

                        dwk = 0.5 * (dwk_i + dwk_j);
                        w = 0.0;

                        /* conduction equation kernel */
                        //                      if((Kappa_i + Kappa[j]) > 0)
                        if ((Kappa_i > 0) && (Kappa[j] > 0))
                        {
                            kappa_mean = 2 * (Kappa_i * Kappa[j]) / (Kappa_i + Kappa[j]);
                            w += 2.0 * P[j].Mass / (rho * SphP[j].Density) * kappa_mean * (-dwk) / r;
                        }

#if defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
                        if ((KappaB_i + KappaB[j]) > 0)
                        {
                            kappa_mean = 2 * (KappaB_i * KappaB[j]) / (KappaB_i + KappaB[j]);
#ifdef CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED
                            // mix of isotropic and anisotropic formulation to ensure 2nd law of thermodynamics
                            b_skp_i = dx * B[0] + dy * B[1] + dz * B[2];
                            b_skp_i = kappa_mean * b_skp_i * b_skp_i;

                            b_skp_j = dx * SphP[j].BPred[0] + dy * SphP[j].BPred[1] + dz * SphP[j].BPred[2];
                            b_skp_j = kappa_mean * b_skp_j * b_skp_j;
#else // CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED
      // fully anisotropic may violate 2nd law of thermodynamics
                            b_skp_i = dx * B[0] + dy * B[1] + dz * B[2];
                            b2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
                            b_skp_i = kappa_mean * (2.5 * b_skp_i * b_skp_i - 0.5 * r2 * b2);

                            b_skp_j = dx * SphP[j].BPred[0] + dy * SphP[j].BPred[1] + dz * SphP[j].BPred[2];
                            b2 = SphP[j].BPred[0] * SphP[j].BPred[0] + SphP[j].BPred[1] * SphP[j].BPred[1] +
                                 SphP[j].BPred[2] * SphP[j].BPred[2];
                            b_skp_j = kappa_mean * (2.5 * b_skp_j * b_skp_j - 0.5 * r2 * b2);
#endif // CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED
                        }
                        if (b_skp_i + b_skp_j != 0)
                        {
                            w += (b_skp_i + b_skp_j) * P[j].Mass * (-dwk) / (rho * SphP[j].Density * r * r2);
                        }
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && !defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
                        out_sum -= w * in[j];
                        w_sum += w;
                    }
                }
            }
        }
    }

    if (mode == 3)
    {
        return 0;
    }
    /* Now collect the result at the right place */
    if (mode == 0 || mode == 2)
    {
#ifdef ACC_INCLUDE_CONDUCTION
        ConductionDataResult[target].Out = out_sum;
        ConductionDataResult[target].Sum = w_sum;
#else // ACC_INCLUDE_CONDUCTION
        out[target] = out_sum;
        sum[target] = w_sum;
#endif // ACC_INCLUDE_CONDUCTION
    }
    else
    {
        ConductionDataResult[target].Out = out_sum;
        ConductionDataResult[target].Sum = w_sum;
    }
    return 0;
}
#endif

#ifdef ACC_INCLUDE_CONDUCTION
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_CONDUCTION

#ifdef ACC_CONDUCTION
#if !defined(ACC_INCLUDE_CONDUCTION) && defined(ACC)
#define ACC_INCLUDE_CONDUCTION 1
#include "conduction.cpp"
#undef ACC_INCLUDE_CONDCUTION
#endif // !defined(ACC_INCLUDE_CONDUCTION)  && defined(ACC)
#endif // ACC_CONDUCTION
