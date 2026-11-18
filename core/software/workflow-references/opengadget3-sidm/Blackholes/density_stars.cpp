/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the functions for calculating the SPH-smoothed stellar density around star particles.
 *
 * @author OpenGadget3-development team, in particular Giuseppe Murante, Klaus Dolag, Luca Sala and Volker Springel.
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
#include "../Hydro/kernel.hpp"
#include "../CodeBase/constants.hpp"

#ifdef GM_STARDENSITY

#define SOLAR_MASS_IN_CODE_UNITS (SOLAR_MASS / All.UnitMass_in_g)

#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#define LOCK_PARTNODEDRIFT
#define UNLOCK_PARTNODEDRIFT

// double GM_STARDENSITY_THRESH=0.004;

// struct kernel_density
// {
//   double dx, dy, dz;
//   double r;
//   double dvx, dvy, dvz;
//   double wk, dwk;
//   double hinv, hinv3, hinv4;
//   double mj_wk, mj_dwk_r;
// };

/*! Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct densdata_in_stars
{
    MyLongDouble Pos[3];
    MyFloat Hsml;
    int NodeList[NODELISTLENGTH];
} *DensDataInStars, *DensDataGetStars;

static struct densdata_out_stars
{
    MyLongDouble Rho;
    MyLongDouble Ngb;
#ifdef EVALPOTENTIAL
    MyLongDouble SmoothPot;
#endif // EVALPOTENTIAL
} *DensDataResultStars, *DensDataOutStars;

void out2particle_density_stars(struct densdata_out_stars *, int, int);
void particle2in_density_stars(struct densdata_in_stars *, int);

/**
 * @brief Copies particle position data into the density input structure for star particles and sets the
 *        smoothing length for star density calculations.
 *
 * This function is responsible for copying the position data of a star particle into a density input
 * structure, which is used in star density calculations. Additionally, it handles the setting of the
 * smoothing length (`Hsml`) for the star particle based on a predefined fixed aperture (if enabled).
 *
 * @param in Pointer to the `densdata_in_stars` structure where the position and smoothing length of
 *                   the star particle will be stored.
 * @param i Index of the star particle in the particle array `P`.
 */
void particle2in_density_stars(struct densdata_in_stars *in, int i)
{
    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
    }

#ifdef GM_STARDENSITY_FIXAPERTURE
    P[i].StarHsml = CM_PER_KPC * GM_STARDENSITY_FIXAPERTURE / All.UnitLength_in_cm;
    if (All.ComovingIntegrationOn)
    {
        P[i].StarHsml *= All.Time / All.HubbleParam;
    }
#endif // GM_STARDENSITY_FIXAPERTURE
    in->Hsml = P[i].StarHsml;
}

/**
 * @brief Updates star particle properties with density calculation results from the density output structure.
 *
 * This function takes the results of density calculations (stored in the `densdata_out_stars` structure)
 * and updates the corresponding properties of the star particle in the particle array `P`.
 *
 * @param[in] out Pointer to the `densdata_out_stars` structure containing the results of the density
 *                calculations, such as the number of neighbors (`Ngb`) and density (`Rho`).
 * @param i Index of the star particle in the particle array `P` that will be updated.
 * @param mode Determines how the data should be updated:
 *                 - Mode 0: Direct assignment of values.
 *                 - Mode 1: Addition of values to the existing properties of the particle.
 */
void out2particle_density_stars(struct densdata_out_stars *out, int i, int mode)
{
    ASSIGN_ADD(P[i].StarNumNgb, out->Ngb, mode);

    if (P[i].Type == 4)
    {
        ASSIGN_ADD(P[i].StarDensity, out->Rho, mode);
#ifdef EVALPOTENTIAL
        ASSIGN_ADD(P[i].SmoothPot, out->SmoothPot, mode);
#endif // EVALPOTENTIAL
    }
}

/**
 * @brief Calculates the SPH-smoothed stellar density for each star particle and determines the smoothing length.
 *
 * This function is a clone of the `density()` function but is specifically designed for star particles.
 * It evaluates the SPH smoothed stellar density for each star particle in the simulation.
 * Additionally, it determines the smoothing length for each particle to ensure an appropriate number of neighbors are
 * considered.
 */
void density_stars(void)
{
    TIMER_PUSH(1, "STARDENSITY");

    MyFloat *Left, *Right;
    int ii, ndone, ndone_flag, npleft, iter = 0;
    int recvTask, place;
    long long ntot;
    double fac;
    int save_NextParticle;
    long long n_exported = 0;
    int redo_particle;

    int numstars = 0, totnumstars = 0;
    int NTaskTimesNumPart;

    /*  check that enough stars do exist for the neighbour search
       (at high redshift this is not true */
    /* I ask for 10 times the number of neighbours */
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Type == 4)
        {
            numstars++;
            if (P[i].StarHsml >= 0.)
            {
                P[i].StarHsml = All.SofteningTable[4];
            }
        }
    }
    MPI_Allreduce(&numstars, &totnumstars, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    if (totnumstars < 10 * All.DesNumNgb)
    {
        VERBOSE(2, " density_stars: not enough star particles for neighbour search (we have %d stars) ", totnumstars);
        TIMER_POP(1, "STARDENSITY");
        return;
    }
    else
    {
        VERBOSE(2, " ...%d stars available ", totnumstars);
    }

    NTaskTimesNumPart = maxThreads * NumPart;
    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    Left = (MyFloat *)mymalloc("Left", NumPart * sizeof(MyFloat));
    Right = (MyFloat *)mymalloc("Right", NumPart * sizeof(MyFloat));

    int i;
#ifdef KD_ACTIVE_PARTICLE_LIST_FOR_OPENMP
#pragma omp parallel for private(i)
    for (int il = 0; il < NActivePart; il++)
    {
        i = ActiveParticleList[il];
#else // KD_ACTIVE_PARTICLE_LIST_FOR_OPENMP
    for (i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // KD_ACTIVE_PARTICLE_LIST_FOR_OPENMP
        if (density_isactive_stars(i))
        {
            Left[i] = Right[i] = 0;
        } // if
    } // for

    /* allocate buffers to arrange communication */

#ifdef KD_BUFFER_MANAGEMENT
    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);
#else // KD_BUFFER_MANAGEMENT
    size_t MyBufferSize = All.BufferSize;
#endif // KD_BUFFER_MANAGEMENT

    All.BunchSize = (int)((MyBufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct densdata_in_stars) +
                           sizeof(struct densdata_out_stars) +
                           sizemax(sizeof(struct densdata_in_stars), sizeof(struct densdata_out_stars))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

#ifdef KD_BUFFER_MANAGEMENT
    VERBOSE(0, "Density_stars: using %g MB for buffering, %g MB remaining for other buffers ", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));
#endif // KD_BUFFER_MANAGEMENT

    double desnumngb = All.DesNumNgb * GM_STARDENSITY_NEIGH;
    double desnumngbdev = desnumngb * All.MaxNumNgbDeviationRel;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {
        NextParticle = FirstActiveParticle; /* begin with this index */

        do
        {
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextParticle = NextParticle;

            TIMER_PUSH(0, "COMPUTE_STARDENS");
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else // _OPENMP
                int mainthreadid = 0;
#endif // _OPENMP
                density_evaluate_primary_stars(&mainthreadid); /* do local particles and prepare export list */
            }
            TIMER_POP(0, "COMPUTE_STARDENS");

            if (BufferFullFlag)
            {
                int last_nextparticle = NextParticle;

                NextParticle = save_NextParticle;

                while (NextParticle >= 0)
                {
                    if (NextParticle == last_nextparticle)
                    {
                        break;
                    }

                    if (ProcessedFlag[NextParticle] != 1)
                    {
                        break;
                    }

                    ProcessedFlag[NextParticle] = 2;

                    NextParticle = NextActiveParticle[NextParticle];
                }

                if (NextParticle == save_NextParticle)
                {
                    /* in this case, the buffer is too small to process even a single particle */
                    VERBOSE_ALL(0, "Task %d: Type=%d pos=(%g,%g,%g) mass=%g ", ThisTask, P[NextParticle].Type,
                                P[NextParticle].Pos[0], P[NextParticle].Pos[1], P[NextParticle].Pos[2],
                                P[NextParticle].Mass);
                    if (P[NextParticle].Type == 0)
                    {
                        VERBOSE(0, "   rho=%g hsml=%g ", P[NextParticle].StarDensity, P[NextParticle].StarHsml);
                    }
                    ENDRUN("Ending as the buffer is too small to process a single particle");
                }

                int new_export = 0;
                int k = 0;
                for (int j = 0; j < Nexport; j++)
                {
                    if (ProcessedFlag[DataIndexTable[j].Index] != 2)
                    {
                        if (k < j + 1)
                        {
                            k = j + 1;
                        }

                        for (; k < Nexport; k++)
                        {
                            if (ProcessedFlag[DataIndexTable[k].Index] == 2)
                            {
                                int old_index = DataIndexTable[j].Index;

                                DataIndexTable[j] = DataIndexTable[k];
                                DataNodeList[j] = DataNodeList[k];
                                DataIndexTable[j].IndexGet = j;
                                new_export++;

                                DataIndexTable[k].Index = old_index;
                                k++;
                                break;
                            }
                        }
                    }
                    else
                    {
                        new_export++;
                    }
                    Nexport = new_export;
                }
            }

            n_exported += Nexport;

            for (int j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
            }

            for (int j = 0; j < Nexport; j++)
            {
                Send_count[DataIndexTable[j].Task]++;
            }
            MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);

            TIMER_PUSH(1, "WAIT");
            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
            TIMER_POP(1, "WAIT");
            Nimport = 0;
            Recv_offset[0] = 0;
            Send_offset[0] = 0;
            for (int j = 0; j < NTask; j++)
            {
                Nimport += Recv_count[j];

                if (j > 0)
                {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }

            DensDataGetStars =
                (struct densdata_in_stars *)mymalloc("DensDataGetStars", Nimport * sizeof(struct densdata_in_stars));
            DensDataInStars =
                (struct densdata_in_stars *)mymalloc("DensDataInStars", Nexport * sizeof(struct densdata_in_stars));

            /* prepare particle data for export */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;

                particle2in_density_stars(&DensDataInStars[j], place);

                memcpy(DensDataInStars[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                       NODELISTLENGTH * sizeof(int));
            }
            /* exchange particle data */
            TIMER_PUSH(1, "BH_COMM");
            templated_exchange_data_Sendrecv(DensDataInStars, DensDataGetStars, TAG_DENS_A, Send_offset, Send_count,
                                             Recv_offset, Recv_count);
            TIMER_POP(1, "BH_COMM");

            myfree(DensDataInStars);
            DensDataResultStars = (struct densdata_out_stars *)mymalloc("DensDataResultStars",
                                                                        Nimport * sizeof(struct densdata_out_stars));
            DensDataOutStars =
                (struct densdata_out_stars *)mymalloc("DensDataOutStars", Nexport * sizeof(struct densdata_out_stars));

            report_memory_usage(&HighMark_sphdensity, "STAR_DENSITY");

            /* now do the particles that were sent to us */

            TIMER_PUSH(1, "COMPUTE");

            NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else // _OPENMP
                int mainthreadid = 0;
#endif // _OPENMP
                density_evaluate_secondary_stars(&mainthreadid);
            }

            TIMER_POP(1, "COMPUTE");

            if (NextParticle < 0)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }

            TIMER_PUSH(1, "WAIT");
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            TIMER_POPPUSH(1, "WAIT", "COMM");
            templated_exchange_data_Sendrecv(DensDataResultStars, DensDataOutStars, TAG_DENS_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POPPUSH(1, "COMM", "COMPUTE");

            /* add the result to the local particles */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;
                out2particle_density_stars(&DensDataOutStars[j], place, 1);
            }
            TIMER_POP(1, "COMPUTE");

            myfree(DensDataOutStars);
            myfree(DensDataResultStars);
            myfree(DensDataGetStars);
        } while (ndone < NTask);

        /* do final operations on results */
        TIMER_PUSH(1, "COMPUTE");

        npleft = 0;

#ifdef KD_ACTIVE_PARTICLE_LIST_FOR_OPENMP
#pragma omp parallel for reduction(+:npleft) private(i,k,desnumngb,desnumngbdev,redo_particle,ii,fac) if(NActivePart>80)
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
#else // KD_ACTIVE_PARTICLE_LIST_FOR_OPENMP
        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
#endif // KD_ACTIVE_PARTICLE_LIST_FOR_OPENMP
            if (density_isactive_stars(i))
            {

                P[i].SmoothPot /= P[i].StarDensity; /* smoothed gravitational potential */

                /* now check whether we had enough neighbours */
                desnumngb = All.DesNumNgb * GM_STARDENSITY_NEIGH;
                desnumngbdev = All.MaxNumNgbDeviation;

                redo_particle = 0;

#ifndef GM_STARDENSITY_FIXAPERTURE /* if this is defined, the stellar Hsml is fixed */
                if (P[i].StarNumNgb < (desnumngb - desnumngbdev) || P[i].StarNumNgb > (desnumngb + desnumngbdev))
                {
                    redo_particle = 1;
                }
#endif // GM_STARDENSITY_FIXAPERTURE

#ifdef MAXHSML
                if (P[i].StarNumNgb < (desnumngb - desnumngbdev) && P[i].StarHsml >= All.MaxHsml)
                {
                    P[i].StarHsml = All.MaxHsml;
                    redo_particle = 0;
                }
#endif // MAXHSML

                if (redo_particle)
                {
                    /* need to redo this particle */
                    npleft++;
                    /* added variable my_continue because the openmp multithreading does not allow "continue" statement
                     * here */
                    short my_continue = 0;

                    if (!my_continue)
                        if (Left[i] > 0 && Right[i] > 0)
                            if ((Right[i] - Left[i]) < 1.0e-3 * Left[i])
                            {
                                /* this one should be ok */
                                npleft--;
                                P[i].TimeBin = -P[i].TimeBin - 1; /* Mark as inactive */
                                my_continue = 1;
                            }
                    if (!my_continue)
                    {
                        if (P[i].StarNumNgb < (desnumngb - desnumngbdev))
                        {
                            Left[i] = max(P[i].StarHsml, Left[i]);
                        }
                        else
                        {
                            if (Right[i] != 0)
                            {
                                if (P[i].StarHsml < Right[i])
                                {
                                    Right[i] = P[i].StarHsml;
                                }
                            }
                            else
                            {
                                Right[i] = P[i].StarHsml;
                            }
                        }
                    }

                    if (!my_continue)
                    {
                        if (iter >= MAXITER - 10)
                        {
                            VERBOSE_ALL(2,
                                        "i=%d task=%d ID=%llu StarHsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   "
                                        "pos=(%g|%g|%g) ",
                                        i, ThisTask, (unsigned long long)P[i].ID, P[i].StarHsml, Left[i], Right[i],
                                        (float)P[i].NumNgb, Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                        }
                    }

                    if (!my_continue)
                    {
                        if (Right[i] > 0 && Left[i] > 0)
                        {
                            P[i].StarHsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
                        }
                        else
                        {
                            if (Right[i] == 0 && Left[i] == 0)
                            {
                                char buf[1000];
                                VERBOSE_ALL(0, "Right[i] == 0 && Left[i] == 0 P[i].StarHsml=%g ", P[i].StarHsml);
                                exit(-11732);
                            }

                            if (Right[i] == 0 && Left[i] > 0)
                            {
                                P[i].StarHsml *= 1.26;
                            }

                            if (Right[i] > 0 && Left[i] == 0)
                            {
                                P[i].StarHsml /= 1.26;
                            }
                        }
                    }

                } // redo particle
                else
                {
                    P[i].TimeBin = -P[i].TimeBin - 1; /* Mark as inactive */
                }
            } // if
        } // for
        TIMER_POP(1, "COMPUTE");

        sumup_large_ints(1, &npleft, &ntot);

        if (ntot > 0)
        {
            iter++;

            if (iter > 0 && ThisTask == 0)
            {
                VERBOSE_ALL(2, "star density ngb iteration %d: need to repeat for %d%09d particles. ", iter,
                            (int)(ntot / 1000000000), (int)(ntot % 1000000000));
            }

            if (iter > MAXITER)
            {
                ENDRUN("Failed to converge in neighbour iteration in density()");
            }
        }
    } while (ntot > 0);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Right);
    myfree(Left);
    myfree(Ngblist);

    /* mark as active again */
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if (P[i].TimeBin < 0)
        {
            P[i].TimeBin = -P[i].TimeBin - 1;
        }
    }
    TIMER_POP(1, "STARDENSITY");
}

/**
 * @brief Computes the stellar density around a target particle using a kernel-based approach.
 *
 * This function is analogous to `density_evaluate`, but specifically for calculating the
 * density of stellar particles around stars. It iterates through neighboring particles
 * within a the smoothing length (`Hsml`) and uses a kernel function to evaluate
 * the local stellar density.
 *
 * @param target       The index of the target particle for which the density is calculated.
 * @param mode         Determines whether the function operates in local mode (0) or non-local
 *                     mode (1). In local mode, data is directly retrieved from the target particle.
 *                     In non-local mode, data is retrieved from the `DensDataGetStars` array.
 * @param exportflag   Array used for marking particles that need to be exported to other processors.
 * @param exportnodecount Array used for counting the number of exported nodes per processor.
 * @param exportindex  Array used for storing the indices of the exported data.
 * @param ngblist      Array containing the indices of neighboring particles within the smoothing length.
 *
 * @return Returns 0 on success or -1 if there is an error during the neighbor search.
 */
int density_evaluate_stars(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
    int startnode, numngb_inbox, listindex = 0;

    struct kernel_density kernel;
    struct densdata_in_stars local;
    struct densdata_out_stars out;

    memset(&out, 0, sizeof(struct densdata_out_stars));

    if (mode == 0)
    {
        particle2in_density_stars(&local, target);
    }
    else
    {
        local = DensDataGetStars[target];
    }

    double h2 = local.Hsml * local.Hsml;

    kernel_hinv(local.Hsml, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = DensDataGetStars[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            /* XXXX controlla che sta roba qui sotto trovi tutto e non solo il  gas */
            numngb_inbox = ngb_treefind_variable_threads_stars(local.Pos, local.Hsml, target, &startnode, mode,
                                                               exportflag, exportnodecount, exportindex, ngblist);

            if (numngb_inbox < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
                int j = ngblist[n];

                kernel.dx = local.Pos[0] - P[j].Pos[0];
                kernel.dy = local.Pos[1] - P[j].Pos[1];
                kernel.dz = local.Pos[2] - P[j].Pos[2];

#ifdef PERIODIC /*  now find the closest image in the given box size  */
                kernel.dx = NEAREST_X(kernel.dx);
                kernel.dy = NEAREST_Y(kernel.dy);
                kernel.dz = NEAREST_Z(kernel.dz);
#endif // PERIODIC
                double r2 = kernel.dx * kernel.dx + kernel.dy * kernel.dy + kernel.dz * kernel.dz;

                if (r2 < h2)
                {
                    kernel.r = sqrt(r2);
                    double u = kernel.r * kernel.hinv;

                    kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, 0);

                    double mass_j = P[j].Mass;

                    kernel.mj_wk = mass_j * kernel.wk;
                    out.Rho += kernel.mj_wk;
                    out.Ngb += NORM_COEFF * kernel.wk / kernel.hinv3; /* NORM_COEFF= 4.0/3 * PI = 4.188790204786 */

#ifdef EVALPOTENTIAL
                    out.SmoothPot += mass_j * kernel.wk * P[j].Potential;
#endif // EVALPOTENTIAL
                }
            }
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = DensDataGetStars[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        out2particle_density_stars(&out, target, 0);
    }
    else
    {
        DensDataResultStars[target] = out;
    }
    return 0;
}

/**
 * @brief Function for evaluating stellar densities.
 *
 * This function is designed to be executed by multiple threads to compute the stellar density
 * around active particles. The function evaluates whether each particle
 * is active, calculates the density using `density_evaluate_stars`, and manages the
 * export of particles to other processors when the local buffer is full.
 *
 * @param p A pointer to the thread ID, which is used to identify and allocate resources
 *          such as neighbor lists and export arrays specific to each thread.
 *
 * @return Returns `NULL` when the function completes execution. This return type is necessary
 *         to comply with the `void *` return type required by threading functions.
 */
void *density_evaluate_primary_stars(void *p)
{
    int thread_id = *(int *)p;
    int i;
    int *exportflag, *exportnodecount, *exportindex, *ngblist;

    ngblist = Ngblist + thread_id * NumPart;
    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    /* Note: exportflag is local to each thread */
    for (int j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }

    while (1)
    {
        int exitFlag = 0;
        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            if (BufferFullFlag != 0 || NextParticle < 0)
            {
                exitFlag = 1;
            }
            else
            {
                i = NextParticle;
                ProcessedFlag[i] = 0;
                NextParticle = NextActiveParticle[NextParticle];
            }
        }
        UNLOCK_NEXPORT;
        if (exitFlag)
        {
            break;
        }

        if (density_isactive_stars(i))
        {
            if (density_evaluate_stars(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0)
            {
                break; /* export buffer has filled up */
            }
        }
        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Secondary function for evaluating stellar densities.
 *
 * This function computes the stellar density
 * around imported particles in a secondary phase of the density calculation. It processes
 * particles in parallel, ensuring that each thread handles its own portion of the workload
 * and evaluates densities for particles that have been imported from other processors.
 *
 * @param p A pointer to the thread ID, which is used to allocate resources
 *          such as the neighbor list specific to each thread.
 *
 * @return Returns `NULL` when the function completes execution.
 */
void *density_evaluate_secondary_stars(void *p)
{
    int thread_id = *(int *)p;
    int j, dummy, *ngblist;
    ngblist = Ngblist + thread_id * NumPart;

    while (1)
    {
        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            j = NextJ;
            NextJ++;
        }
        UNLOCK_NEXPORT;

        if (j >= Nimport)
        {
            break;
        }

        density_evaluate_stars(j, 1, &dummy, &dummy, &dummy, ngblist);
    }

    return NULL;
}

/**
 * @brief Determines if a particle is an active star particle for density calculations.
 *
 * This function checks whether a given particle is active and of the correct type
 * (i.e., a star particle) for the purposes of stellar density evaluation.
 *
 * @param n Index of the particle to be checked.
 *
 * @return Returns `1` if the particle is active and is a star particle (Type 4),
 *         otherwise returns `0`.
 */
int density_isactive_stars(int n)
{
    if (P[n].TimeBin < 0)
    {
        return 0;
    }

    if (P[n].Type == 4)
    {
        return 1;
    }

    return 0;
}

/**
 * @brief Finds neighboring star particles within a specified search radius using a tree-based search algorithm.
 *
 * This function searches for star particles around a given `searchcenter` within a radius defined by `hsml`.
 * It operates using a tree structure, traversing nodes to identify nearby particles and populating a list of neighbors.
 *
 * @param searchcenter Array of three `MyLongDouble` values representing the coordinates of the search center.
 * @param hsml Search radius (smoothing length) for finding neighbors.
 * @param target Index of the target particle for which neighbors are being searched. It can be used to track exports.
 * @param startnode Pointer to the index of the starting node in the tree structure. The function updates this value.
 * @param mode Search mode:
 *             - `0`: Normal search mode.
 *             - `1`: Secondary search mode, typically used during import/export processes.
 * @param exportflag Pointer to an array used to track export tasks for different domains.
 * @param exportnodecount Pointer to an array that tracks the number of nodes exported for each task.
 * @param exportindex Pointer to an array that stores indices for exported data.
 * @param ngblist Pointer to an array where the indices of found neighboring star particles will be stored.
 *
 * @return Returns the number of neighbors found within the specified search radius.
 *         Returns `-1` if the export buffer is full, indicating the need to retry the operation after flushing the
 * buffer.
 */
int ngb_treefind_variable_threads_stars(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                        int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
    int nexp, p, task;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxNodes = MaxNodes;

#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        MyLongDouble dx, dy, dz, dist;
        integertime ti_Current = All.Ti_Current;
        int maxPart = All.MaxPart;

        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

            /* GM: note: this is the only difference with the normal  function
               I prefer to clone it to make of this file a standalone module */

            if (P[p].Type != 4)
            {
                continue;
            }

            if (P[p].Ti_current != ti_Current)
            {
                LOCK_PARTNODEDRIFT;
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                drift_particle(p, ti_Current);
                UNLOCK_PARTNODEDRIFT;
            }

            dist = hsml;
            dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
            if (dz > dist)
            {
                continue;
            }
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            ngblist[numngb++] = p;
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Ending as mode is 1 and no >= maxPart + maxNodes");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (exportnodecount[task] == NODELISTLENGTH)
                    {
                        int exitFlag = 0;
                        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
                        {
                            if (Nexport >= All.BunchSize)
                            {
                                /* out of buffer space. Need to discard work for this particle and interrupt */
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport++;
                            }
                        }
                        UNLOCK_NEXPORT;
                        if (exitFlag)
                        {
                            return -1;
                        }

                        exportnodecount[task] = 0;
                        exportindex[task] = nexp;
                        DataIndexTable[nexp].Task = task;
                        DataIndexTable[nexp].Index = target;
                        DataIndexTable[nexp].IndexGet = nexp;
                    }

                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                        DomainNodeIndex[no - (maxPart + maxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                    }
                }

                no = Nextnode[no - maxNodes];
                continue;
            }

            current = &Nodes[no];

            if (mode == 1)
            {
                if (current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }

            if (current->Ti_current != ti_Current)
            {
                LOCK_PARTNODEDRIFT;
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                force_drift_node(no, ti_Current);
                UNLOCK_PARTNODEDRIFT;
            }

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            no = current->u.d.sibling; /* in case the node can be discarded */

            dist = hsml + 0.5 * current->len;
            dx = NGB_PERIODIC_LONG_X(current->center[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            dy = NGB_PERIODIC_LONG_Y(current->center[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            dz = NGB_PERIODIC_LONG_Z(current->center[2] - searchcenter[2]);
            if (dz > dist)
            {
                continue;
            }
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}

/**
 * @brief Repositions a black hole based on the maximum stellar density in its vicinity.
 *
 * This function adjusts the position of a black hole  to a new location where the stellar density is highest.
 * It performs repositioning based on a predefined set of conditions and options, which may include shifting
 * the black hole's position according to specific formulas or maintaining its current position if no valid data is
 * present.
 *
 * @param n Index of the black hole particle to be repositioned.
 */
void blackhole_stardens_reposition(int n)
{
    double pbh[3], stardens_in_msun_per_pc3;
#ifdef GM_USE_STARDENS_AND_POTMIN
    double dd1, dd2;
#endif // GM_USE_STARDENS_AND_POTMIN
#ifdef GM_BHMOVE_WT13
    double dxvers, dyvers, dzvers, vmod, rr, shift1, shift2, dt_drift;
#endif // GM_BHMOVE_WT13

    if (BPP(n).BH_StarDens > 0 &&
        P[n].Mass > 0.) /* second condition for the (unlikely) case when BH has no valid star particle neighbour */
    { /* in that case, it does not move */

        /* converting stardensity in Msun/pc^3 */
        stardens_in_msun_per_pc3 = BPP(n).BH_StarDens / pow(All.UnitLength_in_cm, 3); // codemass/cm3
        stardens_in_msun_per_pc3 *= CM_PER_PC * CM_PER_PC * CM_PER_PC / SOLAR_MASS_IN_CODE_UNITS; // Msun/pc^3
        if (All.ComovingIntegrationOn) // to physical
            stardens_in_msun_per_pc3 *= All.HubbleParam * All.HubbleParam / pow(All.Time, 3);

        if (stardens_in_msun_per_pc3 >
            GM_STARDENSITY_THRESH) /* the BH only is repositioned if max star density is above threshold, i.e. it does
                                      not move if it is in an underdense environment */
        {
#ifdef GM_BHMOVE_WT13 // shifts BHs according to formula (21) in  Wurster & Thacker 2013
            rr = (P[n].Pos[0] - BPP(n).BH_StarDensPos[0]) * (P[n].Pos[0] - BPP(n).BH_StarDensPos[0]) +
                 (P[n].Pos[1] - BPP(n).BH_StarDensPos[1]) * (P[n].Pos[1] - BPP(n).BH_StarDensPos[1]) +
                 (P[n].Pos[2] - BPP(n).BH_StarDensPos[2]) * (P[n].Pos[2] - BPP(n).BH_StarDensPos[2]);
            dxvers = (BPP(n).BH_StarDensPos[0] - P[n].Pos[0]) / rr;
            dyvers = (BPP(n).BH_StarDensPos[1] - P[n].Pos[1]) / rr;
            dzvers = (BPP(n).BH_StarDensPos[2] - P[n].Pos[2]) / rr;

            vmod = sqrt(P[n].Vel[0] * P[n].Vel[0] + P[n].Vel[1] * P[n].Vel[1] + P[n].Vel[2] * P[n].Vel[2]);
            if (All.ComovingIntegrationOn)
            {
                dt_drift = get_drift_factor(P[n].Ti_current, All.Ti_Current);
            }
            else
            {
                dt_drift = (All.Ti_Current - P[n].Ti_current) * All.Timebase_interval;
            }

            shift1 = 0.01 * All.SofteningTable[5];
            shift2 = 0.03 * vmod * dt_drift;
            if (shift1 <= shift2)
            {
                pbh[0] = P[n].Pos[0] + shift1 * dxvers;
                pbh[1] = P[n].Pos[1] + shift1 * dyvers;
                pbh[2] = P[n].Pos[2] + shift1 * dzvers;
            }
            else
            {
                pbh[0] = P[n].Pos[0] + shift2 * dxvers;
                pbh[1] = P[n].Pos[1] + shift2 * dyvers;
                pbh[2] = P[n].Pos[2] + shift2 * dzvers;
            }
        }
#else // GM_BHMOVE_WT13
            for (int k = 0; k < 3; k++)
            {
                pbh[k] = BPP(n).BH_StarDensPos[k];
            }
        }
#endif // GM_BHMOVE_WT13

        for (int k = 0; k < 3; k++)
        {
            P[n].Pos[k] = pbh[k];
#ifdef GM_USE_VELS_ON_REPOSITIONING
            P[n].Vel[k] = BPP(n).BH_StarDensVel[k];
#endif // GM_USE_VELS_ON_REPOSITIONING
        }

    } // ends the density threshold if
}

#endif // GM_STARDENSITY

/********************** Config Options **************************************************
#--------------------------------------- BH merger fix switches
GM_USE_ABSVAL_IN_VREL      #BH mergers only happen if their relative vel is smaller than
                         #All.BlackHoleMergeCsndFrac set in the param file. In this case
                         #SOUND speed is NOT used

#--------------------------------------- Stellar density switches
GM_STARDENSITY                  #master switch for the evaluation of stellar density around stars
#GM_STARDENSITY_FIXAPERTURE=15   #if set, a PHYSICAL radius is used for the stellar density
                              #otherwise, the a fixed number of stars is used.
                              #in output you have the number of neighbours (1st case) or the stellar
                              #hsml (2nd). You always have RHOS (rho star) in output
GM_STARDENSITY_NEIGH=0.5          #number of star neighbour for the above case, WITH RESCPECT TO THE GAS NEIGHBOURS
                              # maybe <1. e.g. if 200 Ngb for gas and this is 0.5, will require 100 stars

GM_REPOSITION_ON_STARDENSITY_MAX #master switch for continuosly pinning BHs on the top of the
                               #star particle having the maximum stellar density
                               #note that this requires All.BlackHoleRepositioningOn set to 1 in parameter file
GM_STARDENSITY_THRESH=0.004      #density threshold for repositioning BHs (msun/pc^3)
                               #Bovi 2017: in solar neighbourhood is 0.04 Msun/pc^3. This  is one tenth of solar
GM_VREL_REPOSITION=60            #relative velocity threshold for repositioning BHs (km/s)
#GM_BHMOVE_WT13                   #alternative repositioning as in Wurster & Thacker 2013
GM_USE_VELS_ON_REPOSITIONING     #the BH velocity is set to the star particle velocity upon which the BH moves
**********************************End of Config options*********************************/
