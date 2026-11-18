/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Calculation of spherical overdensity (SO) radius and mass.
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
#include <sys/stat.h>
#include <sys/types.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/communication.hpp"

#ifdef SUBFIND

#include "fof.hpp"
#include "subfind.hpp"

#define LOCK_NEXPORT
#define UNLOCK_NEXPORT

/**
 * @brief Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct SOdens_in
{
    MyLongDouble Pos[3];
    MyOutputFloat R200;
    int NodeList[NODELISTLENGTH];
} *SOdensIn, *SOdensGet;

static struct SOdens_out
{
    MyOutputFloat Mass;
#ifdef SO_VEL_DISPERSIONS
    MyOutputFloat Vx200, Vy200, Vz200, Disp200;
#endif // SO_VEL_DISPERSIONS
#ifdef SO_BAR_INFO
    MyOutputFloat gas_mass, star_mass, temp, xlum, ygas;
#endif // SO_BAR_INFO
} *SOdensResult, *SOdensOut;

static MyOutputFloat *R200, *M200;

#ifdef SO_VEL_DISPERSIONS
static MyOutputFloat *Vx200, *Vy200, *Vz200, *Disp200;
#endif // SO_VEL_DISPERSIONS

#ifdef SO_BAR_INFO
static MyOutputFloat *gas_mass, *star_mass, *temp, *xlum, *ygas;
#endif // SO_BAR_INFO

struct dens_eval_arg_t
{
    int threadid;
    char *todo;
    MyFloat *left, *right;
};

/**
 * @brief This function calculates the spherical overdensity (SO) radius and mass for groups of particles.
 *
 * This function is part of the Subfind algorithm, used to identify substructures within halos.
 * It computes various overdensity thresholds (e.g., 200, 500) with respect to both the critical density and the
 * background density. The function utilizes parallel processing and MPI for communication between tasks.
 */
void subfind_overdensity(void)
{
    long long ntot;
    int i, k, ndone, ndone_flag, npleft, iter;
    int place;
#ifdef SO_BAR_INFO
    double Deltas[6];
    double DeltaMean500, DeltaCrit500, DeltaCrit2500;
#else // SO_BAR_INFO
    double Deltas[3];
#endif // SO_BAR_INFO
    double rguess, overdensity, rhoback, z, omegaz, x, DeltaMean200, DeltaCrit200, DeltaTopHat;

    int save_NextGroup;

    struct dens_eval_arg_t dens_eval_arg;

    /* allocate buffers to arrange communication */

    int NTaskTimesNumPart;

    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));

    VERBOSE(2, "NumPartNgroups %d\t%d\n", NumPart, Ngroups);

    dens_eval_arg.left = (MyFloat *)mymalloc("Left", sizeof(MyFloat) * Ngroups);
    dens_eval_arg.right = (MyFloat *)mymalloc("Right", sizeof(MyFloat) * Ngroups);
    R200 = (MyOutputFloat *)mymalloc("R200", sizeof(MyOutputFloat) * Ngroups);
    M200 = (MyOutputFloat *)mymalloc("M200", sizeof(MyOutputFloat) * Ngroups);
#ifdef SO_VEL_DISPERSIONS
    Vx200 = (MyOutputFloat *)mymalloc("Vx200", sizeof(MyOutputFloat) * Ngroups);
    Vy200 = (MyOutputFloat *)mymalloc("Vy200", sizeof(MyOutputFloat) * Ngroups);
    Vz200 = (MyOutputFloat *)mymalloc("Vz200", sizeof(MyOutputFloat) * Ngroups);
    Disp200 = (MyOutputFloat *)mymalloc("Disp200", sizeof(MyOutputFloat) * Ngroups);
#endif // SO_VEL_DISPERSIONS
#ifdef SO_BAR_INFO
    gas_mass = (MyOutputFloat *)mymalloc("GasMass", sizeof(MyOutputFloat) * Ngroups);
    star_mass = (MyOutputFloat *)mymalloc("GasMass", sizeof(MyOutputFloat) * Ngroups);
    temp = (MyOutputFloat *)mymalloc("Temperature", sizeof(MyOutputFloat) * Ngroups);
    xlum = (MyOutputFloat *)mymalloc("XrayLum", sizeof(MyOutputFloat) * Ngroups);
    ygas = (MyOutputFloat *)mymalloc("YGas", sizeof(MyOutputFloat) * Ngroups);
#endif // SO_BAR_INFO

    dens_eval_arg.todo = (char *)mymalloc("Todo", sizeof(char) * Ngroups);

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize = (int)((MyBufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct SOdens_in) +
                           sizeof(struct SOdens_out) + sizemax(sizeof(struct SOdens_in), sizeof(struct SOdens_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "Subfind_SO: using %g MB for buffering, %g MB remaining for other buffers\n", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    if (All.ComovingIntegrationOn)
    {
        z = 1 / All.Time - 1;
    }
    else
    {
        z = 0;
    }

    rhoback = 3 * All.Omega0 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);

    omegaz = All.Omega0 * pow(1 + z, 3) /
             (All.Omega0 * pow(1 + z, 3) + OMEGAR * pow(1 + z, 4) + OMEGAK * pow(1 + z, 2) + All.OmegaLambda);

    DeltaMean200 = 200.0;
    DeltaCrit200 = 200.0 / omegaz;
#ifdef SO_BAR_INFO
    DeltaMean500 = 500.0;
    DeltaCrit500 = 500.0 / omegaz;
    DeltaCrit2500 = 2500.0 / omegaz;
#endif // SO_BAR_INFO

    x = omegaz - 1;
    DeltaTopHat = 18 * M_PI * M_PI + 82 * x - 39 * x * x;
    DeltaTopHat /= omegaz;

#ifndef SO_BAR_INFO
    Deltas[0] = DeltaMean200; /* standard fixed overdensity with respect to background */
    Deltas[1] = DeltaTopHat; /* tophat overdensity with respect to background */
    Deltas[2] = DeltaCrit200; /* overdensity of 200 relative to critical, expressed relative to background density */
#else // SO_BAR_INFO
    Deltas[0] = DeltaTopHat; /* tophat overdensity with respect to background */
    Deltas[1] = DeltaMean200; /* 200 with respect to background */
    Deltas[2] = DeltaMean500; /* 500 with respect to background */
    Deltas[3] = DeltaCrit200; /* 200 with respect to critical */
    Deltas[4] = DeltaCrit500; /* 500 with respect to critical */
    Deltas[5] = DeltaCrit2500; /* 2500 with respect to critical */
#endif // SO_BAR_INFO

#ifdef SO_BAR_INFO
    for (int rep = 0; rep < 6; rep++) /* repeat for all six overdensity values */
#else // SO_BAR_INFO
    for (int rep = 0; rep < 3; rep++) /* repeat for all three overdensity values */
#endif // SO_BAR_INFO
    {
        TIMER_PUSH(4, "SO_Radius_Calculation");

        for (i = 0; i < Ngroups; i++)
        {
            if (Group[i].Nsubs > 0)
            {
                rguess = pow(All.GravityConstantInternal * Group[i].Mass / (100 * All.Hubble * All.Hubble), 1.0 / 3);

                dens_eval_arg.right[i] = 3 * rguess;
                dens_eval_arg.left[i] = 0;

                dens_eval_arg.todo[i] = 1;
            }
            else
            {
                dens_eval_arg.todo[i] = 0;
            }
        }

        iter = 0;

        /* we will repeat the whole thing for those groups where we didn't converge to a SO radius yet */
        do
        {
            double t0 = second();
            TIMER_PUSH(4, "SO_iteration");
            NextGroup = 0; /* begin with this index */

            do
            {
                BufferFullFlag = 0;
                Nexport = 0;
                save_NextGroup = NextGroup;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
                {
#ifdef _OPENMP
                    int mythreadid = omp_get_thread_num();
#else // _OPENMP
                    int mythreadid = 0;
#endif // _OPENMP
                    subfind_overdensity_evaluate_primary(mythreadid, dens_eval_arg.todo, dens_eval_arg.left,
                                                         dens_eval_arg.right);
                }

                if (BufferFullFlag)
                {
                    int last_nextGroup = NextGroup;

                    NextGroup = save_NextGroup;

                    while (NextGroup < Ngroups)
                    {
                        if (NextGroup == last_nextGroup)
                        {
                            break;
                        }

                        // ProcessedFlag -> size NumPart, not Ngroups, but Ngroups<NumPart
                        if (ProcessedFlag[NextGroup] != 1)
                        {
                            break;
                        }

                        ProcessedFlag[NextGroup] = 2;

                        NextGroup += 1;
                    }

                    ENDRUN_IF(NextGroup == save_NextGroup, "Task %d: pos=(%g,%g,%g) mass=%g\n", ThisTask,
                              Group[NextGroup].Pos[0], Group[NextGroup].Pos[1], Group[NextGroup].Pos[2],
                              Group[NextGroup].Mass);

                    int new_export = 0;
                    k = 0;
                    for (int j = 0; j < Nexport; j++)
                        if (ProcessedFlag[DataIndexTable[j].Index] != 2)
                        {
                            if (k < j + 1)
                            {
                                k = j + 1;
                            }

                            for (; k < Nexport; k++)
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
                        else
                        {
                            new_export++;
                        }

                    Nexport = new_export;
                }

                for (int j = 0; j < NTask; j++)
                {
                    Send_count[j] = 0;
                }
                for (int j = 0; j < Nexport; j++)
                {
                    Send_count[DataIndexTable[j].Task]++;
                }

                STD_SORT(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);

                MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

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

                SOdensGet = (struct SOdens_in *)mymalloc("	      SOdensGet", Nimport * sizeof(struct SOdens_in));
                SOdensIn = (struct SOdens_in *)mymalloc("	      SOdensIn", Nexport * sizeof(struct SOdens_in));

                /* prepare particle data for export */
                for (int j = 0; j < Nexport; j++)
                {
                    place = DataIndexTable[j].Index;

                    SOdensIn[j].Pos[0] = Group[place].Pos[0];
                    SOdensIn[j].Pos[1] = Group[place].Pos[1];
                    SOdensIn[j].Pos[2] = Group[place].Pos[2];
                    SOdensIn[j].R200 = R200[place];

                    memcpy(SOdensIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                           NODELISTLENGTH * sizeof(int));
                }

                /* exchange data */
                TIMER_PUSH(4, "subfind_comm");
                templated_exchange_data_Sendrecv(SOdensIn, SOdensGet, TAG_DENS_A, Send_offset, Send_count, Recv_offset,
                                                 Recv_count);
                TIMER_POP(4, "subfind_comm");

                myfree(SOdensIn);
                SOdensResult =
                    (struct SOdens_out *)mymalloc("	      SOdensResult", Nimport * sizeof(struct SOdens_out));
                SOdensOut = (struct SOdens_out *)mymalloc("	      SOdensOut", Nexport * sizeof(struct SOdens_out));

                /* now do the locations that were sent to us */

                NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
                {
#ifdef _OPENMP
                    int mythreadid = omp_get_thread_num();
#else // _OPENMP
                    int mythreadid = 0;
#endif // _OPENMP
                    subfind_overdensity_evaluate_secondary(
                        &mythreadid); /* do local particles and prepare export list */
                }

                if (i >= Ngroups)
                {
                    ndone_flag = 1;
                }
                else
                {
                    ndone_flag = 0;
                }
                MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
                TIMER_PUSH(4, "subfind_comm");
                templated_exchange_data_Sendrecv(SOdensResult, SOdensOut, TAG_DENS_B, Recv_offset, Recv_count,
                                                 Send_offset, Send_count);
                TIMER_POP(4, "subfind_comm");
                /* add the result to the local particles */
                for (int j = 0; j < Nexport; j++)
                {
                    place = DataIndexTable[j].Index;

                    M200[place] += SOdensOut[j].Mass;
                }

                myfree(SOdensOut);
                myfree(SOdensResult);
                myfree(SOdensGet);
            } while (ndone < NTask);

            /* do final operations on results */
            npleft = 0;
            for (i = 0; i < Ngroups; i++)
            {
                if (dens_eval_arg.todo[i])
                {
                    overdensity = M200[i] / (4.0 * M_PI / 3.0 * R200[i] * R200[i] * R200[i]) / rhoback;

                    if ((dens_eval_arg.right[i] - dens_eval_arg.left[i]) > 1.0e-4 * dens_eval_arg.left[i])
                    {
                        /* need to redo this group */
                        npleft++;

                        if (overdensity > Deltas[rep])
                        {
                            dens_eval_arg.left[i] = R200[i];
                        }
                        else
                        {
                            dens_eval_arg.right[i] = R200[i];
                        }

                        if (iter >= MAXITER - 10)
                        {
                            VERBOSE_ALL(
                                2,
                                "gr=%d task=%d  R200=%g Left=%g Right=%g Menclosed=%g Right-Left=%g\n   pos=(%g|%g|%g)",
                                i, ThisTask, R200[i], dens_eval_arg.left[i], dens_eval_arg.right[i], M200[i],
                                dens_eval_arg.right[i] - dens_eval_arg.left[i], Group[i].Pos[0], Group[i].Pos[1],
                                Group[i].Pos[2]);
                        }
                    }
                    else
                    {
                        dens_eval_arg.todo[i] = 0;
                    }
                }
            }

            sumup_large_ints(1, &npleft, &ntot);

            double t1 = second();
            TIMER_POP(4, "SO_iteration");

            if (ntot > 0)
            {
                iter++;

                if (iter > 0 && ThisTask == 0)
                {
                    VERBOSE_ALL(2, "SO iteration %d: need to repeat for %d%09d particles. (took %g sec)", iter,
                                (int)(ntot / 1000000000), (int)(ntot % 1000000000), timediff(t0, t1));
                }

                ENDRUN_IF(iter > MAXITER, "failed to converge in neighbour iteration in density()");
            }
        } while (ntot > 0);

        MPI_Barrier(MYMPI_COMM_WORLD);

        TIMER_POP(4, "SO_Radius_Calculation");

#ifdef SO_VEL_DISPERSIONS
        TIMER_PUSH(4, "SO_VelDisp_Calculation");
        NextGroup = 0; /* begin with this index */
        do
        {
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextGroup = NextGroup;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mythreadid = omp_get_thread_num();
#else // _OPENMP
                int mythreadid = 0;
#endif // _OPENMP
                subfind_overdensity_evaluate_dispersion_primary(
                    &mythreadid); /* do local particles and prepare export list */
            }

            if (BufferFullFlag)
            {
                int last_nextGroup = NextGroup;

                NextGroup = save_NextGroup;

                while (NextGroup < Ngroups)
                {
                    if (NextGroup == last_nextGroup)
                    {
                        break;
                    }

                    if (ProcessedFlag[NextGroup] != 1)
                    {
                        break;
                    }

                    ProcessedFlag[NextGroup] = 2;

                    NextGroup += 1;
                }

                ENDRUN_IF(NextGroup == save_NextGroup, "Task %d: pos=(%g,%g,%g) mass=%g", ThisTask,
                          Group[NextGroup].Pos[0], Group[NextGroup].Pos[1], Group[NextGroup].Pos[2],
                          Group[NextGroup].Mass);

                int new_export = 0;
                k = 0;
                for (int j = 0; j < Nexport; j++)
                    if (ProcessedFlag[DataIndexTable[j].Index] != 2)
                    {
                        if (k < j + 1)
                        {
                            k = j + 1;
                        }

                        for (; k < Nexport; k++)
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
                    else
                    {
                        new_export++;
                    }

                Nexport = new_export;
            }

            for (int j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
            }
            for (int j = 0; j < Nexport; j++)
            {
                Send_count[DataIndexTable[j].Task]++;
            }

            STD_SORT(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);

            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

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

            SOdensGet = (struct SOdens_in *)mymalloc("	  SOdensGet", Nimport * sizeof(struct SOdens_in));
            SOdensIn = (struct SOdens_in *)mymalloc("	  SOdensIn", Nexport * sizeof(struct SOdens_in));

            /* prepare particle data for export */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;

                SOdensIn[j].Pos[0] = Group[place].Pos[0];
                SOdensIn[j].Pos[1] = Group[place].Pos[1];
                SOdensIn[j].Pos[2] = Group[place].Pos[2];
                SOdensIn[j].R200 = R200[place];

                memcpy(SOdensIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                       NODELISTLENGTH * sizeof(int));
            }

            /* exchange data */
            TIMER_PUSH(4, "subfind_comm");
            templated_exchange_data_Sendrecv(SOdensIn, SOdensGet, TAG_DENS_A, Send_offset, Send_count, Recv_offset,
                                             Recv_count);
            TIMER_POP(4, "subfind_comm");

            myfree(SOdensIn);
            SOdensResult = (struct SOdens_out *)mymalloc("	  SOdensResult", Nimport * sizeof(struct SOdens_out));
            SOdensOut = (struct SOdens_out *)mymalloc("	  SOdensOut", Nexport * sizeof(struct SOdens_out));

            /* now do the locations that were sent to us */

            NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mythreadid = omp_get_thread_num();
#else // _OPENMP
                int mythreadid = 0;
#endif // _OPENMP
                subfind_overdensity_evaluate_dispersion_secondary(
                    &mythreadid); /* do local particles and prepare export list */
            }

            if (i >= Ngroups)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            TIMER_PUSH(4, "subfind_comm");
            templated_exchange_data_Sendrecv(SOdensResult, SOdensOut, TAG_DENS_B, Recv_offset, Recv_count, Send_offset,
                                             Send_count);
            TIMER_POP(4, "subfind_comm");
            /* add the result to the local particles */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;

                Vx200[place] += SOdensOut[j].Vx200;
                Vy200[place] += SOdensOut[j].Vy200;
                Vz200[place] += SOdensOut[j].Vz200;
                Disp200[place] += SOdensOut[j].Disp200;
            }

            myfree(SOdensOut);
            myfree(SOdensResult);
            myfree(SOdensGet);
        } while (ndone < NTask);

        MPI_Barrier(MYMPI_COMM_WORLD);

        TIMER_POP(4, "SO_VelDisp_Calculation");
#endif // SO_VEL_DISPERSIONS

#ifdef SO_BAR_INFO
        TIMER_PUSH(4, "SO_BarInfo_Calculation");
        NextGroup = 0; /* begin with this index */
        do
        {
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextGroup = NextGroup;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mythreadid = omp_get_thread_num();
#else // _OPENMP
                int mythreadid = 0;
#endif // _OPENMP
                subfind_overdensity_evaluate_barinfo_primary(
                    &mythreadid); /* do local particles and prepare export list */
            }

            if (BufferFullFlag)
            {
                int last_nextGroup = NextGroup;

                NextGroup = save_NextGroup;

                while (NextGroup < Ngroups)
                {
                    if (NextGroup == last_nextGroup)
                    {
                        break;
                    }

                    if (ProcessedFlag[NextGroup] != 1)
                    {
                        break;
                    }

                    ProcessedFlag[NextGroup] = 2;

                    NextGroup += 1;
                }

                ENDRUN_IF(NextGroup == save_NextGroup, "Task %d: pos=(%g,%g,%g) mass=%g", ThisTask,
                          Group[NextGroup].Pos[0], Group[NextGroup].Pos[1], Group[NextGroup].Pos[2],
                          Group[NextGroup].Mass);

                int new_export = 0;
                k = 0;
                for (int j = 0; j < Nexport; j++)
                    if (ProcessedFlag[DataIndexTable[j].Index] != 2)
                    {
                        if (k < j + 1)
                        {
                            k = j + 1;
                        }

                        for (; k < Nexport; k++)
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
                    else
                    {
                        new_export++;
                    }

                Nexport = new_export;
            }

            for (int j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
            }
            for (int j = 0; j < Nexport; j++)
            {
                Send_count[DataIndexTable[j].Task]++;
            }

            STD_SORT(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);

            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
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

            SOdensGet = (struct SOdens_in *)mymalloc("	  SOdensGet", Nimport * sizeof(struct SOdens_in));
            SOdensIn = (struct SOdens_in *)mymalloc("	  SOdensIn", Nexport * sizeof(struct SOdens_in));

            /* prepare particle data for export */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;

                SOdensIn[j].Pos[0] = Group[place].Pos[0];
                SOdensIn[j].Pos[1] = Group[place].Pos[1];
                SOdensIn[j].Pos[2] = Group[place].Pos[2];
                SOdensIn[j].R200 = R200[place];

                memcpy(SOdensIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                       NODELISTLENGTH * sizeof(int));
            }

            /* exchange data */
            TIMER_PUSH(4, "subfind_comm");
            templated_exchange_data_Sendrecv(SOdensIn, SOdensGet, TAG_DENS_A, Send_offset, Send_count, Recv_offset,
                                             Recv_count);
            TIMER_POP(4, "subfind_comm");

            myfree(SOdensIn);
            SOdensResult = (struct SOdens_out *)mymalloc("	  SOdensResult", Nimport * sizeof(struct SOdens_out));
            SOdensOut = (struct SOdens_out *)mymalloc("	  SOdensOut", Nexport * sizeof(struct SOdens_out));

            /* now do the locations that were sent to us */

            NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mythreadid = omp_get_thread_num();
#else // _OPENMP
                int mythreadid = 0;
#endif // _OPENMP
                subfind_overdensity_evaluate_barinfo_secondary(
                    &mythreadid); /* do local particles and prepare export list */
            }

            if (i >= Ngroups)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            TIMER_PUSH(4, "subfind_comm");
            templated_exchange_data_Sendrecv(SOdensResult, SOdensOut, TAG_DENS_B, Recv_offset, Recv_count, Send_offset,
                                             Send_count);
            TIMER_POP(4, "subfind_comm");
            /* add the result to the local particles */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;
                gas_mass[place] += SOdensOut[j].gas_mass;
                star_mass[place] += SOdensOut[j].star_mass;
                temp[place] += SOdensOut[j].temp;
                xlum[place] += SOdensOut[j].xlum;
                ygas[place] += SOdensOut[j].ygas;
            }

            myfree(SOdensOut);
            myfree(SOdensResult);
            myfree(SOdensGet);
        } while (ndone < NTask);

        MPI_Barrier(MYMPI_COMM_WORLD);

        TIMER_POP(4, "SO_BarInfo_Calculation");

#endif // SO_BAR_INFO

        for (int i = 0; i < Ngroups; i++)
        {
            if (Group[i].Nsubs > 0)
            {
                overdensity = M200[i] / (4.0 * M_PI / 3.0 * R200[i] * R200[i] * R200[i]) / rhoback;

                if ((overdensity - Deltas[rep]) > 0.1 * Deltas[rep])
                {
                    R200[i] = M200[i] = 0;
                }
                else if (M200[i] < 5 * Group[i].Mass / Group[i].Len)
                {
                    R200[i] = M200[i] = 0;
                }
            }
            else
            {
                R200[i] = M200[i] = 0;
            }

#ifdef SO_BAR_INFO
            switch (rep)
            {
            case 0:
                Group[i].M_TopHat200 = M200[i];
                Group[i].R_TopHat200 = R200[i];
                break;
            case 1:
                Group[i].M_Mean200 = M200[i];
                Group[i].R_Mean200 = R200[i];
                break;
            case 2:
                Group[i].M_Mean500 = M200[i];
                Group[i].R_Mean500 = R200[i];
                break;
            case 3:
                Group[i].M_Crit200 = M200[i];
                Group[i].R_Crit200 = R200[i];
                break;
            case 4:
                Group[i].M_Crit500 = M200[i];
                Group[i].R_Crit500 = R200[i];
                break;
            case 5:
                Group[i].M_Crit2500 = M200[i];
                Group[i].R_Crit2500 = R200[i];
                break;
            }
#else // SO_BAR_INFO
            switch (rep)
            {
            case 0:
                Group[i].M_Mean200 = M200[i];
                Group[i].R_Mean200 = R200[i];
                break;
            case 1:
                Group[i].M_TopHat200 = M200[i];
                Group[i].R_TopHat200 = R200[i];
                break;
            case 2:
                Group[i].M_Crit200 = M200[i];
                Group[i].R_Crit200 = R200[i];
                break;
            }
#endif // SO_BAR_INFO

#ifdef SO_VEL_DISPERSIONS
            if (R200[i] > 0 && M200[i] > 0)
            {
                Vx200[i] /= M200[i];
                Vy200[i] /= M200[i];
                Vz200[i] /= M200[i];
                Disp200[i] /= M200[i];
                Disp200[i] -= (Vx200[i] * Vx200[i] + Vy200[i] * Vy200[i] + Vz200[i] * Vz200[i]);
                Disp200[i] = sqrt(Disp200[i] / 3); /* convert to 1D velocity dispersion */
            }
            else
            {
                Disp200[i] = 0;
            }
#ifdef SO_BAR_INFO
            switch (rep)
            {
            case 0:
                Group[i].VelDisp_TopHat200 = Disp200[i];
                break;
            case 1:
                Group[i].VelDisp_Mean200 = Disp200[i];
                break;
            case 2:
                Group[i].VelDisp_Mean500 = Disp200[i];
                break;
            case 3:
                Group[i].VelDisp_Crit200 = Disp200[i];
                break;
            case 4:
                Group[i].VelDisp_Crit500 = Disp200[i];
                break;
            case 5:
                Group[i].VelDisp_Crit2500 = Disp200[i];
                break;
            }
#else // SO_BAR_INFO
            switch (rep)
            {
            case 0:
                Group[i].VelDisp_Mean200 = Disp200[i];
                break;
            case 1:
                Group[i].VelDisp_TopHat200 = Disp200[i];
                break;
            case 2:
                Group[i].VelDisp_Crit200 = Disp200[i];
                break;
            }
#endif // SO_BAR_INFO
#endif // SO_VEL_DISPERSIONS

#ifdef SO_BAR_INFO
            if (gas_mass[i] > 0)
            {
                Group[i].gas_mass[rep] = gas_mass[i];
                Group[i].star_mass[rep] = star_mass[i];
                Group[i].temp[rep] = temp[i] / gas_mass[i];
                Group[i].xlum[rep] = xlum[i];
                if (R200[i] > 0)
                {
                    Group[i].ygas[rep] = ygas[i] / (M_PI * R200[i] * R200[i] * All.Time * All.Time);
                }
                else
                {
                    Group[i].ygas[rep] = 0;
                }
            }
            else
            {
                Group[i].gas_mass[rep] = 0;
                Group[i].star_mass[rep] = 0;
                Group[i].temp[rep] = 0;
                Group[i].xlum[rep] = 0;
                Group[i].ygas[rep] = 0;
            }
#endif // SO_BAR_INFO
        }
    }

    VERBOSE(2, "Finished SO loop.");

    myfree(DataNodeList);
    myfree(DataIndexTable);

    myfree(dens_eval_arg.todo);
#ifdef SO_BAR_INFO
    myfree(ygas);
    myfree(xlum);
    myfree(temp);
    myfree(star_mass);
    myfree(gas_mass);
#endif // SO_BAR_INFO
#ifdef SO_VEL_DISPERSIONS
    myfree(Disp200);
    myfree(Vz200);
    myfree(Vy200);
    myfree(Vx200);
#endif // SO_VEL_DISPERSIONS
    myfree(M200);
    myfree(R200);
    myfree(dens_eval_arg.right);
    myfree(dens_eval_arg.left);

    myfree(Ngblist);
}

/**
 * @brief Evaluates the overdensity for a given target.
 *
 * This function calculates the mass within a specified radius around a target
 * group or spherical overdensity (SO) center. It iterates through the tree
 * structure to sum up the mass of particles or groups within the given radius.
 *
 * @param target The index of the target group or SO center.
 * @param mode The mode of operation. If mode is 0, it operates on groups. If mode is 1, it operates on SO centers.
 * @param exportflag Pointer to an array of export flags, used for marking nodes for export.
 * @param exportnodecount Pointer to an integer keeping count of the nodes to be exported.
 * @param exportindex Pointer to an array of export indices, used for tracking nodes to be exported.
 *
 * @return Returns 0 on successful evaluation, or -1 if an error occurred during tree traversal.
 */
int subfind_overdensity_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex)
{

    double h = 0;
    MyLongDouble *pos;
    if (mode == 0)
    {
        pos = Group[target].Pos;
        h = R200[target];
    }
    else
    {
        pos = SOdensGet[target].Pos;
        h = SOdensGet[target].R200;
    }

    int startnode = 0;
    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = SOdensGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    double mass = 0;
    int listindex = 0;
    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            double massret = subfind_ovderdens_treefind_threads(pos, h, target, &startnode, mode, exportflag,
                                                                exportnodecount, exportindex);

            if (massret < 0)
            {
                return -1;
            }

            mass += massret;
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = SOdensGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        M200[target] = mass;
    }
    else
    {
        SOdensResult[target].Mass = mass;
    }

    return 0;
}

/**
 * @brief Finds and sums the mass of particles within a specified radius around a search center, using a tree structure
 * for efficient searching.
 *
 * This function traverses a hierarchical tree structure to locate and sum the mass of particles that lie within a
 * specified search radius from a given center. It handles both local and pseudo particles, managing export lists for
 * pseudo particles to be processed by other tasks.
 *
 * @param searchcenter An array of three coordinates representing the center of the search region.
 * @param hsml The search radius (smoothing length) within which to sum the particle masses.
 * @param target The target index for which the search is being conducted.
 * @param startnode A pointer to the starting node index for the tree traversal. The value is updated during the
 * function execution.
 * @param mode The mode of operation: 0 for local search, 1 for non-local search.
 * @param exportflag An array used to mark tasks that require data export.
 * @param exportnodecount An array used to count the number of nodes to be exported per task.
 * @param exportindex An array storing the indices of the export tasks.
 * @return The total mass of particles found within the search radius, or -1 if the buffer is full and the search needs
 * to be restarted.
 */
double subfind_ovderdens_treefind_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                          int mode, int *exportflag, int *exportnodecount, int *exportindex)
{
    int p, task, nexp;
    struct NODE *current;
    MyLongDouble dx, dy, dz, dist, r2;

#define FACT2 0.86602540
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    double mass = 0;
    int no = *startnode;

    while (no >= 0)
    {
        if (no < All.MaxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

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

            mass += P[p].Mass;
        }
        else
        {
            if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode equal 1 not supported.");

                if (mode == 0)
                {
                    if (exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
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
                        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                    }
                }
                no = Nextnode[no - MaxNodes];
                continue;
            }

            current = &Nodes[no];

            if (mode == 1)
            {
                if (current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                {
                    *startnode = -1;
                    return mass;
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
                continue;
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if ((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist)
            {
                continue;
            }

            if ((current->u.d.bitflags & ((1 << BITFLAG_TOPLEVEL) + (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))) ==
                0) /* only use fully local nodes */
            {
                /* test whether the node is contained within the sphere */
                dist = hsml - FACT2 * current->len;
                if (dist > 0)
                {
                    if (r2 < dist * dist)
                    {
                        mass += current->u.d.mass;
                        continue;
                    }
                }
            }
            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return mass;
}

/**
 * @brief Evaluates primary overdensity regions in parallel.
 *
 * This function evaluates primary overdensity regions in parallel using multiple threads.
 * It iterates over a list of tasks to determine which regions need to be evaluated. Each
 * task is assigned to a thread, and the evaluation is performed concurrently.
 *
 * @param thread_id The ID of the thread executing the function.
 * @param todo A list indicating which tasks need to be evaluated.
 * @param left An array containing the left boundary of each task.
 * @param right An array containing the right boundary of each task.
 * @return NULL
 */
void *subfind_overdensity_evaluate_primary(int thread_id, char *todo, MyFloat *left, MyFloat *right)
{
    int *exportflag, *exportnodecount, *exportindex, i;
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
#endif
        {
            if (BufferFullFlag != 0 || NextGroup >= Ngroups)
            {
                exitFlag = 1;
            }
            else
            {
#ifdef _OPENMP
                do
                {
                    i = NextGroup;
                    NextGroup += 1;
                    if (NextGroup >= Ngroups)
                    {
                        break;
                    }
                    ProcessedFlag[i] = 1;
                } while (!todo[i]);
                ProcessedFlag[i] = 0;
#else // _OPENMP
                i = NextGroup;
                ProcessedFlag[NextGroup] = 0;
                NextGroup += 1;
#endif // _OPENMP
            }
        }
        UNLOCK_NEXPORT;
        if (exitFlag)
        {
            break;
        }

        if (todo[i])
        {
            R200[i] = 0.5 * (left[i] + right[i]);
            if (subfind_overdensity_evaluate(i, 0, exportflag, exportnodecount, exportindex) < 0)
            {
                break; /* export buffer has filled up */
            }
        }
        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Evaluates secondary overdensity regions.
 *
 * This function evaluates secondary overdensity regions using a single thread. It iterates
 * over a list of tasks to determine which regions need to be evaluated. Each task is evaluated
 * sequentially in this single-threaded implementation.
 *
 * @param p Pointer to the thread ID.
 * @return NULL
 */
void *subfind_overdensity_evaluate_secondary(void *p)
{
    int j, dummy;

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

        subfind_overdensity_evaluate(j, 1, &dummy, &dummy, &dummy);
    }
    return NULL;
}

#ifdef SO_VEL_DISPERSIONS
/**
 * @brief Evaluates the dispersion of velocities within a specified radius around a target.
 *
 * This function calculates the dispersion of velocities for particles within a given radius
 * around a target. It iterates over neighboring particles within the search radius and computes
 * the velocity dispersion based on their masses and velocities.
 *
 * @param target The index of the target particle or group.
 * @param mode The mode of operation (0 for primary, 1 for secondary).
 * @param exportflag Pointer to the export flag array.
 * @param exportnodecount Pointer to the export node count array.
 * @param exportindex Pointer to the export index array.
 * @param ngblist Array storing indices of neighboring particles.
 * @return 0 if successful, -1 otherwise.
 */
int subfind_overdensity_evaluate_dispersion(int target, int mode, int *exportflag, int *exportnodecount,
                                            int *exportindex, int *ngblist)
{
    int startnode, listindex = 0;
    double h, vvx, vvy, vvz;
    MyLongDouble *pos;
    double vel_to_peculiar, H_of_a, atime;

    if (All.ComovingIntegrationOn)
    {
        H_of_a = hubble_function(All.Time);
        vel_to_peculiar = 1.0 / All.Time;
        atime = All.Time;
    }
    else
    {
        H_of_a = 0;
        vel_to_peculiar = atime = 1;
    }

    if (mode == 0)
    {
        pos = Group[target].Pos;
        h = R200[target];
    }
    else
    {
        pos = SOdensGet[target].Pos;
        h = SOdensGet[target].R200;
    }

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = SOdensGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    double vx = 0;
    double vy = 0;
    double vz = 0;
    double v2 = 0;

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            int ngb = subfind_ovderdens_treefind_dispersion_threads(pos, h, target, &startnode, mode, exportflag,
                                                                    exportnodecount, exportindex, ngblist);

            ENDRUN_IF(ngb > NumPart / DENSITY_LESS_NGB_FACTOR_SUB,
                      "ngb>NumPart/DENSITY_LESS_NGB_FACTOR_SUB => %d>%d/%d \n", ngb, NumPart,
                      DENSITY_LESS_NGB_FACTOR_SUB);

            if (ngb < 0)
            {
                return -1;
            }

            for (int n = 0; n < ngb; n++)
            {
                int j = ngblist[n];

                vvx = vel_to_peculiar * P[j].Vel[0] + H_of_a * atime * NEAREST(P[j].Pos[0] - pos[0]);
                vvy = vel_to_peculiar * P[j].Vel[1] + H_of_a * atime * NEAREST(P[j].Pos[1] - pos[1]);
                vvz = vel_to_peculiar * P[j].Vel[2] + H_of_a * atime * NEAREST(P[j].Pos[2] - pos[2]);

                vx += P[j].Mass * vvx;
                vy += P[j].Mass * vvy;
                vz += P[j].Mass * vvz;

                v2 += P[j].Mass * (vvx * vvx + vvy * vvy + vvz * vvz);
            }
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = SOdensGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        Disp200[target] = v2;
        Vx200[target] = vx;
        Vy200[target] = vy;
        Vz200[target] = vz;
    }
    else
    {
        SOdensResult[target].Disp200 = v2;
        SOdensResult[target].Vx200 = vx;
        SOdensResult[target].Vy200 = vy;
        SOdensResult[target].Vz200 = vz;
    }
    return 0;
}

/**
 * @brief Evaluates dispersion of velocities for primary overdensity regions.
 *
 * This function evaluates the dispersion of velocities for primary overdensity regions.
 * It iterates over a list of primary groups to determine which regions need to be evaluated.
 * Each group is assigned to a thread, and the evaluation is performed concurrently.
 *
 * @param p Pointer to the thread ID.
 * @return NULL
 */
void *subfind_overdensity_evaluate_dispersion_primary(void *p)
{
    int thread_id = *((int *)p);
    int *exportflag, *exportnodecount, *exportindex, *ngblist, i;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

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
            if (BufferFullFlag != 0 || NextGroup >= Ngroups)
            {
                exitFlag = 1;
            }
            else
            {
#ifdef _OPENMP
                do
                {
                    i = NextGroup;
                    NextGroup += 1;
                    if (NextGroup >= Ngroups)
                        break;
                    ProcessedFlag[i] = 1;
                } while (Group[i].Nsubs <= 0);
                ProcessedFlag[i] = 0;
#else // _OPENMP
                i = NextGroup;
                ProcessedFlag[NextGroup] = 0;
                NextGroup += 1;
#endif // _OPENMP
            }
        }
        UNLOCK_NEXPORT;
        if (exitFlag)
        {
            break;
        }

        if (Group[i].Nsubs > 0)
        {
            if (subfind_overdensity_evaluate_dispersion(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0)
            {
                break; /* export buffer has filled up */
            }
        }

        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Evaluates dispersion of velocities for secondary overdensity regions.
 *
 * This function evaluates the dispersion of velocities for secondary overdensity regions.
 * It iterates over a list of secondary regions to determine which regions need to be evaluated.
 * Each region is assigned to a thread, and the evaluation is performed sequentially.
 *
 * @param p Pointer to the thread ID.
 * @return NULL
 */
void *subfind_overdensity_evaluate_dispersion_secondary(void *p)
{
    int thread_id = *((int *)p);
    int j, dummy, *ngblist;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

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

        subfind_overdensity_evaluate_dispersion(j, 1, &dummy, &dummy, &dummy, ngblist);
    }
    return NULL;
}
#endif // SO_VEL_DISPERSIONS

#ifdef SO_BAR_INFO
/**
 * @brief Evaluates baryonic information for a given target.
 *
 * This function calculates various baryonic properties for particles within a specified radius around a target.
 * The properties include gas mass, star mass, temperature, X-ray luminosity, and Y-parameter.
 * It iterates over neighboring particles within the search radius and accumulates these properties.
 *
 * @param target The index of the target particle or group.
 * @param mode The mode of operation (0 for primary, 1 for secondary).
 * @param exportflag Pointer to the export flag array.
 * @param exportnodecount Pointer to the export node count array.
 * @param exportindex Pointer to the export index array.
 * @param ngblist Array storing indices of neighboring particles.
 * @return 0 if successful, -1 otherwise.
 */
int subfind_overdensity_evaluate_barinfo(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                         int *ngblist)
{
    int ngb, j, startnode, listindex = 0;
    double h;
    MyLongDouble *pos;
    double a3inv;
    double this_gas_mass = 0, this_star_mass = 0, this_temp = 0, this_xlum = 0, this_ygas = 0;
    double particle_temperature, cutoff, Lx;
    double ukev = 1.1606e7;
    double kevtoerg = 1.6021773e-9;
    float xclouds = 0;
#ifdef COOLING
    double ne;
#ifndef LT_STELLAREVOLUTION
    double dt;
#endif // LT_STELLAREVOLUTION
#endif // COOLING

    double mol = 4 / (5 * HYDROGEN_MASSFRAC + 3);
    double n2ne =
        (HYDROGEN_MASSFRAC + 0.5 * (1 - HYDROGEN_MASSFRAC)) / (2 * HYDROGEN_MASSFRAC + 0.75 * (1 - HYDROGEN_MASSFRAC));
    double gg = 1;
    double C_j = 2.42e-24; /* factor for X-ray emissivity (BS1996, eq.4) */

    if (All.ComovingIntegrationOn)
    {
        a3inv = 1.0 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1.0;
    }

    if (mode == 0)
    {
        pos = Group[target].Pos;
        h = R200[target];
    }
    else
    {
        pos = SOdensGet[target].Pos;
        h = SOdensGet[target].R200;
    }

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = SOdensGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            ngb = subfind_ovderdens_treefind_dispersion_threads(pos, h, target, &startnode, mode, exportflag,
                                                                exportnodecount, exportindex, ngblist);

            ENDRUN_IF(ngb > NumPart / DENSITY_LESS_NGB_FACTOR_SUB,
                      "ngb>NumPart/DENSITY_LESS_NGB_FACTOR_SUB => %d>%d/%d \n", ngb, NumPart,
                      DENSITY_LESS_NGB_FACTOR_SUB);

            if (ngb < 0)
            {
                return -1;
            }

            for (int n = 0; n < ngb; n++)
            {
                j = ngblist[n];
#ifdef BLACK_HOLES
                if (P[j].Type == 0 && P[j].Mass > 0)
#else // BLACK_HOLES
                if (P[j].Type == 0)
#endif // BLACK_HOLES
                {
                    this_gas_mass += P[j].Mass;
#ifdef LT_STELLAREVOLUTION
                    particle_temperature = SphP[j].Temperature;
                    xclouds = SphP[j].XColdCloud;
#else // LT_STELLAREVOLUTION
#ifdef COOLING
                    ne = SphP[j].elec;

                    dt = (P[j].TimeBin ? (((integertime)1) << P[j].TimeBin) : 0) * All.Timebase_interval;
                    particle_temperature = convert_u_to_tempSTD(
                        max(All.MinEgySpec, (SphP[j].Entropy + SphP[j].DtEntropy * dt) / GAMMA_MINUS1 *
                                                pow(SphP[j].Density * a3inv, GAMMA_MINUS1)),
                        SphP[j].Density * a3inv, &ne);
#else // COOLING
                    particle_temperature = SphP[j].Entropy * pow(SphP[j].Density * a3inv, GAMMA_MINUS1) * PROTONMASS *
                                           mol / BOLTZMANN * All.UnitMass_in_g / All.UnitEnergy_in_cgs;
#endif // COOLING
#endif // LT_STELLAREVOLUTION
                    particle_temperature /= ukev; /* now is in keV */
                    this_temp += particle_temperature * P[j].Mass;

                    /*
                       double E0 = 0.3, E1 = 10;
                       cutoff = exp(- E0 / particle_temperature) - exp(- E1 / particle_temperature);
                     */
                    cutoff = 1.0;

                    Lx = P[j].Mass * All.UnitMass_in_g / All.HubbleParam * SphP[j].Density * a3inv * All.UnitMass_in_g /
                         pow(All.UnitLength_in_cm, 3) * pow(All.HubbleParam, 2) * sqrt(particle_temperature) * cutoff;

                    /* Steinmetz & Bartelmann, based on Spizer 1968, gg = 1.0 (!?)
                       Beside the fact, that it is not clear which value they used for the Gaunt factor
                       it is the best formulation, as composition H/He (fr) and conversion from particle
                       number to electron number (n2ne) is explicite formulated. */

                    Lx *= 4 * C_j * gg / (1 + HYDROGEN_MASSFRAC) * pow(n2ne / (mol * PROTONMASS), 2);
                    if (xclouds < 0.1)
                        this_xlum += Lx / 1e44;

#ifdef COOLING
                    ne = SphP[j].elec;

                    /* units of comoving length^2 */
                    this_ygas += ne * P[j].Mass * All.UnitMass_in_g * All.HubbleParam * particle_temperature *
                                 kevtoerg / (ELECTRONMASS * PROTONMASS * C_LIGHT * C_LIGHT) * THOMSON /
                                 (All.UnitLength_in_cm * All.UnitLength_in_cm);
#else // COOLING
                    this_ygas += mol / n2ne * P[j].Mass * All.UnitMass_in_g * All.HubbleParam * particle_temperature *
                                 kevtoerg / (ELECTRONMASS * PROTONMASS * C_LIGHT * C_LIGHT) * THOMSON /
                                 (All.UnitLength_in_cm * All.UnitLength_in_cm);
#endif // COOLING
                }
                if (P[j].Type == 4)
                {
                    this_star_mass += P[j].Mass;
                }
            }
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = SOdensGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        gas_mass[target] = this_gas_mass;
        star_mass[target] = this_star_mass;
        temp[target] = this_temp;
        xlum[target] = this_xlum;
        ygas[target] = this_ygas;
    }
    else
    {
        SOdensResult[target].gas_mass = this_gas_mass;
        SOdensResult[target].star_mass = this_star_mass;
        SOdensResult[target].temp = this_temp;
        SOdensResult[target].xlum = this_xlum;
        SOdensResult[target].ygas = this_ygas;
    }
    return 0;
}

/**
 * @brief Evaluates baryonic information for primary targets in a multithreaded environment.
 *
 * This function is executed by multiple threads to evaluate various baryonic properties (gas mass, star mass,
 * temperature, X-ray luminosity...) for primary targets (groups with substructures). It iterates
 * over the groups, processes each group with substructures, and accumulates the relevant properties.
 *
 * @param p Pointer to the thread ID.
 * @return NULL
 */
void *subfind_overdensity_evaluate_barinfo_primary(void *p)
{
    int thread_id = *((int *)p);

    int i;
    int *exportflag, *exportnodecount, *exportindex, *ngblist;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

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
            if (BufferFullFlag != 0 || NextGroup >= Ngroups)
            {
                exitFlag = 1;
            }
            else
            {
#ifdef _OPENMP
                do
                {
                    i = NextGroup;
                    NextGroup += 1;
                    if (NextGroup >= Ngroups)
                    {
                        break;
                    }
                    ProcessedFlag[i] = 1;
                } while (Group[i].Nsubs <= 0);
                ProcessedFlag[i] = 0;
#else // _OPENMP
                i = NextGroup;
                ProcessedFlag[NextGroup] = 0;
                NextGroup += 1;
#endif // _OPENMP
            }
        }
        UNLOCK_NEXPORT;
        if (exitFlag)
        {
            break;
        }

        if (Group[i].Nsubs > 0)
        {
            if (subfind_overdensity_evaluate_barinfo(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0)
            {
                break; /* export buffer has filled up */
            }
        }
        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Evaluates baryonic information for secondary targets in a multithreaded environment.
 *
 * This function is designed to be executed by multiple threads to evaluate various baryonic properties,
 * such as gas mass, star mass, temperature, X-ray luminosity, ..., for secondary targets.
 * It iterates over the imported data, processes each entry, and accumulates the relevant properties
 * in a thread-safe manner, ensuring that the export buffer is not filled up, thus preventing data loss
 * or corruption during the computation process.
 *
 * @param p Pointer to the thread ID.
 * @return NULL
 */
void *subfind_overdensity_evaluate_barinfo_secondary(void *p)
{
    int thread_id = *((int *)p);

    int j, dummy, *ngblist;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

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

        subfind_overdensity_evaluate_barinfo(j, 1, &dummy, &dummy, &dummy, ngblist);
    }
    return NULL;
}
#endif // SO_BAR_INFO

#if defined(SO_VEL_DISPERSIONS) || defined(SO_BAR_INFO)
/**
 * @brief Finds neighbor particles within a given radius for a given target in the tree using multithreading.
 *
 * This function searches for neighbor particles around a specified search center within a given radius (hsml)
 * and populates the neighbor list (ngblist). It handles both single particles and pseudo particles,
 * and ensures proper handling of the export buffer to avoid overflow.
 *
 * @param searchcenter Array containing the coordinates of the search center [x, y, z].
 * @param hsml Search radius.
 * @param target Index of the target particle for which neighbors are being searched.
 * @param startnode Pointer to the starting node for the tree traversal.
 * @param mode Search mode (0 for primary, 1 for secondary).
 * @param exportflag Array to flag nodes that need to be exported.
 * @param exportnodecount Array to count the nodes that need to be exported.
 * @param exportindex Array to index the nodes that need to be exported.
 * @param ngblist Array to store the indices of the found neighbor particles.
 * @return Number of neighbors found, or -1 if the export buffer is full.
 */
int subfind_ovderdens_treefind_dispersion_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target,
                                                  int *startnode, int mode, int *exportflag, int *exportnodecount,
                                                  int *exportindex, int *ngblist)
{
    int p, task, nexp;
    struct NODE *current;
    MyLongDouble dx, dy, dz, dist;

#define FACT2 0.86602540
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        if (no < All.MaxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

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
            if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode 1 is not supported.");

                if (mode == 0)
                {
                    if (exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
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
                                /* Out of buffer space. Need to discard work for this particle and interrupt. */
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
                        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                    }
                }

                no = Nextnode[no - MaxNodes];
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
            if ((dx * dx + dy * dy + dz * dz) > dist * dist)
            {
                continue;
            }

            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}

#endif // defined(SO_VEL_DISPERSIONS) || defined(SO_BAR_INFO)

#endif // SUBFIND
