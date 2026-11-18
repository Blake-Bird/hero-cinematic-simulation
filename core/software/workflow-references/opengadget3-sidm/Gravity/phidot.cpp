/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief PHIDOT functions.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Margarita Petkova and Benjamin Seidel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/communication.hpp"

#ifdef PHIDOT
/**
 * @brief Compute the time derivative of the gravitational potential for gas particles.
 *
 * This function iterates over gas particles to compute the time derivative of the gravitational potential
 * (phi dot) for each particle. It parallelizes the computation using OpenMP if enabled and handles
 * communication between different processes using MPI. The function exchanges particle data between
 * processes, aggregates results, and updates the local particles with the received phi dot values.
 * It also handles cases where the communication buffer becomes full and adjusts the export list accordingly.
 * After computing phi dot for all gas particles, the function multiplies the computed values by the
 * gravitational constant G.
 */
void phidot(void)
{
    int save_NextParticle;
    int ndone, ndone_flag;
    int place;

    if (All.ComovingIntegrationOn)
    {
        set_softenings();
    }

    TIMER_PUSH(1, "PHIDOT");
    VERBOSE(3, "Start computation of phi dot for all gas particles...");

    /* allocate buffers to arrange communication */
    All.BunchSize =
        (int)((All.BufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct gravdata_in) +
               sizeof(struct phidotdata_out) + sizemax(sizeof(struct gravdata_in), sizeof(struct phidotdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    NextParticle = 0; /* beginn with this index */

    do
    {
        BufferFullFlag = 0;
        Nexport = 0;
        save_NextParticle = NextParticle;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
            int mainthreadid = ThisThread;
            phidot_primary_loop(&mainthreadid); /* do local particles and prepare export list */
        }

        if (BufferFullFlag)
        {
            int last_nextparticle = NextParticle;

            NextParticle = save_NextParticle;

            while (NextParticle < N_gas)
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
                NextParticle++;
            }
            ENDRUN_IF(NextParticle == save_NextParticle, "the buffer is too small to process even a single particle");

            int new_export = 0;
            int k = 0;
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

        MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

        Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
        for (int j = 0; j < NTask; j++)
        {
            Nimport += Recv_count[j];

            if (j > 0)
            {
                Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
            }
        }

        GravDataGet = (struct gravdata_in *)mymalloc("GravDataGet", Nimport * sizeof(struct gravdata_in));
        GravDataIn = (struct gravdata_in *)mymalloc("GravDataIn", Nexport * sizeof(struct gravdata_in));

        /* prepare particle data for export */

        for (int j = 0; j < Nexport; j++)
        {
            place = DataIndexTable[j].Index;

            for (int k = 0; k < 3; k++)
            {
                GravDataIn[j].Pos[k] = P[place].Pos[k];
            }

            GravDataIn[j].Type = P[place].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
            if (P[place].Type == 0)
            {
                GravDataIn[j].Soft = P[place].Hsml;
            }
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#ifdef ADAPTGRAVSOFT
            GravDataIn[j].AGS_zeta = P[place].AGS_zeta;
            GravDataIn[j].AGS_omega = P[place].AGS_omega;
            GravDataIn[j].AGS_Hsml = P[place].AGS_Hsml;
            GravDataIn[j].Mass = P[place].Mass;
#endif // ADAPTGRAVSOFT

            memcpy(GravDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                   NODELISTLENGTH * sizeof(int));
        }
        /* exchange particle data */
        TIMER_PUSH(1, "PHIDOT_COMM");
        templated_exchange_data_Sendrecv(GravDataIn, GravDataGet, TAG_GRAV_A, Send_offset, Send_count, Recv_offset,
                                         Recv_count);
        TIMER_POP(1, "PHIDOT_COMM");
        myfree(GravDataIn);
        PhiDotDataResult =
            (struct phidotdata_out *)mymalloc("PhiDotDataResult", Nimport * sizeof(struct phidotdata_out));
        PhiDotDataOut = (struct phidotdata_out *)mymalloc("PhiDotDataOut", Nexport * sizeof(struct phidotdata_out));

        /* now do the particles that were sent to us */
        NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
            int mainthreadid = ThisThread;
            phidot_secondary_loop(&mainthreadid);
        }

        if (NextParticle >= N_gas)
        {
            ndone_flag = 1;
        }
        else
        {
            ndone_flag = 0;
        }
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        TIMER_PUSH(1, "PHIDOT_COMM");
        templated_exchange_data_Sendrecv(PhiDotDataResult, PhiDotDataOut, TAG_GRAV_B, Recv_offset, Recv_count,
                                         Send_offset, Send_count);
        TIMER_POP(1, "PHIDOT_COMM");
        /* add the results to the local particles */
        for (int j = 0; j < Nexport; j++)
        {
            place = DataIndexTable[j].Index;
            SphP[place].PhiDot += PhiDotDataOut[j].PhiDot;
        }

        myfree(PhiDotDataOut);
        myfree(PhiDotDataResult);
        myfree(GravDataGet);
    } while (ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);

    /* multiply with the gravitational constant */

    for (int i = 0; i < N_gas; i++)
    {
        SphP[i].PhiDot *= All.GravityConstantInternal;
    }

    TIMER_POP(1, "PHIDOT");
}

/**
 * @brief Primary loop for computing phi dot for local gas particles in parallel.
 *
 * This function is a primary loop executed by each thread in parallel to compute the time derivative of
 * the gravitational potential (phi dot) for local gas particles. It iterates over particles in a thread-safe
 * manner and evaluates the phi dot for each gas particle using the `phidot_treeevaluate` function.
 * The function handles buffer full conditions and ensures that each particle is processed exactly once.
 *
 * @param p Pointer to the thread ID.
 * @return NULL after completing the computation for all particles.
 */
void *phidot_primary_loop(void *p)
{
    int thread_id = *(int *)p;
    int *exportflag, *exportnodecount, *exportindex;

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
        int i;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        if (BufferFullFlag != 0 || NextParticle >= N_gas)
        {
            exitFlag = 1;
        }
        else
        {
            i = NextParticle;
            ProcessedFlag[i] = 0;
            NextParticle++;
        }

        if (exitFlag)
        {
            break;
        }

        if (P[i].Type == 0)
        {
            int ret = phidot_treeevaluate(i, 0, exportflag, exportnodecount, exportindex);
            if (ret < 0)
            {
                break;
            } /* export buffer has filled up */
        }
        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Secondary loop for computing phi dot for imported particles in parallel.
 *
 * This function is a secondary loop executed by each thread in parallel to compute the time derivative of
 * the gravitational potential (phi dot) for imported particles. It iterates over imported particles in a
 * thread-safe manner and evaluates the phi dot for each particle using the `phidot_treeevaluate` function.
 *
 * @param p Pointer to the thread ID (unused).
 * @return NULL after completing the computation for all imported particles.
 */
void *phidot_secondary_loop(void *p)
{
    int nodesinlist, dummy;

    while (1)
    {
        int j;
#ifdef _OPENMP
#pragma omp atomic capture
#endif // _OPENMP
        j = NextJ++;

        if (j >= Nimport)
        {
            break;
        }

        phidot_treeevaluate(j, 1, &nodesinlist, &dummy, &dummy);
    }
    return NULL;
}

#endif // PHIDOT
