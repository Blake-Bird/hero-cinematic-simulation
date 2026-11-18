/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Compute gravitational potentials in Subfind.
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

#ifdef SUBFIND

#include "fof.hpp"
#include "subfind.hpp"

/**
 * @brief Compute gravitational potentials for particles.
 *
 * This function computes the gravitational potentials for particles in a distributed computing environment,
 * utilizing an adaptive parallel approach. It iterates over particles, prepares a list of particles to be
 * exported for potential calculation, exchanges particle data between MPI tasks, computes gravitational
 * potentials locally, aggregates results, and finally computes potentials for all particles. The function
 * accounts for cosmological parameters and adjusts potential energies accordingly.
 *
 * @param num Number of particles.
 * @param d Pointer to an array of unbind_data structures containing particle indices.
 * @param phase Phase indicator for potential calculation.
 * @param weakly_bound_limit Threshold for weakly bound particles.
 */
void subfind_potential_compute(int num, struct unbind_data *d, int phase, double weakly_bound_limit)
{
    int ndone, ndone_flag, dummy;
    int place, nexport, nimport;
    double atime;

    /* allocate buffers to arrange communication */

    All.BunchSize =
        (int)((All.BufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct gravdata_in) +
               sizeof(struct potdata_out) + sizemax(sizeof(struct gravdata_in), sizeof(struct potdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    int i = 0; /* beginn with this index */
    do
    {
        for (int j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Exportflag[j] = -1;
        }

        /* do local particles and prepare export list */
        for (nexport = 0; i < num; i++)
        {
            if (phase == 1)
            {
                if (P[d[i].index].v.DM_BindingEnergy <= weakly_bound_limit)
                {
                    continue;
                }
            }

            if (subfind_force_treeevaluate_potential(d[i].index, 0, &nexport, Send_count) < 0)
            {
                break;
            }
        }

        STD_SORT(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);

        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

        nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
        for (int j = 0; j < NTask; j++)
        {
            nimport += Recv_count[j];

            if (j > 0)
            {
                Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
            }
        }

        GravDataGet = (struct gravdata_in *)mymalloc("GravDataGet", nimport * sizeof(struct gravdata_in));
        GravDataIn = (struct gravdata_in *)mymalloc("GravDataIn", nexport * sizeof(struct gravdata_in));

        /* prepare particle data for export */
        for (int j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;

            for (int k = 0; k < 3; k++)
            {
                GravDataIn[j].Pos[k] = P[place].Pos[k];
            }

            for (int k = 0; k < NODELISTLENGTH; k++)
            {
                GravDataIn[j].NodeList[k] = DataNodeList[DataIndexTable[j].IndexGet].NodeList[k];
            }

            GravDataIn[j].Type = P[place].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
            if (P[place].Type == 0)
            {
                GravDataIn[j].Soft = SphP[place].Hsml;
            }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
            GravDataIn[j].AGS_Hsml = P[place].AGS_Hsml;
#endif // ADAPTGRAVSOFT
        }

        /* exchange particle data */
        TIMER_PUSH(4, "subfind_comm");
        templated_exchange_data_Sendrecv(GravDataIn, GravDataGet, TAG_POTENTIAL_A, Send_offset, Send_count, Recv_offset,
                                         Recv_count);
        TIMER_POP(4, "subfind_comm");

        myfree(GravDataIn);
        PotDataResult = (struct potdata_out *)mymalloc("PotDataResult", nimport * sizeof(struct potdata_out));
        PotDataOut = (struct potdata_out *)mymalloc("PotDataOut", nexport * sizeof(struct potdata_out));

        /* now do the particles that were sent to us */
        for (int j = 0; j < nimport; j++)
        {
            subfind_force_treeevaluate_potential(j, 1, &dummy, &dummy);
        }

        if (i >= num)
        {
            ndone_flag = 1;
        }
        else
        {
            ndone_flag = 0;
        }

        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        TIMER_PUSH(4, "subfind_comm");
        templated_exchange_data_Sendrecv(PotDataResult, PotDataOut, TAG_POTENTIAL_B, Recv_offset, Recv_count,
                                         Send_offset, Send_count);
        TIMER_POP(4, "subfind_comm");
        /* add the results to the local particles */
        for (int j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;

            P[place].u.DM_Potential += PotDataOut[j].Potential;
        }
        myfree(PotDataOut);
        myfree(PotDataResult);
        myfree(GravDataGet);
    } while (ndone < NTask);

    if (All.ComovingIntegrationOn)
    {
        atime = All.Time;
    }
    else
    {
        atime = 1;
    }

    for (int i = 0; i < num; i++)
    {
        if (phase == 1)
        {
            if (P[d[i].index].v.DM_BindingEnergy <= weakly_bound_limit)
            {
                continue;
            }
        }

#ifndef ADAPTGRAVSOFT
        P[d[i].index].u.DM_Potential += P[d[i].index].Mass / All.SofteningTable[P[d[i].index].Type];
#else // ADAPTGRAVSOFT
        P[d[i].index].u.DM_Potential += P[d[i].index].Mass * 2.8 / P[d[i].index].AGS_Hsml;
#endif // ADAPTGRAVSOFT
        P[d[i].index].u.DM_Potential *= All.GravityConstantInternal / atime;

        if (All.TotN_gas > 0 && (FOF_SECONDARY_LINK_TYPES & 1) == 0 && (FOF_PRIMARY_LINK_TYPES & 1) == 0 &&
            All.OmegaBaryon > 0)
        {
            P[d[i].index].u.DM_Potential *= All.Omega0 / (All.Omega0 - All.OmegaBaryon);
        }
    }

    myfree(DataNodeList);
    myfree(DataIndexTable);
}
#endif // SUBFIND
