/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Computation of the gravitational potential.
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

#if !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL))

/**
 * @brief Compute the gravitational potential for all particles.
 *
 * This function computes the gravitational potential for ALL the particles.
 * First, the (short-range) tree potential is computed, and then, if needed,
 * the long range PM potential is added.
 */
void compute_potential(void)
{
    TIMER_PUSH(1, "COMPUTE_POTENTIAL");

#ifndef NOGRAVITY
    int ret, recvTask;
    int ndone, ndone_flag, dummy;
    int ngrp, place, nexport, nimport;
    double fac;
    MPI_Status status;

    if (All.ComovingIntegrationOn)
    {
        set_softenings();
    }

    VERBOSE(3, "Start computation of potential for all particles...");

    if (TreeReconstructFlag)
    {
        VERBOSE(2, "Tree construction.");

#if defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)
        rearrange_particle_sequence();
#endif // defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

        force_treebuild(NumPart, NULL);

        TreeReconstructFlag = 0;

        VERBOSE(2, "Tree construction done.");
    }

    TIMER_PUSH(1, "POTENTIAL");

    /* allocate buffers to arrange communication */
    All.BunchSize =
        (int)((All.BufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct gravdata_in) +
               sizeof(struct potdata_out) + sizemax(sizeof(struct gravdata_in), sizeof(struct potdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

#ifndef SUBFIND_RESHUFFLE_AND_POTENTIAL
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Ti_current != All.Ti_Current)
        {
            drift_particle(i, All.Ti_Current);
        }
    }
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
    int i = 0; /* beginn with this index */

    do
    {
        for (int j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Exportflag[j] = -1;
        }

        /* do local particles and prepare export list */
        for (nexport = 0; i < NumPart; i++)
        {
#ifndef PMGRID
            ret = force_treeevaluate_potential(i, 0, &nexport, Send_count);
#else // PMGRID
            ret = force_treeevaluate_potential_shortrange(i, 0, &nexport, Send_count);
#endif // PMGRID
            if (ret < 0)
            {
                break; /* export buffer has filled up */
            }
        }

        MYSORT_DATAINDEX(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);

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
            GravDataIn[j].OldAcc = P[place].OldAcc;

            for (int k = 0; k < NODELISTLENGTH; k++)
            {
                GravDataIn[j].NodeList[k] = DataNodeList[DataIndexTable[j].IndexGet].NodeList[k];
            }
        }

        /* exchange particle data */
        for (ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;

            if (recvTask < NTask)
            {
                if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* get the particles */
                    MPI_Sendrecv(&GravDataIn[Send_offset[recvTask]], Send_count[recvTask] * sizeof(struct gravdata_in),
                                 MPI_BYTE, recvTask, TAG_POTENTIAL_A, &GravDataGet[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct gravdata_in), MPI_BYTE, recvTask, TAG_POTENTIAL_A,
                                 MYMPI_COMM_WORLD, &status);
                }
            }
        }

        myfree(GravDataIn);
        PotDataResult = (struct potdata_out *)mymalloc("PotDataResult", nimport * sizeof(struct potdata_out));
        PotDataOut = (struct potdata_out *)mymalloc("PotDataOut", nexport * sizeof(struct potdata_out));

        /* now do the particles that were sent to us */
        for (int j = 0; j < nimport; j++)
        {
#ifndef PMGRID
            force_treeevaluate_potential(j, 1, &dummy, &dummy);
#else // PMGRID
            force_treeevaluate_potential_shortrange(j, 1, &dummy, &dummy);
#endif // PMGRID
        }

        if (i >= NumPart)
        {
            ndone_flag = 1;
        }
        else
        {
            ndone_flag = 0;
        }
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        TIMER_PUSH(1, "POTENTIAL_COMM");
        templated_exchange_data_Sendrecv(PotDataResult, PotDataOut, TAG_POTENTIAL_B, Recv_offset, Recv_count,
                                         Send_offset, Send_count);
        TIMER_POP(1, "POTENTIAL_COMM");
        /* add the results to the local particles */
        for (int j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;

            // P[place].p.dPotential += PotDataOut[j].Potential;
            P[place].Potential += PotDataOut[j].Potential;
        }

        myfree(PotDataOut);
        myfree(PotDataResult);
        myfree(GravDataGet);
    } while (ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);

#ifndef ADAPTGRAVSOFT
    /* add correction to exclude self-potential */

    for (int i = 0; i < NumPart; i++)
    {
        /* remove self-potential */
        P[i].Potential += P[i].Mass / All.SofteningTable[P[i].Type];

        if (All.ComovingIntegrationOn)
        {
            if (All.PeriodicBoundariesOn)
            {
                P[i].Potential -=
                    2.8372975 * pow(P[i].Mass, 2.0 / 3) *
                    pow(All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal), 1.0 / 3);
            }
        }
    }
#endif // ADAPTGRAVSOFT

    /* multiply with the gravitational constant */

    for (int i = 0; i < NumPart; i++)
    {
        P[i].Potential *= All.GravityConstantInternal;
    }

    if (All.ComovingIntegrationOn)
    {
#ifndef PERIODIC
        fac = -0.5 * All.Omega0 * All.Hubble * All.Hubble;

        for (int i = 0; i < NumPart; i++)
        {
            double r2 = 0.0;
            for (int k = 0; k < 3; k++)
            {
                r2 += P[i].Pos[k] * P[i].Pos[k];
            }

            P[i].Potential += fac * r2;
        }
#endif // PERIODIC
    }
    else
    {
        fac = -0.5 * All.OmegaLambda * All.Hubble * All.Hubble;
        if (fac != 0)
        {
            for (int i = 0; i < NumPart; i++)
            {
                double r2 = 0.0;
                for (int k = 0; k < 3; k++)
                {
                    r2 += P[i].Pos[k] * P[i].Pos[k];
                }

                P[i].Potential += fac * r2;
            }
        }
    }

    VERBOSE(4, "potential done.");

#else // NOGRAVITY
    for (int i = 0; i < NumPart; i++)
    {
        P[i].Potential = 0;
    }
#endif // NOGRAVITY

    TIMER_POP(1, "POTENTIAL");
    TIMER_POP(1, "COMPUTE_POTENTIAL");
}
#endif // !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL))
