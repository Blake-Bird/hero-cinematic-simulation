/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Perform MFM limiter calculation.
 *
 * This file contains the "second loop" of the MFM limiter calculation.
 * Compute slope limiters for all active particles on this step.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../Hydro/kernel.hpp"
#include "../System/communication.hpp"
#include "Matrix.hpp"
#include "mfm_limiters.hpp"

#include "../CodeBase/greentree.hpp"

struct slopedata_in *SlopeDataIn, *SlopeDataGet;
struct slopedata_out *SlopeDataResult, *SlopeDataOut;

/**
 * @brief Create reduced particle data input structure for MFM slope limiter computation.
 *
 * This function prepares input data for the MFM slope limiter computation based on the properties of a given particle.
 *
 * @param in Pointer to the slopedata_in structure to store the input data for the Hydra computation.
 * @param i Index of the particle for which the input data is prepared.
 *
 * The function prepares the following input data for the Hydra computation:
 * - Particle position
 * - primitive Fluid Vector
 * - gradient of primitive Fluid variables
 * - some properties related to Springel 2009 limiter
 */
void particle2in_slope(struct slopedata_in *in, int i)
{
    // in->ID = P[i].ID;
    in->Hsml = P[i].Hsml;
    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
    }
    for (int var = 0; var < numvar; var++)
    {
        in->Wprim[var] = SphP[i].Wprim[var];
        in->gradW[var] = SphP[i].gradW[var];
#if MFM_SLOPE_LIMITER == LIMITER_AREPO
        in->dWmax[var] = SphP[i].Wmax[var] - SphP[i].Wprim[var];
        in->dWmin[var] = SphP[i].Wmin[var] - SphP[i].Wprim[var];
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO
#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            in->Wprim_dust[idust][var] = SphP[i].Wprim_dust[idust][var];
            in->gradW_dust[idust][var] = SphP[i].gradW_dust[idust][var];
#if MFM_SLOPE_LIMITER == LIMITER_AREPO
            in->dWmax_dust[idust][var] = SphP[i].Wmax_dust[idust][var] - SphP[i].Wprim_dust[idust][var];
            in->dWmin_dust[idust][var] = SphP[i].Wmin_dust[idust][var] - SphP[i].Wprim_dust[idust][var];
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO
        }
#endif // GT_DUST
    }
}

/**
 * @brief Updates particle properties using output data from the MFM slope limiter computation.
 *
 * This function updates the slope limiter based on the output data from the MFM slope limiter computation.
 *
 * @param out Pointer to the slopedata_out structure containing the output data from the slope limiter computation.
 * @param i Index of the particle to be updated.
 * @param mode Specifies the mode of updating. A value of 0 indicates a non-assigned particle, while a value of 1
 * indicates an assigned particle.
 *
 * The function updates the following properties of the particle:
 * - slope limiter (depending on the limiter being active)
 */
void out2particle_slope(struct slopedata_out *out, int i, int mode)
{
    if (P[i].Type == type_gas)
    {
#if MFM_SLOPE_LIMITER == LIMITER_TVD_SCALAR || MFM_SLOPE_LIMITER == LIMITER_AREPO
        for (int k = 0; k < numvar; k++)
        {
            ASSIGN_MIN(SphP[i].AlphaSlope[k], out->AlphaSlope[k], mode);
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                ASSIGN_MIN(SphP[i].AlphaSlope_dust[idust][k], out->AlphaSlope_dust[idust][k], mode);
            }
#endif // GT_DUST
        }
#endif // MFM_SLOPE_LIMITER == LIMITER_TVD_SCALAR || MFM_SLOPE_LIMITER == LIMITER_AREPO
    }
}

/**
 * @brief Check if a particle is active for slope limiter computation.
 *
 * A particle is considered active for slope limiters if it is type 0 and, optionally,
 * if it is not a wind particle, has a positive mass (with BLACK_HOLES), and
 * a non-zero ID (FG_REMOVE_BND).
 *
 * @param n Index of the particle.
 * @return True if the particle is active for slope limiter computation, false otherwise.
 */
static inline int slope_isactive(int n)
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

/**
 * @brief Compute the MFM slope limiter of gas particles.
 *
 * This function is the driver routine for the calculation of MFM slope limiters
 * for all active particles. Depending on the limiter, this can require an
 * additional neighbor loop or only a single loop over all particles.
 */
void mfm_limiters(void)
{
#if MFM_SLOPE_LIMITER == LIMITER_TVD_SCALAR ||                                                                         \
    MFM_SLOPE_LIMITER ==                                                                                               \
        LIMITER_AREPO // skip the additional neighbor loop for all limiters, where it is not necessary.

    TIMER_PUSH(2, "Initializing");

    int NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer;", NTaskTimesNumPart * sizeof(int));

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize =
        (int)((MyBufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct slopedata_in) +
               sizeof(struct slopedata_out) + sizemax(sizeof(struct slopedata_in), sizeof(struct slopedata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "MFM limiters: using %g MB for buffering, %g MB remaining for other buffers", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    TIMER_POP(2, "Initializing");

#endif // MFM_SLOPE_LIMITER == LIMITER_TVD_SCALAR || MFM_SLOPE_LIMITER == LIMITER_AREPO

#if MFM_SLOPE_LIMITER == LIMITER_TVD_SCALAR ||                                                                         \
    MFM_SLOPE_LIMITER ==                                                                                               \
        LIMITER_AREPO // skip the additional neighbor loop for all limiters, where it is not necessary.
    // begin with this index
    NextParticle = FirstActiveParticle;

    int ndone, group_particles = 1;

    int DoPrimaryOnGPU = 0;
    int should_drift = 1;
    do
    {
        TIMER_PUSH(2, "Primary");
        BufferFullFlag = 0;
        Nexport = 0;
        int save_NextParticle = NextParticle;

        for (int j = 0; j < NTask; j++)
        {
            Exportflag[j] = -1;
        }

        TIMER_PUSH(1, "MFMLIM_COMPUTE");

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        ar_green_evaluate_primary(slope_isactive, slope_evaluate, Module_MfmLimiters, TreeWalkType_GreenTreeWalk,
                                  NextParticleLoop_ActiveParticleList, NeighbourSearchCriteria_Density, &ThisThread,
                                  DoPrimaryOnGPU, should_drift, group_particles, ProcessedFlag_mfm_limiters);

        TIMER_POP(1, "MFMLIM_COMPUTE");
        TIMER_POPPUSH(2, "Primary", "Exchange");
        TIMER_PUSH(1, "MFMLIM_PREPARE");

        /* if we are not grouping particles and SendRecv fails, then we panic as there is no memory to process even a
         * single particles */
        int panic_on_error = !group_particles;
        int is_send_receive_ok =
            prepare_sendrec_offset_send(save_NextParticle, panic_on_error); // 1 means panic on bufer too small

        TIMER_POP(1, "MFMLIM_PREPARE");

        if (!is_send_receive_ok)
        {
            TIMER_POPPUSH(2, "Exchange", "Primary");
            group_particles = 0;
            continue;
        }

        TIMER_PUSH(1, "MFMLIM_COPY_IN");

        SlopeDataGet = (struct slopedata_in *)mymalloc("SlopeDataGet", Nimport * sizeof(struct slopedata_in));
        DataNodeListGet = (struct data_nodelist *)mymalloc("DataNodeListGet", Nimport * sizeof(struct data_nodelist));
        SlopeDataIn = (struct slopedata_in *)mymalloc("SlopeDataIn", Nexport * sizeof(struct slopedata_in));
        DataNodeListIn = (struct data_nodelist *)mymalloc("DataNodeListIn", Nexport * sizeof(struct data_nodelist));

        /* prepare particle data for export */
        for (int j = 0; j < Nexport; j++)
        {
            int place = DataIndexTable[j].Index;
            particle2in_slope(&SlopeDataIn[j], place);
        }

        copy_data_nodelist();

        TIMER_POPPUSH(1, "MFMLIM_COPY_IN", "MFMLIM_COMM_EXC");
        templated_exchange_data_in_get(SlopeDataIn, SlopeDataGet, 1, TAG_SLOPE_A);
        TIMER_POP(1, "MFMLIM_COMM_EXC");

        TIMER_POPPUSH(2, "Exchange", "Secondary");

        TIMER_PUSH(1, "MFMLIM_ZERO_AND_COPY");
        myfree(DataNodeListIn);
        myfree(SlopeDataIn);
        SlopeDataResult = (struct slopedata_out *)mymalloc("SlopeDataResult", Nimport * sizeof(struct slopedata_out));
        SlopeDataOut = (struct slopedata_out *)mymalloc("SlopeDataOut", Nexport * sizeof(struct slopedata_out));

        report_memory_usage(&HighMark_mfmlim, "MFM_FLUXES");
        TIMER_POPPUSH(1, "MFMLIM_ZERO_AND_COPY", "MFMLIM_COMPUTE");

        // now do the particles that were sent to us
        NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        ar_green_evaluate_secondary(SlopeDataGet, slope_evaluate, Module_MfmLimiters, TreeWalkType_GreenTreeWalk,
                                    NeighbourSearchCriteria_Density, &ThisThread, should_drift);

        TIMER_POPPUSH(1, "MFMLIM_COMPUTE", "MFMLIM_WAIT");

        int ndone_flag;
        if (NextParticle < 0)
        {
            ndone_flag = 1;
        }
        else
        {
            ndone_flag = 0;
        }

        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        TIMER_POP(1, "MFMLIM_WAIT");
        TIMER_POPPUSH(2, "Secondary", "Send_Results");
        TIMER_PUSH(1, "MFMLIM_COMM_EXC");
        templated_exchange_data_Sendrecv(SlopeDataResult, SlopeDataOut, TAG_SLOPE_B, Recv_offset, Recv_count,
                                         Send_offset, Send_count);
        TIMER_POPPUSH(1, "MFMLIM_COMM_EXC", "MFMLIM_COPY");

        // add the result to the local particles
        for (int j = 0; j < Nexport; j++)
        {
            int place = DataIndexTable[j].Index;
            out2particle_slope(&SlopeDataOut[j], place, 1);
        }
        TIMER_POP(1, "MFMLIM_COPY");
        TIMER_POP(2, "Send_Results");

        myfree(SlopeDataOut);
        myfree(SlopeDataResult);
        myfree(DataNodeListGet);
        myfree(SlopeDataGet);
    } while (ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(RecyclingBuffer);
    myfree(Ngblist);
#endif // MFM_SLOPE_LIMITER == LIMITER_TVD_SCALAR || MFM_SLOPE_LIMITER == LIMITER_AREPO

    TIMER_PUSH(2, "Final_Operations");
    TIMER_PUSH(1, "MFMLIM_FINAL");

    // do final operations on results
#ifdef _OPENMP
#pragma omp parallel for
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        if (slope_isactive(i))
        {
            slopeLimiter->slope_limiter_final(i);
        }
    }
    TIMER_POP(1, "MFMLIM_FINAL");
    TIMER_POP(2, "Final_Operations");
}

/**
 * @brief Compute MFM slope limiter for a target particle.
 *
 * This function computes the MFM slope limiter for a target particle. It can handles particles residing in the
 * communication buffer.
 *
 * @param target Index of the target particle.
 * @param mode Flag indicating the mode of operation:
 *             - 0: Normal mode for local particles.
 *             - 1: Active target mode.
 *             - 2: Mode for particles in the communication buffer.
 *             - 3: Return the number of neighbors.
 * @param ngblist Array containing the indices of neighboring particles.
 * @param extended_numngb_inbox Extended number of neighboring particles in the communication buffer.
 * @param i_startnode Pointer to the start node of the tree walk.
 * @param node Current node in the tree walk.
 * @param i_node Index of the current node in the tree walk.
 * @param additional_data Additional data required for computation (not used in this function).
 *
 * @return The number of neighbors if mode is 3, otherwise returns 0.
 */
int slope_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                   int i_node, void *data)
{
#ifdef ACC_INCLUDE_HYDRA
    int target;
    int extended_numngb_inbox = numngb;
#else // ACC_INCLUDE_HYDRA
    int numngb;
#endif // ACC_INCLUDE_HYDRA

    struct slopedata_in local;
    slopedata_out out{};

    for (int k = 0; k < numvar; k++)
    {
        out.AlphaSlope[k] = 1.0;
#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            out.AlphaSlope_dust[idust][k] = 1.0;
        }
#endif // GT_DUST
    }

    if (mode == 0 || mode == 2 || mode == 3)
    {
        particle2in_slope(&local, target);
    }
    else
    {
        local = SlopeDataGet[target];
    }

    MyAtLeastDouble h2 = local.Hsml * local.Hsml;

    int startnode = node;
    if (i_node > 0)
    {
        out = SlopeDataResult[target];
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            if (extended_numngb_inbox == -1)
            {
                return -1;
            }
            numngb = extended_numngb_inbox;
#ifndef ACC_INCLUDE_HYDRA
            startnode = *i_startnode;
            ENDRUN_IF(
                numngb > NumPart / DENSITY_LESS_NGB_FACTOR,
                "hydra: Number of neighbours found (%d) exceeds allowed number (%d) derived from "
                "NumPart/DENSITY_LESS_NGB_FACTOR, to continue, decreas DENSITY_LESS_NGB_FACTOR (currently %f, min "
                "possible val would be 1) !!\n",
                numngb, (int)(NumPart / DENSITY_LESS_NGB_FACTOR), DENSITY_LESS_NGB_FACTOR);
#else // ACC_INCLUDE_HYDRA
            startnode = -1;
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_HYDRA
            if (mode == 3)
            {
                return numngb;
            }
#endif // ACC_HYDRA
            if (numngb < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb; n++)
            {
                int j = ngblist[n];

                // Skip self-interaction
                // if (local.ID == SphP[j].ID) continue;

#ifdef BLACK_HOLES
                if (P[j].Mass == 0)
                {
                    continue;
                }
#endif // BLACK_HOLES
#ifdef FG_REMOVE_BND
                if (P[j].ID == 0)
                {
                    continue;
                }
#endif // FG_REMOVE_BND
#ifdef WINDS
                if (SphP[j].DelayTime > 0) /* decouple wind particles */
                {
                    continue;
                }
#endif // WINDS

                Vector<NUMDIMS, MyAtLeastDouble> dr;
                for (int k = 0; k < NUMDIMS; k++)
                {
                    dr[k] = P[j].Pos[k] - local.Pos[k];
                }
#ifdef PERIODIC
                dr[0] = NEAREST_X(dr[0]);
#if NUMDIMS > 1
                dr[1] = NEAREST_Y(dr[1]);
#endif // NUMDIMS > 1
#if NUMDIMS == 3
                dr[2] = NEAREST_Z(dr[2]);
#endif // NUMDIMS == 3
#endif // PERIODIC
                MyAtLeastDouble r2 = dr.GetMagnitudeSquared();

                if (r2 < h2)
                {
                    slopeLimiter->compute_slope_limiter(dr, local, j, out);
                }
            } // for(int n = 0; n < numngb_inbox; n++)
        } // while(startnode >= 0)
    } // while(startnode >= 0)

    if (mode == 3)
    {
        return 0;
    }

#ifdef ACC_INCLUDE_HYDRA
    target = active_target;
#endif // ACC_INCLUDE_HYDRA
    /* Now collect the result at the right place */
    if (mode == 0 || mode == 2)
    {
#ifdef ACC_INCLUDE_HYDRA
        SlopeDataResult[target] = out;
#else // ACC_INCLUDE_HYDRA
        out2particle_slope(&out, target, 0);
#endif // ACC_INCLUDE_HYDRA
    }
    else
    {
        SlopeDataResult[target] = out;
    }
    return 0;
}
