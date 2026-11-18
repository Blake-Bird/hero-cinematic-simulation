/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Distribute groups when running Subfind in parallel.
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
#include "fof.hpp"
#include "subfind.hpp"

#if defined(SUBFIND) || defined(FOF_EXTENDED_PROPERTIES)
/**
 * @brief Distributes groups between MPI tasks based on specified mode.
 *
 * This function distributes groups between MPI tasks according to the specified mode.
 * It calculates the number of groups to send to each task, allocates memory for sending
 * and receiving buffers, and performs the necessary MPI communication to exchange group data.
 * Depending on the mode, groups are distributed either based on their target task or
 * their original task.
 *
 * @param mode The mode of group distribution:
 *             - 1: Distribute groups based on their target task or original task.
 *             - 2: Distribute groups based on their original task.
 *
 * The function performs the following steps:
 * - Counts the number of groups to be sent to each task.
 * - Allocates memory for sending and receiving buffers.
 * - Performs MPI_Alltoall to exchange the counts of groups to be received.
 * - Determines send and receive offsets for group data.
 * - Copies group data to the appropriate send buffers.
 * - Initiates non-blocking sends and receives for group data exchange.
 * - Waits for all non-blocking communication to complete.
 * - Updates the total number of groups after the exchange.
 */
void subfind_distribute_groups(int mode)
{
    int nexport = 0, nimport = 0, target = -1;
    group_properties *send_Group;

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Ngroups; i++)
    {
        if (mode == 1)
        {
            if (Group[i].GrNr > Ncollective)
            {
                target = (Group[i].GrNr - Ncollective - 1) % (NTask - (Ncollective * ntaskspergroup)) +
                         (Ncollective * ntaskspergroup);
            }
            else
            {
                target = grouptasks[Group[i].GrNr - 1];
            }
        }
        if (mode == 2)
        {
            target = (Group[i].GrNr - 1) % NTask;
        }
        if (target != ThisTask)
        {
            Send_count[target]++;
        }
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    Recv_offset[0] = Send_offset[0] = 0;
    for (int i = 0; i < NTask; i++)
    {
        nimport += Recv_count[i];
        nexport += Send_count[i];

        if (i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    send_Group = (group_properties *)mymalloc("send_Group", nexport * sizeof(group_properties));

    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Ngroups; i++)
    {
        if (mode == 1)
        {
            if (Group[i].GrNr > Ncollective)
                target = (Group[i].GrNr - Ncollective - 1) % (NTask - (Ncollective * ntaskspergroup)) +
                         (Ncollective * ntaskspergroup);
            else
            {
                target = grouptasks[Group[i].GrNr - 1];
            }
        }
        if (mode == 2)
        {
            target = (Group[i].GrNr - 1) % NTask;
        }
        if (target != ThisTask)
        {
            send_Group[Send_offset[target] + Send_count[target]] = Group[i];
            Send_count[target]++;

            Group[i] = Group[Ngroups - 1];
            Ngroups--;
            i--;
        }
    }

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* get the group info */
                if ((Send_offset[recvTask] + Send_count[recvTask]) > nexport)
                    VERBOSE_ALL(2,
                                "SubFind: ThisTask%d WARNING Iteration:%d send_Group  sends [%d, %d] > nexport = %d  ",
                                ThisTask, ngrp, Send_offset[recvTask], Send_offset[recvTask] + Send_count[recvTask] - 1,
                                nexport);

                if ((Ngroups + Recv_offset[recvTask] + Recv_count[recvTask]) > NAllocatedGroups)
                    VERBOSE_ALL(2,
                                "SubFind: ThisTask%d WARNING Iteration:%d receive [ Ngroups(=%d) + "
                                "Recv_offset[recvTask] = %d,  %d] > NAllocatedGroups = %d elements ",
                                ThisTask, ngrp, Ngroups, Ngroups + Recv_offset[recvTask],
                                Ngroups + Recv_offset[recvTask] + Recv_count[recvTask], NAllocatedGroups);
                MPI_Sendrecv(&send_Group[Send_offset[recvTask]], Send_count[recvTask] * sizeof(group_properties),
                             MPI_BYTE, recvTask, TAG_DENS_A, &Group[Ngroups + Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(group_properties), MPI_BYTE, recvTask, TAG_DENS_A,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    Ngroups += nimport;

    myfree(send_Group);
}

/**
 * @brief Distributes particles between MPI tasks based on specified mode.
 *
 * This function distributes particles between MPI tasks according to the specified mode.
 * It calculates the number of particles to send to each task, allocates memory for sending
 * and receiving buffers, and performs the necessary MPI communication to exchange particle data.
 * Depending on the mode, particles are distributed either based on their target task or
 * their origin task.
 *
 * @param mode The mode of particle distribution:
 *             - 0: Distribute particles based on their target task.
 *             - 1: Distribute particles based on their origin task.
 *
 * The function performs the following steps:
 * - Calculates the number of particles to be sent to each task.
 * - Allocates memory for sending and receiving buffers.
 * - Performs MPI_Alltoall to exchange the counts of particles to be received.
 * - Determines send and receive offsets for particle data.
 * - Checks if the total number of particles exceeds the maximum allowed.
 * - Copies particle data to the appropriate send buffers.
 * - Initiates non-blocking sends and receives for particle data exchange.
 * - Waits for all non-blocking communication to complete.
 * - Updates the total number of particles after the exchange.
 */
void subfind_distribute_particles(int mode)
{
    int nexport = 0, nimport = 0;
    int target = 0;
    struct particle_data *partBuf;

    for (int n = 0; n < NTask; n++)
    {
        Send_count[n] = 0;
    }

    for (int n = 0; n < NumPart; n++)
    {
        if (mode == 1)
        {
            if (P[n].targettask != ThisTask)
            {
                Send_count[P[n].targettask] += 1;
            }
        }
        else if (mode == 2)
        {
            if (P[n].origintask != ThisTask)
            {
                Send_count[P[n].origintask] += 1;
            }
        }
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    Send_offset[0] = Recv_offset[0] = 0;
    for (int i = 0; i < NTask; i++)
    {
        nimport += Recv_count[i];
        nexport += Send_count[i];

        if (i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    ENDRUN_IF(NumPart - nexport + nimport > All.MaxPart,
              "on task=%d the maximum particle number All.MaxPart=%d is reached (NumPart=%d togo=%d toget=%d)",
              ThisTask, All.MaxPart, NumPart, nexport, nimport);

    partBuf = (struct particle_data *)mymalloc("partBuf", nexport * sizeof(struct particle_data));

    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int n = 0; n < NumPart; n++)
    {
#ifdef SUBFIND
        if (mode == 0)
        {
            if (!(P[n].GrNr > Ncollective && P[n].GrNr <= TotNgroups)) /* particle is in small group */
            {
                continue;
            }

            target = (P[n].GrNr - 1) % NTask;
        }
        else if (mode == 1)
#else // SUBFIND
        if (mode == 1)
#endif // SUBFIND
        {
            target = P[n].targettask;
        }
        else if (mode == 2)
        {
            target = P[n].origintask;
        }

        if (target != ThisTask)
        {
            partBuf[Send_offset[target] + Send_count[target]] = P[n];
            Send_count[target]++;

            P[n] = P[NumPart - 1];
            NumPart--;
            n--;
        }
    }

#ifdef USE_ISEND_IRECV
    int n_requests = 0, max_requests = 2;

    MPI_Request *requests = (MPI_Request *)mymalloc("requests", max_requests * NTask * sizeof(MPI_Request));

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (Recv_count[target] > 0)
                MPI_Irecv(P + NumPart + Recv_offset[target], Recv_count[target] * sizeof(struct particle_data),
                          MPI_BYTE, target, TAG_PDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
        }
    }

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (Send_count[target] > 0)
                MPI_Isend(partBuf + Send_offset[target], Send_count[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
        }
    }

    MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

    ENDRUN_IF(n_requests > max_requests * NTask, "Not enough memory reserved for requests: %d > %d !", n_requests,
              max_requests * NTask);

    myfree(requests);

#else // USE_ISEND_IRECV

    /*
       double t0, t1;

       t0 = second();
     */

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        /*
           if(NTask > 512)
           {
           if(ThisTask == 0)
           {
           t1 = second();
           printf("ngrp=%d %g sec\n", ngrp, timediff(t0, t1));
           fflush(stdout);
           t0 = second();
           }
           }
         */
        int target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            MPI_Sendrecv(partBuf + Send_offset[target], Send_count[target] * sizeof(struct particle_data), MPI_BYTE,
                         target, TAG_PDATA, P + NumPart + Recv_offset[target],
                         Recv_count[target] * sizeof(struct particle_data), MPI_BYTE, target, TAG_PDATA,
                         MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

#endif // USE_ISEND_IRECV

    NumPart += nimport;

    myfree(partBuf);
}
#endif // defined(SUBFIND) || defined(FOF_EXTENDED_PROPERTIES)

#ifdef SUBFIND
/**
 * @brief Exchange particles between tasks for the subfind algorithm.
 *
 * This function facilitates the exchange of particle data between different MPI tasks
 * as part of the subfind algorithm. It redistributes particles according to their
 * target task, ensuring that each task has the correct particles for subsequent
 * calculations.
 *
 * The function performs the following steps:
 * - Determine the particles to be sent to each task.
 * - Allocate memory for sending and receiving buffers.
 * - Prepare and send particle data to the appropriate tasks.
 * - Receive particle data from other tasks.
 * - Update local particle arrays and adjust memory allocation as necessary.
 */
void subfind_exchange(void)
{
    TIMER_PUSH(1, "SUBFIND_EXCHANGE");
    int count_togo = 0, count_togo_sph = 0, count_get = 0, count_get_sph = 0;
    int *count, *count_sph, *offset, *offset_sph;
    int *count_recv, *count_recv_sph, *offset_recv, *offset_recv_sph;
    int *local_toGet, *local_toGetSph;
    int *local_toGo, *local_toGoSph;
    struct particle_data *partBuf;
    struct sph_particle_data *sphBuf;

#ifdef SFR
    int count_togo_stars = 0, count_get_stars = 0;
    int *count_stars, *offset_stars;
    int *count_recv_stars, *offset_recv_stars;
    int *local_toGetStars;
    int *local_toGoStars;
    struct met_particle_data *metBuf;

    for (int i = 0; i < N_stars; i++)
    {
        if (MetP[i].PID >= NumPart || P[MetP[i].PID].Type != 4)
        {
            ENDRUN("Task %d: i=%d/%d, pid=%u, type=%d", ThisTask, i, N_stars, MetP[i].PID, P[MetP[i].PID].Type);
        }
    }
#endif // SFR

#ifdef BLACK_HOLES
    int count_togo_BHs = 0, count_get_BHs = 0;
    int *count_BHs, *offset_BHs;
    int *count_recv_BHs, *offset_recv_BHs;
    int *local_toGetBHs;
    int *local_toGoBHs;
    struct bh_particle_data *BHBuf;

    for (int i = 0; i < N_BHs; i++)
    {
        if (BHP[i].PID >= NumPart || P[BHP[i].PID].Type != 5)
        {
            ENDRUN(0, "Task %d: i=%d/%d, pid=%u, type=%d", ThisTask, i, N_BHs, BHP[i].PID, P[BHP[i].PID].Type);
        }
    }
#endif // BLACK_HOLES

    count = (int *)mymalloc("count", NTask * sizeof(int));
    count_sph = (int *)mymalloc("count_sph", NTask * sizeof(int));
    offset = (int *)mymalloc("offset", NTask * sizeof(int));
    offset_sph = (int *)mymalloc("offset_sph", NTask * sizeof(int));

    count_recv = (int *)mymalloc("count_recv", NTask * sizeof(int));
    count_recv_sph = (int *)mymalloc("count_recv_sph", NTask * sizeof(int));
    offset_recv = (int *)mymalloc("offset_recv", NTask * sizeof(int));
    offset_recv_sph = (int *)mymalloc("offset_recv_sph", NTask * sizeof(int));

    local_toGet = (int *)mymalloc("local_toGet", NTask * sizeof(int));
    local_toGetSph = (int *)mymalloc("local_toGetSph", NTask * sizeof(int));
    local_toGo = (int *)mymalloc("local_toGo", NTask * sizeof(int));
    local_toGoSph = (int *)mymalloc("local_toGoSph", NTask * sizeof(int));

#if defined(SFR)
    count_stars = (int *)mymalloc("count_stars", NTask * sizeof(int));
    offset_stars = (int *)mymalloc("offset_stars", NTask * sizeof(int));
    count_recv_stars = (int *)mymalloc("count_recv_stars", NTask * sizeof(int));
    offset_recv_stars = (int *)mymalloc("offset_recv_stars", NTask * sizeof(int));
    local_toGetStars = (int *)mymalloc("local_toGetStars", NTask * sizeof(int));
    local_toGoStars = (int *)mymalloc("local_toGoStars", NTask * sizeof(int));
#endif // SFR

#ifdef BLACK_HOLES
    count_BHs = (int *)mymalloc("count_BHs", NTask * sizeof(int));
    offset_BHs = (int *)mymalloc("offset_BHs", NTask * sizeof(int));
    count_recv_BHs = (int *)mymalloc("count_recv_BHs", NTask * sizeof(int));
    offset_recv_BHs = (int *)mymalloc("offset_recv_BHs", NTask * sizeof(int));
    local_toGetBHs = (int *)mymalloc("local_toGetBHs", NTask * sizeof(int));
    local_toGoBHs = (int *)mymalloc("local_toGoBHs", NTask * sizeof(int));
#endif // BLACK_HOLES

    for (int n = 0; n < NTask; n++)
    {
        local_toGo[n] = 0;
        local_toGoSph[n] = 0;
#if defined(SFR)
        local_toGoStars[n] = 0;
#endif // defined(SFR)
#ifdef BLACK_HOLES
        local_toGoBHs[n] = 0;
#endif // BLACK_HOLES
    }

    for (int n = 0; n < NumPart; n++)
    {
        if (P[n].targettask != ThisTask)
        {
            local_toGo[P[n].targettask]++;
            if (P[n].Type == 0)
            {
                local_toGoSph[P[n].targettask]++;
            }
#if defined(SFR)
            if (P[n].Type == 4)
            {
                local_toGoStars[P[n].targettask]++;
            }
#endif // defined(SFR)
#ifdef BLACK_HOLES
            if (P[n].Type == 5)
            {
                local_toGoBHs[P[n].targettask]++;
            }
#endif // BLACK_HOLES
        }
    }

    MPI_Alltoall(local_toGo, 1, MPI_INT, local_toGet, 1, MPI_INT, MYMPI_COMM_WORLD);
    MPI_Alltoall(local_toGoSph, 1, MPI_INT, local_toGetSph, 1, MPI_INT, MYMPI_COMM_WORLD);
#if defined(SFR)
    MPI_Alltoall(local_toGoStars, 1, MPI_INT, local_toGetStars, 1, MPI_INT, MYMPI_COMM_WORLD);
#endif // defined(SFR)
#ifdef BLACK_HOLES
    MPI_Alltoall(local_toGoBHs, 1, MPI_INT, local_toGetBHs, 1, MPI_INT, MYMPI_COMM_WORLD);
#endif // BLACK_HOLES

    int prec_offset, prec_count;
    int *decrease;

    decrease = (int *)mymalloc("decrease", NTask * sizeof(int));
    offset_sph[0] = 0, decrease[0] = 0;
    for (int i = 1; i < NTask; i++)
    {
        offset_sph[i] = offset_sph[i - 1] + local_toGoSph[i - 1];
        decrease[i] = local_toGoSph[i - 1];
    }

    prec_offset = offset_sph[NTask - 1] + local_toGoSph[NTask - 1];

#if defined(SFR)
    offset_stars[0] = prec_offset;

    for (int i = 1; i < NTask; i++)
    {
        offset_stars[i] = offset_stars[i - 1] + local_toGoStars[i - 1];
        decrease[i] += local_toGoStars[i - 1];
    }
    prec_offset = offset_stars[NTask - 1] + local_toGoStars[NTask - 1];
#endif // defined(SFR)

#ifdef BLACK_HOLES
    offset_BHs[0] = prec_offset;

    for (int i = 1; i < NTask; i++)
    {
        offset_BHs[i] = offset_BHs[i - 1] + local_toGoBHs[i - 1];
        decrease[i] += local_toGoBHs[i - 1];
    }
    prec_offset = offset_BHs[NTask - 1] + local_toGoBHs[NTask - 1];
#endif // BLACK_HOLES

    offset[0] = prec_offset;

    for (int i = 1; i < NTask; i++)
    {
        offset[i] = offset[i - 1] + (local_toGo[i - 1] - decrease[i]);
    }

    myfree(decrease);

    for (int i = 0; i < NTask; i++)
    {
        count_togo += local_toGo[i];
        count_togo_sph += local_toGoSph[i];

        count_get += local_toGet[i];
        count_get_sph += local_toGetSph[i];

#if defined(SFR)
        count_togo_stars += local_toGoStars[i];
        count_get_stars += local_toGetStars[i];
#endif // defined(SFR)

#ifdef BLACK_HOLES
        count_togo_BHs += local_toGoBHs[i];
        count_get_BHs += local_toGetBHs[i];
#endif // BLACK_HOLES
    }

    if (NumPart + count_get - count_togo > All.MaxPart)
    {
        ENDRUN("Task=%d NumPart(%d)+count_get(%d)-count_togo(%d)=%d  All.MaxPart=%d\n    you have to increase "
               "PartAllocFactor (currently=%f)\n",
               ThisTask, NumPart, count_get, count_togo, NumPart + count_get - count_togo, All.MaxPart,
               All.PartAllocFactor);
    }

    if (N_gas + count_get_sph - count_togo_sph > All.MaxPartSph)
    {
        ENDRUN("Task=%d N_gas(%d)+count_get_sph(%d)-count_togo_sph(%d)=%d All.MaxPartSph=%d\n    you have to "
               "increase PartAllocFactor (currently=%f)",
               ThisTask, N_gas, count_get_sph, count_togo_sph, N_gas + count_get_sph - count_togo_sph, All.MaxPartSph,
               All.PartAllocFactor);
    }

#ifdef SFR
    if (N_stars + count_get_stars - count_togo_stars > All.MaxPartMet)
    {
        ENDRUN("Task=%d N_stars(%d)+count_get_stars(%d)-count_togo_stars(%d)=%d All.MaxPartMet=%d\n    you can try "
               "to increase SFfactor (currently=%f) or generally increase PartAllocFactor (currently=%f)",
               ThisTask, N_stars, count_get_stars, count_togo_stars, N_stars + count_get_stars - count_togo_stars,
               All.MaxPartMet, All.SFfactor, All.PartAllocFactor);
    }
#endif // SFR

#ifdef BLACK_HOLES
    if (N_BHs + count_get_BHs - count_togo_BHs > All.MaxPartBH)
    {
        VERBOSE_ALL(
            0,
            "Task=%d N_BHs(%d)+count_get_BHs(%d)-count_togo_BHs(%d)=%d All.MaxPartBH=%d \n   you can try to increase "
            "BlackHoleFormationFactor (currently=%f) or generally increase PartAllocFactor (currently=%f)",
            ThisTask, N_BHs, count_get_BHs, count_togo_BHs, N_BHs + count_get_BHs - count_togo_BHs, All.MaxPartBH,
            All.BlackHoleFormationFactor, All.PartAllocFactor);
    }
#endif // BLACK_HOLES

    partBuf = (struct particle_data *)mymalloc("partBuf", count_togo * sizeof(struct particle_data));
    sphBuf = (struct sph_particle_data *)mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));

    for (int i = 0; i < NTask; i++)
    {
        count[i] = count_sph[i] = 0;
    }

#ifdef SFR
    metBuf = (struct met_particle_data *)mymalloc("metBuf", count_togo_stars * sizeof(struct met_particle_data));
    for (int i = 0; i < NTask; i++)
    {
        count_stars[i] = 0;
    }
#endif // SFR

#ifdef BLACK_HOLES
    BHBuf = (struct bh_particle_data *)mymalloc("BHBuf", count_togo_BHs * sizeof(struct bh_particle_data));
    for (int i = 0; i < NTask; i++)
    {
        count_BHs[i] = 0;
    }
#endif // BLACK_HOLES

    for (int n = 0; n < NumPart; n++)
    {
        if (P[n].targettask != ThisTask)
        {
            int target = P[n].targettask;

            if (P[n].Type == 0)
            {
                partBuf[offset_sph[target] + count_sph[target]] = P[n];
                sphBuf[offset_sph[target] + count_sph[target]] = SphP[n];
                count_sph[target]++;
            }
#ifdef SFR
            else if (P[n].Type == 4)
            {
                partBuf[offset_stars[target] + count_stars[target]] = P[n];
                metBuf[offset_stars[target] - offset_stars[0] + count_stars[target]] = MetP[P[n].pt.MetID];
                count_stars[target]++;
            }
#endif // SFR
#ifdef BLACK_HOLES
            else if (P[n].Type == 5)
            {
                partBuf[offset_BHs[target] + count_BHs[target]] = P[n];
                BHBuf[offset_BHs[target] - offset_BHs[0] + count_BHs[target]] = BHP[P[n].pt.BHID];
                count_BHs[target]++;
            }
#endif // BLACK_HOLES
            else
            {
                partBuf[offset[target] + count[target]] = P[n];
                count[target]++;
            }

            if (P[n].Type == 0)
            {
                P[n] = P[N_gas - 1];
                SphP[n] = SphP[N_gas - 1];
                P[N_gas - 1] = P[NumPart - 1];

#ifdef SFR
                if (P[N_gas - 1].Type == 4)
                {
                    MetP[P[N_gas - 1].pt.MetID].PID = N_gas - 1;
                }
#endif // SFR
#ifdef BLACK_HOLES
                if (P[N_gas - 1].Type == 5)
                {
                    BHP[P[N_gas - 1].pt.BHID].PID = N_gas - 1;
                }
#endif // BLACK_HOLES
                NumPart--;
                N_gas--;
                n--;
            }
#ifdef SFR
            else if (P[n].Type == 4)
            {
                MetP[P[n].pt.MetID] = MetP[N_stars - 1];
                P[MetP[N_stars - 1].PID].pt.MetID = P[n].pt.MetID;

                if (n < NumPart - 1)
                {
                    P[n] = P[NumPart - 1];
                    if (P[n].Type == 4)
                    {
                        MetP[P[n].pt.MetID].PID = n;
                    }
#ifdef BLACK_HOLES
                    if (P[n].Type == 5)
                    {
                        BHP[P[n].pt.BHID].PID = n;
                    }
#endif // BLACK_HOLES
                }
                NumPart--;
                N_stars--;
                n--;
            }
#endif // SFR
#ifdef BLACK_HOLES
            else if (P[n].Type == 5)
            {
                BHP[P[n].pt.BHID] = BHP[N_BHs - 1];
                P[BHP[N_BHs - 1].PID].pt.BHID = P[n].pt.BHID;

                if (n < NumPart - 1)
                {
                    P[n] = P[NumPart - 1];
#ifdef SFR
                    if (P[n].Type == 4)
                    {
                        MetP[P[n].pt.MetID].PID = n;
                    }
#endif // SFR
                    if (P[n].Type == 5)
                    {
                        BHP[P[n].pt.BHID].PID = n;
                    }
                }
                NumPart--;
                N_BHs--;
                n--;
            }
#endif // BLACK_HOLES
            else
            {
                P[n] = P[NumPart - 1];
#ifdef SFR
                if (P[n].Type == 4)
                {
                    MetP[P[n].pt.MetID].PID = n;
                }
#endif // SFR
#ifdef BLACK_HOLES
                if (P[n].Type == 5)
                {
                    BHP[P[n].pt.BHID].PID = n;
                }
#endif // BLACK_HOLES
                NumPart--;
                n--;
            }
        }
    }

    int count_totget;

    count_totget = count_get_sph;
#if defined(SFR)
    count_totget += count_get_stars;
#endif // defined(SFR)
#ifdef BLACK_HOLES
    count_totget += count_get_BHs;
#endif // BLACK_HOLES

    if (count_totget)
    {
        memmove(P + N_gas + count_totget, P + N_gas, (NumPart - N_gas) * sizeof(struct particle_data));
    }

#ifdef SFR
    for (int n = 0; n < N_stars; n++)
    {
        MetP[n].PID += count_totget;
        ENDRUN_IF(P[MetP[n].PID].pt.MetID != n,
                  "[Task %d] some serious error in adjusting the memory before particle exchange.", ThisTask);
    }
#endif // SFR

#ifdef BLACK_HOLES
    for (int n = 0; n < N_BHs; n++)
    {
        BHP[n].PID += count_totget;
        ENDRUN_IF(P[BHP[n].PID].pt.BHID != n,
                  "[Task %d] some serious error in adjusting the memory before particle exchange.", ThisTask);
    }
#endif // BLACK_HOLES

    for (int i = 0; i < NTask; i++)
    {
        count_recv_sph[i] = local_toGetSph[i];
        count_recv[i] = local_toGet[i] - local_toGetSph[i];
#if defined(SFR)
        count_recv_stars[i] = local_toGetStars[i];
        count_recv[i] -= local_toGetStars[i];
#endif //  defined(SFR)
#ifdef BLACK_HOLES
        count_recv_BHs[i] = local_toGetBHs[i];
        count_recv[i] -= local_toGetBHs[i];
#endif // BLACK_HOLES
    }

    offset_recv_sph[0] = N_gas;
    for (int i = 1; i < NTask; i++)
    {
        offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];
    }
    prec_count = N_gas + count_get_sph;

#if defined(SFR)
    offset_recv_stars[0] = prec_count;
    for (int i = 1; i < NTask; i++)
    {
        offset_recv_stars[i] = offset_recv_stars[i - 1] + count_recv_stars[i - 1];
    }
    prec_count += count_get_stars;
#endif // defined(SFR)

#ifdef BLACK_HOLES
    offset_recv_BHs[0] = prec_count;
    for (int i = 1; i < NTask; i++)
    {
        offset_recv_BHs[i] = offset_recv_BHs[i - 1] + count_recv_BHs[i - 1];
    }
    prec_count += count_get_BHs;
#endif // BLACK_HOLES

    offset_recv[0] = NumPart - N_gas + prec_count;

    for (int i = 1; i < NTask; i++)
    {
        offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
    }

#ifdef USE_ISEND_IRECV

    int n_requests = 0, max_requests = 6;
    MPI_Request *requests;

#if defined(SFR)
    max_requests += 4;
#endif //  defined(SFR)

#ifdef BLACK_HOLES
    max_requests += 4;
#endif // BLACK_HOLES

    requests = (MPI_Request *)mymalloc("requests", max_requests * NTask * sizeof(MPI_Request));

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (count_recv_sph[target] > 0)
            {
                MPI_Irecv(P + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_SPH, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Irecv(SphP + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct sph_particle_data),
                          MPI_BYTE, target, TAG_SPHDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
            }

#if defined(SFR)
            if (count_recv_stars[target] > 0)
            {
                MPI_Irecv(P + offset_recv_stars[target], count_recv_stars[target] * sizeof(struct particle_data),
                          MPI_BYTE, target, TAG_PDATA_STARS, MYMPI_COMM_WORLD, &requests[n_requests++]);
                MPI_Irecv(MetP + N_stars + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
                              count_recv_sph[NTask - 1],
                          count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
#endif //  defined(SFR)

#ifdef BLACK_HOLES
            if (count_recv_BHs[target] > 0)
            {
                MPI_Irecv(P + offset_recv_BHs[target], count_recv_BHs[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_BHS, MYMPI_COMM_WORLD, &requests[n_requests++]);
#ifdef SFR
                MPI_Irecv(BHP + N_BHs + offset_recv_BHs[target] - offset_recv_stars[NTask - 1] -
                              count_recv_stars[NTask - 1],
                          count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
#else // SFR
                MPI_Irecv(BHP + N_BHs + offset_recv_BHs[target],
                          count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
#endif // SFR
            }
#endif // BLACK_HOLES

            if (count_recv[target] > 0)
            {
                MPI_Irecv(P + offset_recv[target], count_recv[target] * sizeof(struct particle_data), MPI_BYTE, target,
                          TAG_PDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD); /* not really necessary, but this will guarantee that all receives are
                      posted before the sends, which helps the stability of MPI on
                      bluegene, and perhaps some mpich1-clusters */

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (count_sph[target] > 0)
            {
                MPI_Isend(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_SPH, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Isend(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE,
                          target, TAG_SPHDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
            }

#if defined(SFR)
            if (count_stars[target] > 0)
            {
                MPI_Isend(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_STARS, MYMPI_COMM_WORLD, &requests[n_requests++]);
                MPI_Isend(metBuf + offset_stars[target] - offset_sph[NTask - 1] - count_sph[NTask - 1],
                          count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
#endif // defined(SFR)

#ifdef BLACK_HOLES
            if (count_BHs[target] > 0)
            {
                MPI_Isend(partBuf + offset_BHs[target], count_BHs[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_BHS, MYMPI_COMM_WORLD, &requests[n_requests++]);
#ifdef SFR
                MPI_Isend(BHBuf + offset_BHs[target] - offset_stars[NTask - 1] - count_stars[NTask - 1],
                          count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
#else // SFR
                MPI_Isend(BHBuf + offset_BHs[target], count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE,
                          target, TAG_BHDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
#endif // SFR
            }
#endif // BLACK_HOLES

            if (count[target] > 0)
            {
                MPI_Isend(partBuf + offset[target], count[target] * sizeof(struct particle_data), MPI_BYTE, target,
                          TAG_PDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
        }
    }

    MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

    ENDRUN_IF(n_requests > max_requests * NTask, "Not enough memory reserved for requests: %d > %d !\n", n_requests,
              max_requests * NTask);

    myfree(requests);

#else // USE_ISEND_IRECV

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (count_sph[target] > 0 || count_recv_sph[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data), MPI_BYTE,
                             target, TAG_PDATA_SPH, P + offset_recv_sph[target],
                             count_recv_sph[target] * sizeof(struct particle_data), MPI_BYTE, target, TAG_PDATA_SPH,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Sendrecv(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data),
                             MPI_BYTE, target, TAG_SPHDATA, SphP + offset_recv_sph[target],
                             count_recv_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE, target, TAG_SPHDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

#if defined(SFR)
            if (count_stars[target] > 0 || count_recv_stars[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data),
                             MPI_BYTE, target, TAG_PDATA_STARS, P + offset_recv_stars[target],
                             count_recv_stars[target] * sizeof(struct particle_data), MPI_BYTE, target, TAG_PDATA_STARS,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(metBuf + offset_stars[target] - offset_sph[NTask - 1] - count_sph[NTask - 1],
                             count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                             MetP + N_stars + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
                                 count_recv_sph[NTask - 1],
                             count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
#endif // defined(SFR)

#ifdef BLACK_HOLES
            if (count_BHs[target] > 0 || count_recv_BHs[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset_BHs[target], count_BHs[target] * sizeof(struct particle_data), MPI_BYTE,
                             target, TAG_PDATA_BHS, P + offset_recv_BHs[target],
                             count_recv_BHs[target] * sizeof(struct particle_data), MPI_BYTE, target, TAG_PDATA_BHS,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef SFR
                MPI_Sendrecv(BHBuf + offset_BHs[target] - offset_stars[NTask - 1] - count_stars[NTask - 1],
                             count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                             BHP + N_BHs + offset_recv_BHs[target] - offset_recv_stars[NTask - 1] -
                                 count_recv_stars[NTask - 1],
                             count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else // SFR
                MPI_Sendrecv(BHBuf + offset_BHs[target], count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE,
                             target, TAG_BHDATA, BHP + N_BHs + offset_recv_BHs[target],
                             count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif // SFR
            }
#endif // BLACK_HOLES
            if (count[target] > 0 || count_recv[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset[target], count[target] * sizeof(struct particle_data), MPI_BYTE, target,
                             TAG_PDATA, P + offset_recv[target], count_recv[target] * sizeof(struct particle_data),
                             MPI_BYTE, target, TAG_PDATA, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
#endif // USE_ISEND_IRECV

    NumPart += count_get;
    ENDRUN_IF(NumPart > All.MaxPart, "Task=%d NumPart=%d All.MaxPart=%d\n", ThisTask, NumPart, All.MaxPart);

    N_gas += count_get_sph;
    ENDRUN_IF(N_gas > All.MaxPartSph, "Task=%d N_gas=%d All.MaxPartSph=%d\n", ThisTask, N_gas, All.MaxPartSph);

#ifdef SFR
    for (int i = 0; i < count_get_stars; i++)
    {
        P[offset_recv_stars[0] + i].pt.MetID = N_stars + i;
        MetP[N_stars + i].PID = offset_recv_stars[0] + i;
    }
    N_stars += count_get_stars;
    ENDRUN_IF(N_stars > All.MaxPartMet, "Task=%d N_stars=%d All.MaxPartMet=%d\n", ThisTask, N_stars, All.MaxPartMet);
    for (int i = 0; i < N_stars; i++)
    {
        if (MetP[i].PID >= NumPart || P[MetP[i].PID].Type != 4)
        {
            ENDRUN("Error in the number of particles or their type");
        }
    }
#endif // SFR

#ifdef BLACK_HOLES
    for (int i = 0; i < count_get_BHs; i++)
    {
        P[offset_recv_BHs[0] + i].pt.BHID = N_BHs + i;
        BHP[N_BHs + i].PID = offset_recv_BHs[0] + i;
    }
    N_BHs += count_get_BHs;

    ENDRUN_IF(N_BHs > All.MaxPartBH, "Task=%d N_BHs=%d All.MaxPartBH=%d\n", ThisTask, N_BHs, All.MaxPartBH);

    for (int i = 0; i < N_BHs; i++)
    {
        if (BHP[i].PID >= NumPart || P[BHP[i].PID].Type != 5)
        {
            ENDRUN("Error in the number of particles or their type");
        }
    }

    myfree(BHBuf);
#endif // BLACK_HOLES

#ifdef SFR
    myfree(metBuf);
#endif // SFR

    myfree(sphBuf);
    myfree(partBuf);

#ifdef BLACK_HOLES
    myfree(local_toGoBHs);
    myfree(local_toGetBHs);
    myfree(offset_recv_BHs);
    myfree(count_recv_BHs);
    myfree(offset_BHs);
    myfree(count_BHs);
#endif // BLACK_HOLES

#if defined(SFR)
    myfree(local_toGoStars);
    myfree(local_toGetStars);
    myfree(offset_recv_stars);
    myfree(count_recv_stars);
    myfree(offset_stars);
    myfree(count_stars);
#endif // defined(SFR)

    myfree(local_toGoSph);
    myfree(local_toGo);
    myfree(local_toGetSph);
    myfree(local_toGet);

    myfree(offset_recv_sph);
    myfree(offset_recv);
    myfree(count_recv_sph);
    myfree(count_recv);

    myfree(offset_sph);
    myfree(offset);
    myfree(count_sph);
    myfree(count);
    TIMER_POP(1, "SUBFIND_EXCHANGE");
}

#endif // SUBFIND
