/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#if defined(ORDER_SNAPSHOTS_BY_ID) || defined(SUBFIND_READ_FOF) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
/**
 * @brief Auxiliary data structure.
 *
 * This structure contains auxiliary information including ID, group number (GrNr),
 * origin task, origin index, and final task. It is used for storing additional
 * data related to certain operations.
 */
static struct aux_data
{
    MyIDType ID;
    MyIDType GrNr;
    int OriginTask;
    int OriginIndex;
    int FinalTask;
} *Aux;

/**
 * @brief Compare function for sorting auxiliary data based on OriginTask and OriginIndex.
 *
 * This function compares two auxiliary data structures based on their OriginTask
 * and OriginIndex fields. It is used as a comparison function for sorting operations.
 *
 * @param a Pointer to the first auxiliary data structure.
 * @param b Pointer to the second auxiliary data structure.
 * @return An integer indicating the comparison result:
 *         - Negative value if the OriginTask of `a` is less than that of `b`.
 *         - Positive value if the OriginTask of `a` is greater than that of `b`.
 *         - Zero if both OriginTask values are equal, in which case the comparison is based on OriginIndex.
 */
static int compare_Aux_OriginTask_OriginIndex(const void *a, const void *b)
{
    if (((struct aux_data *)a)->OriginTask < (((struct aux_data *)b)->OriginTask))
    {
        return -1;
    }

    if (((struct aux_data *)a)->OriginTask > (((struct aux_data *)b)->OriginTask))
    {
        return +1;
    }

    if (((struct aux_data *)a)->OriginIndex < (((struct aux_data *)b)->OriginIndex))
    {
        return -1;
    }

    if (((struct aux_data *)a)->OriginIndex > (((struct aux_data *)b)->OriginIndex))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compare function for sorting auxiliary data based on GrNr and ID.
 *
 * This function compares two auxiliary data structures based on their GrNr
 * and ID fields. It is used as a comparison function for sorting operations.
 *
 * @param a Pointer to the first auxiliary data structure.
 * @param b Pointer to the second auxiliary data structure.
 * @return An integer indicating the comparison result:
 *         - Negative value if the GrNr of `a` is less than that of `b`.
 *         - Positive value if the GrNr of `a` is greater than that of `b`.
 *         - Zero if both GrNr values are equal, in which case the comparison is based on ID.
 */
static int compare_Aux_GrNr_ID(const void *a, const void *b)
{
    if (((struct aux_data *)a)->GrNr < (((struct aux_data *)b)->GrNr))
    {
        return -1;
    }

    if (((struct aux_data *)a)->GrNr > (((struct aux_data *)b)->GrNr))
    {
        return +1;
    }

    if (((struct aux_data *)a)->ID < (((struct aux_data *)b)->ID))
    {
        return -1;
    }

    if (((struct aux_data *)a)->ID > (((struct aux_data *)b)->ID))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Performs parallel sorting of particle data based on GrNr and ID.
 *
 * This function sorts particle data in parallel based on their Group Number (GrNr)
 * and particle ID (ID). The sorting process involves multiple steps:
 *   - Copying particle data to an auxiliary array.
 *   - Sorting the auxiliary array first by GrNr and ID, then by OriginTask and OriginIndex.
 *   - Determining the final destination task for each particle and calculating send and receive counts.
 *   - Exchanging particle data between tasks using MPI communication.
 *   - Sorting the particle data locally on each task.
 *
 * @details This function is specifically designed for sorting particle data in parallel
 * within a distributed computing environment.
 */
void parallel_sort_special_P_GrNr_ID(void)
{
    TIMER_PUSH(2, "parallel_sort_special_P_GrNr_ID");
    Aux = mymalloc("Aux", NumPart * sizeof(struct aux_data));

    for (int i = 0; i < NumPart; i++)
    {
        Aux[i].GrNr = P[i].GrNr;
        Aux[i].ID = P[i].ID;
        Aux[i].OriginTask = ThisTask;
        Aux[i].OriginIndex = i;
    }

    parallel_sort(Aux, NumPart, sizeof(struct aux_data), compare_Aux_GrNr_ID);

    for (int i = 0; i < NumPart; i++)
    {
        Aux[i].FinalTask = ThisTask;
    }

    parallel_sort(Aux, NumPart, sizeof(struct aux_data), compare_Aux_OriginTask_OriginIndex);

    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < NumPart; i++)
    {
        Send_count[Aux[i].FinalTask]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    int Nimport = 0;
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

    if (Nimport != NumPart)
    {
        ENDRUN_MAIN("Nimport != NumPart");
    }

    struct particle_data *pbuf;

    pbuf = mymalloc("pbuf", NumPart * sizeof(struct particle_data));

    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < NumPart; i++)
    {
        pbuf[Send_offset[Aux[i].FinalTask] + Send_count[Aux[i].FinalTask]++] = P[i];
    }

    memcpy(&P[Recv_offset[ThisTask]], &pbuf[Send_offset[ThisTask]],
           Send_count[ThisTask] * sizeof(struct particle_data));

    TIMER_PUSH(2, "PSORT_COMM");
    templated_exchange_data_Sendrecv(pbuf, P, TAG_DENS_A, Send_offset, Send_count, Recv_offset, Recv_count);
    TIMER_POP(2, "PSORT_COMM");

    myfree(pbuf);
    qsort(P, NumPart, sizeof(struct particle_data), io_compare_P_GrNr_ID);
    myfree(Aux);
    TIMER_POP(2, "parallel_sort_special_P_GrNr_ID");
}
#endif //  defined(ORDER_SNAPSHOTS_BY_ID) || defined(SUBFIND_READ_FOF) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
