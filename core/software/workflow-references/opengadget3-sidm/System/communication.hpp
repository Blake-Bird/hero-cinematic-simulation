/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Geray S. Karademir and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef COMMUNICATION_HPP
#define COMMUNICATION_HPP

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

/*  memcpy(DensDataIn[j].NodeList,
     DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
*/

/**
 * @brief Copy data from the global data node list to the local data node list.
 *
 * This function copies data from the global data node list to the local data node list.
 * It iterates through the indices specified in the `DataIndexTable` and copies the
 * corresponding data node list entries from `DataNodeList` to `DataNodeListIn`.
 *
 * @note This function assumes that the following variables are correctly initialized:
 *       - `Nexport`: Number of entries to be exported.
 *       - `DataNodeListIn`: Local data node list.
 *       - `DataNodeList`: Global data node list.
 *       - `DataIndexTable`: Table containing indices for data retrieval.
 *
 * @warning This function assumes that memory for `DataNodeListIn` and `DataIndexTable`
 *          has been allocated before calling this function.
 */
static inline void copy_data_nodelist()
{
    for (int j = 0; j < Nexport; j++)
    {
        memcpy(&DataNodeListIn[j], &DataNodeList[DataIndexTable[j].IndexGet], sizeof(struct data_nodelist));
    }
}

/**
 * @brief Exchange data between tasks using MPI_Sendrecv.
 *
 * This function exchanges data between tasks using MPI_Sendrecv. It sends data from
 * the `DataIn` array to the specified tasks and receives data into the `DataGet` array.
 * Additionally, if `exchange_datanodelist` is set to true, it also exchanges data node
 * lists.
 *
 * @tparam data_in The type of data being exchanged.
 * @param DataIn Pointer to the data array to be sent.
 * @param DataGet Pointer to the data array to receive the data.
 * @param exchange_datanodelist Flag indicating whether to exchange data node lists.
 * @param Tag The MPI tag to use for the data exchange.
 */
template <typename data_in>
static inline void templated_exchange_data_in_get(data_in *DataIn, data_in *DataGet, int exchange_datanodelist,
                                                  int baseTag)
{
    int recvTask;
    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;
        if (recvTask < NTask)
        {
            if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                // create individual tag for each recvTask and sendTask pair
                int Tag = baseTag + std::min(ThisTask, recvTask) * NTask + std::max(ThisTask, recvTask);
                while (Tag > MYMPI_MAX_TAG_UB)
                    Tag -= MYMPI_MAX_TAG_UB;
                /* get the particles */
                MPI_Sendrecv(&DataIn[Send_offset[recvTask]], Send_count[recvTask] * sizeof(data_in), MPI_BYTE, recvTask,
                             Tag, &DataGet[Recv_offset[recvTask]], Recv_count[recvTask] * sizeof(data_in), MPI_BYTE,
                             recvTask, Tag, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (exchange_datanodelist)
                {
                    MPI_Sendrecv(&DataNodeListIn[Send_offset[recvTask]],
                                 Send_count[recvTask] * sizeof(struct data_nodelist), MPI_BYTE, recvTask, Tag,
                                 &DataNodeListGet[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct data_nodelist), MPI_BYTE, recvTask, Tag,
                                 MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }
}

/**
 * @brief Exchanges data between tasks in an MPI environment using send and receive counts and offsets.
 *
 * This function facilitates the exchange of data between MPI tasks by using send and receive
 * counts and offsets. It performs an all-to-all communication pattern where data is sent and
 * received based on the provided counts and offsets arrays. The function uses MPI_Sendrecv
 * for the data transfer, ensuring the correct amount of data is exchanged between tasks.
 *
 * @tparam data The data type to be exchanged, allowing flexibility with templated data types.
 *
 * @param DataResult Pointer to the data array that will store received data.
 * @param DataOut Pointer to the data array from which data will be sent.
 * @param Tag The MPI tag to use for the data exchange.
 * @param Send_offset_var Array of starting indices for sending data to each task.
 * @param Send_count_var Array containing the number of elements to send to each task.
 * @param Recv_offset_var Array of starting indices for receiving data from each task.
 * @param Recv_count_var Array containing the number of elements to receive from each task.
 */
template <typename data>
static inline void templated_exchange_data_Sendrecv(data *DataResult, data *DataOut, int baseTag, int *Send_offset_var,
                                                    int *Send_count_var, int *Recv_offset_var, int *Recv_count_var)
{
    /* get the result */
    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int recvTask = ThisTask ^ ngrp;
        if (recvTask < NTask)
        {
            if (Send_count_var[recvTask] > 0 || Recv_count_var[recvTask] > 0)
            {
                // create individual tag for each recvTask and sendTask pair
                int Tag = baseTag + std::min(ThisTask, recvTask) * NTask + std::max(ThisTask, recvTask);
                while (Tag > MYMPI_MAX_TAG_UB)
                    Tag -= MYMPI_MAX_TAG_UB;
                /* send the results */
                MPI_Sendrecv(&DataResult[Send_offset_var[recvTask]], Send_count_var[recvTask] * sizeof(data), MPI_BYTE,
                             recvTask, Tag, &DataOut[Recv_offset_var[recvTask]],
                             Recv_count_var[recvTask] * sizeof(data), MPI_BYTE, recvTask, Tag, MYMPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
            }
        }
    }
}

/**
 * @brief Prepares data for sending and receiving between MPI tasks.
 *
 * This function prepares data for sending and receiving between MPI tasks. It handles
 * the exchange of particle data and node lists, manages buffer sizes, and computes
 * offsets for sending and receiving data.
 *
 * @param save_nextparticle The index of the next particle to process before buffer overflow.
 * @param panic_on_error Flag indicating whether to panic on buffer overflow errors.
 *
 * @return Returns 1 if data preparation is successful, otherwise returns 0.
 */
static inline int prepare_sendrec_offset_send(const int save_nextparticle, const int panic_on_error)
{
    if (BufferFullFlag)
    {
        int last_nextparticle = NextParticle;

        NextParticle = save_nextparticle;

        while (NextParticle >= 0)
        {
            if (NextParticle == last_nextparticle)
            {
                break;
            }

            if (!(ProcessedFlag[NextParticle] & ProcessedFlag_finished))
            {
                break;
            }

            ProcessedFlag[NextParticle] = (ProcessedFlag[NextParticle] | ProcessedFlag_communication);

            NextParticle = NextActiveParticle[NextParticle];
        }

        if (NextParticle == save_nextparticle)
        {
            /* in this case, the buffer is too small to process even a single particle */
            VERBOSE_ALL(2, "Task %d: Type=%d pos=(%g,%g,%g) mass=%g\n", ThisTask, P[NextParticle].Type,
                        double(P[NextParticle].Pos[0]), double(P[NextParticle].Pos[1]), double(P[NextParticle].Pos[2]),
                        double(P[NextParticle].Mass));
            if (P[NextParticle].Type == 0)
            {
                VERBOSE_ALL(2, "   rho=%g hsml=%g NgB= %g/%d NextParticle=%d/%d/%d\n",
                            double(SphP[NextParticle].Density), double(P[NextParticle].Hsml),
                            double(P[NextParticle].NumNgb), P[NextParticle].TrueNGB, NextParticle, N_gas, NumPart);
            }
            if (panic_on_error)
            {
                ENDRUN_MAIN("  the buffer is too small to process even a single particle ");
            }
            else
            {
                return 0;
            }
        }

        int new_export = 0;
        for (int j = 0, k = 0; j < Nexport; j++)
        {
            if (!(ProcessedFlag[DataIndexTable[j].Index] & ProcessedFlag_communication))
            {
                if (k < j + 1)
                {
                    k = j + 1;
                }

                for (; k < Nexport; k++)
                {
                    if (ProcessedFlag[DataIndexTable[k].Index] & ProcessedFlag_communication)
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

#ifndef IMPORT_ALLtoALLv_FROM_G4
    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
#else // IMPORT_ALLtoALLv_FROM_G4
#ifdef IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
    int non_active_timebin = 0;
    for (int j = TIMEBINS - 1; j >= 0; j--)
    {
        if (TimeBinActive[j])
        {
            break;
        }
        else
        {
            non_active_timebin++;
        }
    }
    if (non_active_timebin > IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER)
    {
        VERBOSE(3, "Sendreceive performed with MPI windows");
#endif // IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
        for (int j = 0; j < NTask; j++)
        {
            Recv_count[j] = 0;
        }
        MPI_Win_create(Recv_count, NTask * sizeof(int), sizeof(int), MPI_INFO_NULL, MYMPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);
        for (int i = 0; i < NTask; ++i)
        {
            if (Send_count[i] > 0)
            {
                MPI_Put(&Send_count[i], 1, MPI_INT, i, ThisTask, 1, MPI_INT, win);
            }
        }
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);
#ifdef IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
    }
    else
    {
        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    }
#endif // IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
#endif // IMPORT_ALLtoALLv_FROM_G4

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
    return 1;
}
/**
 * @brief Prepares data for sending and receiving between MPI tasks in the context of conduction.
 *
 * This function prepares data for sending and receiving between MPI tasks, specifically in the context of conduction.
 * It handles the exchange of particle data, manages buffer sizes, and computes offsets for sending and receiving data.
 *
 * @param save_nextparticle The index of the next particle to process before buffer overflow.
 * @param panic_on_error Flag indicating whether to panic on buffer overflow errors.
 *
 * @return Returns 1 if data preparation is successful, otherwise returns 0.
 */
static inline int prepare_sendrec_offset_send_conduction(const int save_nextparticle, const int panic_on_error)
{
    if (BufferFullFlag)
    {
        int last_nextparticle = NextParticle;

        NextParticle = save_nextparticle;

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

        if (NextParticle == save_nextparticle)
        {
            /* in this case, the buffer is too small to process even a single particle */
            VERBOSE_ALL(2, "Task %d: Type=%d pos=(%g,%g,%g) mass=%g\n", ThisTask, P[NextParticle].Type,
                        double(P[NextParticle].Pos[0]), double(P[NextParticle].Pos[1]), double(P[NextParticle].Pos[2]),
                        double(P[NextParticle].Mass));
            if (P[NextParticle].Type == 0)
            {
                VERBOSE_ALL(2, "   rho=%g hsml=%g NextParticle=%d/%d/%d\n", double(SphP[NextParticle].Density),
                            double(P[NextParticle].Hsml), NextParticle, N_gas, NumPart);
            }
            if (panic_on_error)
            {
                ENDRUN_MAIN("  the buffer is too small to process even a single particle ");
            }
            else
            {
                return 0;
            }
        }

        int new_export = 0;

        for (int j = 0, k = 0; j < Nexport; j++)
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

#ifndef IMPORT_ALLtoALLv_FROM_G4
    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
#else // IMPORT_ALLtoALLv_FROM_G4
#ifdef IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
    int non_active_timebin = 0;
    for (int j = TIMEBINS - 1; j >= 0; j--)
    {
        if (TimeBinActive[j])
        {
            break;
        }
        else
        {
            non_active_timebin++;
        }
    }
    if (non_active_timebin > IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER)
    {
#endif // IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
        for (int j = 0; j < NTask; j++)
        {
            Recv_count[j] = 0;
        }
        MPI_Win_create(Recv_count, NTask * sizeof(int), sizeof(int), MPI_INFO_NULL, MYMPI_COMM_WORLD, &win);
        MPI_Win_fence(0, win);
        for (int i = 0; i < NTask; ++i)
        {
            if (Send_count[i] > 0)
            {
                MPI_Put(&Send_count[i], 1, MPI_INT, i, ThisTask, 1, MPI_INT, win);
            }
        }
        MPI_Win_fence(0, win);
        MPI_Win_free(&win);
#ifdef IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
    }
    else
    {
        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    }
#endif // IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
#endif // IMPORT_ALLtoALLv_FROM_G4

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
    return 1;
}
#endif // COMMUNICATION_HPP
