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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef MPISENDRECV_SIZELIMIT

#undef MPI_Sendrecv

/**
 * @brief Sends and receives messages between MPI processes with size limitations.
 *
 * This function sends a message from the sending process to the receiving process and receives a
 * message from the receiving process to the sending process using MPI_Sendrecv. It handles cases
 * where the message size exceeds a specified limit by splitting the message into smaller chunks
 * and sending/receiving them iteratively.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements in the send buffer.
 * @param sendtype MPI datatype of elements in the send buffer.
 * @param dest Rank of the destination process.
 * @param sendtag Message tag for sending.
 * @param recvbuf Pointer to the receive buffer.
 * @param recvcount Maximum number of elements to receive.
 * @param recvtype MPI datatype of elements in the receive buffer.
 * @param source Rank of the source process.
 * @param recvtag Message tag for receiving.
 * @param comm MPI communicator.
 * @param status Status object to receive status information.
 *
 * @return Returns 0 on success.
 */
int MPI_Sizelimited_Sendrecv(void *sendbuf, size_t sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                             void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int source, int recvtag,
                             MPI_Comm comm, MPI_Status *status)
{
    int iter = 0, size_sendtype, size_recvtype, send_now, recv_now;
    int count_limit;

    ENDRUN_IF(dest != source, "Source eqals destination");

    MPI_Type_size(sendtype, &size_sendtype);
    MPI_Type_size(recvtype, &size_recvtype);

    if (dest == ThisTask)
    {
        memcpy(recvbuf, sendbuf, recvcount * size_recvtype);
        return 0;
    }

    count_limit = (int)((MPISendrecv_Sizelimit * 1024 * 1024) / size_sendtype);
    ENDRUN_IF(count_limit < 0, "Size limit in MPI_Sendrecv failed!");

    while (sendcount > 0 || recvcount > 0)
    {
        if (sendcount > count_limit)
        {
            send_now = count_limit;
            if (iter == 0)
            {
                VERBOSE_ALL(3, "imposing size limit on MPI_Sendrecv() on task=%d (send of size=%zu)", ThisTask,
                            sendcount * size_sendtype);
                fflush(stdout);
            }
            iter++;
        }
        else
        {
            send_now = sendcount;
        }

        if (recvcount > count_limit)
        {
            recv_now = count_limit;
        }
        else
        {
            recv_now = recvcount;
        }

        MPI_Sendrecv(sendbuf, send_now, sendtype, dest, sendtag, recvbuf, recv_now, recvtype, source, recvtag, comm,
                     status);

        sendcount -= send_now;
        recvcount -= recv_now;

        sendbuf = (char *)sendbuf + send_now * size_sendtype;
        recvbuf = (char *)recvbuf + recv_now * size_recvtype;
    }

    return 0;
}

#endif // MPISENDRECV_SIZELIMIT
