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

#ifdef MPISENDRECV_CHECKSUM

#undef MPI_Sendrecv

/**
 * @brief Customized MPI send-receive operation with error checking and data integrity verification.
 *
 * This function performs a customized version of the MPI send-receive operation with extensive error handling
 * and data integrity verification mechanisms. It ensures that data sent from one process is correctly received
 * by another process without corruption or loss.
 *
 * @param sendbuf Pointer to the send buffer.
 * @param sendcount Number of elements to send.
 * @param sendtype MPI datatype of the send buffer elements.
 * @param dest Rank of the destination process.
 * @param sendtag MPI tag for sending data.
 * @param recvbufreal Pointer to the receive buffer.
 * @param recvcount Number of elements to receive.
 * @param recvtype MPI datatype of the receive buffer elements.
 * @param source Rank of the source process.
 * @param recvtag MPI tag for receiving data.
 * @param comm MPI communicator.
 * @param status Pointer to the MPI status object.
 *
 * @return Returns 0 upon successful completion.
 */
int MPI_Check_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbufreal,
                       int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
{
    int checksumtag = 1000, errtag = 2000;
    int i, iter = 0, err_flag, err_flag_imported, size_sendtype, size_recvtype, Local_ThisTask, Local_NTask;
    long long sendCheckSum, recvCheckSum, importedCheckSum;
    unsigned char *p, *buf, *recvbuf;

    ENDRUN_IF(dest != source, "Destination and source are equal");

    MPI_Comm_rank(comm, &Local_ThisTask);
    MPI_Comm_size(comm, &Local_NTask);

    MPI_Type_size(sendtype, &size_sendtype);
    MPI_Type_size(recvtype, &size_recvtype);

    if (dest == Local_ThisTask)
    {
        memcpy(recvbufreal, sendbuf, recvcount * size_recvtype);
        return 0;
    }

    if (!(buf = (unsigned char *)mymalloc("buf", recvcount * size_recvtype + 1024)))
    {
        ENDRUN("failed to allocate memory for buffer");
    }

    for (i = 0, p = buf; i < recvcount * size_recvtype + 1024; i++)
    {
        *p++ = 255;
    }

    recvbuf = buf + 512;

    MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm,
                 status);

    for (i = 0, p = buf; i < 512; i++, p++)
    {
        ENDRUN_IF(*p != 255,
                  "MPI-ERROR: Task=%d/%s: Recv occured before recv buffer. message-size=%d from %d, i=%d c=%d\n",
                  Local_ThisTask, getenv("HOST"), recvcount, dest, i, *p);
    }

    for (i = 0, p = recvbuf + recvcount * size_recvtype; i < 512; i++, p++)
    {
      ENDRUN_IF(*p != 255,
	    ("MPI-ERROR: Task=%d/%s: Recv occured after recv buffer. message-size=%d from %d, i=%d c=%d\n",
	     Local_ThisTask, getenv("HOST"), recvcount, dest, i, *p);
    }

    for (i = 0, p = (unsigned char *)sendbuf, sendCheckSum = 0; i < sendcount * size_sendtype; i++, p++)
    {
      sendCheckSum += *p;
    }

    importedCheckSum = 0;

    if (dest > Local_ThisTask)
    {
      if (sendcount > 0)
      {
          MPI_Ssend(&sendCheckSum, sizeof(sendCheckSum), MPI_BYTE, dest, checksumtag, comm);
      }
      if (recvcount > 0)
      {
          MPI_Recv(&importedCheckSum, sizeof(importedCheckSum), MPI_BYTE, dest, checksumtag, comm, status);
      }
    }
    else
    {
      if (recvcount > 0)
      {
          MPI_Recv(&importedCheckSum, sizeof(importedCheckSum), MPI_BYTE, dest, checksumtag, comm, status);
      }
      if (sendcount > 0)
      {
          MPI_Ssend(&sendCheckSum, sizeof(sendCheckSum), MPI_BYTE, dest, checksumtag, comm);
      }
    }

    checksumtag++;

    for (i = 0, p = recvbuf, recvCheckSum = 0; i < recvcount * size_recvtype; i++, p++)
    {
      recvCheckSum += *p;
    }

    err_flag = err_flag_imported = 0;

    if (recvCheckSum != importedCheckSum)
    {
      VERBOSE_ALL(0,
                  "MPI-ERROR: Receive error on task=%d/%s from task=%d, message size=%d, sendcount=%d checksums= %d %d "
                  " %d %d. Try to fix it...\n",
                  Local_ThisTask, getenv("HOST"), source, recvcount, sendcount, (int)(recvCheckSum >> 32),
                  (int)recvCheckSum, (int)(importedCheckSum >> 32), (int)importedCheckSum);
      err_flag = 1;
    }

    if (dest > Local_ThisTask)
    {
      MPI_Ssend(&err_flag, 1, MPI_INT, dest, errtag, comm);
      MPI_Recv(&err_flag_imported, 1, MPI_INT, dest, errtag, comm, status);
    }
    else
    {
      MPI_Recv(&err_flag_imported, 1, MPI_INT, dest, errtag, comm, status);
      MPI_Ssend(&err_flag, 1, MPI_INT, dest, errtag, comm);
    }
    errtag++;

    if (err_flag > 0 || err_flag_imported > 0)
    {
      VERBOSE_ALL(3, "Task=%d is on %s, wants to send %d and has checksum=%d %d of send data\n", Local_ThisTask,
                  getenv("HOST"), sendcount, (int)(sendCheckSum >> 32), (int)sendCheckSum);
      do
      {
          sendtag++;
          recvtag++;

          for (i = 0, p = (unsigned char *)recvbuf; i < recvcount * size_recvtype; i++, p++)
          {
              *p = 0;
          }

          if ((iter & 1) == 0)
          {
              if (dest > Local_ThisTask)
              {
                  if (sendcount > 0)
                  {
                      MPI_Ssend(sendbuf, sendcount, sendtype, dest, sendtag, comm);
                  }
                  if (recvcount > 0)
                  {
                      MPI_Recv(recvbuf, recvcount, recvtype, dest, recvtag, comm, status);
                  }
              }
              else
              {
                  if (recvcount > 0)
                  {
                      MPI_Recv(recvbuf, recvcount, recvtype, dest, recvtag, comm, status);
                  }
                  if (sendcount > 0)
                  {
                      MPI_Ssend(sendbuf, sendcount, sendtype, dest, sendtag, comm);
                  }
              }
          }
          else
          {
              if (iter > 5)
              {
                  VERBOSE_ALL(3, "we're trying to send each byte now on task=%d (iter=%d)\n", Local_ThisTask, iter);
                  if (dest > Local_ThisTask)
                  {
                      for (i = 0, p = (unsigned char *)sendbuf; i < sendcount * size_sendtype; i++, p++)
                      {
                          MPI_Ssend(p, 1, MPI_BYTE, dest, i, comm);
                      }
                      for (i = 0, p = (unsigned char *)recvbuf; i < recvcount * size_recvtype; i++, p++)
                      {
                          MPI_Recv(p, 1, MPI_BYTE, dest, i, comm, status);
                      }
                  }
                  else
                  {
                      for (i = 0, p = (unsigned char *)recvbuf; i < recvcount * size_recvtype; i++, p++)
                      {
                          MPI_Recv(p, 1, MPI_BYTE, dest, i, comm, status);
                      }
                      for (i = 0, p = (unsigned char *)sendbuf; i < sendcount * size_sendtype; i++, p++)
                      {
                          MPI_Ssend(p, 1, MPI_BYTE, dest, i, comm);
                      }
                  }
              }
              else
              {
                  MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source,
                               recvtag, comm, status);
              }
          }

          importedCheckSum = 0;

          for (i = 0, p = (unsigned char *)sendbuf, sendCheckSum = 0; i < sendcount * size_sendtype; i++, p++)
          {
              sendCheckSum += *p;
          }
          VERBOSE_ALL(3, "Task=%d gas send_checksum=%d %d\n", Local_ThisTask, (int)(sendCheckSum >> 32),
                      (int)sendCheckSum);

          if (dest > Local_ThisTask)
          {
              if (sendcount > 0)
              {
                  MPI_Ssend(&sendCheckSum, sizeof(sendCheckSum), MPI_BYTE, dest, checksumtag, comm);
              }
              if (recvcount > 0)
              {
                  MPI_Recv(&importedCheckSum, sizeof(importedCheckSum), MPI_BYTE, dest, checksumtag, comm, status);
              }
          }
          else
          {
              if (recvcount > 0)
              {
                  MPI_Recv(&importedCheckSum, sizeof(importedCheckSum), MPI_BYTE, dest, checksumtag, comm, status);
              }
              if (sendcount > 0)
              {
                  MPI_Ssend(&sendCheckSum, sizeof(sendCheckSum), MPI_BYTE, dest, checksumtag, comm);
              }
          }

          for (i = 0, p = recvbuf, recvCheckSum = 0; i < recvcount; i++, p++)
          {
              recvCheckSum += *p;
          }

          err_flag = err_flag_imported = 0;

          if (recvCheckSum != importedCheckSum)
          {
              VERBOSE_ALL(0,
                          "MPI-ERROR: Again (iter=%d) a receive error on task=%d/%s from task=%d, message size=%d, "
                          "checksums= %d %d  %d %d. Try to fix it...\n",
                          iter, Local_ThisTask, getenv("HOST"), source, recvcount, (int)(recvCheckSum >> 32),
                          (int)recvCheckSum, (int)(importedCheckSum >> 32), (int)importedCheckSum);
              err_flag = 1;
          }

          if (dest > Local_ThisTask)
          {
              MPI_Ssend(&err_flag, 1, MPI_INT, dest, errtag, comm);
              MPI_Recv(&err_flag_imported, 1, MPI_INT, dest, errtag, comm, status);
          }
          else
          {
              MPI_Recv(&err_flag_imported, 1, MPI_INT, dest, errtag, comm, status);
              MPI_Ssend(&err_flag, 1, MPI_INT, dest, errtag, comm);
          }

          if (err_flag == 0 && err_flag_imported == 0)
          {
              break;
          }

          errtag++;
          checksumtag++;
          iter++;
      } while (iter < 10);

      if (iter >= 10)
      {
          char buf[1000];
          int length;
          FILE *fd;

          sprintf(buf, "send_data_%d.dat", Local_ThisTask);
          fd = fopen(buf, "w");
          length = sendcount * size_sendtype;
          fwrite(&length, 1, sizeof(int), fd);
          fwrite(sendbuf, sendcount, size_sendtype, fd);
          fclose(fd);

          sprintf(buf, "recv_data_%d.dat", Local_ThisTask);
          fd = fopen(buf, "w");
          length = recvcount * size_recvtype;
          fwrite(&length, 1, sizeof(int), fd);
          fwrite(recvbuf, recvcount, size_recvtype, fd);
          fclose(fd);

          ENDRUN("MPI-ERROR: Even 10 trials proved to be insufficient on task=%d/%s. Stopping\n", Local_ThisTask,
                 getenv("HOST"));
      }
    }

    memcpy(recvbufreal, recvbuf, recvcount * size_recvtype);

    myfree(buf);

    return 0;
}

#endif // MPISENDRECV_CHECKSUM
