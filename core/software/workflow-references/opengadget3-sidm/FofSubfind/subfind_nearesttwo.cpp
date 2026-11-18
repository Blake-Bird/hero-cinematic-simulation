/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Find nearest two neighbours for Subfind.
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
#include "subfind.hpp"

/**
 * @brief Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct nearestdata_in
{
    MyLongDouble Pos[3];
    MyIDType ID;
    MyFloat Hsml;
    MyFloat Density;
    MyFloat Dist[2];
    int Count;
    long long Index[2];
    int NodeList[NODELISTLENGTH];
} *NearestDataIn, *NearestDataGet;

/**
 * @brief Structure for communication during the density computation. Holds data that is recieved from other processors.
 */
static struct nearestdata_out
{
    MyFloat Dist[2];
    long long Index[2];
    int Count;
} *NearestDataResult, *NearestDataOut;

/**
 * @brief Find the nearest two neighbors for each particle.
 *
 * This function finds the nearest two neighbors for each particle in a distributed computing environment.
 * It iterates over particles, prepares a list of particles to be exported for neighbor search, exchanges particle
 * data between MPI tasks, evaluates nearest neighbors locally, aggregates results, and updates the nearest neighbor
 * information for each particle. The function continues the process until all particles have found their nearest
 * neighbors.
 */
void subfind_find_nearesttwo(void)
{
    int l, ndone, ndone_flag, dummy;
    int place, nexport, nimport;

    VERBOSE(2, "Start finding nearest two (%d particles on task=%d)", NumPartGroup, ThisTask);

    /* allocate buffers to arrange communication */

    Ngblist = (int *)mymalloc("Ngblist", NumPartGroup * sizeof(int));
    Dist2list = (MyFloat *)mymalloc("Dist2list", NumPartGroup * sizeof(MyFloat));

    All.BunchSize = (int)((All.BufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct nearestdata_in) +
                           sizeof(struct nearestdata_out) +
                           sizemax(sizeof(struct nearestdata_in), sizeof(struct nearestdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    for (int i = 0; i < NumPartGroup; i++)
    {
        NgbLoc[i].count = 0;
    }

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

    int i = 0; /* begin with this index */
    do
    {
        for (int j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Exportflag[j] = -1;
        }

        /* do local particles and prepare export list */

        for (nexport = 0; i < NumPartGroup; i++)
        {
            if (subfind_nearesttwo_evaluate(i, 0, &nexport, Send_count) < 0)
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

        NearestDataGet = (struct nearestdata_in *)mymalloc("NearestDataGet", nimport * sizeof(struct nearestdata_in));
        NearestDataIn = (struct nearestdata_in *)mymalloc("NearestDataIn", nexport * sizeof(struct nearestdata_in));

        /* prepare particle data for export */
        for (int j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;

            NearestDataIn[j].Pos[0] = P[place].Pos[0];
            NearestDataIn[j].Pos[1] = P[place].Pos[1];
            NearestDataIn[j].Pos[2] = P[place].Pos[2];
            NearestDataIn[j].Hsml = P[place].DM_Hsml;
            NearestDataIn[j].ID = P[place].ID;
            NearestDataIn[j].Density = P[place].u.DM_Density;
            NearestDataIn[j].Count = NgbLoc[place].count;
            for (int k = 0; k < NgbLoc[place].count; k++)
            {
                NearestDataIn[j].Dist[k] = R2Loc[place].dist[k];
                NearestDataIn[j].Index[k] = NgbLoc[place].index[k];
            }

            memcpy(NearestDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                   NODELISTLENGTH * sizeof(int));
        }

        /* exchange particle data */
        TIMER_PUSH(4, "subfind_comm");
        templated_exchange_data_Sendrecv(NearestDataIn, NearestDataGet, TAG_DENS_A, Send_offset, Send_count,
                                         Recv_offset, Recv_count);
        TIMER_POP(4, "subfind_comm");

        myfree(NearestDataIn);
        NearestDataResult =
            (struct nearestdata_out *)mymalloc("NearestDataResult", nimport * sizeof(struct nearestdata_out));
        NearestDataOut = (struct nearestdata_out *)mymalloc("NearestDataOut", nexport * sizeof(struct nearestdata_out));

        /* now do the particles that were sent to us */
        for (int j = 0; j < nimport; j++)
        {
            subfind_nearesttwo_evaluate(j, 1, &dummy, &dummy);
        }

        if (i >= NumPartGroup)
        {
            ndone_flag = 1;
        }
        else
        {
            ndone_flag = 0;
        }

        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        TIMER_PUSH(4, "subfind_comm");
        templated_exchange_data_Sendrecv(NearestDataResult, NearestDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                         Send_offset, Send_count);
        TIMER_POP(4, "subfind_comm");
        /* add the result to the local particles */
        for (int j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;

            for (int k = 0; k < NearestDataOut[j].Count; k++)
            {
                if (NgbLoc[place].count >= 1)
                {
                    if (NgbLoc[place].index[0] == NearestDataOut[j].Index[k])
                    {
                        continue;
                    }
                }

                if (NgbLoc[place].count == 2)
                {
                    if (NgbLoc[place].index[1] == NearestDataOut[j].Index[k])
                    {
                        continue;
                    }
                }

                if (NgbLoc[place].count < 2)
                {
                    l = NgbLoc[place].count;
                    NgbLoc[place].count++;
                }
                else
                {
                    if (R2Loc[place].dist[0] > R2Loc[place].dist[1])
                    {
                        l = 0;
                    }
                    else
                    {
                        l = 1;
                    }

                    if (NearestDataOut[j].Dist[k] >= R2Loc[place].dist[l])
                    {
                        continue;
                    }
                }

                R2Loc[place].dist[l] = NearestDataOut[j].Dist[k];
                NgbLoc[place].index[l] = NearestDataOut[j].Index[k];

                if (NgbLoc[place].count == 2)
                {
                    if (NgbLoc[place].index[0] == NgbLoc[place].index[1])
                    {
                        /*
                           printf("taaa=%d i=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
                           ThisTask, place,
                           NgbLoc[place].task[0], NgbLoc[place].index[0],
                           NgbLoc[place].task[1], NgbLoc[place].index[1]);

                           printf
                           ("NearestDataOut[j].Count=%d  l=%d k=%d task_0=%d index_0=%d  task_1=%d index_1=%d\n",
                           NearestDataOut[j].Count, l, k, NearestDataOut[j].Task[0], NearestDataOut[j].Index[0],
                           NearestDataOut[j].Task[1], NearestDataOut[j].Index[1]);
                         */

                        ENDRUN("Error.");
                    }
                }
            }
        }
        myfree(NearestDataOut);
        myfree(NearestDataResult);
        myfree(NearestDataGet);
    } while (ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);

    myfree(Dist2list);
    myfree(Ngblist);
}

/**
 * @brief Evaluate the nearest two neighbors for a particle.
 *
 * This function evaluates the nearest two neighbors for a given particle in a distributed computing environment.
 * It searches for neighbors within a certain smoothing length around the particle's position, excluding the
 * self-particle. It considers only neighbors that are denser than the particle and updates the nearest neighbor
 * information accordingly. The function operates in two modes: local and remote. In the local mode, it evaluates
 * the nearest neighbors for a particle stored locally. In the remote mode, it evaluates the nearest neighbors for
 * a particle received from another MPI task.
 *
 * @param target The index of the particle for which to evaluate the nearest neighbors.
 * @param mode The mode of operation (local or remote).
 * @param nexport Pointer to the variable storing the number of particles to be exported to other MPI tasks.
 * @param nsend_local Pointer to the variable storing the number of particles to be sent to other MPI tasks.
 * @return 0 if successful, -1 if an error occurs.
 */
int subfind_nearesttwo_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
    int k, count;
    MyIDType ID;
    long long index[2];
    double dist[2];
    int startnode, numngb_inbox, listindex = 0;
    double hmax, h, density;
    MyLongDouble *pos;

    if (mode == 0)
    {
        ID = P[target].ID;
        density = P[target].u.DM_Density;
        pos = P[target].Pos;
        h = P[target].DM_Hsml / FAC_HSML;
        count = NgbLoc[target].count;
        for (int k = 0; k < count; k++)
        {
            dist[k] = R2Loc[target].dist[k];
            index[k] = NgbLoc[target].index[k];
        }
    }
    else
    {
        ID = NearestDataGet[target].ID;
        density = NearestDataGet[target].Density;
        pos = NearestDataGet[target].Pos;
        h = NearestDataGet[target].Hsml / FAC_HSML;
        count = NearestDataGet[target].Count;
        for (int k = 0; k < count; k++)
        {
            dist[k] = NearestDataGet[target].Dist[k];
            index[k] = NearestDataGet[target].Index[k];
        }
    }

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = NearestDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    if (count == 2)
    {
        if (index[0] == index[1])
        {
            /*      printf("T-A-S-K=%d target=%d mode=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
               ThisTask, target, mode, task[0], index[0], task[1], index[1]);
             */
            ENDRUN("Error Indexes are equal.");
        }
    }

    // int numngb = 0;
    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            numngb_inbox =
                subfind_ngb_treefind_nearesttwo(pos, h, target, &startnode, mode, &hmax, nexport, nsend_local);

            if (numngb_inbox < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
                int j = Ngblist[n];
                int r2 = Dist2list[n];

                if (P[j].ID != ID) /* exclude the self-particle */
                {
                    if (P[j].u.DM_Density > density && r2 < h * h) /* we only look at neighbours that are denser */
                    {
                        if (count < 2)
                        {
                            dist[count] = r2;
                            index[count] = (((long long)ThisTask) << 32) + j;
                            count++;
                        }
                        else
                        {
                            if (dist[0] > dist[1])
                            {
                                k = 0;
                            }
                            else
                            {
                                k = 1;
                            }

                            if (r2 < dist[k])
                            {
                                dist[k] = r2;
                                index[k] = (((long long)ThisTask) << 32) + j;
                            }
                        }
                    }
                    // numngb++;
                }
            }
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = NearestDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        NgbLoc[target].count = count;
        for (int k = 0; k < count; k++)
        {
            R2Loc[target].dist[k] = dist[k];
            NgbLoc[target].index[k] = index[k];
        }

        if (count == 2)
        {
            if (NgbLoc[target].index[0] == NgbLoc[target].index[1])
            {
                /*
                   printf("TASK=%d target=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
                   ThisTask, target,
                   NgbLoc[target].task[0], NgbLoc[target].index[0],
                   NgbLoc[target].task[1], NgbLoc[target].index[1]);
                 */
                ENDRUN("Indexes are equal.");
            }
        }
    }
    else
    {
        NearestDataResult[target].Count = count;
        for (int k = 0; k < count; k++)
        {
            NearestDataResult[target].Dist[k] = dist[k];
            NearestDataResult[target].Index[k] = index[k];
        }

        if (count == 2)
        {
            if (NearestDataResult[target].Index[0] == NearestDataResult[target].Index[1])
            {
                /*
                   printf("TASK!!=%d target=%d  task_0=%d index_0=%d  task_1=%d index_1=%d\n",
                   ThisTask, target,
                   NearestDataResult[target].Task[0], NearestDataResult[target].Index[0],
                   NearestDataResult[target].Task[1], NearestDataResult[target].Index[1]);
                 */
                ENDRUN("Indexes are equal.");
            }
        }
    }
    return 0;
}

/**
 * @brief Find the nearest two neighbors for a particle using the tree search algorithm.
 *
 * This function finds the nearest two neighbors for a given particle by performing a tree search algorithm.
 * It searches for neighbors within a certain smoothing length around the particle's position, excluding the
 * self-particle. The search is conducted recursively through the octree structure until the nearest neighbors
 * are found. It operates in two modes: local and remote. In the local mode, it searches for neighbors of a particle
 * stored locally. In the remote mode, it searches for neighbors of a particle received from another MPI task.
 *
 * @param searchcenter The coordinates of the particle for which to find the nearest neighbors.
 * @param hsml The smoothing length used to search for neighbors.
 * @param target The index of the particle for which to find the nearest neighbors.
 * @param startnode Pointer to the index of the starting node in the octree structure.
 * @param mode The mode of operation (local or remote).
 * @param hmax Pointer to the variable storing the maximum smoothing length encountered during the search.
 * @param nexport Pointer to the variable storing the number of particles to be exported to other MPI tasks.
 * @param nsend_local Pointer to the variable storing the number of particles to be sent to other MPI tasks.
 * @return The number of neighbors found for the particle.
 */
int subfind_ngb_treefind_nearesttwo(MyLongDouble searchcenter[3], double hsml, int target, int *startnode, int mode,
                                    double *hmax, int *nexport, int *nsend_local)
{
    int p, task;
    struct NODE *current;
    double dx, dy, dz, dist, r2;

#ifdef PERIODIC
    double xtmp;
#endif // PERIODIC

    int nexport_save = *nexport;

    *hmax = 0;
    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        if (no < All.MaxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

#ifdef DENSITY_SPLIT_BY_TYPE
            if (!((1 << P[p].Type) & (DENSITY_SPLIT_BY_TYPE)))
            {
                continue;
            }
#else // DENSITY_SPLIT_BY_TYPE
            if (!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
            {
                continue;
            }
#endif // DENSITY_SPLIT_BY_TYPE

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
            if ((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist)
            {
                continue;
            }

            Dist2list[numngb] = r2;
            Ngblist[numngb++] = p;
        }
        else
        {
            if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode one is not supported");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    if (Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                    {
                        Exportflag[task] = target;
                        Exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (Exportnodecount[task] == NODELISTLENGTH)
                    {
                        if (*nexport >= All.BunchSize)
                        {
                            *nexport = nexport_save;
                            ENDRUN_IF(nexport_save == 0, "Buffer too small to process single particle.");
                            for (int task = 0; task < NTask; task++)
                            {
                                nsend_local[task] = 0;
                            }
                            for (int no = 0; no < nexport_save; no++)
                            {
                                nsend_local[DataIndexTable[no].Task]++;
                            }
                            return -1;
                        }
                        Exportnodecount[task] = 0;
                        Exportindex[task] = *nexport;
                        DataIndexTable[*nexport].Task = task;
                        DataIndexTable[*nexport].Index = target;
                        DataIndexTable[*nexport].IndexGet = *nexport;
                        *nexport = *nexport + 1;
                        nsend_local[task]++;
                    }

                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                    if (Exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
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
            ;
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
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}

#endif // SUBFIND
