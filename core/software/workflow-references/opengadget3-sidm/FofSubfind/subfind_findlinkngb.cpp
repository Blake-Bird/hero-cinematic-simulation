/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Functions to find linked neighbours.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/communication.hpp"
#include "subfind.hpp"

#ifdef SUBFIND

#define LOCK_NEXPORT
#define UNLOCK_NEXPORT

/**
 * @brief Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct linkngbdata_in
{
    MyLongDouble Pos[3];
    MyFloat DM_Hsml;
    int NodeList[NODELISTLENGTH];
} *LinkngbDataIn, *LinkngbDataGet;

/**
 * @brief Structure for communication during the density computation. Holds data that is recieved from other processors.
 */
static struct linkngbdata_out
{
    int Ngb;
} *LinkngbDataResult, *LinkngbDataOut;

/**
 * @brief Find linked neighbors for particles.
 *
 * This function identifies linked neighbors for particles by iteratively searching for neighbors until the desired
 * number of neighbors (DesLinkNgb) is achieved for each particle. It allocates buffers to arrange communication,
 * prepares particle data for export, exchanges particle data among tasks, and processes the received data to add it to
 * the local particles. The process continues iteratively until the desired number of neighbors is achieved for all
 * particles or until the maximum number of iterations (MAXITER) is reached. After the iteration, the function updates
 * the particle smoothing lengths (DM_Hsml) based on the achieved number of neighbors. If convergence is not achieved
 * within the maximum number of iterations, the function terminates with an error.
 */
void subfind_find_linkngb(void)
{
    long long ntot;
    int ndone, ndone_flag, npleft, dummy, iter = 0;
    MyFloat *Left, *Right;
    char *Todo;
    int place, nexport, nimport;
    double t0, t1;

    VERBOSE(2, "Start find_linkngb (%d particles on task=%d)", NumPartGroup, ThisTask);

    int save_DesNumNgb = All.DesNumNgb;
    All.DesNumNgb = All.DesLinkNgb; /* for simplicity, reset this value */

    /* allocate buffers to arrange communication */

#ifdef AR_PARALLEL_FINDLINKNGB
    Ngblist = (int *)mymalloc("Ngblist", maxThreads * NumPartGroup * sizeof(int));
    Dist2list = (MyFloat *)mymalloc("Dist2list", maxThreads * NumPartGroup * sizeof(MyFloat));
#else // AR_PARALLEL_FINDLINKNGB
    Ngblist = (int *)mymalloc("Ngblist", NumPartGroup * sizeof(int));
    Dist2list = (MyFloat *)mymalloc("Dist2list", NumPartGroup * sizeof(MyFloat));
#endif // AR_PARALLEL_FINDLINKNGB

    Left = (MyFloat *)mymalloc("Left", sizeof(MyFloat) * NumPartGroup);
    Right = (MyFloat *)mymalloc("Right", sizeof(MyFloat) * NumPartGroup);
    Todo = (char *)mymalloc("Todo", sizeof(char) * NumPartGroup);

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize =
        (int)((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                              sizeof(struct linkngbdata_in) + sizeof(struct linkngbdata_out) +
                                              sizemax(sizeof(struct linkngbdata_in), sizeof(struct linkngbdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "SubFind FindLinkNGB: using %g MB for buffering, %g MB remaining for other buffers", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    for (int i = 0; i < NumPartGroup; i++)
    {
        Left[i] = Right[i] = 0;
        Todo[i] = 1;
    }

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {
        t0 = second();

        int i = 0; /* begin with this index */

        do
        {
            for (int j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
                Exportflag[j] = -1;
            }

            /* do local particles and prepare export list */

            nexport = 0; // why in the previous version we did declare nexport in the first parameter of for?

#ifdef AR_PARALLEL_FINDLINKNGB
            BufferFullFlag = 0;
            Nexport = 0;
            /*
               TODO: if the buffer is full, in the next iteration the loop starts again from i=0,
               ofc it doesn't  process  the already-done particles twice (it checks for Todo[i]).
               we can get an improvement it by adding a ParticleList there too.
               AR.
             */

            VERBOSE(2, "starting inner loop of subfind_linkngb.");

            i = 0;
            nexport = 0;
#pragma omp parallel
            while (1)
            {
                int myi;
#pragma omp critical
                myi = i++;

                if (myi >= NumPartGroup)
                {
                    break;
                }
                if (Todo[myi])
                {
                    if (subfind_linkngb_evaluate(myi, 0, &nexport, Send_count) < 0)
                    {
                        break;
                    }
                }
                if (myi % 100000 == 0)
                    VERBOSE_ALL(2, "Task %d: %d/%d done in subfind_linkngb.", ThisTask, myi, NumPartGroup);
            }
            nexport = Nexport;
#else // AR_PARALLEL_FINDLINKNGB
            for (nexport = 0; i < NumPartGroup; i++)
            {
                if (Todo[i])
                {
                    if (subfind_linkngb_evaluate(i, 0, &nexport, Send_count) < 0)
                    {
                        break;
                    }
                }
            }
#endif // AR_PARALLEL_FINDLINKNGB

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

            LinkngbDataGet =
                (struct linkngbdata_in *)mymalloc("	  LinkngbDataGet", nimport * sizeof(struct linkngbdata_in));
            LinkngbDataIn =
                (struct linkngbdata_in *)mymalloc("	  LinkngbDataIn", nexport * sizeof(struct linkngbdata_in));

            /* prepare particle data for export */
            for (int j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;

                LinkngbDataIn[j].Pos[0] = P[place].Pos[0];
                LinkngbDataIn[j].Pos[1] = P[place].Pos[1];
                LinkngbDataIn[j].Pos[2] = P[place].Pos[2];
                LinkngbDataIn[j].DM_Hsml = P[place].DM_Hsml;

                memcpy(LinkngbDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                       NODELISTLENGTH * sizeof(int));
            }

            /* exchange particle data */
            TIMER_PUSH(4, "subfind_comm");
            templated_exchange_data_Sendrecv(LinkngbDataIn, LinkngbDataGet, TAG_DENS_A, Send_offset, Send_count,
                                             Recv_offset, Recv_count);
            TIMER_POP(4, "subfind_comm");

            myfree(LinkngbDataIn);
            LinkngbDataResult =
                (struct linkngbdata_out *)mymalloc("	  LinkngbDataResult", nimport * sizeof(struct linkngbdata_out));
            LinkngbDataOut =
                (struct linkngbdata_out *)mymalloc("	  LinkngbDataOut", nexport * sizeof(struct linkngbdata_out));

            /* now do the particles that were sent to us */
#ifdef AR_PARALLEL_FINDLINKNGB
#pragma omp parallel for
#endif // AR_PARALLEL_FINDLINKNGB
            for (int j = 0; j < nimport; j++)
            {
                subfind_linkngb_evaluate(j, 1, &dummy, &dummy);
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
            templated_exchange_data_Sendrecv(LinkngbDataResult, LinkngbDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POP(4, "subfind_comm");

            /* add the result to the local particles */
            for (int j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;
                P[place].DM_NumNgb += LinkngbDataOut[j].Ngb;
            }

            myfree(LinkngbDataOut);
            myfree(LinkngbDataResult);
            myfree(LinkngbDataGet);
        } while (ndone < NTask);

        /* do final operations on results */
        npleft = 0;
        for (int i = 0; i < NumPartGroup; i++)
        {
            /* now check whether we had enough neighbours */
            if (Todo[i])
            {
                if (P[i].DM_NumNgb != All.DesLinkNgb &&
                    ((Right[i] - Left[i]) > 1.0e-3 * Left[i] || Left[i] == 0 || Right[i] == 0))
                {
                    /* need to redo this particle */
                    npleft++;

                    if (P[i].DM_NumNgb < All.DesLinkNgb)
                    {
                        Left[i] = max(P[i].DM_Hsml, Left[i]);
                    }
                    else
                    {
                        if (Right[i] != 0)
                        {
                            if (P[i].DM_Hsml < Right[i])
                                Right[i] = P[i].DM_Hsml;
                        }
                        else
                            Right[i] = P[i].DM_Hsml;
                    }

                    if (iter >= MAXITER - 10)
                    {
                        VERBOSE_ALL(
                            2,
                            "i=%d task=%d ID=%d DM_Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)",
                            i, ThisTask, (int)P[i].ID, P[i].DM_Hsml, Left[i], Right[i], (double)P[i].DM_NumNgb,
                            Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                    }

                    if (Right[i] > 0 && Left[i] > 0)
                    {
                        P[i].DM_Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
                    }
                    else
                    {
                        if (Right[i] == 0 && Left[i] == 0)
                        {
                            ENDRUN("Error in Right and Left as both are zero."); /* can't occur */
                        }

                        if (Right[i] == 0 && Left[i] > 0)
                        {
                            P[i].DM_Hsml *= 1.26;
                        }

                        if (Right[i] > 0 && Left[i] == 0)
                        {
                            P[i].DM_Hsml /= 1.26;
                        }
                    }
                }
                else
                {
                    Todo[i] = 0;
                }
            }
        }

        sumup_large_ints(1, &npleft, &ntot);

        t1 = second();

        if (ntot > 0)
        {
            iter++;

            if (iter > 0)
            {
                VERBOSE(2, "find linkngb iteration %d: need to repeat for %d%09d particles. (took %g sec)", iter,
                        (int)(ntot / 1000000000), (int)(ntot % 1000000000), timediff(t0, t1));
            }

            ENDRUN_IF(iter > MAXITER, "failed to converge in neighbour iteration in density()");
        }
    } while (ntot > 0);

    myfree(DataNodeList);
    myfree(DataIndexTable);

    myfree(Todo);
    myfree(Right);
    myfree(Left);

    myfree(Dist2list);
    myfree(Ngblist);

    All.DesNumNgb = save_DesNumNgb; /* restore it */
}

/**
 * @brief Evaluate linked neighbors for a target particle.
 *
 * This function evaluates linked neighbors for a target particle. It iterates through the tree nodes to find
 * neighboring particles within the smoothing length (h) of the target particle. If the mode is 0, the function operates
 * on the local particle data; if the mode is 1, it operates on the received particle data. The function updates the
 * number of neighbors (DM_NumNgb) for the target particle in mode 0 and stores the number of neighbors (Ngb) for the
 * target particle in mode 1. If the maximum smoothing length (hmax) encountered during the neighbor search in mode 0 is
 * greater than zero, it updates the smoothing length (DM_Hsml) for the target particle. The function returns 0 upon
 * successful completion and -1 if an error occurs during the neighbor search.
 *
 * @param target The index of the target particle.
 * @param mode The mode of operation (0 for local data, 1 for received data).
 * @param nexport Pointer to the number of particles to be exported.
 * @param nsend_local Pointer to the number of particles sent locally.
 * @return 0 if successful, -1 if an error occurs.
 */
int subfind_linkngb_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
    int startnode, ngbs, listindex = 0;
    double hmax;
    double h;
    MyLongDouble *pos;

    if (mode == 0)
    {
        pos = P[target].Pos;
        h = P[target].DM_Hsml;
    }
    else
    {
        pos = LinkngbDataGet[target].Pos;
        h = LinkngbDataGet[target].DM_Hsml;
    }

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = LinkngbDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    int numngb = 0;

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
#ifdef AR_PARALLEL_FINDLINKNGB
            int thread_id = THREAD_ID_OR_ZERO();
            int *ngblist = Ngblist + thread_id * NumPartGroup;
            MyFloat *dist2list = Dist2list + thread_id * NumPartGroup;
            if (mode == 0)
            {
                int *exportflag = Exportflag + thread_id * NTask;
                int *exportnodecount = Exportnodecount + thread_id * NTask;
                int *exportindex = Exportindex + thread_id * NTask;
                ngbs = subfind_ngb_treefind_linkngb_threads(pos, h, target, &startnode, mode, &hmax, exportflag,
                                                            exportnodecount, exportindex, ngblist, dist2list);

                //      ENDRUN_IF(ngbs<1, "ngb=%d",mode);
            }
            else if (mode == 1)
            {
                ngbs = subfind_ngb_treefind_linkngb_threads(pos, h, target, &startnode, mode, &hmax, NULL, NULL, NULL,
                                                            ngblist, dist2list);
            }
            else
            {
                ENDRUN_MAIN("mode==%d", mode);
            }
#else // AR_PARALLEL_FINDLINKNGB
            ngbs = subfind_ngb_treefind_linkngb(pos, h, target, &startnode, mode, &hmax, nexport, nsend_local);
#endif // AR_PARALLEL_FINDLINKNGB
            if (ngbs < 0)
            {
                return -1;
            }

            if (mode == 0 && hmax > 0)
            {
                P[target].DM_Hsml = hmax;
            }

            numngb += ngbs;
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = LinkngbDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        P[target].DM_NumNgb = numngb;
    }
    else
    {
        LinkngbDataResult[target].Ngb = numngb;
    }

    return 0;
}

/**
 * @brief Find neighboring particles within the smoothing length around a given search center.
 *
 * This function finds neighboring particles within the given smoothing length (hsml) around a specified search center.
 * It traverses the tree nodes to locate particles that fall within the search radius.
 * The function updates the number of neighbors found (numngb) and stores their indices and squared distances in Ngblist
 * and Dist2list arrays, respectively. If the mode is 0 (local particle), and if all the neighbors are local, the
 * function sorts the neighbors based on their squared distances and updates the maximum smoothing length (hmax)
 * encountered among the neighbors. The function handles exported particles and ensures the completion of the search by
 * setting the startnode to -1. It returns the total number of neighbors found.
 *
 * @param searchcenter The coordinates of the search center.
 * @param hsml The smoothing length within which neighbors are searched.
 * @param target The index of the target particle.
 * @param startnode Pointer to the starting node of the tree traversal.
 * @param mode The mode of operation (0 for local particle, 1 for exported particle).
 * @param hmax Pointer to store the maximum smoothing length encountered among the neighbors.
 * @param nexport Pointer to the number of particles to be exported.
 * @param nsend_local Pointer to the number of particles sent locally.
 * @return The total number of neighbors found.
 */
int subfind_ngb_treefind_linkngb(MyLongDouble searchcenter[3], double hsml, int target, int *startnode, int mode,
                                 double *hmax, int *nexport, int *nsend_local)
{
    int p, task, exported = 0;
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
#else // DENSITY_SPLIT_BY_TYPE
            if (!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif // DENSITY_SPLIT_BY_TYPE
                continue;

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
                ENDRUN_IF(mode == 1, "Mode equal one not supported.");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    exported = 1;
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
                            ENDRUN_IF(nexport_save == 0, "the buffer is too small to process even a single particle");
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

    if (mode == 0) /* local particle */
    {
        if (exported == 0) /* completely local */
            if (numngb >= All.DesNumNgb)
            {
                R2list = (struct r2data *)mymalloc("	  R2list", sizeof(struct r2data) * numngb);
                for (int i = 0; i < numngb; i++)
                {
                    R2list[i].index = Ngblist[i];
                    R2list[i].r2 = Dist2list[i];
                }

                STD_SORT(R2list, numngb, sizeof(struct r2data), subfind_ngb_compare_dist);

                *hmax = sqrt(R2list[All.DesNumNgb - 1].r2);
                numngb = All.DesNumNgb;

                for (int i = 0; i < numngb; i++)
                {
                    Ngblist[i] = R2list[i].index;
                    Dist2list[i] = R2list[i].r2;
                }

                myfree(R2list);
            }
    }

    *startnode = -1;
    return numngb;
}

/**
 * @brief Find neighboring particles within a given search radius in a tree-based data structure.
 *
 * This function traverses the tree structure to locate neighboring particles or pseudo-particles
 * within the specified search radius around a target particle. It can operate in two modes:
 *  - Mode 0: Find neighbors for local particles.
 *  - Mode 1: Export data for particles to be processed by other tasks.
 *
 * @param searchcenter The coordinates of the center of the search region.
 * @param hsml The smoothing length used for the search.
 * @param target The index of the target particle.
 * @param startnode Pointer to the index of the starting node in the tree.
 * @param mode The operation mode: 0 for local particles, 1 for exporting data.
 * @param hmax Pointer to store the maximum smoothing length encountered during the search.
 * @param exportflag Array to store export flags for each task.
 * @param exportnodecount Array to store the count of exported nodes for each task.
 * @param exportindex Array to store the export index for each task.
 * @param ngblist Array to store the indices of neighboring particles.
 * @param dist2list Array to store the squared distances of neighboring particles.
 * @return The number of neighboring particles found.
 */
int subfind_ngb_treefind_linkngb_threads(MyLongDouble searchcenter[3], double hsml, int target, int *startnode,
                                         int mode, double *hmax, int *exportflag, int *exportnodecount,
                                         int *exportindex, int *ngblist, MyFloat *dist2list)
{
    int p, task, nexp, exported = 0;
    struct NODE *current;
    double dx, dy, dz, dist, r2;

#ifdef PERIODIC
    double xtmp;
#endif // PERIODIC

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
#else // DENSITY_SPLIT_BY_TYPE
            if (!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif // DENSITY_SPLIT_BY_TYPE
                continue;

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

            dist2list[numngb] = r2;
            ngblist[numngb++] = p;
        }
        else
        {
            if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode equal one not supported.");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    exported = 1;
                    if (exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (exportnodecount[task] == NODELISTLENGTH)
                    {
                        int exitFlag = 0;
                        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
                        {
                            if (Nexport >= All.BunchSize)
                            {
                                /* Out of buffer space. Need to discard work for this particle and interrupt. */
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport++;
                            }
                        }
                        UNLOCK_NEXPORT;
                        if (exitFlag)
                        {
                            return -1;
                        }

                        exportnodecount[task] = 0;
                        exportindex[task] = nexp;
                        DataIndexTable[nexp].Task = task;
                        DataIndexTable[nexp].Index = target;
                        DataIndexTable[nexp].IndexGet = nexp;
                    }

                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
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

    if (mode == 0) /* local particle */
    {
        if (exported == 0) /* completely local */
        {
            if (numngb >= All.DesNumNgb)
            {
                // R2list = (struct r2data *)mymalloc("          R2list", sizeof(struct r2data) * numngb);
                //  use local variable + normal malloc because of the multithreading
                //  BTW, what was the advantage of using global variable here?
                struct r2data *r2list = (struct r2data *)malloc(sizeof(struct r2data) * numngb);
                for (int i = 0; i < numngb; i++)
                {
                    r2list[i].index = ngblist[i];
                    r2list[i].r2 = dist2list[i];
                }

                /************** This function will be called from a multithreaded region, should not use parallel
                 * sorting !!!! **************/
                qsort(r2list, numngb, sizeof(struct r2data), subfind_ngb_compare_dist);
                *hmax = sqrt(r2list[All.DesNumNgb - 1].r2);
                numngb = All.DesNumNgb;

                for (int i = 0; i < numngb; i++)
                {
                    ngblist[i] = r2list[i].index;
                    dist2list[i] = r2list[i].r2;
                }

                // myfree(R2list);
                free(r2list);
            }
        }
    }

    *startnode = -1;
    return numngb;
}

#ifdef DENSITY_SPLIT_BY_TYPE
/**
 * @brief Find neighboring particles within a given search radius in a tree-based data structure.
 *
 * This function traverses the tree structure to locate neighboring particles or pseudo-particles
 * within the specified search radius around a target particle. It can operate in two modes:
 *  - Mode 0: Find neighbors for local particles.
 *  - Mode 1: Export data for particles to be processed by other tasks.
 *
 * @param searchcenter The coordinates of the center of the search region.
 * @param hsml The smoothing length used for the search.
 * @param target The index of the target particle.
 * @param startnode Pointer to the index of the starting node in the tree.
 * @param mode The operation mode: 0 for local particles, 1 for exporting data.
 * @param hmax Pointer to store the maximum smoothing length encountered during the search.
 * @param exportflag Array to store export flags for each task.
 * @param exportnodecount Array to store the count of exported nodes for each task.
 * @param exportindex Array to store the export index for each task.
 * @param ngblist Array to store the indices of neighboring particles.
 * @param dist2list Array to store the squared distances of neighboring particles.
 * @return The number of neighboring particles found.
 */
int subfind_ngb_treefind_linkpairs_threads(MyLongDouble searchcenter[3], double hsml, int target, int *startnode,
                                           int mode, double *hmax, int *exportflag, int *exportnodecount,
                                           int *exportindex, int *ngblist, MyFloat *dist2list)
{
    int p, task, nexp, exported = 0;
    struct NODE *current;
    double dx, dy, dz, dist, r2;

    int startnode_backup = *startnode;
    int cpt = 0;

#ifdef PERIODIC
    double xtmp;
#endif // PERIODIC

    *hmax = 0;
    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        cpt++;
        if (no < All.MaxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

#ifdef DENSITY_SPLIT_BY_TYPE
            if (!((1 << P[p].Type) & (DENSITY_SPLIT_BY_TYPE)))
#else // DENSITY_SPLIT_BY_TYPE
            if (!((1 << P[p].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif // DENSITY_SPLIT_BY_TYPE
                continue;

            dist = max((double)P[p].DM_Hsml, hsml);
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

            dist2list[numngb] = r2;
            ngblist[numngb++] = p;
        }
        else
        {
            if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "no=%d\tstartnode_backup=%d\tcpt=%d", no, startnode_backup, cpt);

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    exported = 1;
                    if (exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (exportnodecount[task] == NODELISTLENGTH)
                    {
                        int exitFlag = 0;
                        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
                        {
                            if (Nexport >= All.BunchSize)
                            {
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport++;
                            }
                        }
                        UNLOCK_NEXPORT;
                        if (exitFlag)
                        {
                            return -1;
                        }
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp;
                        DataIndexTable[nexp].Task = task;
                        DataIndexTable[nexp].Index = target;
                        DataIndexTable[nexp].IndexGet = nexp;
                    }

                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                        DomainNodeIndex[no - (All.MaxPart + MaxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
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

            dist = max((double)Extnodes[no].hmax, hsml) + 0.5 * current->len;
            no = current->u.d.sibling; /* in case the node can be discarded */
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

    if (mode == 0) /* local particle */
    {
        if (exported == 0) /* completely local */
        {
            if (numngb >= All.DesNumNgb)
            {
                // R2list = (struct r2data *)mymalloc("          R2list", sizeof(struct r2data) * numngb);
                struct r2data *r2list = (struct r2data *)malloc(sizeof(struct r2data) * numngb);
                for (int i = 0; i < numngb; i++)
                {
                    r2list[i].index = ngblist[i];
                    r2list[i].r2 = dist2list[i];
                }

                /************** This function will be called from a multithreaded region, should not use parallel
                 * sorting !!!! **************/
                qsort(r2list, numngb, sizeof(struct r2data), subfind_ngb_compare_dist);

                *hmax = sqrt(r2list[All.DesNumNgb - 1].r2);
                numngb = All.DesNumNgb;

                for (int i = 0; i < numngb; i++)
                {
                    ngblist[i] = r2list[i].index;
                    dist2list[i] = r2list[i].r2;
                }

                // myfree(R2list);
                free(r2list);
            }
        }
    }

    *startnode = -1;
    return numngb;
}
#endif // DENSITY_SPLIT_BY_TYPE

/**
 * @brief Compare function for sorting elements based on squared distances.
 *
 * This function compares two elements of type struct r2data based on their squared distances (r2 values). It is used as
 * a comparison function for sorting an array of struct r2data elements. The function returns -1 if the squared distance
 * of the first element is less than that of the second element, +1 if it is greater, and 0 if they are equal.
 *
 * @param a Pointer to the first struct r2data element.
 * @param b Pointer to the second struct r2data element.
 * @return -1 if the squared distance of the first element is less than that of the second element, +1 if it is greater,
 * and 0 if they are equal.
 */
int subfind_ngb_compare_dist(const void *a, const void *b)
{
    if (((struct r2data *)a)->r2 < (((struct r2data *)b)->r2))
    {
        return -1;
    }

    if (((struct r2data *)a)->r2 > (((struct r2data *)b)->r2))
    {
        return +1;
    }

    return 0;
}

#endif // SUBFIND
