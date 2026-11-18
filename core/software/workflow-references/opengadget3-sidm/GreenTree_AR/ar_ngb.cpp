/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Antonio Ragagnin.
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
#include <time.h>
#ifdef NUM_THREADS
#include <pthread.h>
#endif

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/vector.hpp"

/*! \file ngb.c
 *  \brief neighbour search by means of the tree
 *
 *  This file contains routines for neighbour finding.  We use the
 *  gravity-tree and a range-searching technique to find neighbours.
 */

#ifdef NUM_THREADS
extern pthread_mutex_t mutex_nexport, mutex_partnodedrift;

#define LOCK_NEXPORT pthread_mutex_lock(&mutex_nexport);
#define UNLOCK_NEXPORT pthread_mutex_unlock(&mutex_nexport);
#define LOCK_PARTNODEDRIFT pthread_mutex_lock(&mutex_partnodedrift);
#define UNLOCK_PARTNODEDRIFT pthread_mutex_unlock(&mutex_partnodedrift);
#else // NUM_THREADS
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT
#define LOCK_PARTNODEDRIFT
#define UNLOCK_PARTNODEDRIFT
#endif // NUM_THREADS

#define KD_EXTEND_SEARCH 1.00001

/**
 * @brief Finds neighbors of a particle within a specified search radius using the tree-based.
 *
 * This function performs a neighbor search within a spherical volume centered at `searchcenter` with a radius
 * defined by `hsml` (the smoothing length). It traverses the tree structure to find particles or nodes that fall
 * within this volume and records them in `Ngblist`. The search is conducted in a parallel fashion, and particles
 * or nodes are processed according to the specified mode and drift correction requirements.
 *
 * @param searchcenter An array of length 3 representing the center of the search sphere.
 * @param phsml Pointer to the smoothing length used for determining the search radius.
 * @param startnode Pointer to the starting node index for the tree traversal.
 * @param mode An integer flag indicating the mode of the function. The behavior may vary based on this value.
 * @param exportflag Array used to mark which tasks should receive exported data.
 * @param exportnodecount Array counting the number of nodes to export to each task.
 * @param exportindex Array holding the starting index for export data in the receiving tasks.
 * @param ngblist Array to store the indices of neighboring particles found within the search radius.
 * @param targets Array of target particle indices for exporting data.
 * @param n_targets Number of target particles to consider for export.
 * @param should_drift Flag indicating whether drift correction should be applied to particles.
 * @param MyFOF_PRIMARY_LINK_TYPES A bitmask specifying the types of particles to consider as primary link types.
 * @return int The number of neighbors found within the search radius.
 *
 * @note Actually, particles in a box of half side length hsml are
 *  returned, i.e. the reduction to a sphere still needs to be done in the
 *  calling routine.
 */
int ar_green_ngb_treefind_fof_variable_threads(MyLongDouble searchcenter[3], MyFloat *phsml, int *startnode, int mode,
                                               int *exportflag, int *exportnodecount, int *exportindex, int *ngblist,
                                               int *targets, int n_targets, int should_drift,
                                               int MyFOF_PRIMARY_LINK_TYPES);

int ar_green_ngb_treefind_fof_variable_threads(MyLongDouble searchcenter[3], MyFloat *phsml, int *startnode, int mode,
                                               int *exportflag, int *exportnodecount, int *exportindex, int *ngblist,
                                               int *targets, int n_targets, int should_drift,
                                               int MyFOF_PRIMARY_LINK_TYPES)
{
    int numngb, no, nexp, p, task;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    MyFloat hsml = *phsml * KD_EXTEND_SEARCH;
    integertime ti_Current = All.Ti_Current;
    //  printf("walking for t=%d+%d, len=%f\n",targets[0],n_targets,hsml);
    MyLongDouble dx, dy, dz, dist;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    numngb = 0;
    no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
#ifdef GROUP_LEAVES
            no = NextGroupedNode[no];
#else // GROUP_LEAVES
            no = Nextnode[no];
#endif // GROUP_LEAVES

            if (!((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
            {
                continue;
            }

            if (should_drift)
                if (P[p].Ti_current != ti_Current)
                {
#pragma omp critical(_partnodedrift_)
                    drift_particle(p, ti_Current);
                }
            if (mode == 3)
            {
                continue;
            }

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
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            ngblist[numngb++] = p;
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                //              printf("no=%d, maxpart+maxnodes=%d\n",no,maxPart + maxNodes);
                int i_target = 0;

                ENDRUN_IF(mode == 1, "Mode 1 is not supported");

                if (targets[i_target] >= 0) /* if no target is given, export will not occur */
                {

                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != targets[i_target])
                    {
                        exportflag[task] = targets[i_target];
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
                            if (Nexport + n_targets - 1 >= bunchSize)
                            {
                                /* out of buffer space. Need to discard work for this particle and interrupt */
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport += n_targets;
                            }
                        }
                        UNLOCK_NEXPORT;
                        if (exitFlag)
                        {
                            //  for(;i_target<n_targets;i_target++)
                            // returns[i_target]=-1;
                            return -1;
                        }
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp; // per questo task, indice della coppia nexp->particella
                        for (i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataIndexTable[nexp + i_target].Task = task;
                            DataIndexTable[nexp + i_target].Index = targets[i_target];
                            DataIndexTable[nexp + i_target].IndexGet = nexp + i_target;
                        }
                    }
                    for (i_target = 0; i_target < n_targets; i_target++)
                    {
                        DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];
                    }
                    exportnodecount[task]++;
                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        for (i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] = -1;
                        }
                    }
                }
                no = Nextnode[no - maxNodes];
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

            if (should_drift)
                if (current->Ti_current != ti_Current)
                {
#pragma omp critical(_partnodedrift_)
                    force_drift_node(no, ti_Current);
                }

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            if (mode == 3 && !(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
            {
                no = Nodes[no].u.d.sibling;
                continue;
            }

            no = current->u.d.sibling; /* in case the node can be discarded */

            dist = hsml + 0.5 * current->len;
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

/**
 * @brief Finds neighbors of a particle within a variable search radius using the tree.
 *
 * This function performs a neighbor search within a spherical volume centered at `searchcenter` with a radius
 * defined by `hsml` (the smoothing length) extended by a factor `KD_EXTEND_SEARCH`. It traverses the tree structure
 * to find particles or nodes within this volume and records them in `Ngblist`. The search is parallelized and includes
 * options for drift correction and mode-specific behavior.
 *
 * @param searchcenter An array of length 3 representing the center of the search sphere.
 * @param phsml Pointer to the smoothing length used to define the search radius.
 * @param startnode Pointer to the starting node index for the tree traversal.
 * @param mode An integer flag indicating the mode of the function. The behavior may vary based on this value.
 * @param exportflag Array used to mark which tasks should receive exported data.
 * @param exportnodecount Array counting the number of nodes to export to each task.
 * @param exportindex Array holding the starting index for export data in the receiving tasks.
 * @param ngblist Array to store the indices of neighboring particles found within the search radius.
 * @param targets Array of target particle indices for exporting data.
 * @param n_targets Number of target particles to consider for export.
 * @param should_drift Flag indicating whether drift correction should be applied to particles.
 * @return int The number of neighbors found within the search radius.
 */
int ar_green_ngb_treefind_variable_threads(MyLongDouble searchcenter[3], MyFloat *phsml, int *startnode, int mode,
                                           int *exportflag, int *exportnodecount, int *exportindex, int *ngblist,
                                           int *targets, int n_targets, int should_drift)
{
    int numngb, no, nexp, p, task;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    MyFloat hsml = *phsml * KD_EXTEND_SEARCH;
    integertime ti_Current = All.Ti_Current;
    //  printf("walking for t=%d+%d, len=%f\n",targets[0],n_targets,hsml);
    MyLongDouble dx, dy, dz, dist;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    numngb = 0;
    no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
#ifdef GROUP_LEAVES
            no = NextGroupedNode[no];
#else // GROUP_LEAVES
            no = Nextnode[no];
#endif // GROUP_LEAVES

            if (P[p].Type > 0)
            {
                continue;
            }

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

            if (should_drift)
                if (P[p].Ti_current != ti_Current)
                {
#pragma omp critical(_partnodedrift_)
                    drift_particle(p, ti_Current);
                }
            if (mode == 3)
            {
                continue;
            }

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
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

#ifdef AR_GRAVCOST_IN_DENSITY
            P[p].GravCost[TakeLevel] += 0.2;
#endif // AR_GRAVCOST_IN_DENSITY

            ngblist[numngb++] = p;
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                //              printf("no=%d, maxpart+maxnodes=%d\n",no,maxPart + maxNodes);
                int i_target = 0;

                ENDRUN_IF(mode == 1, "Mode 1 is not supported");

                if (targets[i_target] >= 0) /* if no target is given, export will not occur */
                {
                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != targets[i_target])
                    {
                        exportflag[task] = targets[i_target];
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
                            if (Nexport + n_targets - 1 >= bunchSize)
                            {
                                /* out of buffer space. Need to discard work for this particle and interrupt */
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport += n_targets;
                            }
                        }
                        UNLOCK_NEXPORT;
                        if (exitFlag)
                        {
                            //  for(;i_target<n_targets;i_target++)
                            // returns[i_target]=-1;
                            return -1;
                        }
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp; // per questo task, indice della coppia nexp->particella
                        for (i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataIndexTable[nexp + i_target].Task = task;
                            DataIndexTable[nexp + i_target].Index = targets[i_target];
                            DataIndexTable[nexp + i_target].IndexGet = nexp + i_target;
                        }
                    }
                    for (i_target = 0; i_target < n_targets; i_target++)
                    {
                        DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];
                    }
                    exportnodecount[task]++;
                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        for (i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] = -1;
                        }
                    }
                }
                no = Nextnode[no - maxNodes];
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

            if (should_drift)
                if (current->Ti_current != ti_Current)
                {
#pragma omp critical(_partnodedrift_)
                    force_drift_node(no, ti_Current);
                }

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            if (mode == 3 && !(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
            {
                no = Nodes[no].u.d.sibling;
                continue;
            }

            no = current->u.d.sibling; /* in case the node can be discarded */

            dist = hsml + 0.5 * current->len;
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

/**
 * @brief Finds neighboring particles within a given search radius using a tree structure.
 *
 * This function traverses a tree structure to identify all particles that lie within a specified search radius from a
 * given center. It also handles periodic boundary conditions and can export node information for further processing.
 *
 * @param searchcenter Coordinates of the center of the search sphere.
 * @param phsml Pointer to the smoothing length of the particle.
 * @param pradius Pointer to the radius of the search sphere.
 * @param startnode Pointer to the starting node for the search. On output, it will be set to -1 when the search is
 * complete.
 * @param mode Mode of operation affecting the search behavior.
 * @param exportflag Array indicating which tasks should export data.
 * @param exportnodecount Array to count the number of nodes to export for each task.
 * @param exportindex Array of indices where node export data begins for each task.
 * @param ngblist Array to store the indices of neighboring particles found.
 * @param targets Array of target particles for export.
 * @param n_targets Number of target particles.
 * @param should_drift Flag indicating if particles should be drifted according to the current time.
 *
 * @return The number of neighboring particles found.
 */
int ar_green_ngb_treefind_pairs_threads(MyLongDouble searchcenter[3], MyFloat *phsml, MyFloat *pradius, int *startnode,
                                        int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist,
                                        int *targets, int n_targets, int should_drift)
{
    int no, p, numngb, task, nexp;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    MyFloat hsml = *phsml;
    MyFloat radius = *pradius * KD_EXTEND_SEARCH;
    MyFloat hsmlpradius = hsml + radius;
    integertime ti_Current = All.Ti_Current;
    MyLongDouble dist, dx, dy, dz;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    numngb = 0;

    no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
#ifdef GROUP_LEAVES
            no = NextGroupedNode[no];
#else // GROUP_LEAVES
            no = Nextnode[no];
#endif // GROUP_LEAVES

            if (P[p].Type > 0)
            {
                continue;
            }

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

            if (should_drift)
                if (P[p].Ti_current != ti_Current)
                {
#pragma omp critical(_partnodedrift_)
                    drift_particle(p, ti_Current);
                }

            if (mode == 3)
            {
                continue;
            }

            dist = max(P[p].Hsml, hsml) + radius;

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
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            ngblist[numngb++] = p; /* Note: unlike in previous versions of the code, the buffer
                          can hold up to all particles */
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                int i_target = 0;
#ifdef DONOTUSENODELIST
                if (mode == 1)
                {
                    no = Nextnode[no - maxNodes];
                    continue;
                }
#endif // DONOTUSENODELIST

                ENDRUN_IF(mode == 1, "mode =%d task=%d no=%d maxpart=%d maxnodes=%d +=%d startnode=%d\n", mode,
                          ThisTask, no, All.MaxPart, maxNodes, maxPart + maxNodes, *startnode);

                if (targets[i_target] >= 0) /* if no target is given, export will not occur */
                {
                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != targets[i_target])
                    {
                        exportflag[task] = targets[i_target];
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
                            if (Nexport + n_targets - 1 >= bunchSize)
                            {
                                /* out of buffer space. Need to discard work for this particle and interrupt */
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport += n_targets;
                            }
                        }
                        UNLOCK_NEXPORT;
                        if (exitFlag)
                        {
                            return -1;
                        }
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp; // per questo task, indice della coppia nexp->particella
                        for (i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataIndexTable[nexp + i_target].Task = task;
                            DataIndexTable[nexp + i_target].Index = targets[i_target];
                            DataIndexTable[nexp + i_target].IndexGet = nexp + i_target;
                        }
                    }
#ifndef DONOTUSENODELIST
                    for (i_target = 0; i_target < n_targets; i_target++)
                    {
                        DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];
                    }
                    exportnodecount[task]++;
                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        for (i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] = -1;
                        }
                    }
#endif // DONOTUSENODELIST
                }

                no = Nextnode[no - maxNodes];
                continue;
            }

            current = &Nodes[no];

#ifndef DONOTUSENODELIST
            if (mode == 1)
            {
                if (current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }
#endif // DONOTUSENODELIST

            if (should_drift)
                if (current->Ti_current != ti_Current)
                {
#pragma omp critical(_partnodedrift_)
                    force_drift_node(no, ti_Current);
                }

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            if (mode == 3 && !(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
            {
                no = Nodes[no].u.d.sibling;
                continue;
            }
            dist = max(Extnodes[no].hmax, hsml) + 0.5 * current->len + radius;

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
    *startnode = -1;
    return numngb;
}
