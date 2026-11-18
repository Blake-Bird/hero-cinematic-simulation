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

#ifndef OPENACC_NGB_H
#define OPENACC_NGB_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef NUM_THREADS
#include <pthread.h>
#endif // NUM_THREADS

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/vector.hpp"

#include "gpuallvars_branch.hpp"

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

/**
 * @brief Finds neighboring particles within a specified interaction range using multiple threads.
 *
 * This function performs a neighbor search for particle `i` in the communication buffer,
 * finding all particles `j` that can interact with `i` based on their smoothing lengths (`hsml`).
 * The search takes into account both conditions: \f$ r_{ij} < h_i \f$ or \f$ r_{ij} < h_j \f$.
 *
 * The function can also handle multi-threaded environments and pseudo-particles. It ensures that
 * interactions are calculated for all valid neighboring particles in a domain, using periodic boundary
 * corrections if necessary.
 *
 * @param searchcenter The 3D coordinates of the particle for which neighbors are being searched.
 * @param hsml The smoothing length of the particle.
 * @param target The target index of the particle in the communication buffer.
 * @param startnode Pointer to the start node for the tree search.
 * @param mode Search mode that defines how the tree traversal should proceed.
 * @param exportflag Pointer to array tracking the export status of nodes.
 * @param exportnodecount Pointer to the array of node counts for export.
 * @param exportindex Pointer to array of export indices.
 * @param ngblist Pointer to the list that will store found neighbor indices.
 * @return The number of neighboring particles found.
 */
#ifndef AXION_DM
int ngb_treefind_pairs_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                               int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
#else // AXION_DM
int ngb_treefind_pairs_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                               int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int targettype)
#endif // AXION_DM
{
    int no, p, numngb, task, nexp;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
#ifndef ACC
    integertime ti_Current = All.Ti_Current;
#endif // !ACC
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
            no = Nextnode[no];

#ifndef AXION_DM
            if (P[p].Type > 0)
            {
                continue;
            }
#else // AXION_DM
            if (P[p].Type != targettype)
            {
                continue;
            }
#endif // AXION_DM

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

#ifndef ACC
            if (P[p].Ti_current != ti_Current)
            {
                LOCK_PARTNODEDRIFT;
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                drift_particle(p, ti_Current);
                UNLOCK_PARTNODEDRIFT;
            }
#endif // #if !defined(ACC)

            dist = max(P[p].Hsml, hsml);

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
#ifdef DONOTUSENODELIST
                if (mode == 1)
                {
                    no = Nextnode[no - maxNodes];
                    continue;
                }
#endif // DONOTUSENODELIST
                ENDRUN_IF(mode == 1, "Mode 1 is not supported in this function.");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
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
                            if (Nexport >= bunchSize)
                            {
                                /* out of buffer space. Need to discard work for this particle and interrupt */
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

#ifndef DONOTUSENODELIST
                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                        DomainNodeIndex[no - (maxPart + maxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
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

#ifndef ACC
            if (current->Ti_current != ti_Current)
            {
                LOCK_PARTNODEDRIFT;
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                force_drift_node(no, ti_Current);
                UNLOCK_PARTNODEDRIFT;
            }
#endif // !ACC

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

#ifdef ACC // we do not want to interfer with the standard Gadget3 Version //or maybe bafter the next open?
            if (mode == 3 && !(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
            {
                no = Nodes[no].u.d.sibling;
                continue;
            }
#endif // ACC

            dist = max(Extnodes[no].hmax, hsml) + 0.5 * current->len;

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

/**
 * @brief Finds neighbors within a specified distance and returns them in Ngblist.
 *
 * This function searches for particles within a cubic volume of side length `hsml` centered at `searchcenter`.
 * It then filters these particles to those within a spherical region of radius `hsml`. The resulting particle
 * indices are stored in `Ngblist`. The function also handles node export if necessary.
 *
 * @param searchcenter The center of the search region.
 * @param hsml The smoothing length defining the search radius.
 * @param target The target particle index for export (if applicable).
 * @param startnode The starting node index for the search.
 * @param mode Determines the behavior of the search (e.g., whether to stop at top-level nodes).
 * @param nexport Pointer to the count of particles to export.
 * @param nsend_local Pointer to an array where the number of particles to send to each task is stored.
 * @param targettype (Optional) The type of particles to target (only if AXION_DM is defined).
 *
 * @return The number of neighbors found.
 */
#if !defined(AXION_DM)
int ngb_treefind_variable(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                          int *nexport, int *nsend_local)
#else // AXION_DM
int ngb_treefind_variable(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                          int *nexport, int *nsend_local, int targettype)
#endif // AXION_DM
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
#if !defined(ACC)
    integertime ti_Current = All.Ti_Current;
#endif // !defined(ACC)
    MyLongDouble dist, dx, dy, dz;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    nexport_save = *nexport;

    numngb = 0;
    no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

#if !defined(AXION_DM)
            if (P[p].Type > 0)
            {
                continue;
            }
#else // AXIOM_DM
            if (P[p].Type != targettype)
            {
                continue;
            }
#endif // AXIOM_DM

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

#if !defined(ACC)
            if (P[p].Ti_current != ti_Current)
            {
                drift_particle(p, ti_Current);
            }
#endif // #if !defined(ACC)

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

            Ngblist[numngb++] = p;
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode 1 is not supported.");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    if (Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                    {
                        Exportflag[task] = target;
                        Exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (Exportnodecount[task] == NODELISTLENGTH)
                    {
                        if (*nexport >= bunchSize)
                        {
                            *nexport = nexport_save;
                            ENDRUN_IF(nexport_save == 0, "Buffer is too mall to process a single particle.");
                            for (task = 0; task < NTask; task++)
                            {
                                nsend_local[task] = 0;
                            }
                            for (no = 0; no < nexport_save; no++)
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
                        DomainNodeIndex[no - (maxPart + maxNodes)];

                    if (Exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
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
#if !defined(ACC)

            if (current->Ti_current != ti_Current)
                force_drift_node(no, ti_Current);
#endif // #if !defined(ACC)

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
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

    // printf("%d\n", numngb);
    *startnode = -1;
    return numngb;
}

/**
 * @brief Constructs the neighbor tree.
 *
 * This function builds the neighbor tree required for efficient neighbor searches.
 * Internally, it constructs the gravitational tree since it is utilized for neighbor
 * searches in this context. The function starts by calling `force_treebuild` to
 * build the tree based on the number of particles.
 *
 * The verbosity levels of the log provide feedback on the start and completion
 * of the neighbor tree construction process.
 */
void ngb_treebuild(void)
{
    VERBOSE(3, "Begin Ngb-tree construction.");

    force_treebuild(NumPart, NULL);

    VERBOSE(4, "Ngb-Tree contruction finished.");
}

/**
 * @brief Finds neighbors for the Friends-of-Friends (FoF) algorithm within a specified radius.
 *
 * This function searches for neighbors within a specified smoothing length (hsml) from a given
 * center point, `searchcenter`, and stores them in `Ngblist`. The function is tailored for the
 * Friends-of-Friends (FoF) algorithm and handles nodes based on their type as specified by
 * `MyFOF_PRIMARY_LINK_TYPES`. It also supports different modes to control the behavior of
 * neighbor searches and data export.
 *
 * The function traverses the tree structure, checking if particles or nodes fall within the
 * defined radius. It handles different modes for data export and processing, ensuring
 * that nodes are efficiently processed based on their local or global status.
 *
 * @param searchcenter The center of the search sphere (3D coordinates).
 * @param hsml The smoothing length defining the search radius.
 * @param target The target particle index for export operations.
 * @param startnode The node to start the search from.
 * @param mode The mode controlling the function's behavior (e.g., export, local processing).
 * @param nexport Pointer to the number of nodes to export.
 * @param nsend_local Array to keep track of the number of nodes to send to each task.
 * @param MyFOF_PRIMARY_LINK_TYPES Bitmask specifying valid particle types for FoF linking.
 * @return The number of neighbors found.
 */
int ngb_treefind_fof_primary(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                             int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    MyLongDouble dx, dy, dz, dist, r2;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;

#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    nexport_save = *nexport;

    numngb = 0;
    no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

            if (!((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
            {
                continue;
            }

            if (mode == 0)
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

            Ngblist[numngb++] = p;
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode 1 is not supported.");

                if (mode == 0)
                {
                    if (Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                    {
                        Exportflag[task] = target;
                        Exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (Exportnodecount[task] == NODELISTLENGTH)
                    {
                        if (*nexport >= bunchSize)
                        {
                            *nexport = nexport_save;
                            ENDRUN_IF(nexport_save == 0, "Buffer is too small to process a single particle.");
                            for (task = 0; task < NTask; task++)
                            {
                                nsend_local[task] = 0;
                            }
                            for (no = 0; no < nexport_save; no++)
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
                        DomainNodeIndex[no - (maxPart + maxNodes)];

                    if (Exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                    }
                }

                if (mode == -1)
                {
                    *nexport = 1;
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

            if (mode == 0)
            {
                if (!(current->u.d.bitflags &
                      (1 << BITFLAG_TOPLEVEL))) /* we have a node with only local particles, can skip branch */
                {
                    no = current->u.d.sibling;
                    continue;
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
            if ((r2 = (dx * dx + dy * dy + dz * dz)) > dist * dist)
            {
                continue;
            }

            if ((current->u.d.bitflags & ((1 << BITFLAG_TOPLEVEL) + (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))) ==
                0) /* only use fully local nodes */
            {
                /* test whether the node is contained within the sphere */
                dist = hsml - FACT2 * current->len;
                if (dist > 0)
                {
                    if (r2 < dist * dist)
                    {
                        if (current->u.d.bitflags & (1 << BITFLAG_INSIDE_LINKINGLENGTH)) /* already flagged */
                        {
                            /* sufficient to return only one particle inside this cell */

                            p = current->u.d.nextnode;
                            while (p >= 0)
                            {
                                if (p < maxPart)
                                {
                                    if (((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
                                    {
                                        dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
                                        dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
                                        dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
                                        if (dx * dx + dy * dy + dz * dz > hsml * hsml)
                                            break;

                                        Ngblist[numngb++] = p;
                                        break;
                                    }
                                    p = Nextnode[p];
                                }
                                else if (p >= maxPart + maxNodes)
                                {
                                    p = Nextnode[p - maxNodes];
                                }
                                else
                                {
                                    p = Nodes[p].u.d.nextnode;
                                }
                            }
                            continue;
                        }
                        else
                        {
                            /* flag it now */
                            current->u.d.bitflags |= (1 << BITFLAG_INSIDE_LINKINGLENGTH);
                        }
                    }
                }
            }

            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}

/**
 * @brief Finds particles of a specified type within a given smoothing length.
 *
 * This function searches for particles of type specified by `MyFOF_PRIMARY_LINK_TYPES` within a
 * specified smoothing length (hsml) from a given center point, `searchcenter`. It aims to find
 * and list these particles that fall within the search radius.
 *
 * It handles different operational modes, such as local processing and data export. If particles
 * or nodes fall within the smoothing length, they are added to the `Ngblist`. For nodes, the function
 * checks whether they are within the search radius and whether they should be opened to further
 * processing based on their local or global status.
 *
 * @param searchcenter The center of the search sphere (3D coordinates).
 * @param hsml The smoothing length defining the search radius.
 * @param target The target particle index for export operations.
 * @param startnode The node to start the search from.
 * @param mode The mode controlling the function's behavior (e.g., export, local processing).
 * @param nexport Pointer to the number of nodes to export.
 * @param nsend_local Array to keep track of the number of nodes to send to each task.
 * @param MyFOF_PRIMARY_LINK_TYPES Bitmask specifying valid particle types for the search.
 *
 * @return The number of neighbors found.
 */
int ngb_treefind_fof_nearest(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                             int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    MyLongDouble dx, dy, dz, dist;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

#define FACT2 0.86602540

    nexport_save = *nexport;

    numngb = 0;
    no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

            if (!((1 << P[p].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
            {
                continue;
            }

            dist = hsml;
            dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
            if (dx > dist)
                continue;
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

            Ngblist[numngb++] = p;
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode 1 is not supported.");

                if (Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                {
                    Exportflag[task] = target;
                    Exportnodecount[task] = NODELISTLENGTH;
                }

                if (Exportnodecount[task] == NODELISTLENGTH)
                {
                    if (*nexport >= bunchSize)
                    {
                        *nexport = nexport_save;
                        ENDRUN_IF(nexport_save == 0, "Buffer is too small to handle a single particle.");
                        for (task = 0; task < NTask; task++)
                        {
                            nsend_local[task] = 0;
                        }
                        for (no = 0; no < nexport_save; no++)
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
                    DomainNodeIndex[no - (maxPart + maxNodes)];

                if (Exportnodecount[task] < NODELISTLENGTH)
                {
                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
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
            if ((dx * dx + dy * dy + dz * dz) > dist * dist)
            {
                continue;
            }

            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}

#if defined(RADTRANSFER) && defined(SFR)
/**
 * @brief Finds star particles within a given smoothing length from a specified center.
 *
 * This function searches for star particles within a specified smoothing length (hsml) from
 * a given center (`searchcenter`). It operates on a hierarchical tree structure, processing both
 * individual particles and nodes in the tree. The function performs the search based on particle
 * types and checks for density thresholds before considering particles for inclusion.
 *
 * The function handles different modes, such as local processing and data export. If particles
 * are within the smoothing length and meet density criteria, they are added to `Ngblist`. For nodes,
 * the function checks if they should be processed further based on their local or global status,
 * opening nodes as necessary to search for particles.
 *
 * @param searchcenter The center of the search sphere (3D coordinates).
 * @param hsml The smoothing length defining the search radius.
 * @param target The target particle index for export operations.
 * @param startnode The node to start the search from.
 * @param mode The mode controlling the function's behavior (e.g., export, local processing).
 * @param nexport Pointer to the number of nodes to export.
 * @param nsend_local Array to keep track of the number of nodes to send to each task.
 *
 * @return The number of neighbors (star particles) found.
 */
int ngb_treefind_stars(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode, int *nexport,
                       int *nsend_local)
{
    int numngb, no, p, task, nexport_save;
    struct NODE *current;
    t_vector box, hbox, vcenter;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    integertime ti_Current = All.Ti_Current;
    MyLongDouble dx, dy, dz, dist;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    nexport_save = *nexport;

    numngb = 0;
    no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

            if (P[p].Type > 0)
                continue;

            double a;

            if (All.ComovingIntegrationOn == 1)
            {
                a = All.Time;
            }
            else
            {
                a = 1.0;
            }

            if ((SphP[p].d.Density / (a * a * a)) >= All.PhysDensThresh)
            {
                continue;
            }
#if !defined(ACC)
            if (P[p].Ti_current != ti_Current)
            {
                drift_particle(p, ti_Current);
            }
#endif // #if !defined(ACC)

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

            Ngblist[numngb++] = p;
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode 1 is not supported.");

                if (Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                {
                    Exportflag[task] = target;
                    Exportnodecount[task] = NODELISTLENGTH;
                }

                if (Exportnodecount[task] == NODELISTLENGTH)
                {
                    if (*nexport >= bunchSize)
                    {
                        *nexport = nexport_save;
                        ENDRUN_IF(nexport_save == 0, "Buffer is too small to handle one particle.");
                        for (task = 0; task < NTask; task++)
                        {
                            nsend_local[task] = 0;
                        }
                        for (no = 0; no < nexport_save; no++)
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
                    DomainNodeIndex[no - (maxPart + maxNodes)];

                if (Exportnodecount[task] < NODELISTLENGTH)
                {
                    DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
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
#if !defined(ACC)
            if (current->Ti_current != ti_Current)
                force_drift_node(no, ti_Current);
#endif // #if !defined(ACC)

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
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
#endif // defined(RADTRANSFER) && defined(SFR)

#ifdef ADAPTGRAVSOFT
#include "AdaptGravSoft/ads_ngb.cpp"
#endif // ADAPTGRAVSOFT

/**
 * @brief Finds neighboring particles within a smoothing length using a local tree-based search.
 *
 * This function searches for particles within a specified smoothing length (`hsml`) from a given center
 * (`searchcenter`). It operates on a hierarchical tree structure and processes both individual particles
 * and nodes in the tree. It is designed to handle different types of particles and nodes, considering
 * specific types and constraints as defined by the function's parameters and compilation options.
 *
 * The function supports two modes:
 * - Mode 0: Basic search mode where it processes nodes and particles without additional constraints.
 * - Mode 1: Advanced search mode where it may update the start node and handle special conditions.
 *
 * It includes options for handling black holes and different types of particles, depending on the
 * compilation flags and provided parameters. The function also utilizes periodic boundary conditions
 * if enabled.
 *
 * @param searchcenter Array of length 3 representing the center of the search sphere (3D coordinates).
 * @param hsml The smoothing length defining the search radius.
 * @param mode The mode controlling the function's behavior (0 or 1).
 * @param startnode Pointer to the node to start the search from. Updated to the next node if mode is 1.
 * @param P Array of particle data structures, used to retrieve properties of particles.
 * @param ngblist Array to store the list of neighboring particles found.
 * @param maxnumngb The maximum number of neighbors to find.
 * @param targettype (Optional) Particle type to filter by (only applicable if `AXION_DM` is defined).
 * @param i_node Index of the current node in the tree (used in mode 1).
 *
 * @return The number of neighbors found within the smoothing length.
 */
ACC_DECLARE_ROUTINE_BEFORE
#ifndef AXION_DM
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
int ngb_treefind_pairs_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *P, int *ngblist, int maxnumngb, int i_node);
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP
#else // AXION_DM
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
int ngb_treefind_pairs_threads_local(MyLongDouble searchcenter[3], MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *P, int *ngblist, int maxnumngb, int targettype,
                                     int i_node);
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP
#endif // AXION_DM
ACC_DECLARE_ROUTINE_AFTER

#if defined(ACC_OPENACC) && defined(ACC_NGB)
#pragma acc routine
#endif // ACC_OPENACC && ACC_NGB)

#ifndef AXION_DM
int ngb_treefind_pairs_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *P, int *ngblist, int maxnumngb, int i_node)
#else // AXION_DM
int ngb_treefind_pairs_threads_local(MyLongDouble searchcenter[3], MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *P, int *ngblist, int maxnumngb, int targettype,
                                     int i_node)
#endif // AXIOM_DM
{
    int no, p, numngb;
    struct NODE *current;
    struct NODE *Nodes = Nodes_base - All.MaxPart;
    struct extNODE *Extnodes = Extnodes_base - All.MaxPart;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    if (mode == 1 && i_node == 1)
        (*startnode) = Nodes[*startnode].u.d.nextnode;

    MyLongDouble dist, dx, dy, dz;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    numngb = 0;

    no = *startnode;

    while (no >= 0)
    {
        if (numngb >= maxnumngb)
        {
            *startnode = no;
            return numngb;
        }

        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

#ifndef AXION_DM
            if (p >= N_gas)
            {
                continue;
            }
            if (P[p].Type > 0)
            {
                continue;
            }
#else // AXION_DM
            if (P[p].Type != targettype)
            {
                continue;
            }
#endif // AXION_DM

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

            dist = max(P[p].Hsml, hsml);

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

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            dist = max(Extnodes[no].hmax, hsml) + 0.5 * current->len;

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
// ACC_DECLARE_ROUTINE_AFTER

/**
 * @brief Finds neighbors within a variable smoothing length and returns them in `ngblist`.
 *
 * This function searches for neighboring particles within a variable smoothing length (`hsml`) from a specified center
 * (`searchcenter`). It operates on a hierarchical tree structure and considers both individual particles and tree
 * nodes. The function returns particles that are within a box of half side length `hsml`, but the reduction to a sphere
 * should be performed by the calling routine.
 *
 * The function can handle different particle types and constraints, depending on the compilation flags and input
 * parameters.
 *
 * @param searchcenter Array of length 3 representing the center of the search box (3D coordinates).
 * @param hsml The smoothing length defining the search box's half side length.
 * @param mode The mode controlling the function's behavior (0 or 1).
 * @param startnode Pointer to the node to start the search from. Updated to the next node if mode is 1.
 * @param P Array of particle data structures, used to retrieve properties of particles.
 * @param ngblist Array to store the list of neighboring particles found.
 * @param maxnumngb The maximum number of neighbors to find.
 * @param i_node Index of the current node in the tree (used in mode 1).
 * @param targettype (Optional) Particle type to filter by (only applicable if `AXION_DM` is defined).
 *
 * @return The number of neighbors found within the smoothing length.
 */
ACC_DECLARE_ROUTINE_BEFORE
#ifndef AXION_DM
int ngb_treefind_variable_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                        struct gravity_walkdata *P, int *ngblist, int maxnumngb, int i_node);
#else // AXION_DM
int ngb_treefind_variable_threads_local(MyLongDouble searchcenter[3], MyFloat hsml, int mode, int target,
                                        int *startnode, struct gravity_walkdata *P, int *ngblist, int targettype,
                                        int i_node);
#endif // AXION_DM
ACC_DECLARE_ROUTINE_AFTER

/**
 * @brief Finds neighbors within a variable smoothing length and returns them in `ngblist`.
 *
 * This function searches for neighboring particles within a specified smoothing length (`hsml`) from a given center
 * (`searchcenter`). It traverses a hierarchical tree structure, considering both individual particles and tree nodes.
 * The function fills the `ngblist` with particles that fall within a box of half side length `hsml`, and the reduction
 * to a sphere should be handled by the calling routine.
 *
 * The function can handle different particle types and constraints based on compilation flags and input parameters.
 *
 * @param searchcenter Array of length 3 representing the center of the search region (3D coordinates).
 * @param hsml The smoothing length defining the search box's half side length.
 * @param mode The mode controlling the function's behavior (0 or 1). If mode is 1 and `i_node` is 1, the function
 * continues from the next node.
 * @param startnode Pointer to the node to start the search from. This is updated to the next node if mode is 1.
 * @param P Array of particle data structures, used to retrieve properties of particles.
 * @param ngblist Array to store the list of neighboring particles found.
 * @param maxnumngb The maximum number of neighbors to find.
 * @param i_node Index of the current node in the tree. Used for continuing an old walk in GPU mode.
 * @param targettype (Optional) Particle type to filter by (only applicable if `AXION_DM` is defined).
 *
 * @return The number of neighbors found within the smoothing length. If the buffer `ngblist` is full, the function
 *         updates `startnode` to the current node and returns the number of neighbors found.
 */

// ACC_DECLARE_ROUTINE_BEFORE

#if defined(ACC_OPENACC) && defined(ACC_NGB)
#pragma acc routine
#endif // ACC_OPENACC && ACC_NGB
#ifndef AXION_DM
int ngb_treefind_variable_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                        struct gravity_walkdata *P, int *ngblist, int maxnumngb, int i_node)
#else // AXION_DM
int ngb_treefind_variable_threads_local(MyLongDouble searchcenter[3], MyFloat hsml, int mode, int target,
                                        int *startnode, struct gravity_walkdata *P, int *ngblist, int targettype,
                                        int i_node)
#endif // AXIOM_DM
{
    int numngb, no, p;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    struct NODE *Nodes = Nodes_base - All.MaxPart;
    // i_node indicates first or not iteration in GPU mode. if it is not the first your are continuing an old walk
    if (mode == 1 && i_node == 1)
    {
        (*startnode) = Nodes[*startnode].u.d.nextnode;
    }

    MyLongDouble dx, dy, dz, dist;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    numngb = 0;
    no = *startnode;

    while (no >= 0)
    {
        if (numngb >= maxnumngb)
        {
            *startnode = no;
            return numngb;
        }
        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

#ifndef AXION_DM
            if (p >= N_gas)
            {
                continue;
            }
            if (P[p].Type > 0)
            {
                continue;
            }
#else // AXIOM_DM
            if (P[p].Type != targettype)
            {
                continue;
            }
#endif // AXIOM_DM

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

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

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
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
// ACC_DECLARE_ROUTINE_AFTER

#endif // OPENACC_NGB_H
