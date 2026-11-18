/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Neighbour search by means of the tree.
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
#include <time.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/vector.hpp"
#include "../System/communication.hpp"

/**
 * @brief Finds neighboring particles or nodes within a search radius using the tree.
 *
 * This function returns neighbours with distance <= hsml and returns them in
 * Ngblist. Actually, particles in a box of half side length hsml are
 * returned, i.e. the reduction to a sphere still needs to be done in the
 * calling routine.
 *
 * @param searchcenter The center of the search sphere.
 * @param hsml The smoothing length.
 * @param target The target particle index.
 * @param startnode Pointer to the starting node of the search.
 * @param mode The search mode.
 * @param nexport Pointer to the number of exported particles or nodes.
 * @param nsend_local Array to store the number of particles or nodes sent to each task.
 *
 * @return The number of neighboring particles or nodes found.
 */
int ngb_treefind_variable(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                          int *nexport, int *nsend_local)
{
    int p, task;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    integertime ti_Current = All.Ti_Current;
    MyLongDouble dist, dx, dy, dz;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    int nexport_save = *nexport;
    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

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

            if (P[p].Ti_current != ti_Current)
            {
                drift_particle(p, ti_Current);
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

                ENDRUN_IF(mode == 1, "Error in ngb_treefind_variable: mode == 1 not supported for pseudo particles.");

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

                            ENDRUN_IF(nexport_save == 0, "The buffer is too small to process even a single particle.");

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

            if (current->Ti_current != ti_Current)
            {
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

#ifdef GM_DMVELS
/**
 * @brief Finds neighboring particles or nodes within a search radius using the tree (with OpenMP parallelization).
 *
 * This function searches for neighboring particles or nodes within a specified search radius
 * centered around a given point.
 *
 * @param searchcenter The center of the search sphere.
 * @param hsml The smoothing length.
 * @param target The target particle index.
 * @param startnode Pointer to the starting node of the search.
 * @param mode The search mode.
 * @param exportflag Array indicating export flags.
 * @param exportnodecount Array indicating export node counts.
 * @param exportindex Array indicating export indices.
 * @param ngblist Array to store neighboring particle or node indices.
 *
 * @return The number of neighboring particles or nodes found.
 */
int ngb_treefind_variable_threads_dmvelfb(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                          int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                          int *ngblist)
{
    int nexp, p, task;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    integertime ti_Current = All.Ti_Current;

    MyLongDouble dx, dy, dz, dist;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    int numngb = 0;
    int no = *startnode;
    while (no >= 0)
    {

        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

            if (P[p].Type > 1) /* GM WARNING: including DM particles, only called by smooth_simple */
            {
                continue;
            }

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

            if (P[p].Ti_current != ti_Current)
            {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                drift_particle(p, ti_Current);
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
                ENDRUN_IF(
                    mode == 1,
                    "Error in ngb_treefind_variable_threads_dmvelfb: mode == 1 not supported for pseudo particles.");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (exportnodecount[task] == NODELISTLENGTH)
                    {
#pragma omp atomic capture
                        nexp = Nexport++;
                        if (nexp >= bunchSize)
                        {
                            BufferFullFlag = 1;
                            Nexport = bunchSize;
                            return -1;
                        }
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp;
                        DataIndexTable[nexp].Task = task;
                        DataIndexTable[nexp].Index = target;
                        DataIndexTable[nexp].IndexGet = nexp;
                    }

                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                        DomainNodeIndex[no - (maxPart + maxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
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

            if (current->Ti_current != ti_Current)
            {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
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
#endif // GM_DMVELS
#ifdef AR_GREEN_TREE_FOF
/**
 * @brief Finds neighboring particles or nodes within a search radius for FOF groups using OpenMP parallelization.
 *
 * This function searches for neighboring particles or nodes within a specified search radius
 * centered around a given point, specifically for FOF groups. It utilizes a tree-based algorithm
 * for efficient search.
 *
 * @param searchcenter The center of the search sphere.
 * @param hsml The smoothing length.
 * @param target The target particle index.
 * @param startnode Pointer to the starting node of the search.
 * @param mode The search mode.
 * @param exportflag Array indicating export flags.
 * @param exportnodecount Array indicating export node counts.
 * @param exportindex Array indicating export indices.
 * @param ngblist Array to store neighboring particle or node indices.
 * @param MyFOF_PRIMARY_LINK_TYPES Type of FOF primary link.
 *
 * @return The number of neighboring particles or nodes found.
 */
int ngb_treefind_fof_nearest_openmp(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                                    int *exportflag, int *exportnodecount, int *exportindex, int *ngblist,
                                    int MyFOF_PRIMARY_LINK_TYPES)
{
    int nexp, p, task;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    integertime ti_Current = All.Ti_Current;
    MyLongDouble dx, dy, dz, dist;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    int numngb = 0;
    int no = *startnode;
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
                ENDRUN_IF(mode == 1,
                          "Error in ngb_treefind_fof_nearest_openmp: mode == 1 not supported for pseudo particles.");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (exportnodecount[task] == NODELISTLENGTH)
                    {
#pragma omp atomic capture
                        nexp = Nexport++;
                        if (nexp >= bunchSize)
                        {
                            BufferFullFlag = 1;
                            Nexport = bunchSize;
                            return -1;
                        }
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp;
                        DataIndexTable[nexp].Task = task;
                        DataIndexTable[nexp].Index = target;
                        DataIndexTable[nexp].IndexGet = nexp;
                    }

                    DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                        DomainNodeIndex[no - (maxPart + maxNodes)];

                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
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

            if (current->Ti_current != ti_Current)
            {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
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
#endif // AR_GREEN_TREE_FOF


#define n_last_nodes 20

/**
 * @brief Build tree.
 *
 * This function constructs the neighbour tree. To this end, we actually need
 * to construct the gravitational tree, because we use it now for the
 * neighbour search.
 */
void ngb_treebuild(void)
{
    TIMER_PUSH(1, "NGB_TREEBUILD");
    VERBOSE(3, "Begin Ngb-tree construction.");
    force_treebuild(NumPart, NULL);
    VERBOSE(4, "Ngb-Tree contruction finished.");
    TIMER_POP(1, "NGB_TREEBUILD");
}

/**
 * @brief Finds FOF primary neighbors within a given search radius and returns them in Ngblist.
 *
 * This function returns FOF primary neighbors with a distance less than or equal to the given
 * smoothing length (hsml) around the specified search center. Particles within a
 * box of half side length hsml are considered, and the reduction to a sphere still
 * needs to be done in the calling routine.
 *
 * @param searchcenter Array containing the coordinates of the search center.
 * @param hsml The smoothing length.
 * @param target The target particle index.
 * @param startnode The starting node of the search.
 * @param mode The search mode.
 * @param nexport Pointer to the number of particles to be exported.
 * @param nsend_local Pointer to the number of particles to be sent locally.
 * @param MyFOF_PRIMARY_LINK_TYPES Bitmask indicating FOF primary link types.
 *
 * @return Returns the number of FOF primary neighbors found.
 */
int ngb_treefind_fof_primary(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                             int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES)
{
    int p, task, nexport_save;
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

    int numngb = 0;
    int no = *startnode;

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
                ENDRUN_IF(mode == 1, "Error in ngb_treefind_fof_primary");

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

                            ENDRUN_IF(nexport_save == 0, "The buffer is too small to process even a single particle.");

                            for (int task = 0; task < NTask; task++)
                                nsend_local[task] = 0;
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
                                        {
                                            break;
                                        }
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
 * @brief Finds all particles of type FOF_PRIMARY_LINK_TYPES within a given smoothing length to find the nearest one.
 *
 * This function finds all particles of type FOF_PRIMARY_LINK_TYPES within a given smoothing length (hsml) around
 * the specified search center in order to find the nearest one. Particles within a box of half side length hsml
 * are considered, and the reduction to a sphere still needs to be done in the calling routine.
 *
 * @param searchcenter Array containing the coordinates of the search center.
 * @param hsml The smoothing length.
 * @param target The target particle index.
 * @param startnode The starting node of the search.
 * @param mode The search mode.
 * @param nexport Pointer to the number of particles to be exported.
 * @param nsend_local Pointer to the number of particles to be sent locally.
 * @param MyFOF_PRIMARY_LINK_TYPES Bitmask indicating FOF primary link types.
 *
 * @return Returns the number of particles of type FOF_PRIMARY_LINK_TYPES found within the smoothing length.
 */
int ngb_treefind_fof_nearest(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                             int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES)
{
    int p, task;
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

    int nexport_save = *nexport;
    int numngb = 0;
    int no = *startnode;

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
                ENDRUN_IF(mode == 1, "Error in ngb_treefind.");

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

                        ENDRUN_IF(nexport_save == 0, "The buffer is too small to process even a single particle.");

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

#ifdef ADAPTGRAVSOFT
#include "AdaptGravSoft/ads_ngb.cpp"
#endif // ADAPTGRAVSOFT
