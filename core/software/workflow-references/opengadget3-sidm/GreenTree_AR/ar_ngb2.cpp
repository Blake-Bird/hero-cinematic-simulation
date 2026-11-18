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

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/vector.hpp"

/**
 * @brief Finds neighboring particles within a search radius using a tree structure.
 *
 * This function performs a neighbor search within a specified radius from a given center using a tree-based structure.
 * It supports periodic boundary conditions and can handle both single particles and pseudo particles. It also manages
 * export tasks for parallel processing.
 *
 * @param searchcenter Coordinates of the center of the search sphere.
 * @param hsml Smoothing length of the particle.
 * @param radius Search radius, must be greater than zero for hydro loop searches.
 * @param startnode Pointer to the starting node for the search. On completion, it will be set to -1.
 * @param mode Mode of operation affecting the search behavior (e.g., hybrid GPU run).
 * @param exportflag Array indicating which tasks should export data.
 * @param exportnodecount Array to count the number of nodes to export for each task.
 * @param exportindex Array of indices where node export data begins for each task.
 * @param ngblist Array to store the indices of neighboring particles found.
 * @param targettype Type of target particles for export.
 * @param targets Array of target particle indices for export.
 * @param returns Array to store return values for each target.
 * @param n_targets Number of target particles.
 * @param should_drift Flag indicating if particles should be drifted according to the current time.
 *
 * @return The number of neighboring particles found.
 */
int ar_green_ngb_treefind_pairs_threads(MyLongDouble searchcenter[3], MyFloat hsml,
                                        MyFloat radius, /* radius>0 is used in hydro loop search */
                                        int *startnode, int mode, int *exportflag, int *exportnodecount,
                                        int *exportindex, int *ngblist, int targettype, int *targets, int *returns,
                                        int n_targets, int should_drift)
{

    int numngb, no, nexp, p, task;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    if (radius < 0)
    {
        radius = 0.;
    }
    MyFloat hsmlpradius = hsml + radius;

    integertime ti_Current = All.Ti_Current;
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
            /* mode==3 implies we are in a hybrid GPU run and we only fill export buffer */

            dist = radius > 0 ? (max(PPP[p].Hsml, hsml) + radius) : (hsml);
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

            dist =
                radius > 0 ? (max(Extnodes[no].hmax, hsml) + 0.5 * current->len + radius) : (hsml + 0.5 * current->len);

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
