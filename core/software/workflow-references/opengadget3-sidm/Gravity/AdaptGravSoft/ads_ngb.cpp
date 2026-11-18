/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Functions for the computaton of the adaptive gravitational softening.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Francesca Iannuzzi.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifdef ADAPTGRAVSOFT

#include "../../CodeBase/allvars.hpp"
#include "../../CodeBase/proto.hpp"

/**
 * @brief Find neighboring particles within a given search radius around a specified center.
 *
 * This function searches for neighboring particles or nodes within a specified
 * radius `hsml` around `searchcenter`. It updates the `ngblist` with indices of
 * neighboring particles found.
 *
 * @param searchcenter The center coordinates around which to search for neighbors.
 * @param hsml The smoothing length or search radius.
 * @param target The target flag for exporting nodes.
 * @param startnode Pointer to the starting node index for the search.
 * @param mode The mode of operation (1 for special behavior, 0 for default).
 * @param exportflag Array indicating export status for tasks.
 * @param exportnodecount Array tracking the count of exported nodes per task.
 * @param exportindex Array tracking the export index per task.
 * @param ngblist Array to store indices of neighboring particles.
 * @return Number of neighboring particles found.
 */
int ags_ngb_treefind_variable_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                                      int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
    int p, task;
    struct NODE *current;

#ifdef PERIODIC
    MyDouble xtmp;
#endif // PERIODIC

    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        if (no < All.MaxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

            if (P[p].Ti_current != All.Ti_Current)
            {
                drift_particle(p, All.Ti_Current);
            }

            MyDouble dist = hsml;
            MyDouble dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            MyDouble dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            MyDouble dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
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
            if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
            {
                ENDRUN_IF(mode == 1, "Mode 1 is not supported.");

                if (target >= 0) /* if no target is given, export will not occur */
                {
                    if (exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                    {
                        exportflag[task] = target;
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (exportnodecount[task] == NODELISTLENGTH)
                    {
                        if (Nexport >= All.BunchSize)
                        {
                            /* out if buffer space. Need to discard work for this particle and interrupt */
                            BufferFullFlag = 1;
                            return -1;
                        }
                        int nexp = Nexport;
                        Nexport++;
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

            if (current->Ti_current != All.Ti_Current)
            {
                force_drift_node(no, All.Ti_Current);
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

            MyDouble dist = hsml + 0.5 * current->len;
            ;
            MyDouble dx = NGB_PERIODIC_LONG_X(current->center[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            MyDouble dy = NGB_PERIODIC_LONG_Y(current->center[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            MyDouble dz = NGB_PERIODIC_LONG_Z(current->center[2] - searchcenter[2]);
            if (dz > dist)
                continue;
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

#endif // ADAPTGRAVSOFT
