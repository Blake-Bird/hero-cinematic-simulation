/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Functions to update the force tree.
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

/**
 * @brief Updates the tree structure dynamically after applying momentum kicks to particles.
 *
 * This function updates the tree structure dynamically after applying momentum kicks to particles.
 * It iterates over active particles, applying their momentum kicks to the tree nodes and updating
 * relevant properties such as maximum velocity, softening, and soundspeed. Once all kicks are applied,
 * it finishes the kick process for all affected tree nodes.
 */
void force_update_tree(void)
{
    TIMER_PUSH(1, "TREEUPDATE");
    VERBOSE(2, "kicks will prepare for dynamic update of tree\n");

    GlobFlag++;
    DomainNumChanged = 0;
    DomainList = (int *)mymalloc("DomainList", NTopleaves * sizeof(int));

    /* note: the current list of active particles still refers to that
     * synchronized at the previous time.
     */
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        force_kick_node(i, P[i].dp); /* kick the parent nodes with this momentum
                        difference, also updated maximum velocity, softening and soundspeed, if needed */
        for (int j = 0; j < 3; j++)
        {
            P[i].dp[j] = 0;
        }
    }

    force_finish_kick_nodes();
    myfree(DomainList);

    VERBOSE(2, "Tree has been updated dynamically.\n");
    TIMER_POP(1, "TREEUPDATE");
}

/**
 * @brief Apply momentum kick to a particle and updates the corresponding tree nodes.
 *
 * This function applies a momentum kick to a particle and updates the corresponding tree nodes
 * accordingly. It iterates up the tree hierarchy until it reaches the top-level nodes, updating
 * their properties and marking them for potential further processing.
 *
 * @param i Index of the particle to apply the kick to.
 * @param dp Array containing the momentum kick in each direction.
 */
void force_kick_node(int i, MyFloat *dp)
{
    MyFloat v, vmax = 0;

#ifdef NEUTRINOS
    if ((P[i].Type == type_neutrino) && (All.Time <= All.Time_tree_on_nu))
    {
        VERBOSE(5, "Situation 1\n");
        return;
    }
#endif // NEUTRINOS

    for (int j = 0; j < 3; j++)
    {
        if ((v = abs(P[i].Vel[j])) > vmax)
        {
            vmax = v;
        }
    }

    int no = Father[i];

    ENDRUN_IF(no < All.MaxPart, "error: at i=%d, no=%d < All.MaxPart=%d\n", i, no, All.MaxPart);

    while (no >= 0)
    {
        force_drift_node(no, All.Ti_Current);

        for (int j = 0; j < 3; j++)
        {
            Extnodes[no].dp[j] += dp[j];
        }

        if (Extnodes[no].vmax < vmax)
        {
            Extnodes[no].vmax = vmax;
        }

        Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);

        Extnodes[no].Ti_lastkicked = All.Ti_Current;

        if (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* top-level tree-node reached */
        {
            if (Extnodes[no].Flag != GlobFlag)
            {
                Extnodes[no].Flag = GlobFlag;
                DomainList[DomainNumChanged++] = no;
            }
            break;
        }

        ENDRUN_IF(no == Nodes[no].u.d.father, "Task=%d Fixed point  no == Nodes[no].u.d.father  detected, no=%d\n",
                  ThisTask, no);
        no = Nodes[no].u.d.father;
    }
}

/**
 * @brief Finishes the kicking process for tree nodes.
 *
 * This function completes the kicking process for tree nodes by exchanging momentum data
 * across CPUs and constructing momentum kicks in the top-level tree.
 */
void force_finish_kick_nodes(void)
{
    int no, ta, totDomainNumChanged;
    int *domainList_all;
    int *counts, *counts_dp, *offset_list, *offset_dp, *offset_vmax;
    MyLongDouble *domainDp_loc, *domainDp_all;

    MyFloat *domainVmax_loc, *domainVmax_all;

    /* share the momentum-data of the pseudo-particles accross CPUs */

    counts = (int *)mymalloc("counts", sizeof(int) * NTask);
    counts_dp = (int *)mymalloc("counts_dp", sizeof(int) * NTask);
    offset_list = (int *)mymalloc("offset_list", sizeof(int) * NTask);
    offset_dp = (int *)mymalloc("offset_dp", sizeof(int) * NTask);
    offset_vmax = (int *)mymalloc("offset_vmax", sizeof(int) * NTask);

    domainDp_loc = (MyLongDouble *)mymalloc("domainDp_loc", DomainNumChanged * 3 * sizeof(MyLongDouble));

    domainVmax_loc = (MyFloat *)mymalloc("domainVmax_loc", DomainNumChanged * sizeof(MyFloat));

    for (int i = 0; i < DomainNumChanged; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            domainDp_loc[i * 3 + j] = Extnodes[DomainList[i]].dp[j];
        }
        domainVmax_loc[i] = Extnodes[DomainList[i]].vmax;
    }

    MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MYMPI_COMM_WORLD);

    for (ta = 0, totDomainNumChanged = 0, offset_list[0] = 0, offset_dp[0] = 0, offset_vmax[0] = 0; ta < NTask; ta++)
    {
        totDomainNumChanged += counts[ta];
        if (ta > 0)
        {
            offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
            offset_dp[ta] = offset_dp[ta - 1] + counts[ta - 1] * 3 * sizeof(MyLongDouble);
            offset_vmax[ta] = offset_vmax[ta - 1] + counts[ta - 1] * sizeof(MyFloat);
        }
    }

    VERBOSE(2, "I exchange kick momenta for %d top-level nodes out of %d\n", totDomainNumChanged, NTopleaves);

    domainDp_all = (MyLongDouble *)mymalloc("domainDp_all", totDomainNumChanged * 3 * sizeof(MyLongDouble));

    domainVmax_all = (MyFloat *)mymalloc("domainVmax_all", totDomainNumChanged * sizeof(MyFloat));

    domainList_all = (int *)mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

    MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT, domainList_all, counts, offset_list, MPI_INT,
                   MYMPI_COMM_WORLD);

    for (int ta = 0; ta < NTask; ta++)
    {
        counts_dp[ta] = counts[ta] * 3 * sizeof(MyLongDouble);
        counts[ta] *= sizeof(MyFloat);
    }

    MPI_Allgatherv(domainDp_loc, DomainNumChanged * 3 * sizeof(MyLongDouble), MPI_BYTE, domainDp_all, counts_dp,
                   offset_dp, MPI_BYTE, MYMPI_COMM_WORLD);

    MPI_Allgatherv(domainVmax_loc, DomainNumChanged * sizeof(MyFloat), MPI_BYTE, domainVmax_all, counts, offset_vmax,
                   MPI_BYTE, MYMPI_COMM_WORLD);

    /* construct momentum kicks in top-level tree */
    for (int i = 0; i < totDomainNumChanged; i++)
    {
        no = domainList_all[i];

        if (Nodes[no].u.d.bitflags &
            (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS)) /* to avoid that the local one is kicked twice */
        {
            no = Nodes[no].u.d.father;
        }

        while (no >= 0)
        {
            force_drift_node(no, All.Ti_Current);

            for (int j = 0; j < 3; j++)
            {
                Extnodes[no].dp[j] += domainDp_all[3 * i + j];
            }

            if (Extnodes[no].vmax < domainVmax_all[i])
            {
                Extnodes[no].vmax = domainVmax_all[i];
            }

            Nodes[no].u.d.bitflags |= (1 << BITFLAG_NODEHASBEENKICKED);
            Extnodes[no].Ti_lastkicked = All.Ti_Current;

            no = Nodes[no].u.d.father;
        }
    }

    myfree(domainList_all);
    myfree(domainVmax_all);
    myfree(domainDp_all);
    myfree(domainVmax_loc);
    myfree(domainDp_loc);
    myfree(offset_vmax);
    myfree(offset_dp);
    myfree(offset_list);
    myfree(counts_dp);
    myfree(counts);
}

/**
 * @brief Drifts the properties of a tree node forward in time.
 *
 * This function drifts the properties of a tree node forward in time from `Nodes[no].Ti_current` to `time1`.
 *
 * @param no The index of the tree node.
 * @param time1 The target time to drift the node's properties to.
 */
void force_drift_node(int no, int time1)
{
    double dt_drift, dt_drift_hmax, fac;
#ifdef AR_XMAS_TREE // xmas tree has holes
    if (Nodes[no].allocated != 1)
    {
        return;
    }
#endif
    if (time1 == Nodes[no].Ti_current)
    {
        return;
    }

    // int time0 = Extnodes[no].Ti_lastkicked;

    if (Nodes[no].u.d.bitflags & (1 << BITFLAG_NODEHASBEENKICKED))
    {
        if (Extnodes[no].Ti_lastkicked != Nodes[no].Ti_current)
        {
            VERBOSE(0, "Task=%d Extnodes[no].Ti_lastkicked=%d  Nodes[no].Ti_current=%d\n", ThisTask,
                    (int)Extnodes[no].Ti_lastkicked, (int)Nodes[no].Ti_current);
            ENDRUN_MAIN("Inconsistency in drift node");
        }

        if (Nodes[no].u.d.mass)
        {
            fac = 1 / Nodes[no].u.d.mass;
        }
        else
        {
            fac = 0;
        }

        for (int j = 0; j < 3; j++)
        {
            Extnodes[no].vs[j] += fac * Extnodes[no].dp[j];
            Extnodes[no].dp[j] = 0;
        }
        Nodes[no].u.d.bitflags &= (~(1 << BITFLAG_NODEHASBEENKICKED));
    }

    if (All.ComovingIntegrationOn)
    {
        dt_drift_hmax = get_drift_factor(Nodes[no].Ti_current, time1);
        dt_drift = dt_drift_hmax;
    }
    else
    {
        dt_drift_hmax = (time1 - Nodes[no].Ti_current) * All.Timebase_interval;
        dt_drift = dt_drift_hmax;
    }

    for (int j = 0; j < 3; j++)
    {
        Nodes[no].u.d.s[j] += Extnodes[no].vs[j] * dt_drift;
    }
    Nodes[no].len += 2 * Extnodes[no].vmax * dt_drift;

    Nodes[no].Ti_current = time1;
}

/**
 * @brief Update hmax-values in tree nodes that hold SPH particles.
 *
 * This function updates the hmax-values in tree nodes that hold SPH
 * particles. These values are needed to find all neighbors in the
 * hydro-force computation.  Since the Hsml-values are potentially changed
 * in the SPH-denity computation, force_update_hmax() should be carried
 * out just before the hydrodynamical SPH forces are computed, i.e. after
 * density().
 */
void force_update_hmax(void)
{
    TIMER_PUSH(1, "TREEHMAXUPDATE");
    int no, ta;
    int *domainList_all;
    int *counts, *offset_list, *offset_hmax;
    MyFloat *domainHmax_loc, *domainHmax_all;
    int OffsetSIZE = 1;
    double dt_drift_hmax;
    MyFloat hmax_est;
    GlobFlag++;

    DomainNumChanged = 0;
    DomainList = (int *)mymalloc("DomainList", NTopleaves * sizeof(int));

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef BLACK_HOLES
        if (P[i].Type == 0 && P[i].Mass > 0)
#else // BLACK_HOLES
        if (P[i].Type == 0)
#endif // BLACK_HOLES
        {
            no = Father[i];
            while (no >= 0)
            {
                force_drift_node(no, All.Ti_Current);
                integertime time0 = P[i].Ti_current;
                integertime time1 = All.Ti_Current;
                if (time0 != time1)
                {
                    dt_drift_hmax = get_drift_factor(time0, time1);
                    MyAtLeastDouble hsml_evol_fak = exp(0.333333333333 * SphP[i].DivVel * dt_drift_hmax);
                    if (hsml_evol_fak > (1 + All.CourantFac))
                    {
#ifdef DEBUG
                        VERBOSE_ALL(0,
                                    "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange hsml_max evol "
                                    "value in force_update: %e",
                                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].DivVel,
                                    dt_drift_hmax, hsml_evol_fak);
#endif // DEBUG
                        hsml_evol_fak = 1 + All.CourantFac;
                    }
                    if (hsml_evol_fak < 1. / (1 + All.CourantFac))
                    {
#ifdef DEBUG
                        VERBOSE_ALL(0,
                                    "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange hsml_max evol "
                                    "value in force_update: %e",
                                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].DivVel,
                                    dt_drift_hmax, hsml_evol_fak);
#endif // DEBUG
                        hsml_evol_fak = 1. / (1 + All.CourantFac);
                    }
                    hmax_est = P[i].Hsml * hsml_evol_fak;
                }
                else
                {
                    hmax_est = P[i].Hsml;
                }

                if (P[i].Hsml > Extnodes[no].hmax || hmax_est > Extnodes[no].hmax)
                {
                    if (P[i].Hsml > Extnodes[no].hmax)
                    {
                        Extnodes[no].hmax = P[i].Hsml;
                    }

                    if (hmax_est > Extnodes[no].hmax)
                    {
                        Extnodes[no].hmax = hmax_est;
                    }
                    if (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node */
                    {
                        if (Extnodes[no].Flag != GlobFlag)
                        {
                            Extnodes[no].Flag = GlobFlag;
                            DomainList[DomainNumChanged++] = no;
                        }
                        break;
                    }
                }
                else
                {
                    break;
                }

                no = Nodes[no].u.d.father;
            }
        }
    }

    /* share the hmax-data of the pseudo-particles accross CPUs */
    counts = (int *)mymalloc("counts", sizeof(int) * NTask);
    offset_list = (int *)mymalloc("offset_list", sizeof(int) * NTask);
    offset_hmax = (int *)mymalloc("offset_hmax", sizeof(int) * NTask);

    domainHmax_loc = (MyFloat *)mymalloc("domainHmax_loc", DomainNumChanged * OffsetSIZE * sizeof(MyFloat));

    for (int i = 0; i < DomainNumChanged; i++)
    {
        domainHmax_loc[OffsetSIZE * i] = Extnodes[DomainList[i]].hmax;
    }

    MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MYMPI_COMM_WORLD);

    int totDomainNumChanged = 0;
    for (ta = 0, offset_list[0] = 0, offset_hmax[0] = 0; ta < NTask; ta++)
    {
        totDomainNumChanged += counts[ta];
        if (ta > 0)
        {
            offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
            offset_hmax[ta] = offset_hmax[ta - 1] + counts[ta - 1] * OffsetSIZE * sizeof(MyFloat);
        }
    }

    VERBOSE(2, "Hmax exchange: %d topleaves out of %d\n", totDomainNumChanged, NTopleaves);

    domainHmax_all = (MyFloat *)mymalloc("domainHmax_all", totDomainNumChanged * OffsetSIZE * sizeof(MyFloat));
    domainList_all = (int *)mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

    MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT, domainList_all, counts, offset_list, MPI_INT,
                   MYMPI_COMM_WORLD);

    for (int ta = 0; ta < NTask; ta++)
    {
        counts[ta] *= OffsetSIZE * sizeof(MyFloat);
    }

    MPI_Allgatherv(domainHmax_loc, OffsetSIZE * DomainNumChanged * sizeof(MyFloat), MPI_BYTE, domainHmax_all, counts,
                   offset_hmax, MPI_BYTE, MYMPI_COMM_WORLD);

    for (int i = 0; i < totDomainNumChanged; i++)
    {
        no = domainList_all[i];

        if (Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS)) /* to avoid that the hmax is updated twice */
        {
            no = Nodes[no].u.d.father;
        }

        while (no >= 0)
        {
            force_drift_node(no, All.Ti_Current);

            if (domainHmax_all[OffsetSIZE * i] > Extnodes[no].hmax)
            {
                Extnodes[no].hmax = domainHmax_all[OffsetSIZE * i];
            }
            else
            {
                break;
            }

            no = Nodes[no].u.d.father;
        }
    }

    myfree(domainList_all);
    myfree(domainHmax_all);
    myfree(domainHmax_loc);
    myfree(offset_hmax);
    myfree(offset_list);
    myfree(counts);
    myfree(DomainList);

    TIMER_POP(1, "TREEHMAXUPDATE");
}
