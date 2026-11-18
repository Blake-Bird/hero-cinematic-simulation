/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Serial version of substructure finding algorithm (Subfind).
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

#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef SUBFIND
#include "subfind.hpp"
#include "fof.hpp"

#ifndef MAX_NGB_CHECK
#define MAX_NGB_CHECK 3
#endif // MAX_NGB_CHECK

static int *Head, *Next, *Tail, *Len;

#ifdef KD_MAIN_HALO
static int *LenMain;
#endif // KD_MAIN_HALO

/**
 * @brief Structure to store candidate data.
 *
 * This structure holds information about candidates, including head,
 * length, number of sub-elements, rank, sub-number, parent, and bound length.
 */
static struct cand_dat
{
    int head;
    int len;
    int nsub;
    int rank, subnr, parent;
    int bound_length;
} *candidates;

/**
 * @brief Processes a group of particles to identify substructures serially.
 *
 * This function processes a group of particles, identifies substructures (subhalos) within
 * the group, and computes their properties. It performs a series of steps including neighbor
 * searching, candidate subhalo identification, and unbinding operations to determine
 * gravitationally bound substructures.
 *
 * @param nsnap The snapshot number.
 * @param gr The group index within the snapshot.
 * @param Offs The offset in the particle array where the group starts.
 * @return The updated offset after processing the group.
 *
 * The function follows these main steps:
 * 1. Finds the starting particle of the group.
 * 2. Allocates memory for temporary data structures.
 * 3. Builds a spatial tree for the particles in the group.
 * 4. Identifies subhalo candidates by examining local density maxima and linking particles.
 * 5. Merges groups of particles to form candidate subhalos.
 * 6. Unbinds subhalos to identify gravitationally bound structures.
 * 7. Assigns subgroup numbers and computes subhalo properties.
 * 8. Frees allocated memory.
 */
int subfind_process_group_serial(int nsnap, int gr, int Offs)
{
    int k, p, len, subnr, totlen, ss, ngbs, ndiff, N, head = 0, head_attach, len_non_gas;
    int listofdifferent[2], count, prev;
    int ngb_index, part_index, nsubs, rank;

    struct subgroup_properties this_group;

    static struct unbind_data *ud;
    int switch_heads;
    TIMER_PUSH(3, "preparing");

    while (P[Offs].GrNr != Group[gr].GrNr)
    {
        Offs++;
        ENDRUN_IF(Offs >= NumPart, "don't find a particle for groupnr=%d", Group[gr].GrNr);
    }

    N = Group[gr].Len;
    GrNr = Group[gr].GrNr;

    for (int i = 0; i < N; i++)
    {
        ENDRUN_IF(P[Offs + i].GrNr != Group[gr].GrNr,
                  "task=%d, gr=%d: don't have the number of particles for GrNr=%d group-len=%d found=%d before=%d",
                  ThisTask, gr, Group[gr].GrNr, N, P[Offs + i].GrNr, P[Offs - 1].GrNr);
    }

    candidates = (struct cand_dat *)mymalloc("candidates", N * sizeof(struct cand_dat));

    Head = (int *)mymalloc("Head", N * sizeof(int));
    Next = (int *)mymalloc("Next", N * sizeof(int));
    Tail = (int *)mymalloc("Tail", N * sizeof(int));
    Len = (int *)mymalloc("Len", N * sizeof(int));
#ifdef KD_MAIN_HALO
    LenMain = (int *)mymalloc("LenMain", N * sizeof(int));
#endif // KD_MAIN_HALO
    ud = (struct unbind_data *)mymalloc("ud", N * sizeof(struct unbind_data));

    Head -= Offs;
    Next -= Offs;
    Tail -= Offs;
    Len -= Offs;
#ifdef KD_MAIN_HALO
    LenMain -= Offs;
#endif // KD_MAIN_HALO

    for (int i = 0; i < N; i++)
    {
        ud[i].index = Offs + i;
    }

    subfind_loctree_findExtent(N, ud);

    subfind_loctree_treebuild(N, ud); /* build tree for all particles of this group */

    for (int i = Offs; i < Offs + N; i++)
    {
        Head[i] = Next[i] = Tail[i] = -1;
    }

    TIMER_POPPUSH(3, "preparing", "finding_candidates");

    /* note: particles are already ordered in the order of decreasing density */
    int count_cand = 0;
    for (int i = 0; i < N; i++)
    {
        part_index = Offs + i;

        int nfound =
            subfind_locngb_treefind(P[part_index].Pos, All.DesLinkNgb, P[part_index].DM_Hsml / FAC_HSML, part_index);
        //      printf("Task %d: For %d found %d ngbs\n",ThisTask,part_index,nfound);

        /* note: returned neighbours are already sorted by distance */
        for (k = 0, ndiff = 0, ngbs = 0; k < nfound && ngbs < MAX_NGB_CHECK && ndiff < 2; k++)
        {
            ngb_index = R2list[k].index;
            ngbs++;

            if (Head[ngb_index] >= 0) /* neighbor is attached to a group */
            {
                if (ndiff == 1)
                {
                    if (listofdifferent[0] == Head[ngb_index])
                    {
                        continue;
                    }
                }

                /* a new group has been found */
                listofdifferent[ndiff++] = Head[ngb_index];
            }
            else
            {
                ENDRUN("ThisTask=%d gr=%d k=%d i=%d part_index=%d ngb_index = %d  head[ngb_index]=%d "
                       "P[part_index].DM_Density=%g %g GrNrs= %d %d",
                       ThisTask, gr, k, i, part_index, ngb_index, Head[ngb_index], P[part_index].u.DM_Density,
                       P[ngb_index].u.DM_Density, P[part_index].GrNr, P[ngb_index].GrNr);
            }
        }

        //      printf("Task %d:    ndiff=%d  part_index=%d \n",ThisTask,ndiff,part_index);

        switch (ndiff) /* treat the different possible cases */
        {
        case 0: /* this appears to be a lonely maximum -> new group */
            head = part_index;
            Head[part_index] = Tail[part_index] = part_index;
            Len[part_index] = 1;
#ifdef KD_MAIN_HALO
            if (P[part_index].SubLevel == 0)
            {
                LenMain[part_index] = 1;
            }
            else
            {
                LenMain[part_index] = 0;
            }
#endif // KD_MAIN_HALO
            Next[part_index] = -1;
            break;

        case 1: /* the particle is attached to exactly one group */
            head = listofdifferent[0];
            Head[part_index] = head;
            Next[Tail[head]] = part_index;
            Tail[head] = part_index;
            Len[head]++;
#ifdef KD_MAIN_HALO
            if (P[part_index].SubLevel == 0)
            {
                LenMain[head]++;
            }
#endif // KD_MAIN_HALO
            Next[part_index] = -1;
            break;

        case 2: /* the particle merges two groups together */
            head = listofdifferent[0];
            head_attach = listofdifferent[1];
            switch_heads = 0;

#ifdef KD_MAIN_HALO
            if (LenMain[head_attach] + Len[head_attach] >
                LenMain[head] + Len[head]) /* other group is longer, swap them */
            {
                switch_heads = 1;
            }
#else // KD_MAIN_HALO
            if (Len[head_attach] > Len[head]) /* other group is longer, swap them */
            {
                switch_heads = 1;
            }
#endif // KD_MAIN_HALO

            if (switch_heads)
            {
                head = listofdifferent[1];
                head_attach = listofdifferent[0];
            }

            /* only in case the attached group is long enough we bother to register is
               as a subhalo candidate */

            if (Len[head_attach] >= All.DesLinkNgb)
            {
                candidates[count_cand].len = Len[head_attach];
                candidates[count_cand].head = Head[head_attach];
                count_cand++;
            }

            /* now join the two groups */
            Next[Tail[head]] = head_attach;
            Tail[head] = Tail[head_attach];
            Len[head] += Len[head_attach];

            ss = head_attach;
            do
            {
                Head[ss] = head;
            } while ((ss = Next[ss]) >= 0);

            /* finally, attach the particle */
            Head[part_index] = head;
            Next[Tail[head]] = part_index;
            Tail[head] = part_index;
            Len[head]++;
            Next[part_index] = -1;
            break;

        default:
            ENDRUN("Error in ndiff=%d", ndiff);
            break;
        }
    }

    TIMER_POPPUSH(3, "finding_candidates", "finalizing_candidates");

    /* add the full thing as a subhalo candidate */
    prev = -1;
    for (int i = 0; i < N; i++)
    {
        if (Head[Offs + i] == Offs + i)
        {
            if (Next[Tail[Offs + i]] == -1)
            {
                if (prev < 0)
                {
                    head = Offs + i;
                }
                if (prev >= 0)
                {
                    Next[prev] = Offs + i;
                }

                prev = Tail[Offs + i];
            }
        }
    }

    candidates[count_cand].len = N;
    candidates[count_cand].head = head;
    count_cand++;

    /* go through them once and assign the rank */
    p = head;
    rank = 0;
    for (int i = 0; i < N; i++)
    {
        Len[p] = rank++;
        p = Next[p];
    }

    /* for each candidate, we now pull out the rank of its head */
    for (int k = 0; k < count_cand; k++)
    {
        candidates[k].rank = Len[candidates[k].head];
    }

    for (int i = Offs; i < Offs + N; i++)
    {
        Tail[i] = -1;
    }

    TIMER_POPPUSH(3, "finalizing_candidates", "unbinding");
    nsubs = 0;
    for (int k = 0; k < count_cand; k++)
    {
        p = candidates[k].head;
        len = 0;
        for (int i = 0; i < candidates[k].len; i++, p = Next[p])
        {
            if (Tail[p] < 0)
            {
                ud[len++].index = p;
            }
        }

        if (len >= All.DesLinkNgb)
        {
            len = subfind_unbind(ud, len, &len_non_gas);
        }

#ifndef DO_GAS_CLOUDS
        if (len_non_gas >= All.DesLinkNgb)
#else // DO_GAS_CLOUDS
        if (len >= All.DesLinkNgb)
#endif // DO_GAS_CLOUDS
        {
            /* ok, we found a substructure */

            for (int i = 0; i < len; i++)
            {
                Tail[ud[i].index] = nsubs; /* we use this to flag the substructures */
            }

            candidates[k].nsub = nsubs;
            candidates[k].bound_length = len;
            nsubs++;
        }
        else
        {
            candidates[k].nsub = -1;
            candidates[k].bound_length = 0;
        }
    }

    TIMER_POPPUSH(3, "unbinding", "preparing_substructures");

    VERBOSE(1, "\nGroupLen=%d  (gr=%d)\nNumber of substructures: %d", N, gr, nsubs);

    Group[gr].Nsubs = nsubs;
    Group[gr].Pos[0] = Group[gr].CM[0];
    Group[gr].Pos[1] = Group[gr].CM[1];
    Group[gr].Pos[2] = Group[gr].CM[2];

    STD_SORT(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_serial_candidates_boundlength);

    /* now we determine the parent subhalo for each candidate */
    for (int k = 0; k < count_cand; k++)
    {
        candidates[k].subnr = k;
        candidates[k].parent = 0;
    }

    STD_SORT(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_serial_candidates_rank);

    for (int k = 0; k < count_cand; k++)
    {
        for (int j = k + 1; j < count_cand; j++)
        {
            if (candidates[j].rank > candidates[k].rank + candidates[k].len)
            {
                break;
            }

            if (candidates[k].rank + candidates[k].len >= candidates[j].rank + candidates[j].len)
            {
                if (candidates[k].bound_length >= All.DesLinkNgb)
                {
                    candidates[j].parent = candidates[k].subnr;
                }
            }
            else
            {
                ENDRUN("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d bound=%d", k, count_cand,
                       (int)candidates[k].rank, candidates[k].len, (int)candidates[k].bound_length, candidates[j].rank,
                       (int)candidates[j].len, candidates[j].bound_length);
            }
        }
    }

    STD_SORT(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_serial_candidates_subnr);
    /* now determine the properties */

    TIMER_POPPUSH(3, "preparing_substructures", "computing_substructure_properties");
    subnr = 0;
    totlen = 0;
    for (int k = 0; k < nsubs; k++)
    {
        len = candidates[k].bound_length;

        VERBOSE(1, "subnr=%d  SubLen=%d", subnr, len);

        totlen += len;
        p = candidates[k].head;
        count = 0;
        for (int i = 0; i < candidates[k].len; i++)
        {
            if (Tail[p] == candidates[k].nsub)
            {
                ud[count++].index = p;
            }

            p = Next[p];
        }

        ENDRUN_IF(count != len, "Count is uneqal to len: %d != %d", count, len);

        subfind_determine_sub_halo_properties(ud, len, 0, &this_group, k);

        ENDRUN_IF(Nsubgroups >= MaxNsubgroups,
                  "Number of subgroups is larger then the maximum number of subgroups: %d >= %d", Nsubgroups,
                  MaxNsubgroups);

        if (subnr == 0)
        {
            for (int j = 0; j < 3; j++)
            {
                Group[gr].Pos[j] = this_group.Pos[j];
            }
            this_group.Offset = Group[gr].Offset;
        }
        else
        {
            this_group.Offset = SubGroup[Nsubgroups - 1].Offset + SubGroup[Nsubgroups - 1].Len;
        }

        this_group.Len = len;
        this_group.GrNr = GrNr - 1;
        this_group.SubNr = subnr;
        this_group.SubParent = candidates[k].parent;
        this_group.Len = len;

        SubGroup[Nsubgroups] = this_group;

        Nsubgroups++;

        /* Let's now assign the subgroup number */

        for (int i = 0; i < len; i++)
        {
            P[ud[i].index].SubNr = subnr;
        }

        subnr++;
    }

    VERBOSE(1, "Fuzz=%d", N - totlen);

    TIMER_POP(3, "computing_substructure_properties");

    myfree(ud);
#ifdef KD_MAIN_HALO
    myfree(LenMain + Offs);
#endif // KD_MAIN_HALO
    myfree(Len + Offs);
    myfree(Tail + Offs);
    myfree(Next + Offs);
    myfree(Head + Offs);

    myfree(candidates);

    return Offs;
}

/**
 * @brief Unbinds particles based on their binding energy.
 *
 * This function iteratively refines a set of particles by computing their gravitational potentials, binding energies,
 * and removing unbound particles until a stable bound configuration is reached.
 *
 * @param ud Pointer to an array of unbind_data structures representing particles to be unbound.
 * @param len Number of particles in the ud array.
 * @param len_non_gas Pointer to an integer where the function will store the number of non-gas particles remaining
 * after unbinding.
 * @return The number of bound particles remaining after the unbinding process.
 *
 * The function performs the following steps:
 * - Computes the spatial extent of the particles.
 * - Builds a tree structure for efficient potential computation.
 * - Computes gravitational potentials for particles.
 * - Calculates the bulk velocity and center of mass of the particle set.
 * - Computes the binding energy for each particle based on their potential and kinetic energies.
 * - Sorts binding energies and determines an energy limit to identify weakly bound particles.
 * - Removes unbound particles from the set, ensuring no more than 1/4 of the original particles are removed in a single
 * iteration.
 * - Adjusts the phaseflag to control the recomputation of potentials in subsequent iterations.
 */
int subfind_unbind(struct unbind_data *ud, int len, int *len_non_gas)
{
    double minpot, *bnd_energy, energy_limit, weakly_bound_limit = 0;
    int minindex, unbound, phaseflag;
    double s[3], dx[3], v[3], dv[3], pos[3];
    double vel_to_phys, H_of_a, atime;
    double boxsize;
    double TotMass;
    int len_orig = len;

    double t0_sub, t1_sub, diff1 = 0, diff2 = 0, diff3 = 0, diff4 = 0, diff5 = 0, diff6 = 0, diff7 = 0;

    boxsize = All.BoxSize;

    if (All.ComovingIntegrationOn)
    {
        vel_to_phys = 1.0 / All.Time;
        H_of_a = hubble_function(All.Time);
        atime = All.Time;
    }
    else
    {
        vel_to_phys = atime = 1;
        H_of_a = 0;
    }

    bnd_energy = (double *)mymalloc("bnd_energy", len * sizeof(double));

    phaseflag = 0; /* this means we will recompute the potential for all particles */

    subfind_loctree_findExtent(len, ud);

    do
    {
        t0_sub = second();

        subfind_loctree_treebuild(len, ud);

        t1_sub = second();
        diff1 += (t1_sub - t0_sub);
        t0_sub = second();

        /* let's compute the potential  */
        if (phaseflag == 0) /* redo it for all the particles */
        {
            minindex = -1;
            minpot = 1.0e30;
#pragma omp parallel shared(minindex, minpot)
            {
                int myminindex = -1;
                double myminpot = 1.0e30;
#pragma omp for
                for (int i = 0; i < len; i++)
                {
                    int p = ud[i].index;

                    double pot = subfind_loctree_treeevaluate_potential(p);
                    /* note: add self-energy */
#ifndef ADAPTGRAVSOFT
                    P[p].u.DM_Potential = pot + P[p].Mass / All.SofteningTable[P[p].Type];
#else // ADAPTGRAVSOFT
                    P[p].u.DM_Potential = pot + P[p].Mass * 2.8 / P[p].AGS_Hsml;
#endif // ADAPTGRAVSOFT
                    P[p].u.DM_Potential *= All.GravityConstantInternal / atime;

                    if (All.TotN_gas > 0 && (FOF_PRIMARY_LINK_TYPES & 1) == 0 && (FOF_SECONDARY_LINK_TYPES & 1) == 0 &&
                        All.OmegaBaryon > 0)
                        P[p].u.DM_Potential *= All.Omega0 / (All.Omega0 - All.OmegaBaryon);

                    if (P[p].u.DM_Potential < myminpot)
                    {
                        myminpot = P[p].u.DM_Potential;
                        myminindex = p;
                    }
                }
#pragma omp critical(_minpot_)
                {
                    if (myminpot < minpot)
                    {
                        minpot = myminpot;
                        minindex = myminindex;
                    }
                }
            }

            for (int j = 0; j < 3; j++)
                pos[j] = P[minindex].Pos[j]; /* position of minimum potential */
        }
        else
        {
            /* we only repeat for those close to the unbinding threshold */
#pragma omp parallel for
            for (int i = 0; i < len; i++)
            {
                int p = ud[i].index;

                if (P[p].v.DM_BindingEnergy >= weakly_bound_limit)
                {
                    double pot = subfind_loctree_treeevaluate_potential(p);
                    /* note: add self-energy */
#ifndef ADAPTGRAVSOFT
                    P[p].u.DM_Potential = pot + P[p].Mass / All.SofteningTable[P[p].Type];
#else // ADAPTGRAVSOFT
                    P[p].u.DM_Potential = pot + P[p].Mass * 2.8 / P[p].AGS_Hsml;
#endif // ADAPTGRAVSOFT
                    P[p].u.DM_Potential *= All.GravityConstantInternal / atime;

                    if (All.TotN_gas > 0 && (FOF_PRIMARY_LINK_TYPES & 1) == 0 && (FOF_SECONDARY_LINK_TYPES & 1) == 0 &&
                        All.OmegaBaryon > 0)
                        P[p].u.DM_Potential *= All.Omega0 / (All.Omega0 - All.OmegaBaryon);
                }
            }
        }

        t1_sub = second();
        diff2 += (t1_sub - t0_sub);
        t0_sub = second();

        /* let's get bulk velocity and the center-of-mass */

        v[0] = v[1] = v[2] = 0;
        s[0] = s[1] = s[2] = 0;

        TotMass = 0;
#ifdef _OPENMP
        double s_tmp[3], v_tmp[3];
        for (int j = 0; j < 3; j++)
        {
            s_tmp[j] = 0;
            v_tmp[j] = 0;
        }
#pragma omp parallel shared(s_tmp, v_tmp) firstprivate(s, v) reduction(+ : TotMass)
        {
#pragma omp for
#endif // _OPENMP
            for (int i = 0; i < len; i++)
            {
                int p = ud[i].index;

                for (int j = 0; j < 3; j++)
                {
#ifdef PERIODIC
                    double ddxx = NEAREST(P[p].Pos[j] - pos[j]);
#else // PERIODIC
                double ddxx = P[p].Pos[j] - pos[j];
#endif // PERIODIC
                    s[j] += P[p].Mass * ddxx;
                    v[j] += P[p].Mass * P[p].Vel[j];
                }
                TotMass += P[p].Mass;
            }
#ifdef _OPENMP
            for (int j = 0; j < 3; j++)
            {
#pragma omp critical
                {
                    s_tmp[j] += s[j];
                    v_tmp[j] += v[j];
                }
            }
        }
        for (int j = 0; j < 3; j++)
        {
            s[j] = s_tmp[j];
            v[j] = v_tmp[j];
        }
#endif // _OPENMP

        for (int j = 0; j < 3; j++)
        {
            v[j] /= TotMass;
            s[j] /= TotMass; /* center-of-mass */

            s[j] += pos[j];

#ifdef PERIODIC
            while (s[j] < 0)
            {
                s[j] += boxsize;
            }
            while (s[j] >= boxsize)
            {
                s[j] -= boxsize;
            }
#endif // PERIODIC
        }

        t1_sub = second();
        diff3 += (t1_sub - t0_sub);
        t0_sub = second();

#ifdef _OPENMP
#pragma omp parallel for private(dv, dx)
#endif // _OPENMP
        for (int i = 0; i < len; i++)
        {
            int p = ud[i].index;

            for (int j = 0; j < 3; j++)
            {
                dv[j] = vel_to_phys * (P[p].Vel[j] - v[j]);
#ifdef PERIODIC
                dx[j] = atime * NEAREST(P[p].Pos[j] - s[j]);
#else // PERIODIC
                dx[j] = atime * (P[p].Pos[j] - s[j]);
#endif // PERIODIC
                dv[j] += H_of_a * dx[j];
            }

            P[p].v.DM_BindingEnergy = P[p].u.DM_Potential + 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);

#ifdef DENSITY_SPLIT_BY_TYPE
            if (P[p].Type == 0)
            {
                P[p].v.DM_BindingEnergy += P[p].w.int_energy;
            }
#endif // DENSITY_SPLIT_BY_TYPE
            bnd_energy[i] = P[p].v.DM_BindingEnergy;
        }

        t1_sub = second();
        diff4 += (t1_sub - t0_sub);
        t0_sub = second();

        STD_SORT(bnd_energy, len, sizeof(double), subfind_compare_binding_energy); /* largest comes first! */

        energy_limit = bnd_energy[(int)(0.25 * len)];

        t1_sub = second();
        diff5 += (t1_sub - t0_sub);
        t0_sub = second();

        unbound = 0;
        int ii;
        for (ii = 0; ii < len - 1; ii++)
        {
            if (bnd_energy[ii] > 0)
            {
                unbound++;
            }
            else
            {
                unbound--;
            }

            if (unbound <= 0)
            {
                break;
            }
        }
        weakly_bound_limit = bnd_energy[ii];

        t1_sub = second();
        diff6 += (t1_sub - t0_sub);
        t0_sub = second();

        /* now omit unbound particles,  but at most 1/4 of the original size */

        unbound = 0;
        *len_non_gas = 0;
        for (int i = 0; i < len; i++)
        {
            int p = ud[i].index;
            if (P[p].v.DM_BindingEnergy > 0 && P[p].v.DM_BindingEnergy > energy_limit)
            {
                unbound++;
                ud[i] = ud[len - 1];
                i--;
                len--;
            }
            else if (P[p].Type != 0)
                (*len_non_gas)++;
        }

        t1_sub = second();
        diff7 += (t1_sub - t0_sub);
        t0_sub = second();

        if (len < All.DesLinkNgb)
            break;

        if (phaseflag == 0)
        {
            if (unbound > 0)
                phaseflag = 1;
        }
        else
        {
            if (unbound == 0)
            {
                phaseflag = 0; /* this will make us repeat everything once more for all particles */
                unbound = 1;
            }
        }
    } while (unbound > 0);

    myfree(bnd_energy);

    if (len_orig > 10000)
    {
        VERBOSE(2, "EXTRA TIMER SUBFIND:       different unbinding steps (GrLen=%d/%d) took (%g,%g,%g,%g,%g,%g,%g)sec",
                len, len_orig, diff1, diff2, diff3, diff4, diff5, diff6, diff7);
    }

    return (len);
}

#ifdef LT_ADD_GAL_TO_SUB
/**
 * @brief Loads the CB07 luminosity tables and filters effective wavelengths for different metallicities and observer
 * frames.
 *
 * This function reads the CB07 tables for luminosities and the effective wavelengths of filters.
 * It processes data for multiple metallicities and, if the `INTERP_OBSERVER_FRAME` macro is defined, it also
 * interpolates data for observer frames. The function handles file reading and data initialization required for
 * subsequent luminosity calculations.
 *
 * This function performs the following steps:
 * - Reads filter effective wavelengths from a file.
 * - Reads and processes luminosity tables for multiple metallicities.
 * - Optionally handles interpolation for observer frames if the `INTERP_OBSERVER_FRAME` macro is defined.
 * - Handles error cases where files cannot be opened, ensuring proper termination with relevant messages.
 *
 * @param IMFt The initial mass function type.
 * @param num A numerical identifier used when reading specific observer frame files.
 */
void load_CB07_table(int IMFt, int num)
{
    /* This routine reads the CB07 tables for luminosities */

    FILE *TempiFile, *CB07File, *Filters_Effective_WavelenghtFile;
    int imag = 0, imet = 0;
    char buff[200];
#ifdef INTERP_OBSERVER_FRAME
    float *tempCB07obs;
    int j;
#endif // INTERP_OBSERVER_FRAME

    sprintf(buff, "%s/Filters_Effective_Wavelenght.txt", All.BC_SED_Path);
    if (!(Filters_Effective_WavelenghtFile = fopen(buff, "rb")))
    {
        ENDRUN("Could not open file %s !", buff);
    }

    for (int i = 0; i < LT_ADD_GAL_TO_SUB; i++)
    {
        fscanf(Filters_Effective_WavelenghtFile, "%f", &Filters_Effective_Wavelenght[i]);
        bc_use_index[i] = i;
    }

    fclose(Filters_Effective_WavelenghtFile);

    sprintf(buff, "%s/tempi", All.BC_SED_Path);
    if (!(TempiFile = fopen(buff, "rb")))
    {
        ENDRUN("Could not open file %s !", buff);
    }

    for (imet = 0; imet <= 6; imet++)
    {
        sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_ssp.multi_mag_vega", All.BC_SED_Path, SFs[0].IMFname, 2 + imet);
        if (!(CB07File = fopen(buff, "rb")))
        {
            ENDRUN("Could not open file %s !", buff);
        }

        for (int i = 0; i < 219; i++)
        {
            if (imet == 0)
            {
                fscanf(TempiFile, "%f", &tempiAS[i]);
                tempiAS[i] = pow(10, tempiAS[i]) / 1.e9;
            }
            /* now tempiAS should be in Gyrs */

            for (imag = 0; imag < LT_ADD_GAL_TO_SUB; ++imag)
            {
                fscanf(CB07File, "%f", &CB07[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)]);
            }
        }
        fclose(CB07File);

#ifdef OBSERVER_FRAME
#ifdef INTERP_OBSERVER_FRAME
        if (!flag_readCB07obs_redshifts)
        {
            for (int j = 0; j < INTERP_OBSERVER_FRAME; j++)
            {
                sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_ssp.multi_mag_vega_%03d", All.BC_SED_Path, SFs[0].IMFname,
                        2 + imet, j);
                if (!(CB07File = fopen(buff, "rb")))
                {
                    ENDRUN("Could not open file %s !", buff);
                }
                fscanf(CB07File, "%s %s %f", (char *)&buff, (char *)&buff, &CB07obs_redshifts[j]);
                fclose(CB07File);
            }
            flag_readCB07obs_redshifts = 1;
        }

        for (j = 0; j < INTERP_OBSERVER_FRAME - 1; j++)
        {
            if ((1.0 / All.Time - 1) > CB07obs_redshifts[j] && (1.0 / All.Time - 1) <= CB07obs_redshifts[j + 1])
            {
                break;
            }
        }

        /// read in the lower redshift
        sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_ssp.multi_mag_vega_%03d", All.BC_SED_Path, SFs[0].IMFname, 2 + imet,
                j);

        if (!(CB07File = fopen(buff, "rb")))
        {
            ENDRUN("Could not open file %s !", buff);
        }

        for (int i = 0; i < 219; i++)
        {
            for (imag = 0; imag < LT_ADD_GAL_TO_SUB; ++imag)
            {
                fscanf(CB07File, "%f", &CB07obs[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)]);
            }
        }

        fclose(CB07File);

        /// read in the higher redshift
        tempCB07obs = (float *)mymalloc("tempCB07obs", sizeof(float) * 7 * 219 * LT_ADD_GAL_TO_SUB);

        sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_ssp.multi_mag_vega_%03d", All.BC_SED_Path, SFs[0].IMFname, 2 + imet,
                j + 1);
        if (!(CB07File = fopen(buff, "rb")))
        {
            ENDRUN("Could not open file %s !", buff);
        }

        for (int i = 0; i < 219; i++)
        {
            for (imag = 0; imag < LT_ADD_GAL_TO_SUB; ++imag)
            {
                fscanf(CB07File, "%f", &tempCB07obs[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)]);
            }
        }

        fclose(CB07File);

        // interpolate from the two neighboring redshifts
        for (int i = 0; i < 219; i++)
        {
            for (imag = 0; imag < LT_ADD_GAL_TO_SUB; ++imag)
            {
                CB07obs[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)] +=
                    (tempCB07obs[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)] -
                     CB07obs[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)]) *
                    ((1.0 / All.Time - 1) - CB07obs_redshifts[j]) / (CB07obs_redshifts[j + 1] - CB07obs_redshifts[j]);
            }
        }

        myfree(tempCB07obs);
#else // INTERP_OBSERVER_FRAME

        sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_ssp.multi_mag_vega_%03d", All.BC_SED_Path, SFs[0].IMFname, 2 + imet,
                num);
        if (!(CB07File = fopen(buff, "rb")))
        {
            ENDRUN("Could not open file %s !", buff);
        }

        for (int i = 0; i < 219; i++)
        {
            for (imag = 0; imag < LT_ADD_GAL_TO_SUB; ++imag)
            {
                fscanf(CB07File, "%f", &CB07obs[imag + (i * LT_ADD_GAL_TO_SUB) + (imet * 219 * LT_ADD_GAL_TO_SUB)]);
            }
        }

        fclose(CB07File);
#endif // INTERP_OBSERVER_FRAME
#endif // OBSERVER_FRAME
    }
    fclose(TempiFile);
}

#ifndef OLD_FILTERS
/*These functions will be useful within load_CB07_table_new*/

/**
 * @brief Integrand function for computing the luminosity distance.
 *
 * This function defines the integrand used in the calculation of the luminosity distance as a function of scale factor
 * `a`.
 *
 * @param a The scale factor at which the luminosity distance is being evaluated.
 * @param param Pointer to a structure containing cosmological parameters (e.g., OmegaLambda, Omega0).
 * @return The value of the integrand at the given scale factor `a`.
 */
double integrand_lumdist(double a, void *param)
{
    double OmL = All.OmegaLambda, OmM = All.Omega0;
    return 1. / (a * a * sqrt(OmL + OmM / (a * a * a) + (1. - OmL - OmM) / (a * a)));
}

/**
 * @brief Comoving distance function for flat, closed, or open universes.
 *
 * This function computes the comoving distance function for different cosmological models based on the curvature
 * parameter Omega_K. It is used in the calculation of comoving distances in cosmological simulations or analyses.
 *
 * The comoving distance function is defined differently depending on the curvature of the universe:
 * - For a flat universe (Omega_K = 0), the function returns the argument `x`.
 * - For a closed universe (positive Omega_K), the function returns the hyperbolic sine of the argument `x`.
 * - For an open universe (negative Omega_K), the function returns the sine of the argument `x`.
 *
 * @param x The argument of the comoving distance function.
 * @return The value of the comoving distance function for the given argument `x`.
 */
double f_K(double x)
{
    double OmK = 1. - All.Omega0 - All.OmegaLambda;

    if (OmK > 0)
    {
        VERBOSE_ALL(2, "f_K with positive Omega_K");
        return sinh(x);
    }
    else if (OmK < 0)
    {
        VERBOSE_ALL(2, "f_K with negative Omega_K");
        return sin(x);
    }
    else
    {
        VERBOSE_ALL(2, "Omega_K =0");
        return x;
    }
}

/**
 * @brief Loads CB07 tables for luminosities and filters.
 *
 * This function reads CB07 tables for luminosities and filter information.
 * It determines the appropriate tables based on redshift and cosmological parameters.
 * The function also corrects the luminosity distance for the given redshift and cosmology.
 *
 * The function performs the following steps:
 * 1. Reads age tables and luminosity distance information.
 * 2. Determines the appropriate redshift table based on the current redshift.
 * 3. If no redshift table is found for the current redshift, it selects the table with the largest redshift.
 * 4. Reads filter index information and assigns filter read indices.
 * 5. Reads effective wavelength information for filters.
 * 6. Reads rest-frame filter tables and corrects for luminosity distance.
 *
 * @param IMFt The IMF type.
 */
void load_CB07_table_new(int IMFt)
{
    /* This routine reads the CB07 tables for luminosities */
    FILE *TempiFile, *CB07File, *LumDistFile, *Filters_Effective_WavelenghtFile, *FilterIndexFile;
    int NRedhsiftFilters;
    int read_this = -1; // setting to -1 to check if we found a redshift table
    double z_low, z_high, z_this = (1.0 / All.Time - 1);
    char buff[200];

    VERBOSE(2, "    reading age tables ...");

    sprintf(buff, "%s/tempi", All.BC_SED_Path);
    if (!(TempiFile = fopen(buff, "rb")))
    {
        ENDRUN("Could not open file %s !", buff);
    }
    for (int i = 0; i < LOG_AGE_BINS; i++)
    {
        fscanf(TempiFile, "%f", &tempiAS[i]);
        if (i == 0 || i == LOG_AGE_BINS - 1)
        {
            VERBOSE_ALL(2, "     logage[%d]=%g", i, tempiAS[i]);
        }

        tempiAS[i] = pow(10, tempiAS[i]) / 1.e9;

        if (i == 0 || i == LOG_AGE_BINS - 1)
        {
            VERBOSE_ALL(2, " -> age[%d]=%g", i, tempiAS[i]);
        }
    }
    /* now tempiAS should be in Gyrs */
    fclose(TempiFile);

    VERBOSE(2, "    reading luminosity tables ...");

    sprintf(buff, "%s/luminosity_distance", All.BC_SED_Path);
    if (!(LumDistFile = fopen(buff, "rb")))
    {
        ENDRUN("Could not open file %s' !", buff);
    }
    fscanf(LumDistFile, "# Cosmology: H0 = %lf km / (Mpc s), Omega0 = %lf, Omega_Lambda = %lf ", &H0Filters, &OMFilters,
           &OLFiletrs);
    VERBOSE_ALL(2, "       found cosmology (%g,%g,%g) ...", H0Filters, OMFilters, OLFiletrs);
    fscanf(LumDistFile, "%*[^\n]\n");
    VERBOSE_ALL(2, "       skipped line ...");
    fscanf(LumDistFile, "%d", &NRedhsiftFilters);
    VERBOSE_ALL(2, "       found %d tables ...", NRedhsiftFilters);

    float *CB07redshifts, *CB07lumdists;
    CB07redshifts = (float *)mymalloc("CB07redshifts", sizeof(float) * NRedhsiftFilters);
    CB07lumdists = (float *)mymalloc("CB07lumdists", sizeof(float) * NRedhsiftFilters);

    for (int i = 0; i < NRedhsiftFilters; i++)
    {
        fscanf(LumDistFile, "%f %f", &CB07redshifts[i], &CB07lumdists[i]);
    }
    fclose(LumDistFile);

    VERBOSE_ALL(2, "    deciding for filter files ...");

    for (int j = 0; j < NRedhsiftFilters - 1; j++)
    {
        if (z_this >= CB07redshifts[j] && z_this < CB07redshifts[j + 1])
        {
            read_this = j;
            z_low = CB07redshifts[j];
            z_high = CB07redshifts[j + 1];
            CB07_used_lumdist =
                ((z_high - z_this) * CB07lumdists[j] + (z_this - z_low) * CB07lumdists[j + 1]) / (z_high - z_low);
            break;
        }
    }

    // check if we found a redshift table, if not take the one with the second largest redshift and set redshift
    // correction to largest value availiable
    if (read_this == -1)
    {
        VERBOSE_ALL(0, "Could not find redshift table for z=%g !", z_this);

        int index = 0;
        double maxRedshift = CB07redshifts[0]; // get lowest redshift

        // iterate over all redshift tables to find the one with the largest redshift
        for (int j = 1; j < NRedhsiftFilters; j++)
        {
            // Update maxRedshift if a larger value is encountered
            if (CB07redshifts[j] > maxRedshift)
            {
                maxRedshift = CB07redshifts[j];
                index = j - 1; // index needs to be the lower table -> therefore -1
            }
        }

        z_low = CB07redshifts[index];
        z_high = CB07redshifts[index + 1];

        VERBOSE_ALL(0, "Using largest redshift in tables of z=%g", z_high);
        z_this = z_high;

        CB07_used_lumdist =
            ((z_high - z_this) * CB07lumdists[index] + (z_this - z_low) * CB07lumdists[index + 1]) / (z_high - z_low);

        // set the redshift table to the one with the second largest redshift
        read_this = index;
    }
    myfree(CB07lumdists);
    myfree(CB07redshifts);

    VERBOSE_ALL(2, "    ... using %d,%d [%g,%g,%g] ...", read_this, read_this + 1, z_low, z_this, z_high);
    VERBOSE_ALL(2, "    LumDist(%f) = %f", z_this, CB07_used_lumdist);

    VERBOSE_ALL(2, "    reading filter index table ...");

    sprintf(buff, "./filters.dat");
    if (!(FilterIndexFile = fopen(buff, "rb")))
    {
        ENDRUN("Could not open file %s' !", buff);
    }
    for (int i = 0; i < LT_ADD_GAL_TO_SUB; i++)
    {
        fscanf(FilterIndexFile, "%d %c%c %*s", &bc_use_index[i], &bc_name[i][0], &bc_name[i][1]);
        VERBOSE_ALL(2, "       filter %d: <%c%c> ...", bc_use_index[i], bc_name[i][0], bc_name[i][1]);
    }
    fclose(FilterIndexFile);

    VERBOSE_ALL(2, "    assining filter read index (out of %d)...", CB07_Tot_N_Filers);

    int use_filters[LT_ADD_GAL_TO_SUB];
    int jmin = 0;
    for (int i = 0; i < LT_ADD_GAL_TO_SUB; i++)
    {
        int found = -1;
        for (int j = jmin; j < CB07_Tot_N_Filers; j++)
        {
            found = -1;
            for (int k = 0; k < LT_ADD_GAL_TO_SUB; k++)
            {
                if (bc_use_index[k] == j)
                {
                    found = k;
                    break;
                }
            }
            if (found >= 0)
            {
                jmin = j + 1;
                break;
            }
        }
        if (found >= 0)
        {
            use_filters[i] = found;
            VERBOSE_ALL(2, "       filter %d: -> %d", i, use_filters[i]);
        }
    }

    VERBOSE_ALL(2, "    reading Effective Wavelength file ...");
    sprintf(buff, "%s/Filters_Effective_Wavelenght.txt", All.BC_SED_Path);
    if (!(Filters_Effective_WavelenghtFile = fopen(buff, "rb")))
    {
        ENDRUN("Could not open file %s' !", buff);
    }

    float *filters_effective_wavelength;
    filters_effective_wavelength = (float *)mymalloc("WavelengthFilter", sizeof(float) * NRedhsiftFilters);

    for (int i = 0; i < CB07_Tot_N_Filers; i++)
    {
        fscanf(Filters_Effective_WavelenghtFile, "%f", &filters_effective_wavelength[i]);
    }

    fclose(Filters_Effective_WavelenghtFile);

    for (int k = 0; k < LT_ADD_GAL_TO_SUB; k++)
    {
        Filters_Effective_Wavelenght[k] = filters_effective_wavelength[bc_use_index[k]];
        VERBOSE_ALL(2, "       filter effective wavelength %d: <%c%c> = %f ", bc_use_index[k], bc_name[k][0],
                    bc_name[k][1], Filters_Effective_Wavelenght[k]);
    }
    myfree(filters_effective_wavelength);

    VERBOSE_ALL(2, "    reading restframe filter table ...");

    for (int imet = 0; imet < N_BINS_METAL; imet++)
    {
        sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_%s_ssp.multi_mag_AB_000", All.BC_SED_Path, SFs[0].IMFname, 2 + imet,
                SFs[0].IMFname);

        VERBOSE_ALL(2, "      %s ...", buff);

        if (!(CB07File = fopen(buff, "rb")))
        {
            ENDRUN("Could not open file %s' !", buff);
        }

        fscanf(CB07File, "%*[^\n]\n"); // skipped line

        for (int i = 0; i < LOG_AGE_BINS; i++)
        {
            int imag = 0;

            for (int imag_all = 0; imag_all < CB07_Tot_N_Filers; ++imag_all)
            {
                float dummy_read;
                fscanf(CB07File, "%f", &dummy_read);
                if (imag < LT_ADD_GAL_TO_SUB)
                    if (bc_use_index[use_filters[imag]] == imag_all)
                    {
                        CB07[use_filters[imag] + (i * LT_ADD_GAL_TO_SUB) + (imet * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB)] =
                            dummy_read;
                        imag++;
                    }
            }
        }
        fclose(CB07File);
    }

    /*for(int imag = 0; imag < LT_ADD_GAL_TO_SUB; imag++)
       {
       printf("filte %d imet=%d: ",imag, 0);
       for(int i = 0; i < LOG_AGE_BINS; i++)
       printf("%g ",CB07[imag + (i * LT_ADD_GAL_TO_SUB) + (0 * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB)]);
       printf("\n");
       } */

#ifdef OBSERVER_FRAME
    double f_K(double x);

    double result, abserr;

    auto lumdist_f = [](double a) { return integrand_lumdist(a, nullptr); };

    /* Computing the luminosity distance at this redshift, for this cosmology */
    printf("In observer-frame computing the adequate luminosity distance \n");

    result =
        boost::math::quadrature::gauss_kronrod<double, 41>::integrate(lumdist_f, All.Time, 1.0, 15, 1.0e-8, &abserr);
    double integral = result;
    double LumDist_computed =
        1.0 / All.Time * f_K(C_LIGHT / 1e2 / (All.HubbleParam * 100) * (All.UnitLength_in_cm) * integral);
    double correction_luminosity_distance;
    if (z_this == 0)
    {
        correction_luminosity_distance = 0;
    }
    else
    {
        correction_luminosity_distance = 2.5 * log10(pow(CB07_used_lumdist / LumDist_computed, 2));
    }

    if (LumDist_computed < 0)
    {
        LumDist_computed = 0;
        correction_luminosity_distance = 0;
        read_this = 0;
    }

    VERBOSE_ALL(2, "Check for Luminosity distance: %g", LumDist_computed);
    VERBOSE_ALL(2, "    reading filter tables 1 ...");

    for (int imet = 0; imet < N_BINS_METAL; imet++)
    {
        sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_%s_ssp.multi_mag_AB_%03d", All.BC_SED_Path, SFs[0].IMFname, 2 + imet,
                SFs[0].IMFname, read_this);
        printf("      %s ...\n", buff);
        if (!(CB07File = fopen(buff, "rb")))
        {
            ENDRUN("Could not open file %s' !", buff);
        }

        fscanf(CB07File, "%*[^\n]\n"); // skipped line

        for (int i = 0; i < LOG_AGE_BINS; i++)
        {
            int imag = 0;

            for (int imag_all = 0; imag_all < CB07_Tot_N_Filers; ++imag_all)
            {
                float dummy_read;
                fscanf(CB07File, "%f", &dummy_read);
                if (imag < LT_ADD_GAL_TO_SUB)
                    if (bc_use_index[use_filters[imag]] == imag_all)
                    {
                        CB07obs[use_filters[imag] + (i * LT_ADD_GAL_TO_SUB) +
                                (imet * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB)] = dummy_read;
                        imag++;
                    }
            }
        }
        fclose(CB07File);
    }

    printf("    reading filter tables 2 ...\n");

    for (int imet = 0; imet < N_BINS_METAL; imet++)
    {
        sprintf(buff, "%s/%s/cb2007_hr_stelib_m%d2_%s_ssp.multi_mag_AB_%03d", All.BC_SED_Path, SFs[0].IMFname, 2 + imet,
                SFs[0].IMFname, read_this + 1);
        printf("      %s ...\n", buff);
        if (!(CB07File = fopen(buff, "rb")))
        {
            ENDRUN("Could not open file %s' !\n", buff);
        }

        fscanf(CB07File, "%*[^\n]\n"); // skipped line

        for (int i = 0; i < LOG_AGE_BINS; i++)
        {
            int imag = 0;
            // Loop through all filters
            for (int imag_all = 0; imag_all < CB07_Tot_N_Filers; ++imag_all)
            {
                // Read a float value from CB07File
                float dummy_high;
                fscanf(CB07File, "%f", &dummy_high);
                if (imag < LT_ADD_GAL_TO_SUB)
                {
                    if (bc_use_index[use_filters[imag]] == imag_all)
                    {
                        // Calculate the corrected value for CB07obs
                        double dummy_low = CB07obs[use_filters[imag] + (i * LT_ADD_GAL_TO_SUB) +
                                                   (imet * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB)];
                        CB07obs[use_filters[imag] + (i * LT_ADD_GAL_TO_SUB) +
                                (imet * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB)] =
                            ((z_high - z_this) * dummy_low + (z_this - z_low) * dummy_high) / (z_high - z_low) +
                            correction_luminosity_distance; // correction due to the fact that the magnitudes have a
                                                            // luminosity distance which is cosmology-dependent
                        imag++;
                    }
                }
            }
        }
        fclose(CB07File);
    }

    /*
       for (int imet = 0; imet < 12; imet++)
       {
       printf("filter %d imet = %d: \n", imet,0);
       printf("imag, %d bc_use_index[use_filters[imag]] %d bc_use_index[imag] %d use_filters[imag] %d
       \n",imet,bc_use_index[use_filters[imet]],bc_use_index[imet], use_filters[imet]); for (int i=0; i< LOG_AGE_BINS;
       i++)

       printf("%g ", CB07obs[imet+(i * LT_ADD_GAL_TO_SUB) + (0*LOG_AGE_BINS*LT_ADD_GAL_TO_SUB)]);
       printf("\n");
       }
     */
#endif // OBSERVER_FRAME
}
#endif // OLD_FILTERS

/**
 * @brief Adds luminosity information to a particle.
 *
 * This function calculates and adds luminosity information to a particle based on its age and metallicity.
 * It uses the CB07 tables for luminosities to compute the luminosity in each filter. The function also accounts for the
 * observer frame if enabled.
 *
 * @param PART_index The index of the particle.
 * @param FLAG_AGE Pointer to the flag indicating whether the particle's age is within the valid range.
 * @param lum Pointer to store the calculated luminosity.
 * @param lum_obs Pointer to store the observed luminosity if in the observer frame.
 */
void add_luminosity(int PART_index, int *FLAG_AGE, double *lum, double *lum_obs)
{
    float metCB07[N_BINS_METAL + 1];

    int ind_T, ind_Z;
    float rappt1, rappt, rappz1, rappz, Tstar, Tdiff, Zstar;
    float Lumz1t, Lumzt1, Lumz1t1, Tnow;
#ifdef OBSERVER_FRAME
    float Lumzto, Lumz1to, Lumzt1o, Lumz1t1o;
#endif // OBSERVER_FRAME

    metCB07[0] = 0.0001 / 0.02; // Normalized to Z_sun = 0.02
    metCB07[1] = 0.0004 / 0.02;
    metCB07[2] = 0.004 / 0.02;
    metCB07[3] = 0.008 / 0.02;
    metCB07[4] = 0.02 / 0.02;
    metCB07[5] = 0.0500 / 0.02;
#ifndef OLD_FILTERS
    metCB07[6] = 1.e9;
#else // OLD_FILTERS
    metCB07[6] = 0.1000 / 0.02;
    metCB07[7] = 1.e9;
#endif // OLD_FILTERS

    Tnow = get_age(All.Time);

    Tstar = get_age(MPP(PART_index).StellarAge);
    Tdiff = Tstar - Tnow;
    if (Tdiff > 1.e-2)
    {
        *FLAG_AGE = 0;
    }
    else
    {
        *FLAG_AGE = 1;
    }

    Zstar = get_metallicity_subfind(PART_index) / 0.02;

    int j = 0;
    while (Tdiff >= tempiAS[j])
    {
        j++;
    }

    ind_T = j;
    if (j > 0)
    {
        if (ind_T == LOG_AGE_BINS)
        {
            rappt = 1.;
            rappt1 = 0.;
            ind_T--;
        }
        else
        {
            rappt1 = (tempiAS[ind_T] - Tdiff) / (tempiAS[ind_T] - tempiAS[ind_T - 1]);
            rappt = (Tdiff - tempiAS[ind_T - 1]) / (tempiAS[ind_T] - tempiAS[ind_T - 1]);
        }
    }
    else
    {
        rappt = 0.;
        rappt1 = 1.;
        ind_T++;
    }
    // printf("check indT %d %g %g ", ind_T, rappt,rappt1);
    int k = 0;
    while (Zstar >= metCB07[k])
    {
        k++;
    }

    ind_Z = k;
    if (k > 0)
    {
        if (ind_Z == N_BINS_METAL)
        {
            rappz = 1.;
            rappz1 = 0.;
            ind_Z--;
        }
        else
        {
            rappz1 = (metCB07[ind_Z] - Zstar) / (metCB07[ind_Z] - metCB07[ind_Z - 1]);
            rappz = (Zstar - metCB07[ind_Z - 1]) / (metCB07[ind_Z] - metCB07[ind_Z - 1]);
        }
    }
    else
    {
        rappz = 0.;
        rappz1 = 1.;
        ind_Z++;
    }

    // printf("check indZ %d %g %g\n", ind_Z,rappz,rappz1);

    for (int i = 0; i < LT_ADD_GAL_TO_SUB; i++)
    {
        if (ind_Z == N_BINS_METAL)
        {
            rappz1 = 0.;
            rappz = 1.;
            Lumz1t = 0.;
            Lumz1t1 = 0.;
#ifdef OBSRVER_FRAME
            Lumz1to = 0.;
            Lumz1t1o = 0.;
#endif // OBSRVER_FRAME
        }
        else
        {
            Lumz1t = -CB07[(ind_Z)*LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T - 1) * LT_ADD_GAL_TO_SUB + i];
            Lumz1t1 = -CB07[(ind_Z)*LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T)*LT_ADD_GAL_TO_SUB + i];
#ifdef OBSERVER_FRAME
            Lumz1to = -CB07obs[(ind_Z)*LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T - 1) * LT_ADD_GAL_TO_SUB + i];
            Lumz1t1o = -CB07obs[(ind_Z)*LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T)*LT_ADD_GAL_TO_SUB + i];
#endif // OBSERVER_FRAME
        }
        // Lumzt = -CB07[(ind_Z - 1) * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T - 1) * LT_ADD_GAL_TO_SUB + i];
        Lumzt1 = -CB07[(ind_Z - 1) * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T)*LT_ADD_GAL_TO_SUB + i];
#ifdef OBSERVER_FRAME
        Lumzto = -CB07obs[(ind_Z - 1) * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T - 1) * LT_ADD_GAL_TO_SUB + i];
        Lumzt1o = -CB07obs[(ind_Z - 1) * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB + (ind_T)*LT_ADD_GAL_TO_SUB + i];
#endif // OBSERVER_FRAME
        lum[i] = ((pow(10, Lumzt1 / 2.5) * rappz1 + pow(10, Lumz1t / 2.5) * rappz) * rappt1 +
                  (pow(10, Lumzt1 / 2.5) * rappz1 + pow(10, Lumz1t1 / 2.5) * rappz) * rappt) *
                 MetP[P[PART_index].pt.MetID].iMass * All.UnitMass_in_g / SOLAR_MASS / All.HubbleParam;
#ifdef OBSERVER_FRAME
        lum_obs[i] = ((pow(10, Lumzto / 2.5) * rappz1 + pow(10, Lumz1to / 2.5) * rappz) * rappt1 +
                      (pow(10, Lumzt1o / 2.5) * rappz1 + pow(10, Lumz1t1o / 2.5) * rappz) * rappt) *
                     MetP[P[PART_index].pt.MetID].iMass * All.UnitMass_in_g / SOLAR_MASS / All.HubbleParam;
#endif // OBSERVER_FRAME
        /*
           if(i == 0)
           if(ThisTask == 0)
           printf("%d rappT %g, rappT1 %g, rappz %g, rappz1 %g \n", PART_index,rappt,rappt1,rappz,rappz1);
           printf("LUM_CALC(%d)  Tdiff=%f -> %d/%d  Z=%f -> %d  L0=%f\n", PART_index, Tdiff, ind_T, LOG_AGE_BINS, Zstar,
           ind_Z, lum[i]);
         */
    }
}
#endif // LT_ADD_GAL_TO_SUB

#define SUBFIND_OUTPUT_TRESH 10000000
/**
 * @brief Determines properties of a subhalo.
 *
 * This function calculates various properties of a subhalo based on its constituent particles.
 * These properties include the position, mass, velocity dispersion, most bound particle ID,
 * half-mass radius, maximum circular velocity, angular momentum of stellar component, and more.
 *
 * @param d Pointer to an array of unbind_data structures representing the particles in the subhalo.
 * @param num Number of particles in the subhalo.
 * @param collective Flag indicating whether the calculation should be collective or not.
 * @param this_group Pointer to a subgroup_properties structure to store the calculated properties.
 * @param sublevel Subhalo level.
 */
void subfind_determine_sub_halo_properties(struct unbind_data *d, int num, int collective,
                                           struct subgroup_properties *this_group, int sublevel)
{

    struct subgroup_properties loc_group;
    memset(&loc_group, 0, sizeof(struct subgroup_properties));

    int mincpu;

    double boxsize = All.BoxSize;
    double vel_to_phys, H_of_a, atime;
    if (All.ComovingIntegrationOn)
    {
        vel_to_phys = 1.0 / All.Time;
        H_of_a = hubble_function(All.Time);
        atime = All.Time;
    }
    else
    {
        H_of_a = 0;
        atime = vel_to_phys = 1;
    }

    MyFloat minpot = 1.0e30;
    int minindex = -1;
    int nstarloc = 0, ndmloc = 0, ngasloc = 0, nbhloc = 0;

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                   "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): count partiles in sub halo\n", ThisTask, collective);

    for (int i = 0; i < num; i++)
    {
        int p = d[i].index;
#ifdef KD_MAIN_HALO
        P[p].SubLevel = sublevel;
#endif // KD_MAIN_HALO
        if (P[p].u.DM_Potential < minpot || minindex == -1)
        {
            minpot = P[p].u.DM_Potential;
            minindex = p;
        }
        switch (P[p].Type)
        {
        case 0:
            ngasloc++;
            break;
        case 1:
        case 2:
        case 3:
            ndmloc++;
            break;
        case 4:
            nstarloc++;
            break;
        case 5:
            nbhloc++;
            break;
        }
    }

    int nstar = 0, ndm = 0, ngas = 0, nbh = 0;
    if (collective == 1)
    {
        MyFloat *potlist;
        potlist = (MyFloat *)mymalloc("potlist", NTask * sizeof(MyFloat));

        MPI_Allgather(&minpot, sizeof(MyFloat), MPI_BYTE, potlist, sizeof(MyFloat), MPI_BYTE, MYMPI_COMM_WORLD);
        MPI_Allreduce(&ngasloc, &ngas, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&ndmloc, &ndm, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&nstarloc, &nstar, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&nbhloc, &nbh, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

        mincpu = -1;
        minpot = 1.0e30;
        for (int i = 0; i < NTask; i++)
        {
            if (potlist[i] < minpot)
            {
                mincpu = i;
                minpot = potlist[i];
            }
        }
        ENDRUN_IF(mincpu < 0,
                  "Could not find position of minimum potential in collective mode for SubHalo ta=%d, num=%d", ThisTask,
                  num);
        if (ThisTask == mincpu)
        {
            for (int j = 0; j < 3; j++)
            {
                loc_group.Pos[j] = P[minindex].Pos[j];
            }
        }
        MPI_Bcast(&loc_group.Pos, 3, MPI_DOUBLE, mincpu, MYMPI_COMM_WORLD);

        /* Pos[] now holds the position of minimum potential */
        /* we take that as the center */
        for (int j = 0; j < 3; j++)
        {
            this_group->Pos[j] = loc_group.Pos[j];
        }

        myfree(potlist);
    }
    else // collecive != 1
    {
        ngas = ngasloc;
        ndm = ndmloc;
        nstar = nstarloc;
        nbh = nbhloc;
        ENDRUN_IF(minindex == -1,
                  "Could not find position of minimum potential in serial mode for SubHalo ta=%d, num=%d", ThisTask,
                  num);
        for (int j = 0; j < 3; j++)
        {
            this_group->Pos[j] = P[minindex].Pos[j];
        }
    }

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                   "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): done (%d,%d,%d,%d,%d=?%d)\n", ThisTask, collective,
                   ngas, ndm, nstar, nbh, num, ngas + ndm + nstar + nbh);

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                   "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): find minimum potential\n", ThisTask, collective);

    /* the ID of the most bound particle, we take the minimum binding energy */
    minindex = -1;
    minpot = 1.0e30;
    for (int i = 0; i < num; i++)
    {
        int p = d[i].index;
#ifndef DO_GAS_CLOUDS
        if (P[p].Type > 0)
#endif // DO_GAS_CLOUDS
            if (P[p].v.DM_BindingEnergy < minpot || minindex == -1)
            {
                minpot = P[p].v.DM_BindingEnergy;
                minindex = p;
            }
    }

    if (collective == 1)
    {
        MyFloat *potlist;
        potlist = (MyFloat *)mymalloc("potlist", NTask * sizeof(MyFloat));

        MPI_Allgather(&minpot, sizeof(MyFloat), MPI_BYTE, potlist, sizeof(MyFloat), MPI_BYTE, MYMPI_COMM_WORLD);

        mincpu = -1;
        minpot = 1.0e30;
        for (int i = 0; i < NTask; i++)
        {
            if (potlist[i] < minpot)
            {
                mincpu = i;
                minpot = potlist[i];
            }
        }

        if (ThisTask == mincpu)
        {
            ENDRUN_IF(minindex == -1,
                      "Could not find ID of particle with minimum binding energy in collective mode for SubHalo ta=%d, "
                      "num=%d",
                      ThisTask, num);
            loc_group.SubMostBoundID = P[minindex].ID;
        }

        MPI_Bcast(&loc_group.SubMostBoundID, sizeof(MyIDType), MPI_BYTE, mincpu, MYMPI_COMM_WORLD);

        myfree(potlist);

        this_group->SubMostBoundID = loc_group.SubMostBoundID;
    }
    else // collecive != 1
    {
        ENDRUN_IF(minindex == -1,
                  "Could not find ID of particle with minimum binding energy in serial mode for SubHalo ta=%d, num=%d",
                  ThisTask, num);
        this_group->SubMostBoundID = P[minindex].ID;
    }

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2, "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): done (%llu)\n",
                   ThisTask, collective, (unsigned long long)loc_group.SubMostBoundID);

    /* let's get bulk velocity and the center-of-mass */

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2, "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): find center of mass\n",
                   ThisTask, collective);

    for (int i = 0; i < num; i++)
    {
        int p = d[i].index;

        for (int j = 0; j < 3; j++)
        {
            double ddxx = NEAREST(P[p].Pos[j] - this_group->Pos[j]);

            loc_group.CM[j] += P[p].Mass * ddxx;
            loc_group.Vel[j] += P[p].Mass * P[p].Vel[j];
        }
        loc_group.Mass += P[p].Mass;
#ifdef BLACK_HOLES
        if (P[p].Type == 5)
        {
            loc_group.MassTab[P[p].Type] += BPP(p).BH_Mass;
        }
        else
#endif // BLACK_HOLES
            loc_group.MassTab[P[p].Type] += P[p].Mass;
    }

    if (collective == 1)
    {
        MPI_Allreduce(loc_group.CM, this_group->CM, 3, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(loc_group.Vel, this_group->Vel, 3, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&loc_group.Mass, &this_group->Mass, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(loc_group.MassTab, this_group->MassTab, 6, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
    }
    else // collecive != 1
    {
        for (int j = 0; j < 3; j++)
        {
            this_group->CM[j] = loc_group.CM[j];
            this_group->Vel[j] = loc_group.Vel[j];
        }
        this_group->Mass = loc_group.Mass;
        for (int j = 0; j < 6; j++)
        {
            this_group->MassTab[j] = loc_group.MassTab[j];
        }
    }

    for (int j = 0; j < 3; j++)
    {
        this_group->CM[j] = this_group->CM[j] / this_group->Mass + this_group->Pos[j]; /* center of mass */
#ifdef PERIODIC
        while (this_group->CM[j] < 0)
        {
            this_group->CM[j] += boxsize;
        }
        while (this_group->CM[j] >= boxsize)
        {
            this_group->CM[j] -= boxsize;
        }
#endif // PERIODIC
        this_group->Vel[j] *= vel_to_phys / this_group->Mass;
    }

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2, "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): done (%g,%g,%g)\n",
                   ThisTask, collective, this_group->CM[0], this_group->CM[1], this_group->CM[2]);

    /* Here we have to perform only on the dm particles for consistency */
    int num_use = num;
#ifdef DENSITY_SPLIT_BY_TYPE
    num_use = ndmloc;
#endif // DENSITY_SPLIT_BY_TYPE

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                   "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): find halfe dm mass radius (%d)\n", ThisTask,
                   collective, num_use);

    sort_r2list *loc_rr_list_star;
    if (nstarloc > 0)
    {
        loc_rr_list_star = (sort_r2list *)mymalloc("loc_rr_list_star", sizeof(sort_r2list) * nstarloc);
    }
    int i_star = 0;

    sort_r2list *loc_rr_list_gal;
    loc_rr_list_gal = (sort_r2list *)mymalloc("loc_rr_list_gal", sizeof(sort_r2list) * num);

    sort_r2list *loc_rr_list;
    if (num_use > 0)
    {
        loc_rr_list = (sort_r2list *)mymalloc("loc_rr_list", sizeof(sort_r2list) * num_use);
    }

    int i_use = 0;
    double massloc_used = 0;
    for (int i = 0; i < num; i++)
    {
        int p = d[i].index;

        double rr_tmp = 0, disp_tmp = 0;
        double dx[3], dv[3];

#ifdef DUSTATT
        double rr2_tmp = 0;
#endif // DUSTATT

        for (int j = 0; j < 3; j++)
        {
            dx[j] = atime * NEAREST(P[p].Pos[j] - this_group->CM[j]);
            dv[j] = vel_to_phys * (P[p].Vel[j] - this_group->Vel[j]);
            dv[j] += H_of_a * dx[j];

            disp_tmp += P[p].Mass * dv[j] * dv[j];
            /* for rotation curve computation, take minimum of potential as center */

            double ddxx = atime * NEAREST(P[p].Pos[j] - this_group->Pos[j]);
            rr_tmp += ddxx * ddxx;

#ifdef DUSTATT
            if (j < 2)
            {
                rr2_tmp += ddxx * ddxx;
            }
            if (j == 2)
            {
                loc_rr_list_gal[i].r2 = sqrt(rr2_tmp);
                loc_rr_list_gal[i].z = ddxx;
            }
#endif // DUSTATT
        }

        rr_tmp = sqrt(rr_tmp);
        if (P[p].Type == 4)
        {
            loc_rr_list_star[i_star].r = rr_tmp;
            i_star++;
        }

        loc_rr_list_gal[i].r = rr_tmp;

#ifdef DENSITY_SPLIT_BY_TYPE
        if (P[p].Type >= 1 && P[p].Type <= 3) /*-- only for dm part --*/
        {
#endif // DENSITY_SPLIT_BY_TYPE
            loc_group.SubVelDisp += disp_tmp;
            massloc_used += P[p].Mass;

            loc_group.Spin[0] += P[p].Mass * (dx[1] * dv[2] - dx[2] * dv[1]);
            loc_group.Spin[1] += P[p].Mass * (dx[2] * dv[0] - dx[0] * dv[2]);
            loc_group.Spin[2] += P[p].Mass * (dx[0] * dv[1] - dx[1] * dv[0]);

            loc_rr_list[i_use].mass = P[p].Mass;
            loc_rr_list[i_use].r = rr_tmp;

            i_use++;
#ifdef DENSITY_SPLIT_BY_TYPE
        }
#endif // DENSITY_SPLIT_BY_TYPE
    }

    ENDRUN_IF(i_use != num_use,
              "Something went wrong when calculating the spin of dark matter particles in SubFind i_use/num_use "
              "(%d/%d) ta=%d, num=%d",
              i_use, num_use, ThisTask, num);

    ENDRUN_IF(i_star != nstarloc,
              "Something went wrong when calculating the list of local star particles in SubFind i_star/nstarloc "
              "(%d/%d) ta=%d, num=%d",
              i_star, nstarloc, ThisTask, num);

    double mass_used;
    int numtot = 0;
    int offset = 0;
    int nhalf;

    if (collective == 1)
    {
        /* Sum up contributions from different tasks */
        MPI_Allreduce(loc_group.Spin, this_group->Spin, 3, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&loc_group.SubVelDisp, &this_group->SubVelDisp, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&massloc_used, &mass_used, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

        int *npart;
        npart = (int *)mymalloc("npart", NTask * sizeof(int));

        MPI_Allgather(&num_use, 1, MPI_INT, npart, 1, MPI_INT, MYMPI_COMM_WORLD);

        for (int i = 0; i < NTask; i++)
        {
            numtot += npart[i];
        }

        for (int i = 0; i < ThisTask; i++)
        {
            offset += npart[i];
        }

        /* find index and cpu where halve the partilces sit */
        nhalf = numtot / 2;
        mincpu = 0;

        while (nhalf >= npart[mincpu])
        {
            nhalf -= npart[mincpu];
            mincpu++;
        }

        myfree(npart);
    }
    else // collecive != 1
    {
        mass_used = massloc_used;
        numtot = num_use;
        this_group->SubVelDisp = loc_group.SubVelDisp;
        for (int j = 0; j < 3; j++)
        {
            this_group->Spin[j] = loc_group.Spin[j];
        }
        nhalf = numtot / 2;
    }

    if (mass_used > 0)
    {
        /* finalize computations for VelDisp and Spin */
        this_group->SubVelDisp = sqrt(this_group->SubVelDisp / (3 * mass_used)); /* convert to 1d velocity dispersion */
        for (int j = 0; j < 3; j++)
        {
            this_group->Spin[j] /= mass_used;
        }

        if (collective == 1)
        {
            /* globally sort distance list */
            parallel_sort(loc_rr_list, num_use, sizeof(sort_r2list), subfind_compare_dist_rotcurve);
        }
        else // collecive != 1
        {
            /* locally sort distance list */
            STD_SORT(loc_rr_list, num_use, sizeof(sort_r2list), subfind_compare_dist_rotcurve);
        }

        /*--- Here we still have to fix for individual masses,
                maybe we even want the total mass for this ?
                Note however that even within multi mass DM simulation
                the DM mass within a clean halo should be the same ... ------*/
        if (collective == 0 || ThisTask == mincpu)
        {
            this_group->SubHalfMass = loc_rr_list[nhalf].r;
        }

        if (collective == 1)
        {
            /* compute mass on different tasks */
            double *masslist;
            masslist = (double *)mymalloc("masslist", NTask * sizeof(double));

            MPI_Allgather(&massloc_used, 1, MPI_DOUBLE, masslist, 1, MPI_DOUBLE, MYMPI_COMM_WORLD);

            for (int i = 1; i < NTask; i++)
            {
                masslist[i] += masslist[i - 1];
            }
            /* add mass from other tasks */
            if (ThisTask > 0)
            {
                VERBOSE_IF_ALL(num_use == 0, -1,
                               "WARNING: ThisTask %d has num_use=%d => loc_rr_list was not allocated!", ThisTask,
                               num_use);
                if (num_use > 0)
                    loc_rr_list[0].mass += masslist[ThisTask - 1];
            }
            myfree(masslist);
        }

        /* compute cumulative mass */
        for (int i = 1; i < num_use; i++)
        {
            loc_rr_list[i].mass += loc_rr_list[i - 1].mass;
        }

        /*--- Note that here we might want to correct for the baryon fraction ?? ---*/
        for (int i = num_use - 1; i + offset > 5 && i >= 0; i--)
        {
            ENDRUN_IF(loc_rr_list[i].r <= 0,
                      "Something went wrong when calculating Vmax in SubFind i=%d, ta=%d, num=%d", i, ThisTask,
                      num_use);
            if (loc_rr_list[i].mass / loc_rr_list[i].r > loc_group.SubVmax)
            {
                loc_group.SubVmax = loc_rr_list[i].mass / loc_rr_list[i].r;
                loc_group.SubVmaxRad = loc_rr_list[i].r;
            }
        }
        loc_group.SubVmax = sqrt(All.GravityConstantInternal * loc_group.SubVmax);

        if (collective == 1)
        {
            MPI_Bcast(&this_group->SubHalfMass, 1, MPI_DOUBLE, mincpu, MYMPI_COMM_WORLD);
            MPI_Allreduce(&loc_group.SubVmax, &this_group->SubVmax, 1, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);

            if (loc_group.SubVmax < this_group->SubVmax)
            {
                loc_group.SubVmaxRad = 0;
            }

            MPI_Allreduce(&loc_group.SubVmaxRad, &this_group->SubVmax, 1, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);
        }
        else // collecive != 1
        {
            this_group->SubVmaxRad = loc_group.SubVmaxRad;
            this_group->SubVmax = loc_group.SubVmax;
        }
    }
    else
    {
        this_group->SubVelDisp = this_group->SubHalfMass = this_group->SubVmax = this_group->SubVmaxRad =
            this_group->Spin[0] = this_group->Spin[1] = this_group->Spin[2] = 0;
    }

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2, "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): done (%g,%g,%g)\n",
                   ThisTask, collective, this_group->SubHalfMass, this_group->SubVmax, this_group->SubVmaxRad);

    double r_gal;
    int nlum = (int)(0.83 * nstar);
    int mincpu2 = 0;
    nhalf = nlum / 2;

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                   "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): compute luminosity radius (%d,%d)\n", ThisTask,
                   collective, nstar, nlum);

    if (nstar > 0 && nlum > 20)
    {
        /* Compute radius in which 83% of stars are and define this as outer radius of the galaxy */

        if (collective == 1)
        {
            int *npart;
            npart = (int *)mymalloc("npart", NTask * sizeof(int));

            MPI_Allgather(&nstarloc, 1, MPI_INT, npart, 1, MPI_INT, MYMPI_COMM_WORLD);

            mincpu = 0;
            while (nlum >= npart[mincpu])
            {
                nlum -= npart[mincpu];
                mincpu++;
            }
            mincpu2 = 0;
            while (nhalf >= npart[mincpu2])
            {
                nhalf -= npart[mincpu2];
                mincpu2++;
            }

            myfree(npart);
        }

        if (collective == 1)
        {
            /* globally sort distance list */
            parallel_sort(loc_rr_list_star, nstarloc, sizeof(sort_r2list), subfind_compare_dist_rotcurve);
        }
        else // collecive != 1
        {
            /* locally sort distance list */
            STD_SORT(loc_rr_list_star, nstarloc, sizeof(sort_r2list), subfind_compare_dist_rotcurve);
        }

        /*--- Here we still have to fix for individual masses -----*/
        if (collective == 0 || ThisTask == mincpu)
        {
            r_gal = loc_rr_list_star[nlum].r;
        }
        if (collective == 0 || ThisTask == mincpu2)
        {
            this_group->SubStelHalfMass = loc_rr_list_star[nhalf].r;
        }

        if (collective == 1)
        {
            MPI_Bcast(&r_gal, 1, MPI_DOUBLE, mincpu, MYMPI_COMM_WORLD);
            MPI_Bcast(&this_group->SubStelHalfMass, 1, MPI_DOUBLE, mincpu, MYMPI_COMM_WORLD);
        }
    }
    else
    {
        r_gal = this_group->SubStelHalfMass = 0;
    }

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2, "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): done (%g,%g)\n",
                   ThisTask, collective, r_gal, this_group->SubStelHalfMass);

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                   "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): compute stellar properties (%d,%d)\n", ThisTask,
                   collective, nstar, nlum);

    if (nstar > 0 && nlum > 20)
    {
        double AngMomThis[3], MstarThis;
        AngMomThis[0] = AngMomThis[1] = AngMomThis[2] = MstarThis = 0;
        for (int i = 0; i < num; i++)
        {
            int p = d[i].index;
            if (P[p].Type == 4 && loc_rr_list_gal[i].r < 2 * this_group->SubStelHalfMass)
            {
                double dx[3], dv[3];
                for (int j = 0; j < 3; j++)
                {
                    dx[j] = atime * NEAREST(P[p].Pos[j] - this_group->Pos[j]);
                    dv[j] = vel_to_phys * (P[p].Vel[j] - this_group->Vel[j]);
                    dv[j] += H_of_a * dx[j];
                    dx[j] /= All.HubbleParam;
                }
                MstarThis += P[p].Mass;
                AngMomThis[0] += P[p].Mass * (dx[1] * dv[2] - dx[2] * dv[1]);
                AngMomThis[1] += P[p].Mass * (dx[2] * dv[0] - dx[0] * dv[2]);
                AngMomThis[2] += P[p].Mass * (dx[0] * dv[1] - dx[1] * dv[0]);
            }
        }
        double AngMom[3], Mstar;
        if (collective == 1)
        {
            /* Sum up contributions from different tasks */
            MPI_Allreduce(AngMomThis, AngMom, 3, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&MstarThis, &Mstar, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        }
        else
        {
            Mstar = MstarThis;
            for (int j = 0; j < 3; j++)
            {
                AngMom[j] = AngMomThis[j];
            }
        }
        this_group->SubStelAngMom = 0;
        for (int j = 0; j < 3; j++)
        {
            AngMom[j] /= Mstar;
            this_group->SubStelAngMom += AngMom[j] * AngMom[j];
        }
        this_group->SubStelAngMom = sqrt(this_group->SubStelAngMom);
        this_group->SubBval = log10(this_group->SubStelAngMom) - (2. / 3.) * log10(Mstar);
    }
    else
    {
        this_group->SubStelAngMom = this_group->SubBval = 0;
    }

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                   "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): compute stellar angular momentum (%g,%g)\n", ThisTask,
                   collective, this_group->SubStelAngMom, this_group->SubBval);

#ifdef LT_ADD_GAL_TO_SUB
    if (nstar > 0 && nlum > 20)
    {
#ifdef DUSTATT
        double Mcold_in_bin[DUSTATT], Zgas_in_bin[DUSTATT];
        for (int k = 0; k < DUSTATT; k++)
        {
            Mcold_in_bin[k] = Zgas_in_bin[k] = 0;
        }
#endif // DUSTATT
        int ngas_this = 0, ncoldgas_this = 0;

        for (int i = 0; i < num; i++)
        {
            int p = d[i].index;
            if (P[p].Type == 0)
            {
                ngas_this++;
                float Temperature, xclouds;
                loc_group.SubSFR += get_starformation_rate_subfind(p, &Temperature, &xclouds);

#ifdef LT_METAL_COOLING_WAL
                loc_group.SubHIMass += get_HImass_subfind(p, Redshift, DZ);
#endif // LT_METAL_COOLING_WAL
                double z_gas_this = get_metallicity_subfind(p);
                double mcold = xclouds * P[p].Mass;
                loc_group.SubMeanZGas += z_gas_this / 0.02;
                if (mcold > 0)
                {
                    ncoldgas_this++;
                    loc_group.SubMeanZColdGas += z_gas_this / 0.02;
                }
#ifdef DUSTATT
                double rr2_tmp = 0;
                /*- Only 2D projection ! */
                for (int j = 0; j < 2; j++)
                {
                    /* for luminosity, take minimum of potential as center */
                    double ddxx = atime * NEAREST(P[p].Pos[j] - this_group->Pos[j]);
                    rr2_tmp += ddxx * ddxx;
                }

                rr2_tmp = sqrt(rr2_tmp);

                int rad_bin = DUSTATT - 1;
                if (rr2_tmp <= r_gal)
                {
                    rad_bin = floor(rr2_tmp / r_gal * (DUSTATT - 2));
                }

                if (mcold > 0)
                {
                    Mcold_in_bin[rad_bin] += mcold;
                    Zgas_in_bin[rad_bin] += z_gas_this / 0.02 * mcold;
                }
#endif // DUSTATT
            }
        }

#ifdef DUSTATT
        double TauV_in_bin[DUSTATT];
        for (int k = 0; k < DUSTATT; k++)
        {
            TauV_in_bin[k] = 0;
        }
        /*- Note that impact parameters > r_gal are not needed, therefore last bin is irrelevant -*/
        if (collective == 1)
        {
            for (int k = 0; k < DUSTATT - 1; k++)
            {
                double total_val;
                MPI_Allreduce(&Mcold_in_bin[k], &total_val, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
                Mcold_in_bin[k] = total_val;

                if (Mcold_in_bin[k] > 0)
                {
                    MPI_Allreduce(&Zgas_in_bin[k], &total_val, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
                    Zgas_in_bin[k] = total_val;
                }
            }
        }

        double Unit_corrections_for_dust = All.UnitMass_in_g / All.HubbleParam /
                                           (1.4 * PROTONMASS * (All.UnitLength_in_cm / All.HubbleParam) *
                                            (All.UnitLength_in_cm / All.HubbleParam) * 2.1 * 1.e21);

        for (int k = 0; k < DUSTATT - 1; k++)
        {
            if (Mcold_in_bin[k] > 0)
            {
                Zgas_in_bin[k] /= Mcold_in_bin[k];
                TauV_in_bin[k] =
                    pow(Zgas_in_bin[k], 1.6) *
                    (Mcold_in_bin[k] / ((2 * k + 1) * 3.141516 * pow(r_gal / (DUSTATT - 2) * All.Time, 2))) *
                    Unit_corrections_for_dust;
            }
            else
            {
                TauV_in_bin[k] = 0;
            }

            this_group->DustProfile[k] = TauV_in_bin[k];
        }
#endif // DUSTATT

        int nstar_this = 0;

        for (int i = 0; i < num; i++)
        {
            int p = d[i].index;

            if (P[p].Type == 4)
            {
                nstar_this++;
                double rr_tmp = 0;
                double ddxx = 0;
                for (int j = 0; j < 3; j++)
                {
                    /* for luminosity, take minimum of potential as center */
                    ddxx = atime * NEAREST(P[p].Pos[j] - this_group->Pos[j]);

                    if (j < 2)
                        rr_tmp += ddxx * ddxx;
                }

                rr_tmp = sqrt(rr_tmp);

                if (rr_tmp <= r_gal)
                {
                    int FLAG_AGE;
                    double lum_particle[LT_ADD_GAL_TO_SUB], lum_particle_obs[LT_ADD_GAL_TO_SUB];

                    add_luminosity(p, &FLAG_AGE, &lum_particle[0], &lum_particle_obs[0]);
                    for (int ilum = 0; ilum < LT_ADD_GAL_TO_SUB; ilum++)
                        loc_group.SubLum[ilum] += lum_particle[ilum];
#ifdef OBSERVER_FRAME
                    for (int ilum = 0; ilum < LT_ADD_GAL_TO_SUB; ilum++)
                        loc_group.SubLumObs[ilum] += lum_particle_obs[ilum];
#endif // OBSERVER_FRAME

#ifdef DUSTATT
                    int rad_bin = floor(rr_tmp / r_gal * (DUSTATT - 2));
                    int rad_bin_z = floor(sqrt(ddxx * ddxx + rr_tmp * rr_tmp) / r_gal * (DUSTATT - 2));

                    double mu = -1;
                    if (FLAG_AGE != 1)
                    {
                        long mu_seed = -150;
                        double MUWIDTH = 0.2, MUCENTER = 0.3;

                        mu = -1.;
                        while (mu < 0)
                        {
                            mu = gasdev(&mu_seed) * MUWIDTH + MUCENTER;
                            if (mu < 0.1 || mu > 1.0)
                            {
                                mu = -1.;
                            }
                        }
                    }
                    double term_depth1 = 0, term_depth2 = 0, term_depth_total = 0, depth = 0;
                    for (int k = 0; k < DUSTATT - 1; k++)
                    {
                        if (k >= rad_bin_z)
                        {
                            term_depth1 += TauV_in_bin[k];
                        }
                        if (k <= rad_bin_z)
                        {
                            term_depth2 += TauV_in_bin[k];
                        }
                        term_depth_total += TauV_in_bin[k];
                    }
                    if (term_depth_total > 0.)
                    {
                        if (ddxx >= 0)
                        {
                            depth = term_depth1 / (2 * term_depth_total);
                        }
                        if (ddxx < 0)
                        {
                            depth = (term_depth2 + term_depth_total) / (2 * term_depth_total);
                        }
                    }
                    else
                    {
                        depth = 0.;
                    }
                    for (int ilum = 0; ilum < LT_ADD_GAL_TO_SUB; ilum++)
                    {
#ifdef OBSERVER_FRAME
                        double Tau_Lambda_over_V_obs =
                            mu * pow(Filters_Effective_Wavelenght[bc_use_index[ilum]] * All.Time / 550, -0.7);
                        loc_group.SubLumAtt[ilum] +=
                            lum_particle_obs[ilum] * exp(-depth * Tau_Lambda_over_V_obs * TauV_in_bin[rad_bin]);
#endif // OBSERVER_FRAME
                    }
#endif // OBSERVER_FRAME
                }
                loc_group.SubMeanAge += MPP(p).StellarAge;
                loc_group.SubMeanZStar += get_metallicity_subfind(p) / 0.02;
            }
        }

        int nstar_gal, ngas_gal, ncoldgas_gal;

        if (collective == 1)
        {
            MPI_Allreduce(&loc_group.SubLum, &this_group->SubLum, LT_ADD_GAL_TO_SUB, MPI_DOUBLE, MPI_SUM,
                          MYMPI_COMM_WORLD);
#ifdef OBSERVER_FRAME
            MPI_Allreduce(&loc_group.SubLumObs, &this_group->SubLumObs, LT_ADD_GAL_TO_SUB, MPI_DOUBLE, MPI_SUM,
                          MYMPI_COMM_WORLD);
#endif // OBSERVER_FRAME
#ifdef DUSTATT
            MPI_Allreduce(&loc_group.SubLumAtt, &this_group->SubLumAtt, LT_ADD_GAL_TO_SUB, MPI_DOUBLE, MPI_SUM,
                          MYMPI_COMM_WORLD);
#endif // DUSTATT
            MPI_Allreduce(&loc_group.SubMeanAge, &this_group->SubMeanAge, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&loc_group.SubMeanZStar, &this_group->SubMeanZStar, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&loc_group.SubSFR, &this_group->SubSFR, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&loc_group.SubHIMass, &this_group->SubHIMass, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&loc_group.SubMeanZGas, &this_group->SubMeanZGas, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&loc_group.SubMeanZColdGas, &this_group->SubMeanZColdGas, 1, MPI_DOUBLE, MPI_SUM,
                          MYMPI_COMM_WORLD);
            MPI_Allreduce(&nstar_this, &nstar_gal, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&ngas_this, &ngas_gal, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            MPI_Allreduce(&ncoldgas_this, &ncoldgas_gal, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        }
        else
        {
            for (int k = 0; k < LT_ADD_GAL_TO_SUB; k++)
            {
                this_group->SubLum[k] = loc_group.SubLum[k];
#ifdef OBSERVER_FRAME
                this_group->SubLumObs[k] = loc_group.SubLumObs[k];
#endif // OBSERVER_FRAME
#ifdef DUSTATT
                this_group->SubLumAtt[k] = loc_group.SubLumAtt[k];
#endif // DUSTATT
            }
            this_group->SubMeanAge = loc_group.SubMeanAge;
            this_group->SubMeanZStar = loc_group.SubMeanZStar;
            this_group->SubSFR = loc_group.SubSFR;
            this_group->SubHIMass = loc_group.SubHIMass;
            this_group->SubMeanZGas = loc_group.SubMeanZGas;
            this_group->SubMeanZColdGas = loc_group.SubMeanZColdGas;
            nstar_gal = nstar_this;
            ngas_gal = ngas_this;
            ncoldgas_gal = ncoldgas_this;
        }

        VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2,
                       "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): metalicity properties (%g/%d,%g/%d)\n", ThisTask,
                       collective, this_group->SubMeanZGas, ngas_gal, this_group->SubMeanZColdGas, ncoldgas_gal);

        if (nstar_gal > 0)
        {
            this_group->SubMeanAge /= nstar_gal;
            this_group->SubMeanZStar /= nstar_gal;
        }
        if (ngas_gal > 0)
        {
            this_group->SubMeanZGas /= ngas_gal;
        }
        if (ncoldgas_gal > 0)
        {
            this_group->SubMeanZColdGas /= ncoldgas_gal;
        }

        for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
        {
            if (this_group->SubLum[j] > 0)
            {
                this_group->SubLum[j] = -2.5 * log10(this_group->SubLum[j]);
            }
            else
            {
                this_group->SubLum[j] = 0;
            }
#ifdef DUSTATT
            if (this_group->SubLumAtt[j] > 0)
            {
                this_group->SubLumAtt[j] = -2.5 * log10(this_group->SubLumAtt[j]);
            }
            else
            {
                this_group->SubLumAtt[j] = 0;
            }
#endif // DUSTATT
#ifdef OBSERVER_FRAME
            if (this_group->SubLumObs[j] > 0)
            {
                this_group->SubLumObs[j] = -2.5 * log10(this_group->SubLumObs[j]);
            }
            else
            {
                this_group->SubLumObs[j] = 0;
            }
#endif // OBSERVER_FRAME
        }
    }
    else
    {
        this_group->SubSFR = this_group->SubHIMass = this_group->SubMeanZGas = this_group->SubMeanZColdGas =
            this_group->SubMeanAge = this_group->SubMeanZStar = 0;
        for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
        {
            this_group->SubLum[j] = 0;
#ifdef DUSTATT
            this_group->SubLumAtt[j] = 0;
#endif // DUSTATT
#ifdef OBSERVER_FRAME
            this_group->SubLumObs[j] = 0;
#endif // OBSERVER_FRAME
        }
    }
#endif // LT_ADD_GAL_TO_SUB

    VERBOSE_IF_ALL(num > SUBFIND_OUTPUT_TRESH, 2, "SUBFIND_DETERMINE_SUB_HALO (task=%d,col=%d): done (%g,%g)\n",
                   ThisTask, collective, this_group->SubSFR, this_group->SubHIMass);

    if (num_use > 0)
    {
        myfree(loc_rr_list);
    }

    myfree(loc_rr_list_gal);

    if (nstarloc > 0)
    {
        myfree(loc_rr_list_star);
    }
}

/**
 * @brief Compares two candidates based on their bound length and rank.
 *
 * This function is used as a comparator for sorting candidate structures based on their bound length.
 * If the bound lengths are equal, candidates are further sorted based on their rank.
 *
 * @param a Pointer to the first candidate structure.
 * @param b Pointer to the second candidate structure.
 * @return -1 if the bound length of candidate 'a' is greater than candidate 'b' or if the rank of candidate 'a' is less
 * than candidate 'b'; +1 if the bound length of candidate 'a' is less than candidate 'b' or if the rank of candidate
 * 'a' is greater than candidate 'b'; 0 if the bound length and rank of both candidates are equal.
 */
int subfind_compare_serial_candidates_boundlength(const void *a, const void *b)
{
    if (((struct cand_dat *)a)->bound_length > ((struct cand_dat *)b)->bound_length)
    {
        return -1;
    }

    if (((struct cand_dat *)a)->bound_length < ((struct cand_dat *)b)->bound_length)
    {
        return +1;
    }

    if (((struct cand_dat *)a)->rank < ((struct cand_dat *)b)->rank)
    {
        return -1;
    }

    if (((struct cand_dat *)a)->rank > ((struct cand_dat *)b)->rank)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two candidates based on their rank and length.
 *
 * This function is used as a comparator for sorting candidate structures based on their rank.
 * If the ranks are equal, candidates are further sorted based on their length.
 *
 * @param a Pointer to the first candidate structure.
 * @param b Pointer to the second candidate structure.
 * @return -1 if the rank of candidate 'a' is less than candidate 'b' or if the length of candidate 'a' is greater than
 * candidate 'b'; +1 if the rank of candidate 'a' is greater than candidate 'b' or if the length of candidate 'a' is
 * less than candidate 'b'; 0 if the rank and length of both candidates are equal.
 */
int subfind_compare_serial_candidates_rank(const void *a, const void *b)
{
    if (((struct cand_dat *)a)->rank < ((struct cand_dat *)b)->rank)
    {
        return -1;
    }

    if (((struct cand_dat *)a)->rank > ((struct cand_dat *)b)->rank)
    {
        return +1;
    }

    if (((struct cand_dat *)a)->len > ((struct cand_dat *)b)->len)
    {
        return -1;
    }

    if (((struct cand_dat *)a)->len < ((struct cand_dat *)b)->len)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two candidates based on their subnr.
 *
 * This function is used as a comparator for sorting candidate structures based on their subnr.
 *
 * @param a Pointer to the first candidate structure.
 * @param b Pointer to the second candidate structure.
 * @return -1 if the subnr of candidate 'a' is less than the subnr of candidate 'b';
 *         +1 if the subnr of candidate 'a' is greater than the subnr of candidate 'b';
 *         0 if the subnr of both candidates is equal.
 */
int subfind_compare_serial_candidates_subnr(const void *a, const void *b)
{
    if (((struct cand_dat *)a)->subnr < ((struct cand_dat *)b)->subnr)
    {
        return -1;
    }

    if (((struct cand_dat *)a)->subnr > ((struct cand_dat *)b)->subnr)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Generates uniform random numbers in the range [0, 1).
 *
 * This function generates pseudo-random numbers uniformly distributed between 0 and 1 using a linear congruential
 * method.
 *
 * @param idum Pointer to a long integer seed, which serves as the state for the random number generator.
 * @return A pseudo-random number uniformly distributed between 0 and 1 (excluding 1).
 */
float ran1(long *idum)
{
    int IA = 16807;
    int IM = 2147483647; // INT_MAX
    double AM = 1.0 / IM;
    int IQ = 127773;
    int IR = 2836;
    const int NTAB_Local = 32; // Nitya - NTAB has been defined already
    double NDIV = 1 + (IM - 1) / NTAB_Local;
    double EPS = 1.2e-7;
    double RNMX = 1.0 - EPS;

    static long iy = 0;
    static long iv[NTAB_Local];
    float temp;

    if (*idum <= 0 || !iy)
    {
        if (-(*idum) < 1)
        {
            *idum = 1;
        }
        else
        {
            *idum = -(*idum);
        }
        for (int j = NTAB_Local + 7; j >= 0; j--)
        {
            long k = (*idum) / IQ;
            *idum = IA * (*idum - k * IQ) - IR * k;
            if (*idum < 0)
            {
                *idum += IM;
            }
            if (j < NTAB_Local)
            {
                iv[j] = *idum;
            }
        }
        iy = iv[0];
    }
    long kk = (*idum) / IQ;
    *idum = IA * (*idum - kk * IQ) - IR * kk;
    if (*idum < 0)
    {
        *idum += IM;
    }
    int jj = iy / NDIV;
    iy = iv[jj];
    iv[jj] = *idum;
    if ((temp = AM * iy) > RNMX)
    {
        return RNMX;
    }
    else
    {
        return temp;
    }
}

/**
 * @brief Generates Gaussian-distributed random numbers with mean 0 and standard deviation 1.
 *
 * This function uses the Box-Muller transform to generate normally distributed (Gaussian) random numbers with a mean of
 * 0 and a standard deviation of 1. It generates pairs of independent standard normal deviates and returns them one at a
 * time. The function alternates between generating new pairs of numbers and returning the second number of the pair
 * from the previous call. This is managed through the static variables `iset` and `gset`.
 *
 * @param idum Pointer to a long integer seed, used as the seed for the random number generator `ran1`.
 * @return A Gaussian-distributed random number.
 */
float gasdev(long *idum)
{
    static int iset = 0;
    static float gset;

    if (iset == 0)
    {
        float v1, v2, rsq;
        do
        {
            v1 = 2.0 * ran1(idum) - 1.0;
            v2 = 2.0 * ran1(idum) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        float fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        iset = 1;
        return v2 * fac;
    }
    else
    {
        iset = 0;
        return gset;
    }
}

#endif // SUBFIND
