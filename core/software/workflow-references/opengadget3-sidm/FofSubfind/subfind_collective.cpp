/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Parrallel processing of Subfind groups.
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
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../CodeBase/domain.hpp"

#ifdef SUBFIND
#ifndef SUBFIND_ALTERNATIVE_COLLECTIVE

#include "subfind.hpp"
#include "fof.hpp"

#define TAG_POLLING_DONE 201
#define TAG_SET_ALL 202
#define TAG_GET_NGB_INDICES 204
#define TAG_GET_TAILANDLEN 205
#define TAG_GET_TAILANDLEN_DATA 206
#define TAG_SET_TAILANDLEN 207
#define TAG_SET_HEADANDNEXT 209
#define TAG_SETHEADGETNEXT_DATA 210
#define TAG_SET_NEXT 211
#define TAG_SETHEADGETNEXT 213
#define TAG_GET_NEXT 215
#define TAG_GET_NEXT_DATA 216
#define TAG_GET_HEAD 217
#define TAG_GET_HEAD_DATA 218
#define TAG_ADD_PARTICLE 219
#define TAG_ADDBOUND 220
#define TAG_NID 222
#define TAG_NID_DATA 223
#define TAG_SETRANK 224
#define TAG_SETRANK_OUT 226
#define TAG_GET_RANK 227
#define TAG_GET_RANK_DATA 228
#define TAG_MARK_PARTICLE 229
#define TAG_SET_NEWTAIL 230
#define TAG_GET_OLDTAIL 231
#define TAG_GET_TWOHEADS 232
#define TAG_GET_TWOHEADS_DATA 233

#define MASK ((((long long)1) << 32) - 1)
#define HIGHBIT (1 << 30)

static long long *Head, *Next, *Tail;
static int *Len;
static int LocalLen;
static int count_cand, max_candidates;

/**
 * @brief Structure to store candidate subhalo data.
 */
static struct cand_dat
{
    long long head;
    long long rank;
    int len;
    int nsub;
    int subnr, parent;
    int bound_length;
} *candidates;

static struct unbind_data *ud;

/**
 * @brief Structure to store data used for sorting by density.
 */
static struct sort_density_data
{
    MyFloat density;
    int ngbcount;
    long long index; /* this will store the task in the upper word */
    long long ngb_index1, ngb_index2;
} *sd;

/**
 * @brief Unbinds independent subhalos from the candidate list.
 *
 * This function processes a list of candidate subhalos, unbinding them to identify
 * genuine subhalos. It iterates over particles, marking those that belong to a subhalo,
 * and checks their membership by analyzing their binding properties. If a subhalo meets
 * the required criteria, it is flagged as a bound substructure.
 *
 * @param count_cand The number of candidate subhalos to process.
 */
void subfind_unbind_independent_ones(int count_cand)
{
    int len, nsubs, len_non_gas;

    /*subfind_loctree_treeallocate(All.TreeAllocFactor * NumPart, NumPart); */

    R2list = (struct r2data *)mymalloc("R2list", NumPart * sizeof(struct r2data));
    ud = (struct unbind_data *)mymalloc("ud", NumPart * sizeof(struct unbind_data));

    STD_SORT(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_candidates_nsubs);
    int i = 0;
    for (int k = 0; k < count_cand; k++)
    {
        if (candidates[k].parent == 0)
        {
            while (P[i].submark < candidates[k].nsub)
            {
                i++;
                ENDRUN_IF(i >= NumPart, "index is larger than the number of particles");
            }

            if (P[i].submark >= 0 && P[i].submark < HIGHBIT)
            {
                len = 0;
                nsubs = P[i].submark;

                ENDRUN_IF(nsubs != candidates[k].nsub, "TASK=%d i=%d k=%d nsubs=%d candidates[k].nsub=%d", ThisTask, i,
                          k, nsubs, candidates[k].nsub);

                while (i < NumPart)
                {
                    if (P[i].submark == nsubs)
                    {
                        P[i].submark = HIGHBIT;
                        if ((P[i].origintask & HIGHBIT) == 0)
                        {
                            ud[len].index = i;
                            len++;
                        }
                        i++;
                    }
                    else
                    {
                        break;
                    }
                }

                len = subfind_unbind(ud, len, &len_non_gas);

#ifndef DO_GAS_CLOUDS
                if (len_non_gas >= All.DesLinkNgb)
#else // DO_GAS_CLOUDS
                if (len >= All.DesLinkNgb)
#endif // DO_GAS_CLOUDS
                {
                    /* ok, we found a substructure */
                    candidates[k].bound_length = len;

                    for (int j = 0; j < len; j++)
                    {
                        P[ud[j].index].submark = nsubs; /* we use this to flag the substructures */
                    }
                }
                else
                {
                    candidates[k].bound_length = 0;
                }
            }
        }
    }

    myfree(ud);
    myfree(R2list);
}

/**
 * @brief Processes a group collectively to identify and analyze substructures.
 *
 * This function handles the identification and processing of substructures (subhalos) within a group of particles.
 * It uses MPI for parallel processing across multiple tasks, performing various steps including domain decomposition,
 * tree building, density sorting, and unbinding of subhalo candidates. The function ensures that subhalos are correctly
 * identified and their properties are computed and stored.
 *
 * The steps include:
 * - Broadcasting the group number to all tasks.
 * - Counting the number of particles in the group.
 * - Performing domain decomposition.
 * - Sorting particles by group number and constructing a tree for the group.
 * - Finding the nearest two denser neighbors for each particle.
 * - Sorting densities and initializing link lists.
 * - Finding and sorting subhalo candidates.
 * - Performing independent and collective unbinding of subhalo candidates.
 * - Determining the parent subhalo for each candidate and saving properties of the substructures.
 *
 * @param num An integer parameter used in the subfind_col_find_candidates function to assist in candidate finding.
 */
void subfind_process_group_collectively(int num)
{
    TIMER_PUSH(1, "SUBFIND_COLLHALOS");
    long long p;
    int len, len_non_gas, LocalNonGasLen, totgrouplen;
    int ncand, parent, totcand, nremaining;
    int max_loc_length, max_length;
    int count, countall, *countlist, *offset;
    int nr, grindex = 0, nsubs, subnr;
    int count_leaves, tot_count_leaves;
    struct cand_dat *tmp_candidates = 0;
    struct subgroup_properties local_group;

    double t0, t1, tt0, tt1;

    GrNr = Group[0].GrNr;

    MPI_Bcast(&GrNr, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    VERBOSE(2, "\ncolor %d: collectively doing halo %d on %d tasks", color, GrNr, NTask);
    NumPartGroup = 0;
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].GrNr == GrNr)
        {
            NumPartGroup++;
        }
    }

    MPI_Allreduce(&NumPartGroup, &totgrouplen, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    /* distribute this halo among the processors */
    t0 = second();

    All.DoDynamicUpdate = 0;

    domain_free_trick();

    domain_Decomposition(1, 0);

    t1 = second();
    VERBOSE(2, "domain_Decomposition() took %g sec  (presently allocated=%g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    for (int i = 0; i < NumPart; i++)
    {
        P[i].origindex = i;
    }

    STD_SORT(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNrGrNr);

    /* now we have the particles of the group at the beginning, but SPH particles are not aligned.
       They can however be accessed via SphP[P[i].origindex] */
    NumPartGroup = 0;
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].GrNr == GrNr)
        {
            NumPartGroup++;
        }
    }

    subfind_loctree_copyExtent(); /* this will make sure that all the serial trees start from the same root node
                                     geometry */

    /* construct a tree for the halo */
    t0 = second();
    force_treebuild(NumPartGroup, NULL);
    t1 = second();

    VERBOSE(2, "force_treebuild() took %g sec (presently allocated=%g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    sd = (struct sort_density_data *)mymalloc("sd", NumPartGroup * sizeof(struct sort_density_data));

    /* determine the indices of the nearest two denser neighbours within the link region */
    t0 = second();
    NgbLoc = (struct nearest_ngb_data *)mymalloc("NgbLoc", NumPartGroup * sizeof(struct nearest_ngb_data));
    R2Loc = (struct nearest_r2_data *)mymalloc("R2Loc", NumPartGroup * sizeof(struct nearest_r2_data));
    subfind_find_nearesttwo();
    t1 = second();

    VERBOSE(2, "find_nearesttwo() took %g sec (presently allocated=%g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    for (int i = 0; i < NumPartGroup; i++)
    {
        sd[i].density = P[i].u.DM_Density;
        sd[i].ngbcount = NgbLoc[i].count;
        sd[i].index = (((long long)ThisTask) << 32) + i;
        sd[i].ngb_index1 = NgbLoc[i].index[0];
        sd[i].ngb_index2 = NgbLoc[i].index[1];
    }
    myfree(R2Loc);
    myfree(NgbLoc);

    /* sort the densities */
    t0 = second();
    parallel_sort(sd, NumPartGroup, sizeof(struct sort_density_data), subfind_compare_densities);
    t1 = second();

    VERBOSE(2, "parallel sort of densities done. took %g sec", timediff(t0, t1));

    /* allocate and initialize distributed link list */
    Head = (long long *)mymalloc("Head", NumPartGroup * sizeof(long long));
    Next = (long long *)mymalloc("Next", NumPartGroup * sizeof(long long));
    Tail = (long long *)mymalloc("Tail", NumPartGroup * sizeof(long long));
    Len = (int *)mymalloc("Len", NumPartGroup * sizeof(int));

    for (int i = 0; i < NumPartGroup; i++)
    {
        Head[i] = Next[i] = Tail[i] = -1;
        Len[i] = 0;
    }

    /* allocate a list to store subhalo candidates */
    max_candidates = max((NumPartGroup / 50), 200);
    candidates = (struct cand_dat *)mymalloc("candidates", max_candidates * sizeof(struct cand_dat));
    count_cand = 0;

    subfind_col_find_candidates(num, totgrouplen);

    /* establish total number of candidates */
    MPI_Allreduce(&count_cand, &totcand, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    VERBOSE(2, "\ntotal number of subhalo candidates=%d", totcand);

    nremaining = totcand;

    for (int i = 0; i < NumPartGroup; i++)
    {
        Tail[i] = -1;
    }

    for (int i = 0; i < count_cand; i++)
    {
        candidates[i].parent = 0;
    }

    do
    {
        /* Let's see which candidates can be unbound independent from each other.
           We identify them with those candidates that have no embedded other candidate */
        t0 = second();
        if (ThisTask == 0)
        {
            tmp_candidates = (struct cand_dat *)mymalloc("	tmp_candidates", totcand * sizeof(struct cand_dat));
        }

        count = count_cand;
        count *= sizeof(struct cand_dat);

        countlist = (int *)mymalloc("countlist", NTask * sizeof(int));
        offset = (int *)mymalloc("offset", NTask * sizeof(int));

        MPI_Allgather(&count, 1, MPI_INT, countlist, 1, MPI_INT, MYMPI_COMM_WORLD);
        offset[0] = 0;
        for (int i = 1; i < NTask; i++)
        {
            offset[i] = offset[i - 1] + countlist[i - 1];
        }

        MPI_Gatherv(candidates, countlist[ThisTask], MPI_BYTE, tmp_candidates, countlist, offset, MPI_BYTE, 0,
                    MYMPI_COMM_WORLD);

        if (ThisTask == 0)
        {
            for (int k = 0; k < totcand; k++)
            {
                tmp_candidates[k].nsub = k;
                tmp_candidates[k].subnr = k;
            }
            STD_SORT(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_rank);

            for (int k = 0; k < totcand; k++)
            {
                if (tmp_candidates[k].parent >= 0)
                {
                    tmp_candidates[k].parent = 0;

                    for (int j = k + 1; j < totcand; j++)
                    {
                        if (tmp_candidates[j].rank > tmp_candidates[k].rank + tmp_candidates[k].len)
                        {
                            break;
                        }

                        if (tmp_candidates[j].parent < 0) /* ignore these */
                        {
                            continue;
                        }

                        if (tmp_candidates[k].rank + tmp_candidates[k].len >=
                            tmp_candidates[j].rank + tmp_candidates[j].len)
                        {
                            tmp_candidates[k].parent++; /* we here count the number of subhalos that are enclosed */
                        }
                        else
                        {
                            ENDRUN("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d", k, totcand,
                                   (int)tmp_candidates[k].rank, (int)tmp_candidates[k].len, j,
                                   (int)tmp_candidates[j].rank, (int)tmp_candidates[j].len);
                        }
                    }
                }
            }
            STD_SORT(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_subnr);
        }

        MPI_Scatterv(tmp_candidates, countlist, offset, MPI_BYTE, candidates, countlist[ThisTask], MPI_BYTE, 0,
                     MYMPI_COMM_WORLD);

        myfree(offset);
        myfree(countlist);

        if (ThisTask == 0)
        {
            myfree(tmp_candidates);
        }

        count_leaves = 0, max_loc_length = 0;
        for (int i = 0; i < count_cand; i++)
            if (candidates[i].parent == 0)
            {
                if (candidates[i].len > max_loc_length)
                {
                    max_loc_length = candidates[i].len;
                }

                if (candidates[i].len >
                    0.15 * All.TotNumPart / NTask) /* seems large, let's rather do it collectively */
                {
                    candidates[i].parent++; /* this will ensure that it is not considered in this round */
                }
                else
                {
                    count_leaves++;
                }
            }

        MPI_Allreduce(&count_leaves, &tot_count_leaves, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&max_loc_length, &max_length, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);

        t1 = second();
        VERBOSE(2,
                "\nnumber of subhalo candidates that can be done independently=%d.\n(Largest size is %d, finding them "
                "took %g sec)",
                tot_count_leaves, max_length, timediff(t0, t1));

        if (tot_count_leaves <= 0 * NTask) /* if there are only a few left, let's do them collectively */
        {

            VERBOSE(2, "too few, I do the rest of %d collectively\n", nremaining);
            break;
        }

        if (max_length > 0.8 * All.TotNumPart / NTask) /* seems large, let's rather do it collectively */
        {
            VERBOSE(2, "too big candidates, I do the rest collectively\n");
            break;
        }

        nremaining -= tot_count_leaves;

        for (int i = 0; i < NumPart; i++)
        {
            P[i].origintask = P[i].targettask = ThisTask;
            P[i].submark = HIGHBIT;
            if (i < NumPartGroup)
            {
                if (Tail[i] >= 0) /* this means this particle is already bound to a substructure */
                {
                    P[i].origintask |= HIGHBIT;
                }
            }
        }

        /* we now mark the particles that are in subhalo candidates that can be processed independently in parallel */
        nsubs = 0;
        t0 = second();
        for (int master = 0; master < NTask; master++)
        {
            ncand = count_cand;

            MPI_Bcast(&ncand, sizeof(ncand), MPI_BYTE, master, MYMPI_COMM_WORLD);

            for (int k = 0; k < ncand; k++)
            {
                if (ThisTask == master)
                {
                    len = candidates[k].len;
                    parent = candidates[k].parent; /* this is here actually the daughter count */
                }

                MPI_Bcast(&len, sizeof(len), MPI_BYTE, master, MYMPI_COMM_WORLD);
                MPI_Bcast(&parent, sizeof(parent), MPI_BYTE, master, MYMPI_COMM_WORLD);
                MPI_Barrier(MYMPI_COMM_WORLD);

                if (parent == 0)
                {
                    if (ThisTask != master)
                    {
                        subfind_poll_for_requests();
                    }
                    else
                    {
                        p = candidates[k].head;
                        for (int i = 0; i < candidates[k].len; i++)
                        {
                            subfind_distlinklist_mark_particle(p, master, candidates[k].nsub);

                            ENDRUN_IF(p < 0, "Bummer i=%d ", i);
                            p = subfind_distlinklist_get_next(p);
                        }

                        /* now tell the others to stop polling */
                        for (int i = 0; i < NTask; i++)
                        {
                            if (i != ThisTask)
                            {
                                MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MYMPI_COMM_WORLD);
                            }
                        }
                    }

                    MPI_Barrier(MYMPI_COMM_WORLD);
                }

                nsubs++;
            }
        }
        t1 = second();
        VERBOSE(2, "particles are marked (took %g)", timediff(t0, t1));

        t0 = second();
        subfind_distribute_particles(1); /* assemble the particles on individual processors */
        t1 = second();

        VERBOSE(
            2,
            "independent subhalos are assembled on individual CPUs for unbinding (%g sec, (presently allocated=%g MB)",
            timediff(t0, t1), AllocatedBytes / (1024.0 * 1024.0));

        STD_SORT(P, NumPart, sizeof(struct particle_data),
                 subfind_compare_P_submark); /* groups particles of the same canidate together */

        MPI_Barrier(MYMPI_COMM_WORLD);
        t0 = second();

        subfind_unbind_independent_ones(count_cand);

        MPI_Barrier(MYMPI_COMM_WORLD);
        t1 = second();

        VERBOSE(2, "unbinding of independent ones took %g sec", timediff(t0, t1));

        for (int i = 0; i < NumPart; i++)
        {
            P[i].origintask &= (HIGHBIT - 1); /* clear high bit if set */
        }

        t0 = second();
        subfind_distribute_particles(2); /* bring them back to their original processor */
        t1 = second();

        VERBOSE(2, "particles have returned to their original processor (%g sec, presently allocated %g MB)",
                timediff(t0, t1), AllocatedBytes / (1024.0 * 1024.0));

        /* reestablish the original order */
        STD_SORT(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNrGrNr);

        /* now mark the bound particles */
        for (int i = 0; i < NumPartGroup; i++)
        {
            if (P[i].submark >= 0 && P[i].submark < nsubs)
            {
                Tail[i] = P[i].submark; /* we use this to flag bound parts of substructures */
            }
        }

        for (int i = 0; i < count_cand; i++)
        {
            if (candidates[i].parent == 0)
            {
                candidates[i].parent = -1;
            }
        }
    } while (tot_count_leaves > 0);

    force_treebuild(NumPartGroup, NULL); /* re construct the tree for the collective part */

    /**** now we do the collective unbinding of the subhalo candidates that contain other subhalo candidates ****/
    ud = (struct unbind_data *)mymalloc("ud", NumPartGroup * sizeof(struct unbind_data));

    t0 = second();
    nr = 0;
    for (int master = 0; master < NTask; master++)
    {
        ncand = count_cand;

        MPI_Bcast(&ncand, sizeof(ncand), MPI_BYTE, master, MYMPI_COMM_WORLD);

        for (int k = 0; k < ncand; k++)
        {
            if (ThisTask == master)
            {
                len = candidates[k].len;
                nsubs = candidates[k].nsub;
                parent = candidates[k].parent; /* this is here actually the daughter count */
            }

            MPI_Bcast(&parent, sizeof(parent), MPI_BYTE, master, MYMPI_COMM_WORLD);
            MPI_Barrier(MYMPI_COMM_WORLD);

            if (parent >= 0)
            {
                MPI_Bcast(&len, sizeof(len), MPI_BYTE, master, MYMPI_COMM_WORLD);
                MPI_Bcast(&nsubs, sizeof(nsubs), MPI_BYTE, master, MYMPI_COMM_WORLD);

                VERBOSE(2, "collective unbinding of nr=%d (%d) of length=%d ... ", nr, nremaining, (int)len);

                nr++;

                LocalLen = 0;

                tt0 = second();

                if (ThisTask != master)
                {
                    subfind_poll_for_requests();
                }
                else
                {
                    p = candidates[k].head;
                    for (int i = 0; i < candidates[k].len; i++)
                    {
                        subfind_distlinklist_add_particle(p);

                        ENDRUN_IF(p < 0, "Bummer i=%d ", i);

                        p = subfind_distlinklist_get_next(p);
                    }

                    /* now tell the others to stop polling */
                    for (int i = 0; i < NTask; i++)
                    {
                        if (i != ThisTask)
                        {
                            MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MYMPI_COMM_WORLD);
                        }
                    }
                }

                LocalLen = subfind_col_unbind(ud, LocalLen, &LocalNonGasLen);

                tt1 = second();
                VERBOSE(2, "took %g sec", timediff(tt0, tt1));

                MPI_Allreduce(&LocalLen, &len, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

#ifndef DO_GAS_CLOUDS
                MPI_Allreduce(&LocalNonGasLen, &len_non_gas, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
                if (len_non_gas >= All.DesLinkNgb)
#else // DO_GAS_CLOUDS
                if (len >= All.DesLinkNgb)
#endif // DO_GAS_CLOUDS
                {
                    /* ok, we found a substructure */
                    VERBOSE(1, "substructure of len=%d found", (int)len);

                    for (int i = 0; i < LocalLen; i++)
                    {
                        Tail[ud[i].index] = nsubs; /* we use this to flag the substructures */
                    }

                    if (ThisTask == master)
                    {
                        candidates[k].bound_length = len;
                    }
                }
                else
                {
                    if (ThisTask == master)
                    {
                        candidates[k].bound_length = 0;
                    }
                }
            }
        }
    }
    t1 = second();
    VERBOSE(2, "\nthe collective unbinding of remaining halos took %g sec", timediff(t0, t1));

    count = 0;
    for (int k = 0; k < count_cand; k++)
        if (candidates[k].bound_length >= All.DesLinkNgb)
        {
            if (candidates[k].len < All.DesLinkNgb)
            {
                ENDRUN("candidates[k=%d].len=%d bound=%d", k, candidates[k].len, candidates[k].bound_length);
            }
            count++;
        }

    MPI_Allreduce(&count, &countall, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    VERBOSE(2, "\nfound %d bound substructures in FoF group of length %d", countall, totgrouplen);

    /* now determine the parent subhalo for each candidate */
    t0 = second();
    parallel_sort(candidates, count_cand, sizeof(struct cand_dat), subfind_compare_candidates_boundlength);

    if (ThisTask == 0)
    {
        tmp_candidates = (struct cand_dat *)mymalloc("tmp_candidates", totcand * sizeof(struct cand_dat));
    }

    count = count_cand;
    count *= sizeof(struct cand_dat);

    countlist = (int *)mymalloc("countlist", NTask * sizeof(int));
    offset = (int *)mymalloc("offset", NTask * sizeof(int));

    MPI_Allgather(&count, 1, MPI_INT, countlist, 1, MPI_INT, MYMPI_COMM_WORLD);
    offset[0] = 0;
    for (int i = 1; i < NTask; i++)
    {
        offset[i] = offset[i - 1] + countlist[i - 1];
    }

    MPI_Gatherv(candidates, countlist[ThisTask], MPI_BYTE, tmp_candidates, countlist, offset, MPI_BYTE, 0,
                MYMPI_COMM_WORLD);

    if (ThisTask == 0)
    {
        for (int k = 0; k < totcand; k++)
        {
            tmp_candidates[k].subnr = k;
            tmp_candidates[k].parent = 0;
        }

        STD_SORT(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_rank);

        for (int k = 0; k < totcand; k++)
        {
            for (int j = k + 1; j < totcand; j++)
            {
                if (tmp_candidates[j].rank > tmp_candidates[k].rank + tmp_candidates[k].len)
                {
                    break;
                }

                if (tmp_candidates[k].rank + tmp_candidates[k].len >= tmp_candidates[j].rank + tmp_candidates[j].len)
                {
                    if (tmp_candidates[k].bound_length >= All.DesLinkNgb)
                    {
                        tmp_candidates[j].parent = tmp_candidates[k].subnr;
                    }
                }
                else
                {
                    ENDRUN("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d bound=%d\n", k, countall,
                           (int)tmp_candidates[k].rank, (int)tmp_candidates[k].len, (int)tmp_candidates[k].bound_length,
                           (int)tmp_candidates[j].rank, (int)tmp_candidates[j].len,
                           (int)tmp_candidates[j].bound_length);
                }
            }
        }
        STD_SORT(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_subnr);
    }

    MPI_Scatterv(tmp_candidates, countlist, offset, MPI_BYTE, candidates, countlist[ThisTask], MPI_BYTE, 0,
                 MYMPI_COMM_WORLD);

    myfree(offset);
    myfree(countlist);

    if (ThisTask == 0)
    {
        myfree(tmp_candidates);
    }

    t1 = second();
    VERBOSE(2, "determination of parent subhalo took %g sec (presently allocated %g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    /* Now let's save  some properties of the substructures */
    if (ThisTask == 0)
    {
        Group[grindex].Nsubs = countall;
    }

    t0 = second();
    subnr = 0;
    for (int master = 0; master < NTask; master++)
    {
        ncand = count_cand;
        MPI_Bcast(&ncand, 1, MPI_INT, master, MYMPI_COMM_WORLD);

        for (int k = 0; k < ncand; k++)
        {
            if (ThisTask == master)
            {
                len = candidates[k].bound_length;
                nsubs = candidates[k].nsub;
                parent = candidates[k].parent;
            }

            MPI_Bcast(&len, sizeof(len), MPI_BYTE, master, MYMPI_COMM_WORLD);
            MPI_Barrier(MYMPI_COMM_WORLD);

            if (len > 0)
            {
                MPI_Bcast(&nsubs, sizeof(nsubs), MPI_BYTE, master, MYMPI_COMM_WORLD);
                MPI_Bcast(&parent, sizeof(parent), MPI_BYTE, master, MYMPI_COMM_WORLD);

                LocalLen = 0;

                if (ThisTask != master)
                {
                    subfind_poll_for_requests();
                }
                else
                {
                    p = candidates[k].head;
                    for (int i = 0; i < candidates[k].len; i++)
                    {
                        subfind_distlinklist_add_bound_particles(p, nsubs);
                        p = subfind_distlinklist_get_next(p);
                    }

                    /* now tell the others to stop polling */
                    for (int i = 0; i < NTask; i++)
                    {
                        if (i != ThisTask)
                        {
                            MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MYMPI_COMM_WORLD);
                        }
                    }
                }

                MPI_Barrier(MYMPI_COMM_WORLD);

                tt0 = second();
                subfind_determine_sub_halo_properties(ud, LocalLen, 1, &local_group, k);
                tt1 = second();

                if (timediff(tt0, tt1) > 10.0)
                {
                    VERBOSE(2, "  determining properties of halo of len=%d took %g sec", len, timediff(tt0, tt1));
                }

                /* we have filled into ud the binding energy and the particle ID return */

                if (ThisTask == 0)
                {
                    ENDRUN_IF(Nsubgroups >= MaxNsubgroups,
                              "Number of subgroups exceeds the preset maximum MaxNsubgroups=%d", MaxNsubgroups);

                    if (subnr == 0)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            Group[grindex].Pos[j] = local_group.Pos[j];
                        }
                        local_group.Offset = Group[grindex].Offset;
                    }
                    else
                    {
                        local_group.Offset = SubGroup[Nsubgroups - 1].Offset + SubGroup[Nsubgroups - 1].Len;
                    }

                    local_group.Len = len;
                    local_group.GrNr = GrNr - 1;
                    local_group.SubNr = subnr;
                    local_group.SubParent = parent;
                    local_group.Len = len;

                    SubGroup[Nsubgroups] = local_group;

                    Nsubgroups++;
                }

                /* Let's now assign the subgroup number */

                for (int i = 0; i < LocalLen; i++)
                {
                    P[ud[i].index].SubNr = subnr;
                }

                subnr++;
            }
        }
    }
    t1 = second();
    VERBOSE(2, "determining substructure properties took %g sec (presently allocated %g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    myfree(ud);
    myfree(candidates);
    myfree(Len);
    myfree(Tail);
    myfree(Next);
    myfree(Head);
    myfree(sd);

    force_treefree();
    domain_free();
    domain_allocate_trick();

    STD_SORT(P, NumPart, sizeof(struct particle_data),
             subfind_compare_P_origindex); /* reorder them such that the gas particles match again */

    for (int i = 0; i < NumPart; i++)
    {
        ENDRUN_IF(P[i].origindex != i, "Indexing error.");
        ENDRUN_IF(P[i].origintask != ThisTask, "Task error.");
    }
    TIMER_POP(1, "SUBFIND_COLLHALOS");
}

/**
 * @brief Identifies subhalo candidates by building a distributed linked list from high to low density.
 *
 * This function constructs a distributed linked list to identify subhalo candidates within a cosmological simulation.
 * The function operates in parallel using MPI, with each task participating in the identification process.
 *
 * The function follows these main steps:
 * - Initialize and set up logging.
 * - Iterate over all MPI tasks to build the distributed linked list.
 * - Process particles based on their neighbor count to create or merge groups.
 * - Add the full group as a candidate after initial identification.
 * - Establish a rank order for the identified candidates based on their lengths.
 * @param num An integer representing the number of particles in the current group.
 * @param totgrouplen An integer representing the total length of the group being processed.
 */
void subfind_col_find_candidates(int num, int totgrouplen)
{
    int ngbcount, retcode, len_attach;
    int len, master;
    int tmp_int;
    long long prev, tail, tail_attach, tmp, next, index;
    long long p, ss, head, head_attach, ngb_index1, ngb_index2, rank;
    double t0, t1, tt0, tt1;

    VERBOSE(2, "building distributed linked list. (presently allocated %g MB)", AllocatedBytes / (1024.0 * 1024.0));

    /* now find the subhalo candidates by building up link lists from high density to low density */
    t0 = second();
    for (int master = 0; master < NTask; master++)
    {
        tt0 = second();
        if (ThisTask != master)
        {
            subfind_poll_for_requests();
        }
        else
        {
            for (int k = 0; k < NumPartGroup; k++)
            {
                ngbcount = sd[k].ngbcount;
                ngb_index1 = sd[k].ngb_index1;
                ngb_index2 = sd[k].ngb_index2;

                switch (ngbcount) /* treat the different possible cases */
                {
                case 0: /* this appears to be a lonely maximum -> new group */
                    subfind_distlinklist_set_all(sd[k].index, sd[k].index, sd[k].index, 1, -1);
                    break;

                case 1: /* the particle is attached to exactly one group */
                    head = subfind_distlinklist_get_head(ngb_index1);

                    ENDRUN_IF(head == -1, "We have a problem!  head=%d/%d for k=%d on task=%d", (int)(head >> 32),
                              (int)head, k, ThisTask);

                    retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[k].index);

                    if (!(retcode & 1))
                    {
                        subfind_distlinklist_set_headandnext(sd[k].index, head, -1);
                    }
                    if (!(retcode & 2))
                    {
                        subfind_distlinklist_set_next(tail, sd[k].index);
                    }
                    break;

                case 2: /* the particle merges two groups together */
                    if ((ngb_index1 >> 32) == (ngb_index2 >> 32))
                    {
                        subfind_distlinklist_get_two_heads(ngb_index1, ngb_index2, &head, &head_attach);
                    }
                    else
                    {
                        head = subfind_distlinklist_get_head(ngb_index1);
                        head_attach = subfind_distlinklist_get_head(ngb_index2);
                    }

                    if (head == -1 || head_attach == -1)
                    {
                        ENDRUN("We have a problem!  head=%d/%d head_attach=%d/%d for k=%d on task=%d",
                               (int)(head >> 32), (int)head, (int)(head_attach >> 32), (int)head_attach, k, ThisTask);
                    }

                    if (head != head_attach)
                    {
                        subfind_distlinklist_get_tailandlen(head, &tail, &len);
                        subfind_distlinklist_get_tailandlen(head_attach, &tail_attach, &len_attach);

                        if (len_attach > len) /* other group is longer, swap them */
                        {
                            tmp = head;
                            head = head_attach;
                            head_attach = tmp;

                            tmp = tail;
                            tail = tail_attach;
                            tail_attach = tmp;

                            tmp_int = len;
                            len = len_attach;
                            len_attach = tmp_int;
                        }

                        /* only in case the attached group is long enough we bother to register it
                           as a subhalo candidate */

                        if (len_attach >= All.DesLinkNgb)
                        {
                            if (count_cand < max_candidates)
                            {
                                candidates[count_cand].len = len_attach;
                                candidates[count_cand].head = head_attach;
                                count_cand++;
                            }
                            else
                            {
                                ENDRUN("Task %d: count=%d, max=%d, npartgroup=%d", ThisTask, count_cand, max_candidates,
                                       NumPartGroup);
                            }
                        }

                        /* now join the two groups */
                        subfind_distlinklist_set_tailandlen(head, tail_attach, len + len_attach);
                        subfind_distlinklist_set_next(tail, head_attach);

                        ss = head_attach;
                        do
                        {
                            ss = subfind_distlinklist_set_head_get_next(ss, head);
                        } while (ss >= 0);
                    }

                    /* finally, attach the particle to 'head' */
                    retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[k].index);

                    if (!(retcode & 1))
                    {
                        subfind_distlinklist_set_headandnext(sd[k].index, head, -1);
                    }
                    if (!(retcode & 2))
                    {
                        subfind_distlinklist_set_next(tail, sd[k].index);
                    }

                    break;
                }
            }

            /* now tell the others to stop polling */
            for (int k = 0; k < NTask; k++)
            {
                if (k != ThisTask)
                {
                    MPI_Send(&k, 1, MPI_INT, k, TAG_POLLING_DONE, MYMPI_COMM_WORLD);
                }
            }
        }

        MPI_Barrier(MYMPI_COMM_WORLD);
        tt1 = second();
        VERBOSE(2, "  ma=%d/%d took %g sec", master, NTask, timediff(tt0, tt1));
    }
    t1 = second();

    VERBOSE(2, "identification of primary candidates took %g sec", timediff(t0, t1));

    /* add the full thing as a subhalo candidate */
    t0 = second();
    head = -1, prev = -1;
    for (int master = 0; master < NTask; master++)
    {
        if (ThisTask != master)
        {
            subfind_poll_for_requests();
        }
        else
        {
            for (int i = 0; i < NumPartGroup; i++)
            {
                index = (((long long)ThisTask) << 32) + i;

                if (Head[i] == index)
                {
                    subfind_distlinklist_get_tailandlen(Head[i], &tail, &len);
                    next = subfind_distlinklist_get_next(tail);
                    if (next == -1)
                    {
                        if (prev < 0)
                        {
                            head = index;
                        }

                        if (prev >= 0)
                        {
                            subfind_distlinklist_set_next(prev, index);
                        }

                        prev = tail;
                    }
                }
            }

            /* now tell the others to stop polling */
            for (int k = 0; k < NTask; k++)
            {
                if (k != ThisTask)
                {
                    MPI_Send(&k, 1, MPI_INT, k, TAG_POLLING_DONE, MYMPI_COMM_WORLD);
                }
            }
        }

        MPI_Barrier(MYMPI_COMM_WORLD);
        MPI_Bcast(&head, sizeof(head), MPI_BYTE, master, MYMPI_COMM_WORLD);
        MPI_Bcast(&prev, sizeof(prev), MPI_BYTE, master, MYMPI_COMM_WORLD);
    }

    if (ThisTask == NTask - 1)
    {
        if (count_cand < max_candidates)
        {
            candidates[count_cand].len = totgrouplen;
            candidates[count_cand].head = head;
            count_cand++;
        }
        else
        {
            ENDRUN("count_cand larger then max_candidates");
        }
    }
    t1 = second();
    VERBOSE(2, "adding background as candidate took %g sec", timediff(t0, t1));

    /* go through the whole chain once to establish a rank order. For the rank we use Len[] */
    t0 = second();

    master = (head >> 32);

    if (ThisTask != master)
    {
        subfind_poll_for_requests();
    }
    else
    {
        p = head;
        rank = 0;

        while (p >= 0)
        {
            p = subfind_distlinklist_setrank_and_get_next(p, &rank);
        }

        /* now tell the others to stop polling */
        for (int i = 0; i < NTask; i++)
        {
            if (i != master)
            {
                MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MYMPI_COMM_WORLD);
            }
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD);
    MPI_Bcast(&rank, sizeof(rank), MPI_BYTE, master, MYMPI_COMM_WORLD); /* just for testing */

    /* for each candidate, we now pull out the rank of its head */
    for (int master = 0; master < NTask; master++)
    {
        if (ThisTask != master)
        {
            subfind_poll_for_requests();
        }
        else
        {
            for (int k = 0; k < count_cand; k++)
            {
                candidates[k].rank = subfind_distlinklist_get_rank(candidates[k].head);
            }

            /* now tell the others to stop polling */
            for (int i = 0; i < NTask; i++)
            {
                if (i != ThisTask)
                {
                    MPI_Send(&i, 1, MPI_INT, i, TAG_POLLING_DONE, MYMPI_COMM_WORLD);
                }
            }
        }
    }
    MPI_Barrier(MYMPI_COMM_WORLD);

    t1 = second();

    VERBOSE(2, "establishing of rank order took %g sec  (p=%d, grouplen=%d) presently allocated %g MB",
            timediff(t0, t1), (int)rank, totgrouplen, AllocatedBytes / (1024.0 * 1024.0));

    if (((int)rank) != totgrouplen)
    {
        ENDRUN("ERROR: mismatch");
    }
}

/**
 * @brief Unbinds particles from potential wells to identify unbound particles.
 *
 * This function performs the unbinding process to identify unbound particles within a group of particles.
 * It iteratively removes particles that have positive binding energy and lie beyond a specified energy limit.
 * The unbinding process is performed using MPI parallelism for efficiency.
 *
 * @param d Pointer to the unbind_data structure containing particle data.
 * @param num The number of particles in the group.
 * @param num_non_gas Pointer to an integer to store the number of non-gas particles.
 * @return The updated number of particles after unbinding.
 */
int subfind_col_unbind(struct unbind_data *d, int num, int *num_non_gas)
{
    // int iter = 0;
    int p, part_index, minindex, task;
    int unbound, totunbound, numleft, mincpu;
    int *npart, *offset, *nbu_count, count_bound_unbound, phaseflag;
    double s[3], dx[3], ddxx, v[3], dv[3], sloc[3], vloc[3], pos[3];
    double vel_to_phys, H_of_a, atime;
    MyFloat minpot, *potlist;
    double boxsize;
    double mass, massloc;
    double *bnd_energy, energy_limit, energy_limit_local, weakly_bound_limit_local, weakly_bound_limit = 0;
    double mbs, glob_mbs;

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

    phaseflag = 0; /* this means we will recompute the potential for all particles */

    mbs = AllocatedBytes / (1024.0 * 1024.0);
    MPI_Allreduce(&mbs, &glob_mbs, 1, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);

    VERBOSE(2, "maximum alloacted %g MB", glob_mbs);

    do
    {
        force_treebuild(num, d);

        /* let's compute the potential energy */

        subfind_potential_compute(num, d, phaseflag, weakly_bound_limit);

        if (phaseflag == 0)
        {
            potlist = (MyFloat *)mymalloc("	  potlist", NTask * sizeof(MyFloat));
            minindex = -1, minpot = 1.0e30;
            for (int i = 0; i < num; i++)
            {
                if (P[d[i].index].u.DM_Potential < minpot || minindex == -1)
                {
                    minpot = P[d[i].index].u.DM_Potential;
                    minindex = d[i].index;
                }
            }

            MPI_Allgather(&minpot, sizeof(MyFloat), MPI_BYTE, potlist, sizeof(MyFloat), MPI_BYTE, MYMPI_COMM_WORLD);
            mincpu = -1, minpot = 1.0e30;
            for (int i = 0; i < NTask; i++)
                if (potlist[i] < minpot)
                {
                    mincpu = i;
                    minpot = potlist[i];
                }

            ENDRUN_IF(mincpu < 0, "mincpu less then zero.");

            myfree(potlist);

            if (ThisTask == mincpu)
            {
                for (int j = 0; j < 3; j++)
                {
                    pos[j] = P[minindex].Pos[j];
                }
            }

            MPI_Bcast(&pos[0], 3, MPI_DOUBLE, mincpu, MYMPI_COMM_WORLD);
            /* pos[] now holds the position of minimum potential */
            /* we take that as the center */
        }

        /* let's get bulk velocity and the center-of-mass */

        for (int j = 0; j < 3; j++)
        {
            sloc[j] = vloc[j] = 0;
        }

        massloc = 0;
        for (int i = 0; i < num; i++)
        {
            part_index = d[i].index;

            for (int j = 0; j < 3; j++)
            {
#ifdef PERIODIC
                ddxx = NEAREST(P[part_index].Pos[j] - pos[j]);
#else // PERIODIC
                ddxx = P[part_index].Pos[j] - pos[j];
#endif // PERIODIC
                sloc[j] += P[part_index].Mass * ddxx;
                vloc[j] += P[part_index].Mass * P[part_index].Vel[j];
            }
            massloc += P[part_index].Mass;
        }

        MPI_Allreduce(sloc, s, 3, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(vloc, v, 3, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&massloc, &mass, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

        for (int j = 0; j < 3; j++)
        {
            s[j] /= mass; /* center of mass */
            v[j] /= mass;

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

        bnd_energy = (double *)mymalloc("bnd_energy", num * sizeof(double));

        for (int i = 0; i < num; i++)
        {
            part_index = d[i].index;

            for (int j = 0; j < 3; j++)
            {
                dv[j] = vel_to_phys * (P[part_index].Vel[j] - v[j]);
#ifdef PERIODIC
                dx[j] = atime * NEAREST(P[part_index].Pos[j] - s[j]);
#else // PERIODIC
                dx[j] = atime * (P[part_index].Pos[j] - s[j]);
#endif // PERIODIC
                dv[j] += H_of_a * dx[j];
            }

            P[part_index].v.DM_BindingEnergy =
                P[part_index].u.DM_Potential + 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);

#ifdef DENSITY_SPLIT_BY_TYPE
            if (P[part_index].Type == 0)
            {
                P[part_index].v.DM_BindingEnergy += P[part_index].w.int_energy;
            }
#endif // DENSITY_SPLIT_BY_TYPE
            bnd_energy[i] = P[part_index].v.DM_BindingEnergy;
        }

        parallel_sort(bnd_energy, num, sizeof(double), subfind_compare_binding_energy);

        npart = (int *)mymalloc("npart", NTask * sizeof(int));
        nbu_count = (int *)mymalloc("nbu_count", NTask * sizeof(int));
        offset = (int *)mymalloc("offset", NTask * sizeof(int));

        MPI_Allgather(&num, 1, MPI_INT, npart, 1, MPI_INT, MYMPI_COMM_WORLD);
        MPI_Allreduce(&num, &numleft, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        offset[0] = 0;
        for (int i = 1; i < NTask; i++)
        {
            offset[i] = offset[i - 1] + npart[i - 1];
        }

        int j = (int)(0.25 * numleft); /* index of limiting energy value */

        task = 0;
        while (j >= npart[task])
        {
            j -= npart[task];
            task++;
        }

        if (ThisTask == task)
        {
            energy_limit_local = bnd_energy[j];
        }
        else
        {
            energy_limit_local = 1.0e30;
        }

        MPI_Allreduce(&energy_limit_local, &energy_limit, 1, MPI_DOUBLE, MPI_MIN, MYMPI_COMM_WORLD);
        count_bound_unbound = 0;
        for (int i = 0; i < num; i++)
        {
            if (bnd_energy[i] > 0)
            {
                count_bound_unbound++;
            }
            else
            {
                count_bound_unbound--;
            }
        }

        MPI_Allgather(&count_bound_unbound, 1, MPI_INT, nbu_count, 1, MPI_INT, MYMPI_COMM_WORLD);
        count_bound_unbound = 0;
        for (int i = 0; i < ThisTask; i++)
        {
            count_bound_unbound += nbu_count[i];
        }

        int bnd_energy_index = 0;

        for (int i = 0; i < num - 1; i++)
        {
            if (bnd_energy[i] > 0)
            {
                count_bound_unbound++;
            }
            else
            {
                count_bound_unbound--;
            }
            if (count_bound_unbound <= 0)
            {
                break;
            }
            bnd_energy_index++;
        }

        if (num > 0 && count_bound_unbound <= 0)
        {
            weakly_bound_limit_local = bnd_energy[bnd_energy_index];
        }
        else
        {
            weakly_bound_limit_local = -1.0e30;
        }

        MPI_Allreduce(&weakly_bound_limit_local, &weakly_bound_limit, 1, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);

        unbound = 0;
        for (int i = 0; i < num; i++)
        {
            p = d[i].index;

            if (P[p].v.DM_BindingEnergy > 0 && P[p].v.DM_BindingEnergy > energy_limit)
            {
                unbound++;

                d[i] = d[num - 1];
                num--;
                i--;
            }
            else if (P[p].Type != 0)
            {
                (*num_non_gas)++;
            }
        }

        myfree(offset);
        myfree(nbu_count);
        myfree(npart);
        myfree(bnd_energy);

        MPI_Allreduce(&unbound, &totunbound, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(&num, &numleft, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

        if (phaseflag == 0)
        {
            if (totunbound > 0)
            {
                phaseflag = 1;
            }
        }
        else
        {
            if (totunbound == 0)
            {
                phaseflag = 0; /* this will make us repeat everything once more for all particles */
                totunbound = 1;
            }
        }

        // iter++;
    } while (totunbound > 0 && numleft >= All.DesLinkNgb);

    return num;
}

/**
 * @brief Polls for and handles incoming MPI requests.
 *
 * This function continuously polls for incoming MPI requests from other tasks and processes them accordingly.
 * It handles various types of MPI tags to perform actions such as sending and receiving data related to substructures,
 * particle indices, ranks, and more. This function is crucial for communication and coordination among MPI tasks
 * during the substructure identification process.
 */
void subfind_poll_for_requests(void)
{
    int index, nsub, source, tag, ibuf[3], target, submark, task;
    long long head, next, rank, buf[5];
    long long oldtail, newtail;
    int task_newtail, i_newtail, task_oldtail, i_oldtail;
    MPI_Status status;

    do
    {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MYMPI_COMM_WORLD, &status);

        source = status.MPI_SOURCE;
        tag = status.MPI_TAG;

        /* MPI_Get_count(&status, MPI_BYTE, &count); */
        switch (tag)
        {
        case TAG_GET_TWOHEADS:
            MPI_Recv(ibuf, 2, MPI_INT, source, TAG_GET_TWOHEADS, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buf[0] = Head[ibuf[0]];
            buf[1] = Head[ibuf[1]];
            MPI_Send(buf, 2 * sizeof(long long), MPI_BYTE, source, TAG_GET_TWOHEADS_DATA, MYMPI_COMM_WORLD);
            break;
        case TAG_SET_NEWTAIL:
            MPI_Recv(buf, 2 * sizeof(long long), MPI_BYTE, source, TAG_SET_NEWTAIL, MYMPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            index = buf[0];
            newtail = buf[1];
            oldtail = Tail[index]; /* return old tail */
            Tail[index] = newtail;
            Len[index]++;

            task_newtail = (newtail >> 32);
            if (task_newtail == ThisTask)
            {
                i_newtail = (newtail & MASK);
                Head[i_newtail] = (((long long)ThisTask) << 32) + index;
                Next[i_newtail] = -1;
            }
            task_oldtail = (oldtail >> 32);
            if (task_oldtail == ThisTask)
            {
                i_oldtail = (oldtail & MASK);
                Next[i_oldtail] = newtail;
            }

            buf[0] = oldtail;
            MPI_Send(buf, 1 * sizeof(long long), MPI_BYTE, source, TAG_GET_OLDTAIL, MYMPI_COMM_WORLD);
            break;
        case TAG_SET_ALL:
            MPI_Recv(buf, 5 * sizeof(long long), MPI_BYTE, source, TAG_SET_ALL, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            index = buf[0];
            Head[index] = buf[1];
            Tail[index] = buf[2];
            Len[index] = buf[3];
            Next[index] = buf[4];
            break;
        case TAG_GET_TAILANDLEN:
            MPI_Recv(&index, 1, MPI_INT, source, tag, MYMPI_COMM_WORLD, &status);
            buf[0] = Tail[index];
            buf[1] = Len[index];
            MPI_Send(buf, 2 * sizeof(long long), MPI_BYTE, source, TAG_GET_TAILANDLEN_DATA, MYMPI_COMM_WORLD);
            break;
        case TAG_SET_TAILANDLEN:
            MPI_Recv(buf, 3 * sizeof(long long), MPI_BYTE, source, TAG_SET_TAILANDLEN, MYMPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            index = buf[0];
            Tail[index] = buf[1];
            Len[index] = buf[2];
            break;
        case TAG_SET_HEADANDNEXT:
            MPI_Recv(buf, 3 * sizeof(long long), MPI_BYTE, source, TAG_SET_HEADANDNEXT, MYMPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            index = buf[0];
            Head[index] = buf[1];
            Next[index] = buf[2];
            break;
        case TAG_SET_NEXT:
            MPI_Recv(buf, 2 * sizeof(long long), MPI_BYTE, source, TAG_SET_NEXT, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            index = buf[0];
            Next[index] = buf[1];
            break;
        case TAG_SETHEADGETNEXT:
            MPI_Recv(buf, 2 * sizeof(long long), MPI_BYTE, source, TAG_SETHEADGETNEXT, MYMPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            index = buf[0];
            head = buf[1];
            do
            {
                Head[index] = head;
                next = Next[index];
                task = (next >> 32);
                index = (next & MASK);
            } while (next >= 0 && task == ThisTask);
            MPI_Send(&next, 1 * sizeof(long long), MPI_BYTE, source, TAG_SETHEADGETNEXT_DATA, MYMPI_COMM_WORLD);
            break;
        case TAG_GET_NEXT:
            MPI_Recv(&index, 1, MPI_INT, source, tag, MYMPI_COMM_WORLD, &status);
            MPI_Send(&Next[index], 1 * sizeof(long long), MPI_BYTE, source, TAG_GET_NEXT_DATA, MYMPI_COMM_WORLD);
            break;
        case TAG_GET_HEAD:
            MPI_Recv(&index, 1, MPI_INT, source, tag, MYMPI_COMM_WORLD, &status);
            MPI_Send(&Head[index], 1 * sizeof(long long), MPI_BYTE, source, TAG_GET_HEAD_DATA, MYMPI_COMM_WORLD);
            break;
        case TAG_ADD_PARTICLE:
            MPI_Recv(&index, 1, MPI_INT, source, tag, MYMPI_COMM_WORLD, &status);
            if (Tail[index] < 0) /* consider only particles not already in substructures */
            {
                ud[LocalLen].index = index;

                ENDRUN_IF(index >= NumPartGroup, "What: index=%d NumPartGroup=%d\n", index, NumPartGroup);

                LocalLen++;
            }
            break;
        case TAG_MARK_PARTICLE:
            MPI_Recv(ibuf, 3, MPI_INT, source, TAG_MARK_PARTICLE, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            index = ibuf[0];
            target = ibuf[1];
            submark = ibuf[2];

            ENDRUN_IF(P[index].submark != HIGHBIT, "TasK=%d i=%d P[i].submark=%d?", ThisTask, index, P[index].submark);

            P[index].targettask = target;
            P[index].submark = submark;
            break;
        case TAG_ADDBOUND:
            MPI_Recv(ibuf, 2, MPI_INT, source, TAG_ADDBOUND, MYMPI_COMM_WORLD, &status);
            index = ibuf[0];
            nsub = ibuf[1];
            if (Tail[index] == nsub) /* consider only particles in this substructure */
            {
                ud[LocalLen].index = index;
                LocalLen++;
            }
            break;
        case TAG_SETRANK:
            MPI_Recv(buf, 2 * sizeof(long long), MPI_BYTE, source, TAG_SETRANK, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            index = buf[0];
            rank = buf[1];
            do
            {
                Len[index] = rank++;
                next = Next[index];
                if (next < 0)
                    break;
                index = (next & MASK);
            } while ((next >> 32) == ThisTask);
            buf[0] = next;
            buf[1] = rank;
            MPI_Send(buf, 2 * sizeof(long long), MPI_BYTE, source, TAG_SETRANK_OUT, MYMPI_COMM_WORLD);
            break;
        case TAG_GET_RANK:
            MPI_Recv(&index, 1, MPI_INT, source, tag, MYMPI_COMM_WORLD, &status);
            rank = Len[index];
            MPI_Send(&rank, 1 * sizeof(long long), MPI_BYTE, source, TAG_GET_RANK_DATA, MYMPI_COMM_WORLD);
            break;

        case TAG_POLLING_DONE:
            MPI_Recv(&index, 1, MPI_INT, source, tag, MYMPI_COMM_WORLD, &status);
            break;

        default:
            ENDRUN("Error in TAG.");
            break;
        }
    } while (tag != TAG_POLLING_DONE);
}

/**
 * @brief Sets the rank of a particle in the distributed linked list and retrieves the next particle's index.
 *
 * This function sets the rank of a particle in the distributed linked list identified by the given index.
 * If the current task is responsible for the particle, its rank is set and the index of the next particle is returned.
 * Otherwise, a message is sent to the appropriate task to set the rank, and the next particle's index is received
 * after the rank has been updated.
 *
 * @param index The index of the particle in the distributed linked list.
 * @param rank A pointer to the rank of the particle.
 * @return The index of the next particle in the linked list.
 */
long long subfind_distlinklist_setrank_and_get_next(long long index, long long *rank)
{
    long long next;
    long long buf[2];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        Len[i] = *rank;
        *rank = *rank + 1;
        next = Next[i];
    }
    else
    {
        buf[0] = i;
        buf[1] = *rank;

        MPI_Send(buf, 2 * sizeof(long long), MPI_BYTE, task, TAG_SETRANK, MYMPI_COMM_WORLD);
        MPI_Recv(buf, 2 * sizeof(long long), MPI_BYTE, task, TAG_SETRANK_OUT, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        next = buf[0];
        *rank = buf[1];
    }
    return next;
}

/**
 * @brief Sets the head of a particle in the distributed linked list and retrieves the next particle's index.
 *
 * This function sets the head of a particle in the distributed linked list identified by the given index.
 * If the current task is responsible for the particle, its head is set, and the index of the next particle is returned.
 * Otherwise, a message is sent to the appropriate task to set the head, and the index of the next particle is received
 * after the head has been updated.
 *
 * @param index The index of the particle in the distributed linked list.
 * @param head The head index to be set for the particle.
 * @return The index of the next particle in the linked list.
 */
long long subfind_distlinklist_set_head_get_next(long long index, long long head)
{
    long long buf[2];
    long long next;

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        Head[i] = head;
        next = Next[i];
    }
    else
    {
        buf[0] = i;
        buf[1] = head;
        MPI_Send(buf, 2 * sizeof(long long), MPI_BYTE, task, TAG_SETHEADGETNEXT, MYMPI_COMM_WORLD);
        MPI_Recv(&next, 1 * sizeof(long long), MPI_BYTE, task, TAG_SETHEADGETNEXT_DATA, MYMPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    return next;
}

/**
 * @brief Sets the next index for a particle in the distributed linked list.
 *
 * This function sets the next index for a particle in the distributed linked list identified by the given index.
 * If the current task is responsible for the particle, its next index is set directly.
 * Otherwise, a message is sent to the appropriate task to update the next index.
 *
 * @param index The index of the particle in the distributed linked list.
 * @param next The next index to be set for the particle.
 */
void subfind_distlinklist_set_next(long long index, long long next)
{
    long long buf[2];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        Next[i] = next;
    }
    else
    {
        buf[0] = i;
        buf[1] = next;
        MPI_Send(buf, 2 * sizeof(long long), MPI_BYTE, task, TAG_SET_NEXT, MYMPI_COMM_WORLD);
    }
}

/**
 * @brief Adds a particle to the distributed linked list.
 *
 * This function adds a particle to the distributed linked list if it is not already part of any substructures.
 * If the current task is responsible for the particle, it adds the particle directly to the local list.
 * Otherwise, a message is sent to the appropriate task to add the particle.
 *
 * @param index The index of the particle to be added to the distributed linked list.
 */
void subfind_distlinklist_add_particle(long long index)
{
    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        if (Tail[i] < 0) /* consider only particles not already in substructures */
        {
            ud[LocalLen].index = i;

            ENDRUN_IF(i >= NumPartGroup, "What: index=%d NumPartGroup=%d\n", i, NumPartGroup);

            LocalLen++;
        }
    }
    else
    {
        MPI_Send(&i, 1, MPI_INT, task, TAG_ADD_PARTICLE, MYMPI_COMM_WORLD);
    }
}

/**
 * @brief Marks a particle for substructure identification.
 *
 * This function marks a particle for substructure identification by setting its target task and submark.
 * If the current task is responsible for the particle, it updates the particle's properties directly.
 * Otherwise, a message is sent to the appropriate task to mark the particle.
 *
 * @param index   The index of the particle to be marked.
 * @param target  The target task responsible for identifying the substructure.
 * @param submark The submark value to assign to the particle.
 */
void subfind_distlinklist_mark_particle(long long index, int target, int submark)
{
    int ibuf[3];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        ENDRUN_IF(P[i].submark != HIGHBIT, "Task=%d i=%d P[i].submark=%d?\n", ThisTask, i, P[i].submark);

        P[i].targettask = target;
        P[i].submark = submark;
    }
    else
    {
        ibuf[0] = i;
        ibuf[1] = target;
        ibuf[2] = submark;
        MPI_Send(ibuf, 3, MPI_INT, task, TAG_MARK_PARTICLE, MYMPI_COMM_WORLD);
    }
}

/**
 * @brief Adds bound particles to the substructure with the given index.
 *
 * This function adds bound particles to the substructure with the specified index.
 * If the current task is responsible for the particles, it adds them directly to the local list.
 * Otherwise, a message is sent to the appropriate task to add the particles.
 *
 * @param index The index of the substructure to which bound particles are added.
 * @param nsub  The index of the substructure to consider for adding particles.
 */
void subfind_distlinklist_add_bound_particles(long long index, int nsub)
{
    int ibuf[2];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        if (Tail[i] == nsub) /* consider only particles not already in substructures */
        {
            ud[LocalLen].index = i;
            LocalLen++;
        }
    }
    else
    {
        ibuf[0] = i;
        ibuf[1] = nsub;
        MPI_Send(ibuf, 2, MPI_INT, task, TAG_ADDBOUND, MYMPI_COMM_WORLD);
    }
}

/**
 * @brief Gets the next particle index in the linked list.
 *
 * This function retrieves the index of the next particle in the linked list
 * for the given particle index. If the current task is responsible for the
 * particle, it directly retrieves the next index. Otherwise, a message is sent
 * to the appropriate task to fetch the next index.
 *
 * @param index The index of the current particle.
 *
 * @return The index of the next particle in the linked list.
 */
long long subfind_distlinklist_get_next(long long index)
{
    long long next;

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        next = Next[i];
    }
    else
    {
        MPI_Send(&i, 1, MPI_INT, task, TAG_GET_NEXT, MYMPI_COMM_WORLD);
        MPI_Recv(&next, 1 * sizeof(long long), MPI_BYTE, task, TAG_GET_NEXT_DATA, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return next;
}

/**
 * @brief Gets the rank of the particle in the linked list.
 *
 * This function retrieves the rank of the particle in the linked list,
 * indicating its position in the order of traversal. If the current task is
 * responsible for the particle, it directly retrieves the rank. Otherwise, a
 * message is sent to the appropriate task to fetch the rank.
 *
 * @param index The index of the particle.
 *
 * @return The rank of the particle in the linked list.
 */
long long subfind_distlinklist_get_rank(long long index)
{
    long long rank;

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        rank = Len[i];
    }
    else
    {
        MPI_Send(&i, 1, MPI_INT, task, TAG_GET_RANK, MYMPI_COMM_WORLD);
        MPI_Recv(&rank, 1 * sizeof(long long), MPI_BYTE, task, TAG_GET_RANK_DATA, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return rank;
}

/**
 * @brief Gets the head of the linked list associated with a particle.
 *
 * This function retrieves the head of the linked list associated with a
 * particle. If the current task is responsible for the particle, it directly
 * retrieves the head. Otherwise, a message is sent to the appropriate task to
 * fetch the head.
 *
 * @param index The index of the particle.
 *
 * @return The head of the linked list associated with the particle.
 */
long long subfind_distlinklist_get_head(long long index)
{
    long long head;

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        head = Head[i];
    }
    else
    {
        MPI_Send(&i, 1, MPI_INT, task, TAG_GET_HEAD, MYMPI_COMM_WORLD);
        MPI_Recv(&head, 1 * sizeof(long long), MPI_BYTE, task, TAG_GET_HEAD_DATA, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return head;
}

/**
 * @brief Gets the heads of the linked lists associated with two neighboring particles.
 *
 * This function retrieves the heads of the linked lists associated with two neighboring particles.
 * If the current task is responsible for the first particle, it directly retrieves the head of that particle.
 * Otherwise, a message is sent to the appropriate task to fetch the heads of both particles.
 *
 * @param ngb_index1 The index of the first neighboring particle.
 * @param ngb_index2 The index of the second neighboring particle.
 * @param head Pointer to store the head of the first linked list.
 * @param head_attach Pointer to store the head of the second linked list.
 */
void subfind_distlinklist_get_two_heads(long long ngb_index1, long long ngb_index2, long long *head,
                                        long long *head_attach)
{
    int ibuf[2];
    long long buf[2];

    int task = (ngb_index1 >> 32);
    int i1 = (ngb_index1 & MASK);
    int i2 = (ngb_index2 & MASK);

    if (ThisTask == task)
    {
        *head = Head[i1];
        *head_attach = Head[i2];
    }
    else
    {
        ibuf[0] = i1;
        ibuf[1] = i2;
        MPI_Send(ibuf, 2, MPI_INT, task, TAG_GET_TWOHEADS, MYMPI_COMM_WORLD);
        MPI_Recv(buf, 2 * sizeof(long long), MPI_BYTE, task, TAG_GET_TWOHEADS_DATA, MYMPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        *head = buf[0];
        *head_attach = buf[1];
    }
}

/**
 * @brief Sets the head and next pointers of a linked list associated with a particle.
 *
 * This function sets the head and next pointers of the linked list associated with a particle specified by the given
 * index. If the current task is responsible for the particle, it directly updates the head and next pointers.
 * Otherwise, a message is sent to the appropriate task to update the pointers.
 *
 * @param index The index of the particle whose linked list is being updated.
 * @param head The new head pointer for the linked list.
 * @param next The new next pointer for the linked list.
 */
void subfind_distlinklist_set_headandnext(long long index, long long head, long long next)
{
    long long buf[3];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        Head[i] = head;
        Next[i] = next;
    }
    else
    {
        buf[0] = i;
        buf[1] = head;
        buf[2] = next;
        MPI_Send(buf, 3 * sizeof(long long), MPI_BYTE, task, TAG_SET_HEADANDNEXT, MYMPI_COMM_WORLD);
    }
}

/**
 * @brief Gets the tail, sets a new tail, and increases the length of the linked list associated with a particle.
 *
 * This function gets the current tail, sets a new tail, and increases the length of the linked list associated with a
 * particle specified by the given index. If the current task is responsible for the particle, it directly updates the
 * tail, increases the length, and sets the head and next pointers if necessary. Otherwise, a message is sent to the
 * appropriate task to update the tail and retrieve the old tail.
 *
 * @param index The index of the particle whose linked list is being updated.
 * @param tail Pointer to store the old tail of the linked list.
 * @param newtail The new tail pointer for the linked list.
 * @return An integer indicating which operations were performed: 1 if the new tail is on the current task, 2 if the old
 * tail is on the current task, and 0 otherwise.
 */
int subfind_distlinklist_get_tail_set_tail_increaselen(long long index, long long *tail, long long newtail)
{
    int task_newtail, i_newtail, task_oldtail, i_oldtail;
    long long oldtail;
    long long buf[2];

    int task = (index >> 32);
    int i = (index & MASK);
    int retcode = 0;

    if (ThisTask == task)
    {
        oldtail = Tail[i];
        Tail[i] = newtail;
        Len[i]++;
        *tail = oldtail;

        task_newtail = (newtail >> 32);
        if (task_newtail == ThisTask)
        {
            i_newtail = (newtail & MASK);
            Head[i_newtail] = index;
            Next[i_newtail] = -1;
            retcode |= 1;
        }
        task_oldtail = (oldtail >> 32);
        if (task_oldtail == ThisTask)
        {
            i_oldtail = (oldtail & MASK);
            Next[i_oldtail] = newtail;
            retcode |= 2;
        }
    }
    else
    {
        buf[0] = i;
        buf[1] = newtail;
        MPI_Send(buf, 2 * sizeof(long long), MPI_BYTE, task, TAG_SET_NEWTAIL, MYMPI_COMM_WORLD);
        MPI_Recv(&oldtail, 1 * sizeof(long long), MPI_BYTE, task, TAG_GET_OLDTAIL, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *tail = oldtail;

        if ((newtail >> 32) == task)
        {
            retcode |= 1;
        }
        if ((oldtail >> 32) == task)
        {
            retcode |= 2;
        }
    }

    return retcode;
}

/**
 * @brief Sets the tail and length of the linked list associated with a particle.
 *
 * This function sets the tail and length of the linked list associated with a particle specified by the given index.
 * If the current task is responsible for the particle, it directly updates the tail and length.
 * Otherwise, a message is sent to the appropriate task to update the tail and length.
 *
 * @param index The index of the particle whose linked list tail and length are being updated.
 * @param tail The new tail pointer for the linked list.
 * @param len The new length of the linked list.
 */
void subfind_distlinklist_set_tailandlen(long long index, long long tail, int len)
{
    long long buf[3];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        Tail[i] = tail;
        Len[i] = len;
    }
    else
    {
        buf[0] = i;
        buf[1] = tail;
        buf[2] = len;
        MPI_Send(buf, 3 * sizeof(long long), MPI_BYTE, task, TAG_SET_TAILANDLEN, MYMPI_COMM_WORLD);
    }
}

/**
 * @brief Retrieves the tail and length of the linked list associated with a particle.
 *
 * This function retrieves the tail pointer and length of the linked list associated with a particle specified by the
 * given index. If the current task is responsible for the particle, it directly retrieves the tail and length.
 * Otherwise, a message is sent to the appropriate task to retrieve the tail and length.
 *
 * @param index The index of the particle whose linked list tail and length are being retrieved.
 * @param tail Pointer to store the retrieved tail pointer of the linked list.
 * @param len Pointer to store the retrieved length of the linked list.
 */
void subfind_distlinklist_get_tailandlen(long long index, long long *tail, int *len)
{
    long long buf[2];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        *tail = Tail[i];
        *len = Len[i];
    }
    else
    {
        MPI_Send(&i, 1, MPI_INT, task, TAG_GET_TAILANDLEN, MYMPI_COMM_WORLD);
        MPI_Recv(buf, 2 * sizeof(long long), MPI_BYTE, task, TAG_GET_TAILANDLEN_DATA, MYMPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        *tail = buf[0];
        *len = buf[1];
    }
}

/**
 * @brief Sets all properties of the linked list associated with a particle.
 *
 * This function sets all properties of the linked list associated with a particle specified by the given index.
 * It sets the head pointer, tail pointer, length, and next pointer of the linked list.
 * If the current task is responsible for the particle, it directly sets all properties.
 * Otherwise, a message is sent to the appropriate task to set all properties.
 *
 * @param index The index of the particle whose linked list properties are being set.
 * @param head The head pointer of the linked list.
 * @param tail The tail pointer of the linked list.
 * @param len The length of the linked list.
 * @param next The next pointer of the linked list.
 */
void subfind_distlinklist_set_all(long long index, long long head, long long tail, int len, long long next)
{
    long long buf[5];

    int task = (index >> 32);
    int i = (index & MASK);

    if (ThisTask == task)
    {
        Head[i] = head;
        Tail[i] = tail;
        Len[i] = len;
        Next[i] = next;
    }
    else
    {
        buf[0] = i;
        buf[1] = head;
        buf[2] = tail;
        buf[3] = len;
        buf[4] = next;
        MPI_Send(buf, 5 * sizeof(long long), MPI_BYTE, task, TAG_SET_ALL, MYMPI_COMM_WORLD);
    }
}

/**
 * @brief Compares two particle_data structures based on their group numbers and IDs.
 *
 * This function compares two particle_data structures based on their group numbers and IDs.
 * It calculates the absolute difference between the group number of each particle and a global group number (GrNr).
 * If the absolute difference is smaller for particle 'a' than for particle 'b', 'a' is considered smaller.
 * If the absolute difference is greater for particle 'a' than for particle 'b', 'a' is considered larger.
 * If the absolute differences are equal, the function compares the IDs of the particles.
 *
 * @param a Pointer to the first particle_data structure.
 * @param b Pointer to the second particle_data structure.
 * @return -1 if 'a' is smaller than 'b', +1 if 'a' is larger than 'b', and 0 if they are equal.
 */
int subfind_compare_P_GrNrGrNr(const void *a, const void *b)
{
    if (abs(((struct particle_data *)a)->GrNr - GrNr) < abs(((struct particle_data *)b)->GrNr - GrNr))
    {
        return -1;
    }

    if (abs(((struct particle_data *)a)->GrNr - GrNr) > abs(((struct particle_data *)b)->GrNr - GrNr))
    {
        return +1;
    }

    if (((struct particle_data *)a)->ID < ((struct particle_data *)b)->ID)
    {
        return -1;
    }

    if (((struct particle_data *)a)->ID > ((struct particle_data *)b)->ID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two particle_data structures based on their submark values.
 *
 * This function compares two particle_data structures based on their submark values.
 * It checks if the submark of particle 'a' is less than, greater than, or equal to the submark of particle 'b'.
 *
 * @param a Pointer to the first particle_data structure.
 * @param b Pointer to the second particle_data structure.
 * @return -1 if 'a' is smaller than 'b', +1 if 'a' is larger than 'b', and 0 if they are equal.
 */
int subfind_compare_P_submark(const void *a, const void *b)
{
    if (((struct particle_data *)a)->submark < ((struct particle_data *)b)->submark)
    {
        return -1;
    }

    if (((struct particle_data *)a)->submark > ((struct particle_data *)b)->submark)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two cand_dat structures based on their subnr values.
 *
 * This function compares two cand_dat structures based on their subnr values.
 * It checks if the subnr of candidate 'a' is less than, greater than, or equal to the subnr of candidate 'b'.
 *
 * @param a Pointer to the first cand_dat structure.
 * @param b Pointer to the second cand_dat structure.
 * @return -1 if 'a' is smaller than 'b', +1 if 'a' is larger than 'b', and 0 if they are equal.
 */
int subfind_compare_candidates_subnr(const void *a, const void *b)
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
 * @brief Compares two cand_dat structures based on their nsub values.
 *
 * This function compares two cand_dat structures based on their nsub values.
 * It checks if the nsub of candidate 'a' is less than, greater than, or equal to the nsub of candidate 'b'.
 *
 * @param a Pointer to the first cand_dat structure.
 * @param b Pointer to the second cand_dat structure.
 * @return -1 if 'a' is smaller than 'b', +1 if 'a' is larger than 'b', and 0 if they are equal.
 */
int subfind_compare_candidates_nsubs(const void *a, const void *b)
{
    if (((struct cand_dat *)a)->nsub < ((struct cand_dat *)b)->nsub)
    {
        return -1;
    }

    if (((struct cand_dat *)a)->nsub > ((struct cand_dat *)b)->nsub)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two cand_dat structures based on their bound_length and rank values.
 *
 * This function compares two cand_dat structures based on their bound_length and rank values.
 * It first compares the bound_length of candidates 'a' and 'b'. If they are not equal, it returns the result of this
 * comparison. If the bound_length values are equal, it then compares the rank of candidates 'a' and 'b'.
 *
 * @param a Pointer to the first cand_dat structure.
 * @param b Pointer to the second cand_dat structure.
 * @return -1 if 'a' is smaller than 'b', +1 if 'a' is larger than 'b', and 0 if they are equal.
 */
int subfind_compare_candidates_boundlength(const void *a, const void *b)
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
 * @brief Compares two cand_dat structures based on their rank and len values.
 *
 * This function compares two cand_dat structures based on their rank and len values.
 * It first compares the rank of candidates 'a' and 'b'. If they are not equal, it returns the result of this
 * comparison. If the rank values are equal, it then compares the len of candidates 'a' and 'b'.
 *
 * @param a Pointer to the first cand_dat structure.
 * @param b Pointer to the second cand_dat structure.
 * @return -1 if 'a' is smaller than 'b', +1 if 'a' is larger than 'b', and 0 if they are equal.
 */
int subfind_compare_candidates_rank(const void *a, const void *b)
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
 * @brief Compares two sort_r2list structures based on their radial distance 'r'.
 *
 * This function compares two sort_r2list structures based on their radial distance 'r'.
 * It first compares the radial distance of objects 'a' and 'b'.
 *
 * @param a Pointer to the first sort_r2list structure.
 * @param b Pointer to the second sort_r2list structure.
 * @return -1 if 'a' is smaller than 'b', +1 if 'a' is larger than 'b', and 0 if they are equal.
 */
int subfind_compare_dist_rotcurve(const void *a, const void *b)
{
    if (((sort_r2list *)a)->r < ((sort_r2list *)b)->r)
    {
        return -1;
    }

    if (((sort_r2list *)a)->r > ((sort_r2list *)b)->r)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two double values representing binding energies.
 *
 * This function compares two double values representing binding energies.
 * It first compares the binding energy of objects 'a' and 'b'.
 *
 * @param a Pointer to the first double value representing binding energy.
 * @param b Pointer to the second double value representing binding energy.
 * @return -1 if the binding energy of 'a' is greater than 'b', +1 if the binding energy of 'a' is smaller than 'b', and
 * 0 if they are equal.
 */
int subfind_compare_binding_energy(const void *a, const void *b)
{
    if (*((double *)a) > *((double *)b))
    {
        return -1;
    }

    if (*((double *)a) < *((double *)b))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two density values for sorting.
 *
 * This function compares two density values for sorting. It sorts in descending order,
 * with the largest density values coming first.
 *
 * @param a Pointer to the first density data structure.
 * @param b Pointer to the second density data structure.
 * @return -1 if the density of 'a' is greater than 'b', +1 if the density of 'a' is smaller than 'b', and 0 if they are
 * equal.
 */
int subfind_compare_densities(const void *a, const void *b) /* largest density first */
{
    if (((struct sort_density_data *)a)->density > (((struct sort_density_data *)b)->density))
    {
        return -1;
    }

    if (((struct sort_density_data *)a)->density < (((struct sort_density_data *)b)->density))
    {
        return +1;
    }

    return 0;
}

#endif // SUBFIND_ALTERNATIVE_COLLECTIVE
#endif // SUBFIND
