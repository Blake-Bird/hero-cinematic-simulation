/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Alternative collection functions for Subfind.
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

#ifdef SUBFIND
#ifdef SUBFIND_ALTERNATIVE_COLLECTIVE
#include "subfind.hpp"
#include "fof.hpp"

#define MASK ((((long long)1) << 32) - 1)
#define HIGHBIT (1 << 30)

static long long **Head, **Next, **Tail;
static int **Len;
static int LocalLen;
static int *count_cand, max_candidates, loc_count_cand;
static int TotalGroupLen;
static int *List_NumPartGroup, *Offset_NumPartGroup, *List_LocalLen;

static int **targettask, **submark, **origintask;

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
} **candidates, *loc_candidates;

int subfind_mark_independent_ones(void);

static struct unbind_data *ud, **ud_list;

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
    long long p;
    int len, len_non_gas, LocalNonGasLen, totgrouplen1, totgrouplen2;
    int parent, totcand, nremaining;
    int max_length;
    int countall;
    int i, j, k, nr, grindex = 0, nsubs = 0, subnr;
    int tot_count_leaves;
    int task;
    double SubMass, SubPos[3], SubVel[3], SubCM[3], SubVelDisp, SubVmax;
    double SubVmaxRad, SubSpin[3], SubHalfMass, SubStelHalfMass, SubMassTab[6];
#ifdef LT_ADD_GAL_TO_SUB
    double SubLum[LT_ADD_GAL_TO_SUB], SubLumAtt[LT_ADD_GAL_TO_SUB], SubLumObs[LT_ADD_GAL_TO_SUB], SubMeanAge,
        SubMeanZStar, SubSFR;
#ifdef DUSTATT
    double Dust[DUSTATT];
#else // DUSTATT
    double Dust[1];
#endif // DUSTATT
#endif // LT_ADD_GAL_TO_SUB
    struct cand_dat *tmp_candidates = 0;
    MyIDType SubMostBoundID;
    double t0, t1, tt0, tt1, ttt0, ttt1;

    ttt0 = second();

    VERBOSE(2, "\ncollectively doing halo %d (alternative subfind_collective)", GrNr);
    NumPartGroup = 0;
    for (ind i = 0; i < NumPart; i++)
    {
        if (P[i].GrNr == GrNr)
        {
            NumPartGroup++;
        }
    }

    MPI_Allreduce(&NumPartGroup, &totgrouplen1, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    if (ThisTask == ((GrNr - 1) % NTask))
    {
        for (grindex = 0; grindex < Ngroups; grindex++)
            if (Group[grindex].GrNr == GrNr)
            {
                break;
            }

        ENDRUN_IF(grindex >= Ngroups, "gridindex larger than Ngroups");

        totgrouplen2 = Group[grindex].Len;
    }

    MPI_Bcast(&totgrouplen2, 1, MPI_INT, (GrNr - 1) % NTask, MYMPI_COMM_WORLD);

    ENDRUN_IF(totgrouplen1 != totgrouplen2, "Inconsistency in len of total number of groups.");

    TotalGroupLen = totgrouplen1;

    /* distribute this halo among the processors */
    t0 = second();

    All.DoDynamicUpdate = 0;

    domain_free_trick();

    domain_Decomposition(1, 0);

    t1 = second();
    VERBOSE(2, "coldomain_Decomposition() took %g sec  (presently allocated=%g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    for (int i = 0; i < NumPart; i++)
    {
        P[i].origindex = i;
    }

    qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNrGrNr);

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

    /* determine the radius that encloses a certain number of link particles */
    t0 = second();
    subfind_find_linkngb();
    t1 = second();
    VERBOSE(2, "find_linkngb() took %g sec", timediff(t0, t1));

    List_NumPartGroup = mymalloc("List_NumPartGroup", NTask * sizeof(int));
    Offset_NumPartGroup = mymalloc("Offset_NumPartGroup", NTask * sizeof(int));

    MPI_Allgather(&NumPartGroup, 1, MPI_INT, List_NumPartGroup, 1, MPI_INT, MYMPI_COMM_WORLD);
    Offset_NumPartGroup[0] = 0;
    for (int task = 1; task < NTask; task++)
    {
        Offset_NumPartGroup[task] = Offset_NumPartGroup[task - 1] + List_NumPartGroup[task - 1];
    }

    max_candidates = max((TotalGroupLen / NTask / 50), 200);

    if (ThisTask == 0)
    {
        /* allocate a list to store subhalo candidates */

        count_cand = mymalloc("count_cand", NTask * sizeof(int));
        candidates = mymalloc("candidates", NTask * sizeof(struct cand_dat *));

        for (int i = 0; i < NTask; i++)
        {
            count_cand[i] = 0;
            candidates[i] = mymalloc("	  candidates[i]", max_candidates * sizeof(struct cand_dat));
        }

        /* allocate and initialize global link list on CPU0 */
        Next = mymalloc("Next", NTask * sizeof(long long *));
        Len = mymalloc("Len", NTask * sizeof(int *));

        for (int i = 0; i < NTask; i++)
        {
            Next[i] = mymalloc("	  Next[i]", List_NumPartGroup[i] * sizeof(long long));
            Len[i] = mymalloc("	  Len[i]", List_NumPartGroup[i] * sizeof(int));
        }

        Head = mymalloc("Head", NTask * sizeof(long long *));
        Tail = mymalloc("Tail", NTask * sizeof(long long *));

        for (int i = 0; i < NTask; i++)
        {
            Head[i] = mymalloc("	  Head[i]", List_NumPartGroup[i] * sizeof(long long));
            Tail[i] = mymalloc("	  Tail[i]", List_NumPartGroup[i] * sizeof(long long));
        }
    }

    VERBOSE(2, "Link-list allocated on CPU 0: presently allocated there=%g MB", AllocatedBytes / (1024.0 * 1024.0));

    if (ThisTask == 0)
    {
        sd = mymalloc("sd", TotalGroupLen * sizeof(struct sort_density_data));
    }
    else
    {
        sd = mymalloc("sd", NumPartGroup * sizeof(struct sort_density_data));
    }

    /* determine the indices of the nearest two denser neighbours within the link region */
    t0 = second();
    NgbLoc = mymalloc("NgbLoc", NumPartGroup * sizeof(struct nearest_ngb_data));
    R2Loc = mymalloc("R2Loc", NumPartGroup * sizeof(struct nearest_r2_data));

    VERBOSE(2, "sd-global allocated on CPU 0: presently allocated there=%g MB", AllocatedBytes / (1024.0 * 1024.0));

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

    t0 = second();
    /* now get all the other sd's from other CPUs */
    if (ThisTask == 0)
    {
        for (task = 1; task < NTask; task++)
        {
            MPI_Send(&task, 1, MPI_INT, task, TAG_N, MYMPI_COMM_WORLD);
            MPI_Recv(&sd[Offset_NumPartGroup[task]], List_NumPartGroup[task] * sizeof(struct sort_density_data),
                     MPI_BYTE, task, TAG_PDATA, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        MPI_Recv(&i, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(sd, NumPartGroup * sizeof(struct sort_density_data), MPI_BYTE, 0, TAG_PDATA, MYMPI_COMM_WORLD);
    }
    t1 = second();
    VERBOSE(2, "global sd list assembled on task 0 took %g sec", timediff(t0, t1));

    MPI_Barrier(MYMPI_COMM_WORLD);

    if (ThisTask == 0)
    {
        t0 = second();
        for (int i = 0; i < NTask; i++)
        {
            for (int j = 0; j < List_NumPartGroup[i]; j++)
            {
                Head[i][j] = Next[i][j] = Tail[i][j] = -1;
                Len[i][j] = 0;
            }
        }
        t1 = second();
        VERBOSE(2, "initializing link list took %g sec", timediff(t0, t1));
    }

    VERBOSE(2, "before col_find_candidates on CPU 0: presently allocated there=%g MB",
            AllocatedBytes / (1024.0 * 1024.0));

    if (ThisTask == 0)
    {
        t0 = second();

        subfind_col_find_candidates(TotalGroupLen);

        t1 = second();

        VERBOSE(2, "finding of candidates on task 0 took %g sec", timediff(t0, t1));
    }
    myfree(sd);

    /* establish total number of candidates */
    totcand = 0;
    if (ThisTask == 0)
    {
        for (int i = 0; i < NTask; i++)
        {
            totcand += count_cand[i];
        }
    }

    MPI_Bcast(&totcand, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    VERBOSE(2, "\ntotal number of subhalo candidates=%d", totcand);

    nremaining = totcand;

    if (ThisTask == 0)
    {
        for (int i = NTask - 1; i >= 0; i--)
        {
            myfree(Tail[i]);
            myfree(Head[i]);
        }

        myfree(Tail);
        myfree(Head);
    }

    if (ThisTask == 0)
    {
        for (task = 0; task < NTask; task++)
        {
            for (int i = 0; i < List_NumPartGroup[task]; i++)
            {
                Len[task][i] = -1;
            }

            for (int i = 0; i < count_cand[task]; i++)
            {
                candidates[task][i].parent = 0;
            }
        }
    }

    do
    {
        /* Let's see which candidates can be unbound independent from each other.
           We identify them with those candidates that have no embedded other candidate */

        t0 = second();

        if (ThisTask == 0)
        {
            tmp_candidates = mymalloc("	  tmp_candidates", totcand * sizeof(struct cand_dat));
            j = 0;
            for (int task = 0; task < NTask; task++)
            {
                for (int i = 0; i < count_cand[task]; i++)
                {
                    tmp_candidates[j++] = candidates[task][i];
                }
            }

            for (int k = 0; k < totcand; k++)
            {
                tmp_candidates[k].nsub = k;
                tmp_candidates[k].subnr = k;
            }

            qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_rank);

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
                            ENDRUN_MAIN("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d\n", k, totcand,
                                        (int)tmp_candidates[k].rank, (int)tmp_candidates[k].len, j,
                                        (int)tmp_candidates[j].rank, (int)tmp_candidates[j].len);
                        }
                    }
                }
            }
            qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_subnr);
            j = 0;
            for (int task = 0; task < NTask; task++)
            {
                for (int i = 0; i < count_cand[task]; i++)
                {
                    candidates[task][i] = tmp_candidates[j++];
                }
            }

            myfree(tmp_candidates);

            tot_count_leaves = 0, max_length = 0;
            for (int task = 0; task < NTask; task++)
                for (int i = 0; i < count_cand[task]; i++)
                    if (candidates[task][i].parent == 0)
                    {
                        if (candidates[task][i].len > max_length)
                        {
                            max_length = candidates[task][i].len;
                        }

                        if (candidates[task][i].len >
                            0.15 * All.TotNumPart / NTask) /* seems large, let's rather do it collectively */
                        {
                            candidates[task][i].parent++; /* this will ensure that it is not considered in this round */
                        }
                        else
                        {
                            tot_count_leaves++;
                        }
                    }
        }

        MPI_Bcast(&tot_count_leaves, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(&max_length, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

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
        }

        if (ThisTask == 0)
        {
            t0 = second();

            origintask = mymalloc("	  origintask", NTask * sizeof(int *));
            targettask = mymalloc("	  targettask", NTask * sizeof(int *));
            submark = mymalloc("	  submark", NTask * sizeof(int *));

            for (int task = 0; task < NTask; task++)
            {
                origintask[task] = mymalloc("	      origintask[task]", List_NumPartGroup[task] * sizeof(int));
                targettask[task] = mymalloc("	      targettask[task]", List_NumPartGroup[task] * sizeof(int));
                submark[task] = mymalloc("	      submark[task]", List_NumPartGroup[task] * sizeof(int));
            }

            VERBOSE(2, "before submark on CPU 0: presently allocated there=%g MB\n",
                    AllocatedBytes / (1024.0 * 1024.0));

            for (int task = 0; task < NTask; task++)
            {
                for (int i = 0; i < List_NumPartGroup[task]; i++)
                {
                    submark[task][i] = HIGHBIT;
                    origintask[task][i] = targettask[task][i] = task;

                    if (Len[task][i] >= 0) /* this means this particle is already bound to a substructure */
                    {
                        origintask[task][i] |= HIGHBIT;
                    }
                }
            }

            /* we now mark the particles that are in subhalo candidates that can be processed independently in parallel
             */

            nsubs = subfind_mark_independent_ones();

            t1 = second();

            VERBOSE(2, "particles are marked (took %g)", timediff(t0, t1));
        }

        if (ThisTask == 0)
        {
            t0 = second();

            for (int i = 0; i < NumPartGroup; i++)
            {
                P[i].origintask = origintask[0][i];
                P[i].targettask = targettask[0][i];
                P[i].submark = submark[0][i];
            }

            for (int task = 1; task < NTask; task++)
            {
                MPI_Send(origintask[task], List_NumPartGroup[task], MPI_INT, task, TAG_FOF_C, MYMPI_COMM_WORLD);
                MPI_Send(targettask[task], List_NumPartGroup[task], MPI_INT, task, TAG_FOF_A, MYMPI_COMM_WORLD);
                MPI_Send(submark[task], List_NumPartGroup[task], MPI_INT, task, TAG_FOF_B, MYMPI_COMM_WORLD);
            }

            t1 = second();

            VERBOSE(2, "distributing marks (took %g)", timediff(t0, t1));

            for (int task = NTask - 1; task >= 0; task--)
            {
                myfree(submark[task]);
                myfree(targettask[task]);
                myfree(origintask[task]);
            }

            myfree(submark);
            myfree(targettask);
            myfree(origintask);
        }
        else
        {
            int *origin, *target, *mark;

            origin = mymalloc("	  origin", NumPartGroup * sizeof(int));
            target = mymalloc("	  target", NumPartGroup * sizeof(int));
            mark = mymalloc("	  mark", NumPartGroup * sizeof(int));

            MPI_Recv(origin, NumPartGroup, MPI_INT, 0, TAG_FOF_C, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(target, NumPartGroup, MPI_INT, 0, TAG_FOF_A, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(mark, NumPartGroup, MPI_INT, 0, TAG_FOF_B, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < NumPartGroup; i++)
            {
                P[i].origintask = origin[i];
                P[i].targettask = target[i];
                P[i].submark = mark[i];
            }

            myfree(mark);
            myfree(target);
            myfree(origin);
        }

        t0 = second();
        subfind_distribute_particles(1); /* assemble the particles on individual processors */
        t1 = second();

        VERBOSE(
            2,
            "independent subhalos are assembled on individual CPUs for unbinding (%g sec, (presently allocated=%g MB)",
            timediff(t0, t1), AllocatedBytes / (1024.0 * 1024.0));

        qsort(P, NumPart, sizeof(struct particle_data),
              subfind_compare_P_submark); /* groups particles of the same canidate together */

        MPI_Barrier(MYMPI_COMM_WORLD);
        t0 = second();

        if (ThisTask == 0)
        {
            loc_count_cand = count_cand[0];
            loc_candidates = candidates[0];
            for (task = 1; task < NTask; task++)
            {
                MPI_Send(&count_cand[task], 1, MPI_INT, task, TAG_N, MYMPI_COMM_WORLD);
                MPI_Send(candidates[task], count_cand[task] * sizeof(struct cand_dat), MPI_BYTE, task, TAG_FOF_C,
                         MYMPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Recv(&loc_count_cand, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            loc_candidates = mymalloc("	  loc_candidates", loc_count_cand * sizeof(struct cand_dat));
            MPI_Recv(loc_candidates, loc_count_cand * sizeof(struct cand_dat), MPI_BYTE, 0, TAG_FOF_C, MYMPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        subfind_unbind_independent_ones(loc_count_cand);

        if (ThisTask == 0)
        {
            count_cand[0] = loc_count_cand;

            for (int task = 1; task < NTask; task++)
            {
                MPI_Send(&task, 1, MPI_INT, task, TAG_N, MYMPI_COMM_WORLD);
                MPI_Recv(&count_cand[task], 1, MPI_INT, task, TAG_FOF_E, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(candidates[task], count_cand[task] * sizeof(struct cand_dat), MPI_BYTE, task, TAG_FOF_D,
                         MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else
        {
            MPI_Recv(&i, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&loc_count_cand, 1, MPI_INT, 0, TAG_FOF_E, MYMPI_COMM_WORLD);
            MPI_Send(loc_candidates, loc_count_cand * sizeof(struct cand_dat), MPI_BYTE, 0, TAG_FOF_D,
                     MYMPI_COMM_WORLD);
            myfree(loc_candidates);
        }

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
        qsort(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNrGrNr);

        /* now mark the bound particles */
        if (ThisTask == 0)
        {
            for (int i = 0; i < NumPartGroup; i++)
            {
                if (P[i].submark >= 0 && P[i].submark < nsubs)
                {
                    Len[0][i] = P[i].submark; /* we use this to flag bound parts of substructures */
                }
            }
            for (int task = 1; task < NTask; task++)
            {
                MPI_Send(&task, 1, MPI_INT, task, TAG_N, MYMPI_COMM_WORLD);

                int *submark = mymalloc("	      int *submark", List_NumPartGroup[task] * sizeof(int));

                MPI_Recv(submark, List_NumPartGroup[task], MPI_INT, task, TAG_FOF_E, MYMPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                for (int i = 0; i < List_NumPartGroup[task]; i++)
                {
                    if (submark[i] >= 0 && submark[i] < nsubs)
                    {
                        Len[task][i] = submark[i];
                    }
                }

                myfree(submark);
            }
        }
        else
        {
            int *submark = mymalloc("	  int *submark", NumPartGroup * sizeof(int));

            for (int i = 0; i < NumPartGroup; i++)
            {
                submark[i] = P[i].submark;
            }

            MPI_Recv(&i, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Send(submark, NumPartGroup, MPI_INT, 0, TAG_FOF_E, MYMPI_COMM_WORLD);

            myfree(submark);
        }
    } while (tot_count_leaves > 0);

    force_treebuild(NumPartGroup, NULL); /* re construct the tree for the collective part */

    /**** now we do the collective unbinding of the subhalo candidates that contain other subhalo candidates ****/
    ud = mymalloc("ud", NumPartGroup * sizeof(struct unbind_data));

    t0 = second();

    if (ThisTask == 0)
    {
        List_LocalLen = mymalloc("List_LocalLen", NTask * sizeof(int));
        ud_list = mymalloc("ud_list", NTask * sizeof(struct unbind_data *));

        for (int i = 0; i < NTask; i++)
        {
            ud_list[i] = mymalloc("	ud_list[i]", List_NumPartGroup[i] * sizeof(struct unbind_data));
        }

        nr = 0;
        for (int task = 0; task < NTask; task++)
        {
            for (int k = 0; k < count_cand[task]; k++)
            {
                len = candidates[task][k].len;
                nsubs = candidates[task][k].nsub;
                parent = candidates[task][k].parent; /* this is here actually the daughter count */

                if (parent >= 0)
                {
                    VERBOSE_ALL(2, "collective unbinding of nr=%d (%d) of length=%d ... ", nr, nremaining, (int)len);

                    nr++;

                    for (int i = 0; i < NTask; i++)
                    {
                        List_LocalLen[i] = 0;
                    }

                    p = candidates[task][k].head;
                    for (int i = 0; i < candidates[task][k].len; i++)
                    {
                        subfind_distlinklist_add_particle(p);
                        if (p < 0)
                        {
                            ENDRUN("Bummer i=%d ", i);
                        }
                        p = subfind_distlinklist_get_next(p);
                    }

                    /* inform the others */

                    for (int i = 1; i < NTask; i++)
                    {
                        MPI_Send(&List_LocalLen[i], 1, MPI_INT, i, TAG_N, MYMPI_COMM_WORLD);
                        MPI_Send(ud_list[i], List_LocalLen[i] * sizeof(struct unbind_data), MPI_BYTE, i, TAG_FOF_A,
                                 MYMPI_COMM_WORLD);
                    }
                    LocalLen = List_LocalLen[0];
                    memcpy(ud, ud_list[0], LocalLen * sizeof(struct unbind_data));

                    tt0 = second();

                    LocalLen = subfind_col_unbind(ud, LocalLen, &LocalNonGasLen);
                    MPI_Allreduce(&LocalLen, &len, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

                    tt1 = second();

                    VERBOSE(2, "took %g sec", timediff(tt0, tt1));

#ifndef DO_GAS_CLOUDS
                    MPI_Allreduce(&LocalNonGasLen, &len_non_gas, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
                    if (len_non_gas >= All.DesLinkNgb)
#else // DO_GAS_CLOUDS
                    if (len >= All.DesLinkNgb)
#endif // DO_GAS_CLOUDS
                    {
                        /* ok, we found a substructure */

                        List_LocalLen[0] = LocalLen;
                        memcpy(ud_list[0], ud, LocalLen * sizeof(struct unbind_data));

                        for (int i = 1; i < NTask; i++)
                        {
                            MPI_Recv(&List_LocalLen[i], 1, MPI_INT, i, TAG_HEADER, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            MPI_Recv(ud_list[i], List_LocalLen[i] * sizeof(struct unbind_data), MPI_BYTE, i, TAG_FOF_B,
                                     MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }

                        for (int i = 0; i < NTask; i++)
                        {
                            for (int j = 0; j < List_LocalLen[i]; j++)
                            {
                                Len[i][ud_list[i][j].index] = nsubs; /* we use this to flag the substructures */
                            }
                        }

                        candidates[task][k].bound_length = len;
                    }
                    else
                    {
                        candidates[task][k].bound_length = 0;
                    }
                }
            }
        }

        parent = -1;
        /* inform the others that we are done */

        for (int i = 1; i < NTask; i++)
        {
            MPI_Send(&parent, 1, MPI_INT, i, TAG_N, MYMPI_COMM_WORLD);
        }

        for (int i = NTask - 1; i >= 0; i--)
        {
            myfree(ud_list[i]);
        }

        myfree(ud_list);

        myfree(List_LocalLen);
    }
    else
    {
        do
        {
            MPI_Recv(&LocalLen, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (LocalLen >= 0)
            {
                MPI_Recv(ud, LocalLen * sizeof(struct unbind_data), MPI_BYTE, 0, TAG_FOF_A, MYMPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                LocalLen = subfind_col_unbind(ud, LocalLen, &LocalNonGasLen);

                MPI_Allreduce(&LocalLen, &len, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

#ifndef DO_GAS_CLOUDS
                MPI_Allreduce(&LocalNonGasLen, &len_non_gas, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
                if (len_non_gas >= All.DesLinkNgb)
#else // DO_GAS_CLOUDS
                if (len >= All.DesLinkNgb)
#endif // DO_GAS_CLOUDS
                {
                    MPI_Send(&LocalLen, 1, MPI_INT, 0, TAG_HEADER, MYMPI_COMM_WORLD);
                    MPI_Send(ud, LocalLen * sizeof(struct unbind_data), MPI_BYTE, 0, TAG_FOF_B, MYMPI_COMM_WORLD);
                }
            }
        } while (LocalLen >= 0);
    }

    t1 = second();
    VERBOSE("\nthe collective unbinding of remaining halos took %g sec", timediff(t0, t1));

    /********************/
    if (ThisTask == 0)
    {
        countall = 0;
        for (int task = 0; task < NTask; task++)
        {
            for (int k = 0; k < count_cand[task]; k++)
            {
                if (candidates[task][k].bound_length >= All.DesLinkNgb)
                {
                    countall++;
                }
            }
        }

        VERBOSE_ALL(2, "\nfound %d bound substructures in FoF group of length %d", countall, totgrouplen1);
    }

    MPI_Bcast(&countall, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    /* now determine the parent subhalo for each candidate */
    t0 = second();
    if (ThisTask == 0)
    {
        tmp_candidates = mymalloc("tmp_candidates", totcand * sizeof(struct cand_dat));
        j = 0;
        for (int task = 0; task < NTask; task++)
        {
            for (int i = 0; i < count_cand[task]; i++)
            {
                tmp_candidates[j++] = candidates[task][i];
            }
        }

        qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_boundlength);

        for (int k = 0; k < totcand; k++)
        {
            tmp_candidates[k].subnr = k;
            tmp_candidates[k].parent = 0;
        }

        qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_rank);

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
                        tmp_candidates[j].parent = tmp_candidates[k].subnr;
                }
                else
                {
                    ENDRUN("k=%d|%d has rank=%d and len=%d.  j=%d has rank=%d and len=%d bound=%d", k, countall,
                           (int)tmp_candidates[k].rank, (int)tmp_candidates[k].len, (int)tmp_candidates[k].bound_length,
                           (int)tmp_candidates[j].rank, (int)tmp_candidates[j].len,
                           (int)tmp_candidates[j].bound_length);
                }
            }
        }

        qsort(tmp_candidates, totcand, sizeof(struct cand_dat), subfind_compare_candidates_subnr);

        j = 0;
        for (int task = 0; task < NTask; task++)
        {
            for (int i = 0; i < count_cand[task]; i++)
            {
                candidates[task][i] = tmp_candidates[j++];
            }
        }

        myfree(tmp_candidates);
    }

    t1 = second();
    VERBOSE(2, "determination of parent subhalo took %g sec (presently allocated %g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    /* Now let's save  some properties of the substructures */
    if (ThisTask == ((GrNr - 1) % NTask))
    {
        Group[grindex].Nsubs = countall;
    }

    /************** determine properties ***************/
    t0 = second();

    if (ThisTask == 0)
    {
        List_LocalLen = mymalloc("List_LocalLen", NTask * sizeof(int));
        ud_list = mymalloc("ud_list", NTask * sizeof(struct unbind_data *));

        for (int i = 0; i < NTask; i++)
        {
            ud_list[i] = mymalloc("	ud_list[i]", List_NumPartGroup[i] * sizeof(struct unbind_data));
        }

        subnr = 0;
        for (int task = 0; task < NTask; task++)
        {
            /*
               printf("task=%d...", task); fflush(stdout);
             */

            /*      double ttt0 = second();
             */

            for (int k = 0; k < count_cand[task]; k++)
            {
                len = candidates[task][k].bound_length;
                nsubs = candidates[task][k].nsub;
                parent = candidates[task][k].parent;

                if (len > 0)
                {
                    for (int i = 0; i < NTask; i++)
                    {
                        List_LocalLen[i] = 0;
                    }

                    p = candidates[task][k].head;
                    for (int i = 0; i < candidates[task][k].len; i++)
                    {
                        subfind_distlinklist_add_bound_particles(p, nsubs);
                        p = subfind_distlinklist_get_next(p);
                    }

                    /* inform the others */

                    for (int i = 1; i < NTask; i++)
                    {
                        MPI_Send(&List_LocalLen[i], 1, MPI_INT, i, TAG_N, MYMPI_COMM_WORLD);
                        MPI_Send(&len, 1, MPI_INT, i, TAG_HMAX, MYMPI_COMM_WORLD);
                        MPI_Send(&parent, 1, MPI_INT, i, TAG_SPHDATA, MYMPI_COMM_WORLD);
                        MPI_Send(ud_list[i], List_LocalLen[i] * sizeof(struct unbind_data), MPI_BYTE, i, TAG_FOF_A,
                                 MYMPI_COMM_WORLD);
                    }
                    LocalLen = List_LocalLen[0];
                    memcpy(ud, ud_list[0], LocalLen * sizeof(struct unbind_data));

                    tt0 = second();
#ifdef LT_ADD_GAL_TO_SUB
                    subfind_col_determine_sub_halo_properties(
                        ud, LocalLen, &SubMass, &SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp, &SubVmax, &SubVmaxRad,
                        &SubSpin[0], &SubMostBoundID, &SubHalfMass, &SubStelHalfMass, &SubMassTab[0], &SubLum[0],
                        &SubLumAtt[0], &SubLumObs[0], &Dust[0].& SubMeanAge, &SubMeanZStar, &SubSFR);
#else // LT_ADD_GAL_TO_SUB
                    subfind_col_determine_sub_halo_properties(
                        ud, LocalLen, &SubMass, &SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp, &SubVmax, &SubVmaxRad,
                        &SubSpin[0], &SubMostBoundID, &SubHalfMass, &SubStelHalfMass, &SubMassTab[0]);
#endif // LT_ADD_GAL_TO_SUB
                    tt1 = second();

                    if (timediff(tt0, tt1) > 10.0)
                    {
                        VERBOSE(2, "  determining properties of halo of len=%d took %g sec", len, timediff(tt0, tt1));
                    }

                    /* we have filled into ud the binding energy and the particle ID return */

                    if (((GrNr - 1) % NTask) == ThisTask)
                    {
                        ENDRUN_IF(Nsubgroups >= MaxNsubgroups,
                                  "Number of subgroups is larger then the maximum allowed number of subgroups");

                        if (subnr == 0)
                        {
                            for (int j = 0; j < 3; j++)
                            {
                                Group[grindex].Pos[j] = SubPos[j];
                            }
                        }

                        SubGroup[Nsubgroups].Len = len;
                        if (subnr == 0)
                        {
                            SubGroup[Nsubgroups].Offset = Group[grindex].Offset;
                        }
                        else
                        {
                            SubGroup[Nsubgroups].Offset =
                                SubGroup[Nsubgroups - 1].Offset + SubGroup[Nsubgroups - 1].Len;
                            SubGroup[Nsubgroups].GrNr = GrNr - 1;
                            SubGroup[Nsubgroups].SubNr = subnr;
                            SubGroup[Nsubgroups].SubParent = parent;
                            SubGroup[Nsubgroups].Mass = SubMass;
                            SubGroup[Nsubgroups].SubMostBoundID = SubMostBoundID;
                            SubGroup[Nsubgroups].SubVelDisp = SubVelDisp;
                            SubGroup[Nsubgroups].SubVmax = SubVmax;
                            SubGroup[Nsubgroups].SubVmaxRad = SubVmaxRad;
                            SubGroup[Nsubgroups].SubHalfMass = SubHalfMass;
                            SubGroup[Nsubgroups].SubStelHalfMass = SubStelHalfMass;
                        }

                        for (int j = 0; j < 3; j++)
                        {
                            SubGroup[Nsubgroups].Pos[j] = SubPos[j];
                            SubGroup[Nsubgroups].Vel[j] = SubVel[j];
                            SubGroup[Nsubgroups].CM[j] = SubCM[j];
                            SubGroup[Nsubgroups].Spin[j] = SubSpin[j];
                        }

                        for (int j = 0; j < 6; j++)
                        {
                            SubGroup[Nsubgroups].MassTab[j] = SubMassTab[j];
                        }

#ifdef LT_ADD_GAL_TO_SUB
                        if (SubMassTab[4] > 0.)
                        {
                            for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
                            {
                                if (SubLum[j] > 0)
                                {
                                    SubGroup[Nsubgroups].SubLum[j] = -2.5 * log10(SubLum[j]);
                                }
                                else
                                {
                                    SubGroup[Nsubgroups].SubLum[j] = 0;
                                }
#ifdef DUSTATT
                                if (SubLumAtt[j] > 0)
                                {
                                    SubGroup[Nsubgroups].SubLumAtt[j] = -2.5 * log10(SubLumAtt[j]);
                                }
                                else
                                {
                                    SubGroup[Nsubgroups].SubLumAtt[j] = 0;
                                }
#endif // DUSTATT
#ifdef OBSERVER_FRAME
                                if (SubLumObs[j] > 0)
                                {
                                    SubGroup[Nsubgroups].SubLumObs[j] = -2.5 * log10(SubLumObs[j]);
                                }
                                else
                                {
                                    SubGroup[Nsubgroups].SubLumObs[j] = 0;
                                }
#endif // OBSERVER_FRAME
                            }
#ifdef DUSTATT
                            for (int j = 0; j < DUSTATT; j++)
                            {
                                SubGroup[Nsubgroups].DustProfile[j] = Dust[j];
                            }
#endif // DUSTATT
                            SubGroup[Nsubgroups].SubMeanAge = SubMeanAge;
                            SubGroup[Nsubgroups].SubMeanZStar = SubMeanZStar;
                        }
                        else
                        {
                            for (j = 0; j < LT_ADD_GAL_TO_SUB; j++)
                            {
                                SubGroup[Nsubgroups].SubLum[j] = 0;
#ifdef DUSTATT
                                SubGroup[Nsubgroups].SubLumAtt[j] = 0;
#endif // DUSTATT
#ifdef OBSERVER_FRAME
                                SubGroup[Nsubgroups].SubLumObs[j] = 0;
#endif // OBSERVER_FRAME
                            }
#ifdef DUSTATT
                            for (int j = 0; j < DUSTATT; j++)
                                SubGroup[Nsubgroups].DustProfile[j] = 0;
#endif // DUSTATT
                            SubGroup[Nsubgroups].SubMeanAge = SubMeanAge;
                            SubGroup[Nsubgroups].SubMeanZStar = SubMeanZStar;
                        }
                        if (SubMassTab[0] > 0.)
                        {
                            SubGroup[Nsubgroups].SubSFR = SubSFR;
                        }
                        else
                        {
                            SubGroup[Nsubgroups].SubSFR = 0;
                            ;
                        }
#endif // LT_ADD_GAL_TO_SUB

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

            /*
               double ttt1 = second();
               printf("  took %g sec\n", timediff(ttt0, ttt1));
               fflush(stdout);
             */
        }

        parent = -1;
        /* inform the others that we are done */

        for (int i = 1; i < NTask; i++)
        {
            MPI_Send(&parent, 1, MPI_INT, i, TAG_N, MYMPI_COMM_WORLD);
        }

        for (int i = NTask - 1; i >= 0; i--)
        {
            myfree(ud_list[i]);
        }

        myfree(ud_list);

        myfree(List_LocalLen);
    }
    else
    {
        subnr = 0;
        do
        {
            MPI_Recv(&LocalLen, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (LocalLen >= 0)
            {
                MPI_Recv(&len, 1, MPI_INT, 0, TAG_HMAX, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&parent, 1, MPI_INT, 0, TAG_SPHDATA, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Recv(ud, LocalLen * sizeof(struct unbind_data), MPI_BYTE, 0, TAG_FOF_A, MYMPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

#ifdef LT_ADD_GAL_TO_SUB
                subfind_col_determine_sub_halo_properties(
                    ud, LocalLen, &SubMass, &SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp, &SubVmax, &SubVmaxRad,
                    &SubSpin[0], &SubMostBoundID, &SubHalfMass, &SubStelHalfMass, &SubMassTab[0], &SubLum[0],
                    &SubLumAtt[0], &SubLumObs[0], &Dust, &SubMeanAge, &SubMeanZStar, &SubSFR);
#else // LT_ADD_GAL_TO_SUB
                subfind_col_determine_sub_halo_properties(
                    ud, LocalLen, &SubMass, &SubPos[0], &SubVel[0], &SubCM[0], &SubVelDisp, &SubVmax, &SubVmaxRad,
                    &SubSpin[0], &SubMostBoundID, &SubHalfMass, &SubStelHalfMass, &SubMassTab[0]);
#endif // LT_ADD_GAL_TO_SUB
                if (((GrNr - 1) % NTask) == ThisTask)
                {
                    ENDRUN_IF(Nsubgroups >= MaxNsubgroups,
                              "Number of subgroups is larger then the maximum allowed number of subgroups");

                    if (subnr == 0)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            Group[grindex].Pos[j] = SubPos[j];
                        }
                    }

                    SubGroup[Nsubgroups].Len = len;
                    if (subnr == 0)
                    {
                        SubGroup[Nsubgroups].Offset = Group[grindex].Offset;
                    }
                    else
                    {
                        SubGroup[Nsubgroups].Offset = SubGroup[Nsubgroups - 1].Offset + SubGroup[Nsubgroups - 1].Len;
                    }
                    SubGroup[Nsubgroups].GrNr = GrNr - 1;
                    SubGroup[Nsubgroups].SubNr = subnr;
                    SubGroup[Nsubgroups].SubParent = parent;
                    SubGroup[Nsubgroups].Mass = SubMass;
                    SubGroup[Nsubgroups].SubMostBoundID = SubMostBoundID;
                    SubGroup[Nsubgroups].SubVelDisp = SubVelDisp;
                    SubGroup[Nsubgroups].SubVmax = SubVmax;
                    SubGroup[Nsubgroups].SubVmaxRad = SubVmaxRad;
                    SubGroup[Nsubgroups].SubHalfMass = SubHalfMass;
                    SubGroup[Nsubgroups].SubStelHalfMass = SubStelHalfMass;

                    for (int j = 0; j < 3; j++)
                    {
                        SubGroup[Nsubgroups].Pos[j] = SubPos[j];
                        SubGroup[Nsubgroups].Vel[j] = SubVel[j];
                        SubGroup[Nsubgroups].CM[j] = SubCM[j];
                        SubGroup[Nsubgroups].Spin[j] = SubSpin[j];
                    }

                    for (int j = 0; j < 6; j++)
                    {
                        SubGroup[Nsubgroups].MassTab[j] = SubMassTab[j];
                    }
#ifdef LT_ADD_GAL_TO_SUB
                    if (SubMassTab[4] > 0.)
                    {
                        for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
                        {
                            if (SubLum[j] > 0)
                            {
                                SubGroup[Nsubgroups].SubLum[j] = -2.5 * log10(SubLum[j]);
                            }
                            else
                            {
                                SubGroup[Nsubgroups].SubLum[j] = 0;
                            }
#ifdef DUSTATT
                            if (SubLumAtt[j] > 0)
                            {
                                SubGroup[Nsubgroups].SubLumAtt[j] = -2.5 * log10(SubLumAtt[j]);
                            }
                            else
                            {
                                SubGroup[Nsubgroups].SubLumAtt[j] = 0;
                            }
#endif // DUSTATT
#ifdef OBSERVER_FRAME
                            if (SubLumObs[j] > 0)
                            {
                                SubGroup[Nsubgroups].SubLumObs[j] = -2.5 * log10(SubLumObs[j]);
                            }
                            else
                            {
                                SubGroup[Nsubgroups].SubLumObs[j] = 0;
                            }
#endif // OBSERVER_FRAME
                        }
#ifdef DUSTATT
                        for (int j = 0; j < DUSTATT; j++)
                        {
                            SubGroup[Nsubgroups].DustProfile[j] = Dust[j];
                        }
#endif // DUSTATT
                        SubGroup[Nsubgroups].SubMeanAge = SubMeanAge;
                        SubGroup[Nsubgroups].SubMeanZStar = SubMeanZStar;
                    }
                    else
                    {
                        for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
                        {
                            SubGroup[Nsubgroups].SubLum[j] = 0;
#ifdef DUSTATT
                            SubGroup[Nsubgroups].SubLumAtt[j] = 0;
#endif // DUSTATT
#ifdef OBSERVER_FRAME
                            SubGroup[Nsubgroups].SubLumObs[j] = 0;
#endif // OBSERVER_FRAME
                        }
#ifdef DUSTATT
                        for (int j = 0; j < DUSTATT; j++)
                        {
                            SubGroup[Nsubgroups].DustProfile[j] = 0;
                        }
#endif // DUSTATT
                        SubGroup[Nsubgroups].SubMeanAge = SubMeanAge;
                        SubGroup[Nsubgroups].SubMeanZStar = SubMeanZStar;
                    }
#endif // DUSTATT

                    Nsubgroups++;
                }

                /* Let's now assign the subgroup number */

                for (int i = 0; i < LocalLen; i++)
                {
                    P[ud[i].index].SubNr = subnr;
                }

                subnr++;
            }
        } while (LocalLen >= 0);
    }

    t1 = second();
    VERBOSE(2, "determining substructure properties took %g sec (presently allocated %g MB)", timediff(t0, t1),
            AllocatedBytes / (1024.0 * 1024.0));

    myfree(ud);

    if (ThisTask == 0)
    {
        for (int i = NTask - 1; i >= 0; i--)
        {
            myfree(Len[i]);
            myfree(Next[i]);
        }

        myfree(Len);
        myfree(Next);

        for (int i = NTask - 1; i >= 0; i--)
        {
            myfree(candidates[i]);
        }

        myfree(candidates);
        myfree(count_cand);
    }

    myfree(Offset_NumPartGroup);
    myfree(List_NumPartGroup);

    force_treefree();
    domain_free();
    domain_allocate_trick();

    qsort(P, NumPart, sizeof(struct particle_data),
          subfind_compare_P_origindex); /* reorder them such that the gas particles match again */

    forint(i = 0; i < NumPart; i++)
    {
        ENDRUN_IF(P[i].origindex != i, "Particle indexing is wrong");
    }

    ttt1 = second();

    VERBOSE(2, "\ncollective processing of halo done. Took in total %g sec", timediff(ttt0, ttt1));
}

/**
 * @brief Unbinds independent substructures from candidate list.
 *
 * This function processes a list of candidate substructures, identifies independent ones,
 * and unbinds them if they meet certain criteria. The substructures that meet the criteria
 * are then marked appropriately.
 *
 * The function performs the following steps:
 * - Allocates memory for R2list and ud arrays.
 * - Sorts the candidate substructures based on the number of substructures.
 * - Iterates over the sorted candidate substructures.
 * - For each candidate with a parent of 0, it checks the particles to see if they belong to
 *   the substructure.
 * - Marks particles that belong to a substructure with HIGHBIT.
 * - Collects particles that belong to a substructure and attempts to unbind them.
 * - If the number of bound particles exceeds a threshold (All.DesLinkNgb), the substructure
 *   is considered valid and its bound length is set.
 * - Marks processed substructures and frees allocated memory.
 *
 * @param count_cand Number of candidate substructures to process.
 */
void subfind_unbind_independent_ones(int count_cand)
{
    int j, k, len, nsubs, len_non_gas;

    /*subfind_loctree_treeallocate(All.TreeAllocFactor * NumPart, NumPart); */

    R2list = mymalloc("R2list", NumPart * sizeof(struct r2data));
    ud = mymalloc("ud", NumPart * sizeof(struct unbind_data));

    qsort(loc_candidates, count_cand, sizeof(struct cand_dat), subfind_compare_candidates_nsubs);
    int i = 0;
    for (int k = 0; k < count_cand; k++)
        if (loc_candidates[k].parent == 0)
        {
            while (P[i].submark < loc_candidates[k].nsub)
            {
                i++;
                ENDRUN_IF(i >= NumPart, "Index is larger than the number of particles");
            }

            if (P[i].submark >= 0 && P[i].submark < HIGHBIT)
            {
                len = 0;
                nsubs = P[i].submark;

                ENDRUN_IF(nsubs != loc_candidates[k].nsub, "TASK=%d i=%d k=%d nsubs=%d loc_candidates[k].nsub=%d",
                          ThisTask, i, k, nsubs, loc_candidates[k].nsub);

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

                if (len >= All.DesLinkNgb)
                {
                    /* ok, we found a substructure */
                    loc_candidates[k].bound_length = len;

                    for (int j = 0; j < len; j++)
                        P[ud[j].index].submark = nsubs; /* we use this to flag the substructures */
                }
                else
                {
                    loc_candidates[k].bound_length = 0;
                }
            }

            loc_candidates[k].parent = -1;
        }

    myfree(ud);
    myfree(R2list);
}

/**
 * @brief Marks independent substructures.
 *
 * This function iterates over candidate substructures across multiple tasks, identifying and marking
 * independent substructures. It marks particles belonging to independent substructures for further processing.
 *
 * The function performs the following steps:
 * - Initializes the substructure count to zero.
 * - Iterates over each task and its respective candidate substructures.
 * - For each candidate substructure with a parent value of 0, it iterates through its particles.
 * - Marks each particle in the substructure using `subfind_distlinklist_mark_particle`.
 * - Increments the substructure count for each candidate processed.
 *
 * @return The total number of substructures marked.
 */
int subfind_mark_independent_ones(void)
{
    int nsubs = 0;

    for (int task = 0; task < NTask; task++)
    {
        for (int k = 0; k < count_cand[task]; k++)
        {
            int len = candidates[task][k].len;
            int parent = candidates[task][k].parent; /* this is here actually the daughter count */

            if (parent == 0)
            {
                long long p = candidates[task][k].head;
                for (int i = 0; i < len; i++)
                {
                    subfind_distlinklist_mark_particle(p, task, nsubs);

                    p = subfind_distlinklist_get_next(p);
                }
            }

            nsubs++;
        }
    }
    return nsubs;
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
    int i, j, p, part_index, minindex, task;
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

            ENDRUN_IF(mincpu < 0, "Mincpu is less than 0");

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

        j = (int)(0.25 * numleft); /* index of limiting energy value */

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

        for (i = 0; i < num - 1; i++)
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
        }

        if (num > 0 && count_bound_unbound <= 0)
        {
            weakly_bound_limit_local = bnd_energy[i];
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
 * @brief Adds a particle to the distributed link list if it is not already part of a substructure.
 *
 * This function takes a particle index, determines its task and local index, and adds it to the
 * unbind data list of the corresponding task if it is not already part of a substructure.
 *
 * @param index The global index of the particle to be added.
 */
void subfind_distlinklist_add_particle(long long index)
{
    int task = (index >> 32);
    int i = (index & MASK);

    if (Len[task][i] < 0) /* consider only particles not already in substructures */
    {
        ud_list[task][List_LocalLen[task]].index = i;
        List_LocalLen[task]++;
    }
}

/**
 * @brief Marks a particle in the distributed link list with a specified target and mark.
 *
 * This function takes a particle index, target substructure identifier, and a mark value,
 * and assigns the target and mark to the particle if it is not already marked with HIGHBIT.
 *
 * @param index The global index of the particle to be marked.
 * @param target The target substructure identifier to be assigned to the particle.
 * @param mark The mark value to be assigned to the particle.
 */
void subfind_distlinklist_mark_particle(long long index, int target, int mark)
{
    int task = (index >> 32);
    int i = (index & MASK);

    ENDRUN_IF(submark[task][i] != HIGHBIT, "Task=%d i=%d submark[task][i]=%d?", task, i, submark[task][i]);

    targettask[task][i] = target;
    submark[task][i] = mark;
}

/**
 * @brief Adds a particle to the unbind data list if it belongs to a specified substructure.
 *
 * This function takes a particle index and a substructure identifier, and adds the particle
 * to the unbind data list of the corresponding task if it is part of the specified substructure.
 *
 * @param index The global index of the particle to be added.
 * @param nsub The substructure identifier that the particle must belong to.
 */
void subfind_distlinklist_add_bound_particles(long long index, int nsub)
{
    int task = (index >> 32);
    int i = (index & MASK);

    if (Len[task][i] == nsub) /* consider only particles not already in substructures */
    {
        ud_list[task][List_LocalLen[task]].index = i;
        List_LocalLen[task]++;
    }
}

/**
 * @brief Identifies subhalo candidates from particle data by constructing link lists.
 *
 * This function identifies subhalo candidates by building up link lists from high-density to low-density regions.
 * It processes particle data across multiple tasks to find potential subhalos and merges or attaches particles
 * to subhalos based on their neighborhood properties.
 *
 * The function performs the following steps:
 * - Iterates over each task to process its particles.
 * - For each particle, determines its neighbors and processes it based on the number of neighbors:
 *   - Case 0: The particle is a lonely maximum and starts a new group.
 *   - Case 1: The particle is attached to exactly one existing group.
 *   - Case 2: The particle merges two groups together.
 * - Adds the entire group of particles as a subhalo candidate.
 * - Establishes a rank order for the particles in the group.
 * - Updates the ranks of the subhalo candidates.
 *
 * @param totgrouplen The total length of the group of particles to be processed.
 */
void subfind_col_find_candidates(int totgrouplen)
{
    int ngbcount, retcode, len_attach;
    int i, k, len, task, off;
    long long prev, tail, tail_attach, tmp, next, index;
    long long p, ss, head, head_attach, ngb_index1, ngb_index2, rank;
    double t0, t1;

    /* now find the subhalo candidates by building up link lists from high density to low density */

    for (int task = 0; task < NTask; task++)
    {
        /*      printf("begin task=%d...", task); fflush(stdout); */
        t0 = second();

        for (int k = 0; k < List_NumPartGroup[task]; k++)
        {
            /*      if((k % 1000) == 0)
               printf("k=%d|%d\n", k,  List_NumPartGroup[task]);
             */

            off = Offset_NumPartGroup[task] + k;

            ngbcount = sd[off].ngbcount;
            ngb_index1 = sd[off].ngb_index1;
            ngb_index2 = sd[off].ngb_index2;

            switch (ngbcount) /* treat the different possible cases */
            {
            case 0: /* this appears to be a lonely maximum -> new group */
                subfind_distlinklist_set_all(sd[off].index, sd[off].index, sd[off].index, 1, -1);
                break;

            case 1: /* the particle is attached to exactly one group */
                head = subfind_distlinklist_get_head(ngb_index1);

                ENDRUN_IF(head == -1, "We have a problem!  head=%d/%d for k=%d on task=%d", (int)(head >> 32),
                          (int)head, k, task);

                retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[off].index);

                if (!(retcode & 1))
                {
                    subfind_distlinklist_set_headandnext(sd[off].index, head, -1);
                }
                if (!(retcode & 2))
                {
                    subfind_distlinklist_set_next(tail, sd[off].index);
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
                    ENDRUN("We have a problem!  head=%d/%d head_attach=%d/%d for k=%d on task=%d", (int)(head >> 32),
                           (int)head, (int)(head_attach >> 32), (int)head_attach, k, task);
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
                        tmp = len;
                        len = len_attach;
                        len_attach = tmp;
                    }

                    /* only in case the attached group is long enough we bother to register it
                       as a subhalo candidate */

                    if (len_attach >= All.DesLinkNgb)
                    {
                        if (count_cand[task] < max_candidates)
                        {
                            candidates[task][count_cand[task]].len = len_attach;
                            candidates[task][count_cand[task]].head = head_attach;
                            count_cand[task]++;
                        }
                        else
                        {
                            ENDRUN("Error: count_cand[task]=%d max_candidates=%d", count_cand[task], max_candidates);
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
                retcode = subfind_distlinklist_get_tail_set_tail_increaselen(head, &tail, sd[off].index);

                if (!(retcode & 1))
                {
                    subfind_distlinklist_set_headandnext(sd[off].index, head, -1);
                }
                if (!(retcode & 2))
                {
                    subfind_distlinklist_set_next(tail, sd[off].index);
                }
                break;
            }
        }

        t1 = second();
        /*
           printf("%g sec\n", timediff(t0, t1)); fflush(stdout);
         */
    }

    VERBOSE(2, "identification of primary candidates finished");

    /* add the full thing as a subhalo candidate */

    t0 = second();
    head = -1, prev = -1;
    for (int task = 0; task < NTask; task++)
    {
        for (int i = 0; i < List_NumPartGroup[task]; i++)
        {
            index = (((long long)task) << 32) + i;

            if (Head[task][i] == index)
            {
                subfind_distlinklist_get_tailandlen(Head[task][i], &tail, &len);
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
    }

    if (count_cand[NTask - 1] < max_candidates)
    {
        candidates[NTask - 1][count_cand[NTask - 1]].len = totgrouplen;
        candidates[NTask - 1][count_cand[NTask - 1]].head = head;
        count_cand[NTask - 1]++;
    }
    else
    {
        ENDRUN("count_cand[NTask - 1]=%d max_candidates=%d", count_cand[NTask - 1], max_candidates);
    }

    t1 = second();
    VERBOSE(2, "adding background as candidate finished. (%g sec)", timediff(t0, t1));
    t0 = second();

    /* go through the whole chain once to establish a rank order. For the rank we use Len[] */

    task = (head >> 32);

    p = head;
    rank = 0;

    while (p >= 0)
    {
        p = subfind_distlinklist_setrank_and_get_next(p, &rank);
    }

    /* for each candidate, we now pull out the rank of its head */
    for (int task = 0; task < NTask; task++)
    {
        for (int k = 0; k < count_cand[task]; k++)
        {
            candidates[task][k].rank = subfind_distlinklist_get_rank(candidates[task][k].head);
        }
    }

    t1 = second();

    VERBOSE(2, "establishing of rank order finished (%g sec)", timediff(t0, t1));

    if (((int)rank) != totgrouplen)
    {
        ENDRUN("Mismatch");
    }
}

/**
 * @brief Sets the rank of a particle and returns the index of the next particle in the link list.
 *
 * This function assigns a rank to a particle based on the provided rank pointer and then increments the rank.
 * It returns the index of the next particle in the link list.
 *
 * @param index The global index of the particle whose rank is to be set.
 * @param rank A pointer to the current rank, which will be incremented after assigning the rank to the particle.
 * @return The global index of the next particle in the link list.
 */
long long subfind_distlinklist_setrank_and_get_next(long long index, long long *rank)
{
    int task = (index >> 32);
    int i = (index & MASK);

    Len[task][i] = *rank;
    *rank = *rank + 1;
    long long next = Next[task][i];

    return next;
}

/**
 * @brief Sets the head of a particle's link list and returns the index of the next particle.
 *
 * This function assigns a new head to the particle specified by the global index and returns
 * the index of the next particle in the link list.
 *
 * @param index The global index of the particle whose head is to be set.
 * @param head The global index to be set as the new head of the particle's link list.
 * @return The global index of the next particle in the link list.
 */
long long subfind_distlinklist_set_head_get_next(long long index, long long head)
{
    int task = (index >> 32);
    int i = (index & MASK);

    Head[task][i] = head;
    long long next = Next[task][i];

    return next;
}

/**
 * @brief Sets the index of the next particle in the link list for a given particle.
 *
 * This function assigns the specified next index to the particle identified by the given global index.
 *
 * @param index The global index of the particle whose next index is to be set.
 * @param next The global index to be set as the next particle in the link list.
 */
void subfind_distlinklist_set_next(long long index, long long next)
{
    int task = (index >> 32);
    int i = (index & MASK);
    Next[task][i] = next;
}

/**
 * @brief Retrieves the index of the next particle in the link list for a given particle.
 *
 * This function returns the index of the next particle in the link list for the particle identified by the given global
 * index.
 *
 * @param index The global index of the particle whose next index is to be retrieved.
 * @return The global index of the next particle in the link list.
 */
long long subfind_distlinklist_get_next(long long index)
{
    int task = (index >> 32);
    int i = (index & MASK);

    long long next = Next[task][i];

    return next;
}

/**
 * @brief Retrieves the rank of a particle in the link list.
 *
 * This function returns the rank of the particle identified by the given global index in the link list.
 *
 * @param index The global index of the particle whose rank is to be retrieved.
 * @return The rank of the particle in the link list.
 */
long long subfind_distlinklist_get_rank(long long index)
{
    int task = (index >> 32);
    int i = (index & MASK);

    long long rank = Len[task][i];

    return rank;
}

/**
 * @brief Retrieves the head of the link list for a given particle.
 *
 * This function returns the index of the head of the link list for the particle identified by the given global index.
 *
 * @param index The global index of the particle whose head is to be retrieved.
 * @return The global index of the head of the link list.
 */
long long subfind_distlinklist_get_head(long long index)
{
    int task = (index >> 32);
    int i = (index & MASK);

    long long head = Head[task][i];

    return head;
}

/**
 * @brief Retrieves the heads of the link lists for two particles.
 *
 * This function retrieves the indices of the heads of the link lists for two particles identified
 * by the given global indices.
 *
 * @param ngb_index1 The global index of the first particle.
 * @param ngb_index2 The global index of the second particle.
 * @param head A pointer to store the index of the head of the link list for the first particle.
 * @param head_attach A pointer to store the index of the head of the link list for the second particle.
 */
void subfind_distlinklist_get_two_heads(long long ngb_index1, long long ngb_index2, long long *head,
                                        long long *head_attach)
{
    int task = (ngb_index1 >> 32);
    int i1 = (ngb_index1 & MASK);
    int i2 = (ngb_index2 & MASK);

    *head = Head[task][i1];
    *head_attach = Head[task][i2];
}

/**
 * @brief Sets the head and next indices for a given particle in the link list.
 *
 * This function sets the head and next indices for the particle identified by the given global index
 * in the link list.
 *
 * @param index The global index of the particle whose head and next indices are to be set.
 * @param head The global index to be set as the head of the particle's link list.
 * @param next The global index to be set as the next particle in the link list.
 */
void subfind_distlinklist_set_headandnext(long long index, long long head, long long next)
{
    int task = (index >> 32);
    int i = (index & MASK);

    Head[task][i] = head;
    Next[task][i] = next;
}

/**
 * @brief Retrieves the tail, sets a new tail, and increases the length of a link list for a given particle.
 *
 * This function retrieves the tail index, sets a new tail index, and increases the length of the link list
 * for the particle identified by the given global index. It also updates the head and next indices accordingly.
 *
 * @param index The global index of the particle whose tail, length, and link list are to be modified.
 * @param tail A pointer to store the global index of the old tail.
 * @param newtail The global index to be set as the new tail of the particle's link list.
 * @return An integer indicating which actions were performed:
 *         - Bit 0 (1): Set a new tail.
 *         - Bit 1 (2): Update the next index of the old tail.
 */
int subfind_distlinklist_get_tail_set_tail_increaselen(long long index, long long *tail, long long newtail)
{
    int task = (index >> 32);
    int i = (index & MASK);

    int retcode = 0;

    long long oldtail = Tail[task][i];
    Tail[task][i] = newtail;
    Len[task][i]++;
    *tail = oldtail;

    int task_newtail = (newtail >> 32);
    int i_newtail = (newtail & MASK);
    Head[task_newtail][i_newtail] = index;
    Next[task_newtail][i_newtail] = -1;
    retcode |= 1;

    int task_oldtail = (oldtail >> 32);
    int i_oldtail = (oldtail & MASK);
    Next[task_oldtail][i_oldtail] = newtail;
    retcode |= 2;

    return retcode;
}

/**
 * @brief Sets the tail and length of a link list for a given particle.
 *
 * This function sets the tail index and length of the link list for the particle identified by the given global index.
 *
 * @param index The global index of the particle whose tail and length are to be set.
 * @param tail The global index to be set as the tail of the particle's link list.
 * @param len The length to be set for the particle's link list.
 */
void subfind_distlinklist_set_tailandlen(long long index, long long tail, int len)
{
    int task = (index >> 32);
    int i = (index & MASK);

    Tail[task][i] = tail;
    Len[task][i] = len;
}

/**
 * @brief Retrieves the tail and length of a link list for a given particle.
 *
 * This function retrieves the tail index and length of the link list for the particle identified
 * by the given global index.
 *
 * @param index The global index of the particle whose tail and length are to be retrieved.
 * @param tail A pointer to store the global index of the tail of the particle's link list.
 * @param len A pointer to store the length of the particle's link list.
 */
void subfind_distlinklist_get_tailandlen(long long index, long long *tail, int *len)
{
    int task = (index >> 32);
    int i = (index & MASK);

    *tail = Tail[task][i];
    *len = Len[task][i];
}

/**
 * @brief Sets all parameters of a link list for a given particle.
 *
 * This function sets all parameters of the link list for the particle identified by the given global index.
 *
 * @param index The global index of the particle whose link list parameters are to be set.
 * @param head The global index to be set as the head of the particle's link list.
 * @param tail The global index to be set as the tail of the particle's link list.
 * @param len The length to be set for the particle's link list.
 * @param next The global index to be set as the next particle in the link list.
 */
void subfind_distlinklist_set_all(long long index, long long head, long long tail, int len, long long next)
{
    int task = (index >> 32);
    int i = (index & MASK);

    Head[task][i] = head;
    Tail[task][i] = tail;
    Len[task][i] = len;
    Next[task][i] = next;
}

/**
 * @brief Compares two particles based on their group numbers and keys.
 *
 * This function compares two particle_data structures based on their group numbers and keys.
 *
 * @param a Pointer to the first particle_data structure.
 * @param b Pointer to the second particle_data structure.
 * @return -1 if the first particle has a smaller difference in group number from GrNr,
 *         +1 if the second particle has a smaller difference,
 *         -1 if the key of the first particle is smaller than that of the second particle,
 *         +1 if the key of the first particle is larger than that of the second particle,
 *         -1 if the ID of the first particle is smaller than that of the second particle,
 *         +1 if the ID of the first particle is larger than that of the second particle,
 *         0 if all compared attributes are equal.
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

    if (((struct particle_data *)a)->Key < ((struct particle_data *)b)->Key)
    {
        return -1;
    }

    if (((struct particle_data *)a)->Key > ((struct particle_data *)b)->Key)
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
 * @brief Compares two particles based on their submark values and keys.
 *
 * This function compares two particle_data structures based on their submark values and keys.
 *
 * @param a Pointer to the first particle_data structure.
 * @param b Pointer to the second particle_data structure.
 * @return -1 if the submark of the first particle is smaller than that of the second particle,
 *         +1 if the submark of the first particle is larger than that of the second particle,
 *         -1 if the key of the first particle is smaller than that of the second particle,
 *         +1 if the key of the first particle is larger than that of the second particle,
 *         0 if both the submarks and keys are equal.
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

    if (((struct particle_data *)a)->Key < ((struct particle_data *)b)->Key)
    {
        return -1;
    }

    if (((struct particle_data *)a)->Key > ((struct particle_data *)b)->Key)
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
