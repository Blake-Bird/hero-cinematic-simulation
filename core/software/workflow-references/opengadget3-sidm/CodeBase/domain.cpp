/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Code for domain decomposition.
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

#include "allvars.hpp"
#include "proto.hpp"
#include <set>
#include <vector>
#ifdef DOMAIN_NUMA
#include <map>
#include "../System/numa.hpp"
#endif // DOMAIN_NUMA

#define REDUC_FAC 0.98

/*! toGo[task*NTask + partner] gives the number of particles in task 'task'
 *  that have to go to task 'partner'
 */
static int *toGo, *toGoSph;
static int *toGet, *toGetSph;
static int *list_NumPart;
static int *list_N_gas;
static int *list_load;
static int *list_loadsph;
static double *list_work;
static double *list_worksph;

extern int old_MaxPart, new_MaxPart;

#if defined(SFR)
static int *toGoStars, *toGetStars, *list_N_stars, *list_loadstars;
#endif // SFR

#ifdef BLACK_HOLES
static int *toGoBHs, *toGetBHs, *list_N_BHs, *list_loadBHs;
#endif // BLACK_HOLES

/**
 * @brief Structure representing local topnode data.
 */
static struct local_topnode_data
{
    peanokey Size; /*!< number of Peano-Hilbert mesh-cells represented by top-level node */
    peanokey StartKey; /*!< first Peano-Hilbert key in top-level node */
    long long Count; /*!< counts the number of particles in this top-level node */
    double Cost;
    double SphCost;
    int Daughter; /*!< index of first daughter cell (out of 8) of top-level node */
    int Leaf; /*!< if the node is a leaf, this gives its number when all leaves are traversed in Peano-Hilbert order */
    int Parent;
    int PIndex; /*!< first particle in node */
} *topNodes; /*!< points to the root node of the top-level tree */

/**
 * @brief Structure representing Peano-Hilbert data.
 */
static struct peano_hilbert_data
{
    peanokey key;
    int index;
} *mp;

static void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB);
static void domain_add_cost(struct local_topnode_data *treeA, int noA, long long count, double cost, double sphcost);

static float *domainWork; /*!< a table that gives the total "work" due to the particles stored by each processor */
static float *domainWorkSph; /*!< a table that gives the total "work" due to the particles stored by each processor */
static int *domainCount; /*!< a table that gives the total number of particles held by each processor */
static int *domainCountSph; /*!< a table that gives the total number of SPH particles held by each processor */

#if defined(SFR)
static int *domainCountStars;
#endif // SFR

#ifdef BLACK_HOLES
static int *domainCountBHs;
#endif // BLACK_HOLES

static int domain_allocated_flag = 0;
static int maxLoad, maxLoadsph;

#if defined(SFR)
static int maxLoadstars;
#endif // SFR

#ifdef BLACK_HOLES
static int maxLoadBHs;
#endif // BLACK_HOLES

static double totgravcost, gravcost, totsphcost, sphcost;
static long long totpartcount;

static int UseAllParticles;

/**
 * @brief Calculates the cost factor for a particle in the domain.
 *
 * This function calculates the cost factor for a particle in the domain based on its gravitational cost at a specific
 * level.
 *
 * @param i Index of the particle.
 * @return The cost factor for the particle.
 */
static inline double domain_particle_costfactor(int i)
{
    return 0.1 + P[i].GravCost[TakeLevel];
}

#include "../GreenTree_AR/recompose_domain.cpp"

/**
 * @brief Main routine for the domain decomposition.
 *
 * This function acts as a driver routine that allocates various temporary buffers,
 * maps the particles back onto the periodic box if needed, and then does the
 * domain decomposition, and a final Peano-Hilbert order of all particles
 * as a tuning measure.
 *
 * @param UseAllTimeBins Flag to use all timebins.
 * @param SaveKeys Flag to save keys.
 */
void domain_Decomposition(int UseAllTimeBins, int SaveKeys)
{
    TIMER_PUSH(1, "DOMAIN");
#ifdef TIMER_DOMAIN
    TIMER_PUSH_CLASS_MAINONLY(DomainTimer, 0, "DOMAIN");
#endif // TIMER_DOMAIN
    int ret, retsum, highest_bin_to_include;
    size_t bytes;

    UseAllParticles = UseAllTimeBins;

#ifdef WRITE_KEY_FILES
    peanokey old_key;
    int key_count, previous_keys;
#endif // WRITE_KEY_FILES

    TIMER_PUSH(3, "drift_particles");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Ti_current != All.Ti_Current)
        {
            drift_particle(i, All.Ti_Current);
        }
    }

    // Free memory
    force_treefree();
    domain_free();

    if (old_MaxPart)
    {
        All.MaxPart = new_MaxPart;
        old_MaxPart = 0;
    }

    TIMER_POP(3, "drift_particles"); // drift

#if defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)
    TIMER_PUSH(3, "rearrange_particles");
    rearrange_particle_sequence();
    TIMER_POP(3, "rearrange_particles"); // re-arrange
#endif // defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

#ifdef PERIODIC
    TIMER_PUSH(3, "box_wrap");
    do_box_wrapping(); /* map the particles back onto the box */
    TIMER_POP(3, "box_wrap"); // periodic_box_wrap
#endif // PERIODIC

    TIMER_PUSH(3, "check_particles1");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        ENDRUN_IF((P[i].Type > 5 || P[i].Type < 0), "task=%d:  P[i=%d].Type=%d", ThisTask, i, P[i].Type);
#ifdef SFR
        if (P[i].Type == 4)
        {
            ENDRUN_IF((MetP[P[i].pt.MetID].PID != i),
                      "task=%d:  error in cross-indexes for star-particle %d ID %llu MetID %d PID[MetID] %d", ThisTask,
                      i, (unsigned long long)P[i].ID, P[i].pt.MetID, MetP[P[i].pt.MetID].PID);
        }
#endif // SFR

#ifdef BLACK_HOLES
        if (P[i].Type == 5)
        {
            ENDRUN_IF(BHP[P[i].pt.BHID].PID != i, "task=%d:  error in cross-indexes for bh-particle %d ID %llu",
                      ThisTask, i, (unsigned long long)P[i].ID);
        }
#endif // BLACK_HOLES
    }

    TIMER_POPPUSH(3, "check_particles1", "find_levels"); // check_particle_integrety

    TreeReconstructFlag = 1; /* ensures that new tree will be constructed */

    /* we take the closest cost factor */
    if (UseAllParticles)
    {
        highest_bin_to_include = All.HighestOccupiedTimeBin;
    }
    else
    {
        highest_bin_to_include = All.HighestActiveTimeBin;
    }

    int diff = abs(All.LevelToTimeBin[0] - highest_bin_to_include);
    TakeLevel = 0;
    for (int i = 1; i < Gravcostlevels; i++)
    {
        if (diff > abs(All.LevelToTimeBin[i] - highest_bin_to_include))
        {
            TakeLevel = i;
            diff = abs(All.LevelToTimeBin[i] - highest_bin_to_include);
        }
    }

    VERBOSE(2, "DOMAIN: domain decomposition... LevelToTimeBin[TakeLevel=%d]=%d  (presently allocated=%g MB)",
            TakeLevel, All.LevelToTimeBin[TakeLevel], AllocatedBytes / (1024.0 * 1024.0));

    TIMER_POP(3, "find_levels");
    TIMER_PUSH(1, "DOM_EXCHANGE");
    do
    {
        domain_allocate();

        size_t all_bytes = 0;

        Key = (peanokey *)mymalloc("domain_key", bytes = (sizeof(peanokey) * All.MaxPart));
        all_bytes += bytes;

        toGo = (int *)mymalloc("toGo", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        toGoSph = (int *)mymalloc("toGoSph", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        toGet = (int *)mymalloc("toGet", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        toGetSph = (int *)mymalloc("toGetSph", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_NumPart = (int *)mymalloc("list_NumPart", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_N_gas = (int *)mymalloc("list_N_gas", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_load = (int *)mymalloc("list_load", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_loadsph = (int *)mymalloc("list_loadsph", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_work = (double *)mymalloc("list_work", bytes = (sizeof(double) * NTask));
        all_bytes += bytes;
        list_worksph = (double *)mymalloc("list_worksph", bytes = (sizeof(double) * NTask));
        all_bytes += bytes;
        domainWork = (float *)mymalloc("domainWork", bytes = (MaxTopNodes * sizeof(float)));
        all_bytes += bytes;
        domainWorkSph = (float *)mymalloc("domainWorkSph", bytes = (MaxTopNodes * sizeof(float)));
        all_bytes += bytes;
        domainCount = (int *)mymalloc("domainCount", bytes = (MaxTopNodes * sizeof(int)));
        all_bytes += bytes;
        domainCountSph = (int *)mymalloc("domainCountSph", bytes = (MaxTopNodes * sizeof(int)));
        all_bytes += bytes;

#if defined(SFR)
        toGoStars = (int *)mymalloc("toGoStars", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        toGetStars = (int *)mymalloc("toGetStars", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_N_stars = (int *)mymalloc("list_N_stars", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_loadstars = (int *)mymalloc("list_loadstars", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        domainCountStars = (int *)mymalloc("domainCountStars", bytes = (MaxTopNodes * sizeof(int)));
        all_bytes += bytes;
#endif // SFR

#ifdef BLACK_HOLES
        toGoBHs = (int *)mymalloc("toGoBHs", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        toGetBHs = (int *)mymalloc("toGetBHs", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_N_BHs = (int *)mymalloc("list_N_bhs", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        list_loadBHs = (int *)mymalloc("list_loadbhs", bytes = (sizeof(int) * NTask));
        all_bytes += bytes;
        domainCountBHs = (int *)mymalloc("domainCountBHs", bytes = (MaxTopNodes * sizeof(int)));
        all_bytes += bytes;
#endif // BLACK_HOLES

        topNodes = (struct local_topnode_data *)mymalloc("topNodes",
                                                         bytes = (MaxTopNodes * sizeof(struct local_topnode_data)));
        all_bytes += bytes;

        VERBOSE(2, "DOMAIN: use of %g MB of temporary storage for domain decomposition... (presently allocated=%g MB)",
                all_bytes / (1024.0 * 1024.0), AllocatedBytes / (1024.0 * 1024.0));

        maxLoad = (int)(All.MaxPart * REDUC_FAC);
        maxLoadsph = (int)(All.MaxPartSph * REDUC_FAC);
#ifdef SFR
        maxLoadstars = (int)(All.MaxPartMet * REDUC_FAC);
#endif // SFR

#ifdef BLACK_HOLES
        maxLoadBHs = (int)(All.MaxPartBH * REDUC_FAC);
#endif // BLACK_HOLES

        report_memory_usage(&HighMark_domain, "DOMAIN");

#ifdef WRITE_KEY_FILES
        ret = domain_decompose(SaveKeys);
#else // WRITE_KEY_FILES
        ret = domain_decompose();
#endif // WRITE_KEY_FILES

        /* copy what we need for the topnodes */
        for (int i = 0; i < NTopnodes; i++)
        {
            TopNodes[i].StartKey = topNodes[i].StartKey;
            TopNodes[i].Size = topNodes[i].Size;
            TopNodes[i].Daughter = topNodes[i].Daughter;
            TopNodes[i].Leaf = topNodes[i].Leaf;
        }

        myfree(topNodes);

#ifdef BLACK_HOLES
        myfree(domainCountBHs);
        myfree(list_loadBHs);
        myfree(list_N_BHs);
        myfree(toGetBHs);
        myfree(toGoBHs);
#endif // BLACK_HOLES

#if defined(SFR)
        myfree(domainCountStars);
        myfree(list_loadstars);
        myfree(list_N_stars);
        myfree(toGetStars);
        myfree(toGoStars);
#endif // SFR

        myfree(domainCountSph);
        myfree(domainCount);
        myfree(domainWorkSph);
        myfree(domainWork);
        myfree(list_worksph);
        myfree(list_work);
        myfree(list_loadsph);
        myfree(list_load);
        myfree(list_N_gas);
        myfree(list_NumPart);
        myfree(toGetSph);
        myfree(toGet);
        myfree(toGoSph);
        myfree(toGo);

        MPI_Allreduce(&ret, &retsum, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        if (retsum)
        {
            myfree(Key);

            domain_free();

            VERBOSE(2, "DOMAIN: Increasing TopNodeAllocFactor=%g  ", All.TopNodeAllocFactor);

            All.TopNodeAllocFactor *= 1.3;

            VERBOSE(2, "DOMAIN: new value=%g", All.TopNodeAllocFactor);

            ENDRUN_IF(All.TopNodeAllocFactor > 1000, "something seems to be going seriously wrong here. Stopping.");
        }
    } while (retsum);
    TIMER_POP(1, "DOM_EXCHANGE");
    TIMER_PUSH(3, "check_particles2"); // sending_particles

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        ENDRUN_IF((P[i].Type > 5 || P[i].Type < 0), "task=%d:  P[i=%d].Type=%d", ThisTask, i, P[i].Type);

#ifdef SFR
        if (P[i].Type == 4)
        {
            ENDRUN_IF(MetP[P[i].pt.MetID].PID != i, "task=%d:  error in cross-indexes for star-particle %d ID %llu",
                      ThisTask, i, (unsigned long long)P[i].ID);
        }
#endif // SFR

#ifdef BLACK_HOLES
        if (P[i].Type == 5)
        {
            ENDRUN_IF(BHP[P[i].pt.BHID].PID != i, "task=%d:  error in cross-indexes for bh-particle %d ID %llu",
                      ThisTask, i, (unsigned long long)P[i].ID);
        }
#endif // BLACK_HOLES
    }

    TIMER_POP(3, "check_particles2"); // extra_check
#ifdef SUBFIND
    if (GrNr < 0) /* we don't do it when SUBFIND is executed for a certain group */
#endif // SUBFIND
        peano_hilbert_order();

#ifdef WRITE_KEY_FILES
    TIMER_PUSH(1, "DOM_COMPUTE_KEY");
#ifdef SUBFIND
    if (GrNr < 0)
    {
#endif // SUBFIND
        /* Count Keys prestent, etc ... */

        if (SaveKeys != 0)
        {
            key_count = 0;

            if (N_gas > 0) /* First the Gas particles */
            {
                old_key = Key[0];
                KeyIndex[key_count] = old_key;
                NPartPerKey[key_count] = 1;
                PartKeyOffset[key_count] = 0;
                int i = 1;
                while (i < N_gas)
                {
                    if (Key[i] == old_key)
                    {
                        NPartPerKey[key_count]++;
                    }
                    else
                    {
                        old_key = Key[i];
                        key_count++;
                        NPartPerKey[key_count] = 1;
                        KeyIndex[key_count] = old_key;
                        PartKeyOffset[key_count] = i;
                    }
                    i++;
                }
                key_count++;
                NKeys[0] = key_count;
            }

            previous_keys = NKeys[0];

            if (NumPart > N_gas) /* Now the rest */
            {
                for (int k = 1; k < 6; k++)
                {
                    int offset = 0; /* Let the user calculate the shift between different species ! */
                    int i = N_gas; /* Always skip the gas particles */
                    while (P[i].Type != k && i < NumPart - 1) /* Find first particle of type k */
                    {
                        i++;
                    }
                    if (i == NumPart - 1 && P[i].Type != k) /* No particle of type k present !! */
                    {
                        NKeys[k] = 0;
                    }
                    else
                    {
                        old_key = Key[i];
                        KeyIndex[key_count] = old_key;
                        NPartPerKey[key_count] = 1;
                        PartKeyOffset[key_count] = offset;
                        i++;
                        offset++;
                        while (i < NumPart)
                        {
                            if (P[i].Type == k)
                            {
                                if (Key[i] == old_key)
                                {
                                    NPartPerKey[key_count]++;
                                }
                                else
                                {
                                    old_key = Key[i];
                                    key_count++;
                                    NPartPerKey[key_count] = 1;
                                    KeyIndex[key_count] = old_key;
                                    PartKeyOffset[key_count] = offset;
                                }
                                offset++;
                            }
                            i++;
                        }
                        key_count++;
                        ENDRUN_IF(key_count > NumPart, "Keycount : %d > %d !! ", key_count, NumPart);

                        NKeys[k] = key_count - previous_keys;
                        previous_keys += NKeys[k];
                    }
                }
            }
        }

#ifdef SUBFIND
    }
#endif // SUBFIND
    TIMER_POP(1, "DOM_COMPUTE_KEY"); // extra_computation_for_key_files
#endif // WRITE_KEY_FILES

    myfree(Key);

    memmove(TopNodes + NTopnodes, DomainTask, NTopnodes * sizeof(int));

    TopNodes = (struct topnode_data *)myrealloc(
        TopNodes, bytes = (NTopnodes * sizeof(struct topnode_data) + NTopnodes * sizeof(int)));
    VERBOSE(2, "DOMAIN: Freed %g MByte in top-level domain structure",
            (MaxTopNodes - NTopnodes) * sizeof(struct topnode_data) / (1024.0 * 1024.0));

    DomainTask = (int *)(TopNodes + NTopnodes);

    force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    reconstruct_timebins();

    TIMER_POP(1, "DOMAIN");

#ifdef TIMER_DOMAIN
    TIMER_POP_CLASS_MAINONLY(DomainTimer, 0, "DOMAIN");
#endif // TIMER_DOMAIN
}

/**
 * @brief Allocate memory
 *
 * This function allocates all the memory that will be required for the tree-construction/walk later on
 */
void domain_allocate(void)
{
    size_t bytes, all_bytes = 0;

    MaxTopNodes = (int)(All.TopNodeAllocFactor * All.MaxPart + 1);

    DomainStartList = (int *)mymalloc("DomainStartList", bytes = (NTask * Multipledomains * sizeof(int)));
    all_bytes += bytes;

    DomainEndList = (int *)mymalloc("DomainEndList", bytes = (NTask * Multipledomains * sizeof(int)));
    all_bytes += bytes;

    TopNodes = (struct topnode_data *)mymalloc(
        "TopNodes", bytes = (MaxTopNodes * sizeof(struct topnode_data) + MaxTopNodes * sizeof(int)));
    all_bytes += bytes;

    DomainTask = (int *)(TopNodes + MaxTopNodes);

    VERBOSE(2, "DOMAIN: Allocated %g MByte for top-level domain structure", all_bytes / (1024.0 * 1024.0));

    domain_allocated_flag = 1;
}

/**
 * @brief Frees memory allocated for domain-related data structures.
 */
void domain_free(void)
{
    if (domain_allocated_flag)
    {
        myfree(TopNodes);
        myfree(DomainEndList);
        myfree(DomainStartList);
        domain_allocated_flag = 0;
    }
}

static struct topnode_data *save_TopNodes;
static int *save_DomainStartList, *save_DomainEndList;

/**
 * @brief Frees memory allocated for domain-related data structures and saves pointers for later use.
 *
 * This function frees memory allocated for domain-related data structures if they were allocated
 * and saves the pointers to them for later use. It also resets the domain_allocated_flag.
 * If the data structures were not allocated, it triggers an endrun.
 */
void domain_free_trick(void)
{
    if (domain_allocated_flag)
    {
        save_TopNodes = TopNodes;
        save_DomainEndList = DomainEndList;
        save_DomainStartList = DomainStartList;
        domain_allocated_flag = 0;
    }
    else
    {
        ENDRUN("Problem with domain_free_trick");
    }
}

/**
 * @brief Allocates memory for domain-related data structures using saved pointers.
 *
 * This function sets the domain_allocated_flag to 1 to indicate that domain-related data structures are allocated.
 * It then assigns the saved pointers to the corresponding data structures.
 */
void domain_allocate_trick(void)
{
    domain_allocated_flag = 1;

    TopNodes = save_TopNodes;
    DomainEndList = save_DomainEndList;
    DomainStartList = save_DomainStartList;
}

/**
 * @brief Domain decomposition for all particle types.
 *
 * This function carries out the actual domain decomposition for all
 * particle types. It will try to balance the work-load for each domain,
 * as estimated based on the P[i]-GravCost values.  The decomposition will
 * respect the maximum allowed memory-imbalance given by the value of
 * PartAllocFactor.
 *
 * @param SaveKeys Flag to save keys.
 * @return 0 if the domain decomposition was successful, 1 otherwise.
 */
#ifdef WRITE_KEY_FILES
int domain_decompose(int SaveKeys)
#else // WRITE_KEY_FILES
int domain_decompose(void)
#endif // WRITE_KEY_FILES
{
    int status;
    long long sumtogo, sumload, sumloadsph;
    int maxload, maxloadsph, multipledomains = Multipledomains;
    double sumwork, maxwork, sumworksph, maxworksph;
#if defined(SFR)
    long long sumloadstars;
    int maxloadstars;
#endif // SFR

#ifdef BLACK_HOLES
    long long sumloadbhs;
    int maxloadbhs;
#endif // BLACK_HOLES

    TIMER_PUSH(3, "compute_gravcost");

#ifdef WRITE_KEY_FILES
    if (SaveKeys == 1)
    {
        multipledomains = WRITE_KEY_FILES;
    }
#endif // WRITE_KEY_FILES

    for (int i = 0; i < 6; i++)
    {
        NtypeLocal[i] = 0;
    }

    gravcost = sphcost = 0;

#ifdef SUBFIND
    if (GrNr >= 0)
    {
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].GrNr != GrNr)
            {
                continue;
            }
            NtypeLocal[P[i].Type]++;

            gravcost += domain_particle_costfactor(i);

            if (P[i].Type == 0)
            {
                if (TimeBinActive[P[i].TimeBin] || UseAllParticles)
                {
                    sphcost += 1.0;
                }
            }
        }
    }
    else
    {
#endif // SUBFIND

#ifdef _OPENMP
        int NtypeLocal_tmp[6];

        for (int k = 0; k < 6; k++)
        {
            NtypeLocal_tmp[k] = 0;
        }
#pragma omp parallel shared(NtypeLocal_tmp) firstprivate(NtypeLocal)
        {
#pragma omp for reduction(+ : gravcost, sphcost)
#pragma ivdep
#endif // _OPENMP
            for (int i = 0; i < NumPart; i++)
            {
                NtypeLocal[P[i].Type]++;

                gravcost += domain_particle_costfactor(i);
                if (P[i].Type == 0)
                {
                    if (TimeBinActive[P[i].TimeBin] || UseAllParticles)
                    {
                        sphcost += 1.0;
                    }
                }
            }
#ifdef _OPENMP
            for (int k = 0; k < 6; k++)
#pragma omp atomic
                NtypeLocal_tmp[k] += NtypeLocal[k];
        }
        for (int k = 0; k < 6; k++)
        {
            NtypeLocal[k] = NtypeLocal_tmp[k];
        }
#endif // _OPENMP

#ifdef SUBFIND
    }
#endif // SUBFIND

    /* because Ntype[] is of type `long long', we cannot do a simple
     * MPI_Allreduce() to sum the total particle numbers
     */
    sumup_large_ints(6, NtypeLocal, Ntype);

    totpartcount = 0;
    for (int i = 0; i < 6; i++)
    {
        totpartcount += Ntype[i];
    }

    MPI_Allreduce(&gravcost, &totgravcost, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&sphcost, &totsphcost, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    TIMER_POP(3, "compute_gravcost"); // compute_gravcost

    /* determine global dimensions of domain grid */
#ifdef WRITE_KEY_FILES
    TIMER_PUSH(3, "find_extend");
    domain_findExtent(SaveKeys);
    TIMER_POP(3, "find_extend");
    TIMER_PUSH(1, "DOM_DET_TOP");

    if (domain_determineTopTree(SaveKeys, multipledomains))
    {
        TIMER_POP(1, "DOM_DET_TOP");
        return 1;
    }
    TIMER_POP(1, "DOM_DET_TOP");
#else // WRITE_KEY_FILES
    TIMER_PUSH(3, "find_extend");
    domain_findExtent();
    TIMER_POP(3, "find_extend");
    TIMER_PUSH(1, "DOM_DET_TOP");
    if (domain_determineTopTree())
    {
        TIMER_POP(1, "DOM_DET_TOP");
        return 1;
    }
    TIMER_POP(1, "DOM_DET_TOP");
#endif // WRITE_KEY_FILES

#ifdef WRITE_KEY_FILES
    if (SaveKeys == 1)
    {
        TIMER_PUSH(3, "find_split_work");
        /* find the split of the domain grid */
        domain_findSplit_load_balanced(multipledomains * NTask, NTopleaves);
        TIMER_POPPUSH(3, "find_split_work", "assign_load_or_work"); // DOMAIN_find_Split_work
        domain_assign_load_or_work_balanced(0, multipledomains);
        TIMER_POPPUSH(3, "assign_load_or_work", "check_memory_bound"); // DOMAIN_assign_load_or_work
        status = domain_check_memory_bound(multipledomains);
        TIMER_POP(3, "find_split_work"); // DOMAIN_check_memory_bound

        ENDRUN_IF(status != 0, "No domain decomposition for key files that stays within memory bounds is possible.");
    }
    else
    {
#endif // WRITE_KEY_FILES
        /* find the split of the domain grid */
        TIMER_PUSH(3, "find_split_work");
        domain_findSplit_work_balanced(multipledomains * NTask, NTopleaves);
        TIMER_POPPUSH(3, "find_split_work", "assign_load_or_work"); // find_Split_work

        domain_assign_load_or_work_balanced(1, multipledomains);

        TIMER_POPPUSH(3, "assign_load_or_work", "check_memory_bound"); // assign_load_or_work
        status = domain_check_memory_bound(multipledomains);

        TIMER_POP(3, "check_memory_bound"); // DOMAIN_check_memory_bound

        if (status != 0) /* the optimum balanced solution violates memory constraint, let's try something different */
        {
            VERBOSE(0, "DOMAIN: Note: the domain decomposition is suboptimum because the ceiling for memory-imbalance "
                       "is reached");
            TIMER_PUSH(3, "find_split_load");
            domain_findSplit_load_balanced(multipledomains * NTask, NTopleaves);
            TIMER_POPPUSH(3, "find_split_load", "assign_load_or_work"); // find_Split_load
            domain_assign_load_or_work_balanced(0, multipledomains);
            TIMER_POPPUSH(3, "assign_load_or_work", "check_memory"); // assign_load_or_work
            status = domain_check_memory_bound(multipledomains);
            TIMER_POP(3, "check_memory"); // check_memory

            ENDRUN_IF(status != 0, "No domain decomposition that stays within memory bounds is possible.");
        }
#ifdef WRITE_KEY_FILES
    }
#endif // WRITE_KEY_FILES

    if (ThisTask == 0)
    {
        sumload = maxload = sumloadsph = maxloadsph = 0;
        sumwork = sumworksph = maxwork = maxworksph = 0;
#if defined(SFR)
        sumloadstars = maxloadstars = 0;
#endif // SFR

#ifdef BLACK_HOLES
        sumloadbhs = maxloadbhs = 0;
#endif // BLACK_HOLES

        for (int i = 0; i < NTask; i++)
        {
            sumload += list_load[i];
            sumloadsph += list_loadsph[i];
            sumwork += list_work[i];
            sumworksph += list_worksph[i];
#if defined(SFR)
            sumloadstars += list_loadstars[i];
#endif // SFR

#ifdef BLACK_HOLES
            sumloadbhs += list_loadBHs[i];
#endif // BLACK_HOLES

            if (list_load[i] > maxload)
            {
                maxload = list_load[i];
            }

            if (list_loadsph[i] > maxloadsph)
            {
                maxloadsph = list_loadsph[i];
            }

#if defined(SFR)
            if (list_loadstars[i] > maxloadstars)
            {
                maxloadstars = list_loadstars[i];
            }
#endif // SFR

#ifdef BLACK_HOLES
            if (list_loadBHs[i] > maxloadbhs)
            {
                maxloadbhs = list_loadBHs[i];
            }
#endif // BLACK_HOLES

            if (list_work[i] > maxwork)
            {
                maxwork = list_work[i];
            }

            if (list_worksph[i] > maxworksph)
            {
                maxworksph = list_worksph[i];
            }
        }

#ifdef WRITE_KEY_FILES
        if (SaveKeys == 1)
        {
            VERBOSE_ALL(2, "snap files using keys: memory-balance=%g, memory-balance-sph=%g",
                        maxload / (((double)sumload) / NTask), maxloadsph / (((double)sumloadsph) / NTask));
#if defined(SFR)
            VERBOSE_ALL(2, ", memory-balance-stars=%g", maxloadstars / (((double)sumloadstars) / NTask));
#endif // SFR

#ifdef BLACK_HOLES
            VERBOSE_ALL(2, ", memory-balance-bhs=%g", maxloadbhs / (((double)sumloadbhs) / NTask));
#endif // BLACK_HOLES
        }
        else
#endif // WRITE_KEY_FILES
        {
            VERBOSE_ALL(2, "gravity work-load balance=%g   memory-balance=%g   SPH work-load balance=%g",
                        maxwork / (sumwork / NTask), maxload / (((double)sumload) / NTask),
                        maxworksph / ((sumworksph + 1.0e-30) / NTask));
#if defined(SFR)
            VERBOSE_ALL(2, ", memory-balance-stars=%g", maxloadstars / (((double)sumloadstars) / NTask));
#endif // SFR

#ifdef BLACK_HOLES
            VERBOSE_ALL(2, ", memory-balance-bhs=%g", maxloadbhs / (((double)sumloadbhs) / NTask));
#endif // BLACK_HOLES
        }
    }

    /* flag the particles that need to be exported */
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
#ifdef SUBFIND
        if (GrNr >= 0 && P[i].GrNr != GrNr)
        {
            continue;
        }
#endif // SUBFIND

        int no = 0;

        while (topNodes[no].Daughter >= 0)
        {
            no = topNodes[no].Daughter + (Key[i] - topNodes[no].StartKey) / (topNodes[no].Size / 8);
        }

        no = topNodes[no].Leaf;

        int task = DomainTask[no];

        if (task != ThisTask)
        {
            P[i].Type |= 32;
        }
    }
    int iter = 0, ret;
    size_t exchange_limit;

    TIMER_PUSH(1, "DOM_COMM");
    do
    {
        exchange_limit = FreeBytes - NTask * (24 * sizeof(int) + 22 * sizeof(MPI_Request));

#ifdef MPISENDRECV_CHECKSUM
        exchange_limit -= All.BufferSize * 1024 * 1024;
#endif // MPISENDRECV_CHECKSUM
        ENDRUN_IF(exchange_limit <= 0, "task=%d: exchange_limit=%d", ThisTask, (int)exchange_limit);

        /* determine for each cpu how many particles have to be shifted to other cpus */
        ret = domain_countToGo(exchange_limit);

        sumtogo = 0;
        for (int i = 0; i < NTask; i++)
        {
            sumtogo += toGo[i];
        }

        sumup_longs(1, &sumtogo, &sumtogo);

        VERBOSE(2, "iter=%d exchange of %d%09d particles (ret=%d)", iter, (int)(sumtogo / 1000000000),
                (int)(sumtogo % 1000000000), ret);

        domain_exchange();
        iter++;
    } while (ret > 0);
    TIMER_POP(1, "DOM_COMM"); // domain_exchange

    return 0;
}

/**
 * @brief Checks if the memory usage of domains is within bounds.
 *
 * This function checks if the memory usage of domains is within specified bounds.
 * If the memory usage exceeds the limits, it prints the desired memory imbalance and domain information.
 *
 * @param multipledomains Number of multiple domains.
 * @return Returns 1 if memory usage exceeds the limits, otherwise returns 0.
 */
int domain_check_memory_bound(int multipledomains)
{
    int load, sphload;
    int max_load = 0, max_sphload = 0;

#if defined(SFR)
    int max_starsload = 0;
#endif // SFR

#ifdef BLACK_HOLES
    int max_BHsload = 0;
#endif // BLACK_HOLES

    for (int ta = 0; ta < NTask; ta++)
    {
#if defined(SFR)
        int starsload = 0;
#endif // SFR
#ifdef BLACK_HOLES
        int BHsload = 0;
#endif // BLACK_HOLES
        load = sphload = 0;
        double work = 0;
        double worksph = 0;

        for (int m = 0; m < multipledomains; m++)
        {
            for (int i = DomainStartList[ta * multipledomains + m]; i <= DomainEndList[ta * multipledomains + m]; i++)
            {
                load += domainCount[i];
                sphload += domainCountSph[i];
                work += domainWork[i];
                worksph += domainWorkSph[i];
#if defined(SFR)
                starsload += domainCountStars[i];
#endif // SFR

#ifdef BLACK_HOLES
                BHsload += domainCountBHs[i];
#endif // BLACK_HOLES
            }
        }

        list_load[ta] = load;
        list_loadsph[ta] = sphload;
        list_work[ta] = work;
        list_worksph[ta] = worksph;
#if defined(SFR)
        list_loadstars[ta] = starsload;
#endif // SFR

#ifdef BLACK_HOLES
        list_loadBHs[ta] = BHsload;
#endif // BLACK_HOLES

        if (load > max_load)
        {
            max_load = load;
        }
        if (sphload > max_sphload)
        {
            max_sphload = sphload;
        }
#if defined(SFR)
        if (starsload > max_starsload)
        {
            max_starsload = starsload;
        }
#endif // SFR

#ifdef BLACK_HOLES
        if (BHsload > max_BHsload)
        {
            max_BHsload = BHsload;
        }
#endif // BLACK_HOLES
    }

#ifdef SUBFIND
    if (GrNr >= 0)
    {
        [[maybe_unused]] int starsload = 0;
        [[maybe_unused]] int BHsload = 0;
        load = sphload = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : load, sphload, starsload, BHsload)
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].GrNr != GrNr)
            {
                load++;
                if (P[i].Type == 0)
                {
                    sphload++;
                }
#if defined(SFR)
                if (P[i].Type == 4)
                {
                    starsload++;
                }
#endif // SFR

#ifdef BLACK_HOLES
                if (P[i].Type == 5)
                {
                    BHsload++;
                }
#endif // BLACK_HOLES
            }
        }

        load += list_load[ThisTask];
        if (load > max_load)
        {
            max_load = load;
        }
        else
        {
            load = max_load;
        }

        sphload += list_loadsph[ThisTask];
        if (sphload > max_sphload)
        {
            max_sphload = sphload;
        }
        else
        {
            sphload = max_sphload;
        }

#if defined(SFR)
        starsload += list_loadstars[ThisTask];
        if (starsload > max_starsload)
        {
            max_starsload = starsload;
        }
        else
        {
            starsload = max_starsload;
        }
#endif // SFR

#ifdef BLACK_HOLES
        BHsload += list_loadBHs[ThisTask];
        if (BHsload > max_BHsload)
        {
            max_BHsload = BHsload;
        }
        else
        {
            BHsload = max_BHsload;
        }
#endif // BLACK_HOLES

        MPI_Allreduce(&load, &max_load, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
        MPI_Allreduce(&sphload, &max_sphload, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);

#if defined(SFR)
        MPI_Allreduce(&starsload, &max_starsload, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
#endif // SFR

#ifdef BLACK_HOLES
        MPI_Allreduce(&BHsload, &max_BHsload, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
#endif // BLACK_HOLES
    }
#endif // SUBFIND

    if (max_load > maxLoad)
    {
        VERBOSE(2, "desired memory imbalance=%g  (limit=%d, needed=%d)", (max_load * All.PartAllocFactor) / maxLoad,
                maxLoad, max_load);
        domain_print();
        return 1;
    }

    if (max_sphload > maxLoadsph)
    {
        VERBOSE(2, "desired memory imbalance=%g  (SPH) (limit=%d, needed=%d)",
                (max_sphload * All.PartAllocFactor) / maxLoadsph, maxLoadsph, max_sphload);
        domain_print();
        return 1;
    }

#if defined(SFR)
    if (max_starsload > maxLoadstars)
    {
        VERBOSE(2, "   Task %d: desired memory imbalance=%g  (STARS) (limit=%d, needed=%d)", ThisTask,
                (max_starsload * All.PartAllocFactor) / maxLoadstars, maxLoadstars, max_starsload);
        domain_print();
        return 1;
    }
#endif // SFR

#ifdef BLACK_HOLES
    if (max_BHsload > maxLoadBHs)
    {
        VERBOSE(2, "desired memory imbalance=%g  (BHs) (limit=%d, needed=%d)",
                (max_BHsload * All.PartAllocFactor) / maxLoadBHs, maxLoadBHs, max_BHsload);
        domain_print();
        return 1;
    }
#endif // BLACK_HOLES
    return 0;
}

/**
 * @brief Prints detailed load distribution and workload information across tasks.
 *
 * The function provides verbose logging of the load distribution and workload associated with different tasks.
 * The output includes information on different types of loads, such as particle loads, stellar loads (if applicable),
 * and black hole loads (if applicable).
 */
void domain_print(void)
{
    VERBOSE(2, "load loadsph work worksph");
#if defined(SFR)
    VERBOSE(2, " loadstars");
#endif // SFR
#ifdef BLACK_HOLES
    VERBOSE(2, " loadBHs");
#endif // BLACK_HOLES

    MPI_Barrier(MYMPI_COMM_WORLD);

    for (int ta = 0; ta < NTask; ta++)
    {

        VERBOSE(2, "%d ", ThisTask);
        VERBOSE(2, "%d %d %g %g ", list_load[ta], list_loadsph[ta], list_work[ta], list_worksph[ta]);
#if defined(SFR)
        VERBOSE(2, "%d ", list_loadstars[ta]);
#endif // SFR

#ifdef BLACK_HOLES
        VERBOSE(2, "%d ", list_loadBHs[ta]);
#endif // BLACK_HOLES

        MPI_Barrier(MYMPI_COMM_WORLD);
    }

    MPI_Barrier(MYMPI_COMM_WORLD);

#ifdef SUBFIND
    VERBOSE(2, "load loadsph");
#if defined(SFR)
    VERBOSE(2, " loadstars");
#endif // SFR
#ifdef BLACK_HOLES
    VERBOSE(2, " loadBHs");
#endif // BLACK_HOLES
    VERBOSE(2, "  (GrNr=%d) ", GrNr);

    if (GrNr >= 0)
    {
        MPI_Barrier(MYMPI_COMM_WORLD);

        for (int ta = 0; ta < NTask; ta++)
        {
            if (ThisTask == ta)
            {
                int load = 0;
                int sphload = 0;
#if defined(SFR)
                int starsload = 0;
#endif // SFR

#ifdef BLACK_HOLES
                int BHsload = 0;
#endif // BLACK_HOLES

                for (int i = 0; i < NumPart; i++)
                {
                    if (P[i].GrNr != GrNr)
                    {
                        load++;
                        if (P[i].Type == 0)
                        {
                            sphload++;
                        }
#if defined(SFR)
                        if (P[i].Type == 4)
                        {
                            starsload++;
                        }
#endif // SFR

#ifdef BLACK_HOLES
                        if (P[i].Type == 5)
                        {
                            BHsload++;
                        }
#endif // BLACK_HOLES
                    }
                }
                VERBOSE_ALL(2, "%d ", ThisTask);
                VERBOSE_ALL(2, "%d %d ", load, sphload);
#if defined(SFR)
                VERBOSE_ALL(2, "%d ", starsload);
#endif // SFR

#ifdef BLACK_HOLES
                VERBOSE_ALL(2, "%d ", BHsload);
#endif // BLACK_HOLES
            }
            MPI_Barrier(MYMPI_COMM_WORLD);
        }
    }
#endif // SUBFIND
    MPI_Barrier(MYMPI_COMM_WORLD);
}

/**
 * @brief Exchange particles between domain tasks.
 *
 * This function exchanges particles between domain tasks in a parallel simulation.
 * It allocates memory for various arrays and buffers, assigns particles to their
 * corresponding tasks based on their position in the domain, and then deallocates
 * the memory after the exchange is complete.
 */
void domain_exchange(void)
{
    int count_togo = 0, count_togo_sph = 0, count_get = 0, count_get_sph = 0;
    int *count, *count_sph, *offset, *offset_sph;
    int *count_recv, *count_recv_sph, *offset_recv, *offset_recv_sph;
    int no, target;
    struct particle_data *partBuf;
    struct sph_particle_data *sphBuf;
    peanokey *keyBuf;

    count = (int *)mymalloc("count", NTask * sizeof(int));
    count_sph = (int *)mymalloc("count_sph", NTask * sizeof(int));
    offset = (int *)mymalloc("offset", NTask * sizeof(int));
    offset_sph = (int *)mymalloc("offset_sph", NTask * sizeof(int));

    count_recv = (int *)mymalloc("count_recv", NTask * sizeof(int));
    count_recv_sph = (int *)mymalloc("count_recv_sph", NTask * sizeof(int));
    offset_recv = (int *)mymalloc("offset_recv", NTask * sizeof(int));
    offset_recv_sph = (int *)mymalloc("offset_recv_sph", NTask * sizeof(int));

#ifdef SFR
    int count_togo_stars = 0, count_get_stars = 0;
    int *count_stars, *offset_stars;
    int *count_recv_stars, *offset_recv_stars;
    struct met_particle_data *metBuf;

    count_stars = (int *)mymalloc("count_stars", NTask * sizeof(int));
    offset_stars = (int *)mymalloc("offset_stars", NTask * sizeof(int));
    count_recv_stars = (int *)mymalloc("count_recv_stars", NTask * sizeof(int));
    offset_recv_stars = (int *)mymalloc("offset_recv_stars", NTask * sizeof(int));

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < N_stars; i++)
    {
        ENDRUN_IF((MetP[i].PID >= NumPart || (P[MetP[i].PID].Type & 15) != 4), "Task %d: i=%d/%d, pid=%u, type=%d",
                  ThisTask, i, N_stars, MetP[i].PID, P[MetP[i].PID].Type);
    }
#endif // SFR

#ifdef BLACK_HOLES
    int count_togo_BHs = 0, count_get_BHs = 0;
    int *count_BHs, *offset_BHs;
    int *count_recv_BHs, *offset_recv_BHs;
    struct bh_particle_data *BHBuf;

    count_BHs = (int *)mymalloc("count_BHs", NTask * sizeof(int));
    offset_BHs = (int *)mymalloc("offset_BHs", NTask * sizeof(int));
    count_recv_BHs = (int *)mymalloc("count_recv_BHs", NTask * sizeof(int));
    offset_recv_BHs = (int *)mymalloc("offset_recv_BHs", NTask * sizeof(int));

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < N_BHs; i++)
    {
        ENDRUN_IF((BHP[i].PID >= NumPart || (P[BHP[i].PID].Type & 15) != 5), "Task %d: i=%d/%d, pid=%u, type=%d",
                  ThisTask, i, N_BHs, BHP[i].PID, P[BHP[i].PID].Type);
    }
#endif // BLACK_HOLES

    int prec_offset, prec_count;
    int *decrease;

    decrease = (int *)mymalloc("decrease", NTask * sizeof(int));

    offset_sph[0] = 0;
    decrease[0] = 0;

    for (int i = 1; i < NTask; i++)
    {
        offset_sph[i] = offset_sph[i - 1] + toGoSph[i - 1];
        decrease[i] = toGoSph[i - 1];
    }

    prec_offset = offset_sph[NTask - 1] + toGoSph[NTask - 1];

#if defined(SFR)
    offset_stars[0] = prec_offset;
    for (int i = 1; i < NTask; i++)
    {
        offset_stars[i] = offset_stars[i - 1] + toGoStars[i - 1];
        decrease[i] += toGoStars[i - 1];
    }
    prec_offset = offset_stars[NTask - 1] + toGoStars[NTask - 1];
#endif // SFR

#ifdef BLACK_HOLES
    offset_BHs[0] = prec_offset;
    for (int i = 1; i < NTask; i++)
    {
        offset_BHs[i] = offset_BHs[i - 1] + toGoBHs[i - 1];
        decrease[i] += toGoBHs[i - 1];
    }
    prec_offset = offset_BHs[NTask - 1] + toGoBHs[NTask - 1];
#endif // BLACK_HOLES

    offset[0] = prec_offset;
    for (int i = 1; i < NTask; i++)
    {
        offset[i] = offset[i - 1] + (toGo[i - 1] - decrease[i]);
    }

    myfree(decrease);

    for (int i = 0; i < NTask; i++)
    {
        count_togo += toGo[i];
        count_togo_sph += toGoSph[i];

        count_get += toGet[i];
        count_get_sph += toGetSph[i];

#if defined(SFR)
        count_togo_stars += toGoStars[i];
        count_get_stars += toGetStars[i];
#endif // SFR

#ifdef BLACK_HOLES
        count_togo_BHs += toGoBHs[i];
        count_get_BHs += toGetBHs[i];
#endif // BLACK_HOLES
    }

    partBuf = (struct particle_data *)mymalloc("partBuf", count_togo * sizeof(struct particle_data));
    sphBuf = (struct sph_particle_data *)mymalloc("sphBuf", count_togo_sph * sizeof(struct sph_particle_data));
    keyBuf = (peanokey *)mymalloc("keyBuf", count_togo * sizeof(peanokey));

    for (int i = 0; i < NTask; i++)
    {
        count[i] = count_sph[i] = 0;
    }

#ifdef SFR
    metBuf = (struct met_particle_data *)mymalloc("metBuf", count_togo_stars * sizeof(struct met_particle_data));
    for (int i = 0; i < NTask; i++)
    {
        count_stars[i] = 0;
    }
#endif // SFR

#ifdef BLACK_HOLES
    BHBuf = (struct bh_particle_data *)mymalloc("BHBuf", count_togo_BHs * sizeof(struct bh_particle_data));
    for (int i = 0; i < NTask; i++)
    {
        count_BHs[i] = 0;
    }
#endif // BLACK_HOLES

    for (int n = 0; n < NumPart; n++)
    {
        if ((P[n].Type & (32 + 16)) == (32 + 16))
        {
            P[n].Type &= 15;

            no = 0;

            while (topNodes[no].Daughter >= 0)
            {
                no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size / 8);
            }

            no = topNodes[no].Leaf;

            target = DomainTask[no];

            if (P[n].Type == 0)
            {
                partBuf[offset_sph[target] + count_sph[target]] = P[n];
                keyBuf[offset_sph[target] + count_sph[target]] = Key[n];
                memcpy(&sphBuf[offset_sph[target] + count_sph[target]], &SphP[n], sizeof(sph_particle_data));
                count_sph[target]++;
            }
#ifdef SFR
            else if (P[n].Type == 4)
            {
                partBuf[offset_stars[target] + count_stars[target]] = P[n];
                keyBuf[offset_stars[target] + count_stars[target]] = Key[n];
                metBuf[offset_stars[target] - offset_stars[0] + count_stars[target]] = MetP[P[n].pt.MetID];
                count_stars[target]++;
            }
#endif // SFR
#ifdef BLACK_HOLES
            else if (P[n].Type == 5)
            {
                partBuf[offset_BHs[target] + count_BHs[target]] = P[n];
                keyBuf[offset_BHs[target] + count_BHs[target]] = Key[n];
                BHBuf[offset_BHs[target] - offset_BHs[0] + count_BHs[target]] = BHP[P[n].pt.BHID];
                count_BHs[target]++;
            }
#endif // BLACK_HOLES
            else
            {
                partBuf[offset[target] + count[target]] = P[n];
                keyBuf[offset[target] + count[target]] = Key[n];
                count[target]++;
            }

            if (P[n].Type == 0)
            {
                P[n] = P[N_gas - 1];
                SphP[n] = SphP[N_gas - 1];
                Key[n] = Key[N_gas - 1];

                P[N_gas - 1] = P[NumPart - 1];
                Key[N_gas - 1] = Key[NumPart - 1];
#ifdef SFR
                if ((P[N_gas - 1].Type & 15) == 4)
                {
                    MetP[P[N_gas - 1].pt.MetID].PID = N_gas - 1;
                }
#endif // SFR

#ifdef BLACK_HOLES
                if ((P[N_gas - 1].Type & 15) == 5)
                {
                    BHP[P[N_gas - 1].pt.BHID].PID = N_gas - 1;
                }
#endif // BLACK_HOLES

                NumPart--;
                N_gas--;
                n--;
            }
#ifdef SFR
            else if (P[n].Type == 4)
            {
                MetP[P[n].pt.MetID] = MetP[N_stars - 1];
                P[MetP[N_stars - 1].PID].pt.MetID = P[n].pt.MetID;

                if (n < NumPart - 1)
                {
                    P[n] = P[NumPart - 1];
                    Key[n] = Key[NumPart - 1];
                    if ((P[n].Type & 15) == 4)
                    {
                        MetP[P[n].pt.MetID].PID = n;
                    }
#ifdef BLACK_HOLES
                    if ((P[n].Type & 15) == 5)
                    {
                        BHP[P[n].pt.BHID].PID = n;
                    }
#endif // BLACK_HOLES
                }

                NumPart--;
                N_stars--;
                n--;
            }
#endif // SFR
#ifdef BLACK_HOLES
            else if (P[n].Type == 5)
            {
                BHP[P[n].pt.BHID] = BHP[N_BHs - 1];
                P[BHP[N_BHs - 1].PID].pt.BHID = P[n].pt.BHID;

                if (n < NumPart - 1)
                {
                    P[n] = P[NumPart - 1];
                    Key[n] = Key[NumPart - 1];
                    if ((P[n].Type & 15) == 5)
                    {
                        BHP[P[n].pt.BHID].PID = n;
                    }
#ifdef SFR
                    if ((P[n].Type & 15) == 4)
                    {
                        MetP[P[n].pt.MetID].PID = n;
                    }
#endif // SFR
                }

                NumPart--;
                N_BHs--;
                n--;
            }
#endif // BLACK_HOLES
            else
            {
                P[n] = P[NumPart - 1];
                Key[n] = Key[NumPart - 1];
#ifdef SFR
                if ((P[n].Type & 15) == 4)
                {
                    MetP[P[n].pt.MetID].PID = n;
                }
#endif // SFR

#ifdef BLACK_HOLES
                if ((P[n].Type & 15) == 5)
                {
                    BHP[P[n].pt.BHID].PID = n;
                }
#endif // BLACK_HOLES
                NumPart--;
                n--;
            }
        }
    }

    int count_totget;

    count_totget = count_get_sph;
#if defined(SFR)
    count_totget += count_get_stars;
#endif // SFR

#ifdef BLACK_HOLES
    count_totget += count_get_BHs;
#endif // BLACK_HOLES

    if (count_totget)
    {
        memmove(P + N_gas + count_totget, P + N_gas, (NumPart - N_gas) * sizeof(struct particle_data));
        memmove(Key + N_gas + count_totget, Key + N_gas, (NumPart - N_gas) * sizeof(peanokey));
    }

#ifdef SFR
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int n = 0; n < N_stars; n++)
    {
        MetP[n].PID += count_totget;
        ENDRUN_IF(P[MetP[n].PID].pt.MetID != n,
                  "[Task %d] some serious error in adjusting the memory before particle exchange", ThisTask);
    }
#endif // SFR

#ifdef BLACK_HOLES
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int n = 0; n < N_BHs; n++)
    {
        BHP[n].PID += count_totget;
        ENDRUN_IF(P[BHP[n].PID].pt.BHID != n,
                  "[Task %d] some serious error in adjusting the memory before particle exchange", ThisTask);
    }
#endif // BLACK_HOLES

    for (int i = 0; i < NTask; i++)
    {
        count_recv_sph[i] = toGetSph[i];
        count_recv[i] = toGet[i] - toGetSph[i];
#if defined(SFR)
        count_recv_stars[i] = toGetStars[i];
        count_recv[i] -= toGetStars[i];
#endif // SFR

#ifdef BLACK_HOLES
        count_recv_BHs[i] = toGetBHs[i];
        count_recv[i] -= toGetBHs[i];
#endif // BLACK_HOLES
    }

    offset_recv_sph[0] = N_gas;
    for (int i = 1; i < NTask; i++)
    {
        offset_recv_sph[i] = offset_recv_sph[i - 1] + count_recv_sph[i - 1];
    }
    prec_count = N_gas + count_get_sph;

#if defined(SFR)
    offset_recv_stars[0] = prec_count;
    for (int i = 1; i < NTask; i++)
    {
        offset_recv_stars[i] = offset_recv_stars[i - 1] + count_recv_stars[i - 1];
    }
    prec_count += count_get_stars;
#endif // SFR

#ifdef BLACK_HOLES
    offset_recv_BHs[0] = prec_count;
    for (int i = 1; i < NTask; i++)
    {
        offset_recv_BHs[i] = offset_recv_BHs[i - 1] + count_recv_BHs[i - 1];
    }
    prec_count += count_get_BHs;
#endif // BLACK_HOLES

    offset_recv[0] = NumPart - N_gas + prec_count;

    for (int i = 1; i < NTask; i++)
    {
        offset_recv[i] = offset_recv[i - 1] + count_recv[i - 1];
    }

#ifdef USE_ISEND_IRECV
    int n_requests = 0, max_requests = 10;
    MPI_Request *requests;

#if defined(SFR)
    max_requests += 6;
#endif // SFR

#ifdef BLACK_HOLES
    max_requests += 6;
#endif // BLACK_HOLES

    requests = (MPI_Request *)mymalloc("requests", max_requests * NTask * sizeof(MPI_Request));

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (count_recv_sph[target] > 0)
            {
                MPI_Irecv(P + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_SPH, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Irecv(Key + offset_recv_sph[target], count_recv_sph[target] * sizeof(peanokey), MPI_BYTE, target,
                          TAG_KEY_SPH, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Irecv(SphP + offset_recv_sph[target], count_recv_sph[target] * sizeof(struct sph_particle_data),
                          MPI_BYTE, target, TAG_SPHDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
            }

#if defined(SFR)
            if (count_recv_stars[target] > 0)
            {
                MPI_Irecv(P + offset_recv_stars[target], count_recv_stars[target] * sizeof(struct particle_data),
                          MPI_BYTE, target, TAG_PDATA_STARS, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Irecv(Key + offset_recv_stars[target], count_recv_stars[target] * sizeof(peanokey), MPI_BYTE,
                          target, TAG_KEY_STARS, MYMPI_COMM_WORLD, &requests[n_requests++]);
                MPI_Irecv(MetP + N_stars + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
                              count_recv_sph[NTask - 1],
                          count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
#endif // SFR

#ifdef BLACK_HOLES
            if (count_recv_BHs[target] > 0)
            {
                MPI_Irecv(P + offset_recv_BHs[target], count_recv_BHs[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_BHS, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Irecv(Key + offset_recv_BHs[target], count_recv_BHs[target] * sizeof(peanokey), MPI_BYTE, target,
                          TAG_KEY_BHS, MYMPI_COMM_WORLD, &requests[n_requests++]);

#ifdef SFR
                MPI_Irecv(BHP + N_BHs + offset_recv_BHs[target] - offset_recv_stars[NTask - 1] -
                              count_recv_stars[NTask - 1],
                          count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
#else // SFR
                MPI_Irecv(BHP + N_BHs + offset_recv_BHs[target],
                          count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
#endif // SFR
            }
#endif // BLACK_HOLES

            if (count_recv[target] > 0)
            {
                MPI_Irecv(P + offset_recv[target], count_recv[target] * sizeof(struct particle_data), MPI_BYTE, target,
                          TAG_PDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Irecv(Key + offset_recv[target], count_recv[target] * sizeof(peanokey), MPI_BYTE, target, TAG_KEY,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD); /* not really necessary, but this will guarantee that all receives are
                      posted before the sends, which helps the stability of MPI on
                      bluegene, and perhaps some mpich1-clusters */

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (count_sph[target] > 0)
            {
                MPI_Isend(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_SPH, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Isend(keyBuf + offset_sph[target], count_sph[target] * sizeof(peanokey), MPI_BYTE, target,
                          TAG_KEY_SPH, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Isend(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE,
                          target, TAG_SPHDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
            }

#if defined(SFR)

            if (count_stars[target] > 0)
            {
                MPI_Isend(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_STARS, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Isend(keyBuf + offset_stars[target], count_stars[target] * sizeof(peanokey), MPI_BYTE, target,
                          TAG_KEY_STARS, MYMPI_COMM_WORLD, &requests[n_requests++]);
                MPI_Isend(metBuf + offset_stars[target] - offset_sph[NTask - 1] - count_sph[NTask - 1],
                          count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
#endif // SFR

#ifdef BLACK_HOLES
            if (count_BHs[target] > 0)
            {
                MPI_Isend(partBuf + offset_BHs[target], count_BHs[target] * sizeof(struct particle_data), MPI_BYTE,
                          target, TAG_PDATA_BHS, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Isend(keyBuf + offset_BHs[target], count_BHs[target] * sizeof(peanokey), MPI_BYTE, target,
                          TAG_KEY_BHS, MYMPI_COMM_WORLD, &requests[n_requests++]);

#ifdef SFR
                MPI_Isend(BHBuf + offset_BHs[target] - offset_stars[NTask - 1] - count_stars[NTask - 1],
                          count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
#else // SFR
                MPI_Isend(BHBuf + offset_BHs[target], count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE,
                          target, TAG_BHDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);
#endif // SFR
            }
#endif // BLACK_HOLES

            if (count[target] > 0)
            {
                MPI_Isend(partBuf + offset[target], count[target] * sizeof(struct particle_data), MPI_BYTE, target,
                          TAG_PDATA, MYMPI_COMM_WORLD, &requests[n_requests++]);

                MPI_Isend(keyBuf + offset[target], count[target] * sizeof(peanokey), MPI_BYTE, target, TAG_KEY,
                          MYMPI_COMM_WORLD, &requests[n_requests++]);
            }
        }
    }

    MPI_Waitall(n_requests, requests, MPI_STATUSES_IGNORE);

    ENDRUN_IF(n_requests > max_requests * NTask, "Not enough memory reserved for requests: %d > %d !", n_requests,
              max_requests * NTask);

    myfree(requests);
#else // USE_ISEND_IRECV

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        target = ThisTask ^ ngrp;

        if (target < NTask)
        {
            if (count_sph[target] > 0 || count_recv_sph[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset_sph[target], count_sph[target] * sizeof(struct particle_data), MPI_BYTE,
                             target, TAG_PDATA_SPH, P + offset_recv_sph[target],
                             count_recv_sph[target] * sizeof(struct particle_data), MPI_BYTE, target, TAG_PDATA_SPH,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Sendrecv(sphBuf + offset_sph[target], count_sph[target] * sizeof(struct sph_particle_data),
                             MPI_BYTE, target, TAG_SPHDATA, SphP + offset_recv_sph[target],
                             count_recv_sph[target] * sizeof(struct sph_particle_data), MPI_BYTE, target, TAG_SPHDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Sendrecv(keyBuf + offset_sph[target], count_sph[target] * sizeof(peanokey), MPI_BYTE, target,
                             TAG_KEY_SPH, Key + offset_recv_sph[target], count_recv_sph[target] * sizeof(peanokey),
                             MPI_BYTE, target, TAG_KEY_SPH, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

#if defined(SFR)
            if (count_stars[target] > 0 || count_recv_stars[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset_stars[target], count_stars[target] * sizeof(struct particle_data),
                             MPI_BYTE, target, TAG_PDATA_STARS, P + offset_recv_stars[target],
                             count_recv_stars[target] * sizeof(struct particle_data), MPI_BYTE, target, TAG_PDATA_STARS,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(metBuf + offset_stars[target] - offset_sph[NTask - 1] - count_sph[NTask - 1],
                             count_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                             MetP + N_stars + offset_recv_stars[target] - offset_recv_sph[NTask - 1] -
                                 count_recv_sph[NTask - 1],
                             count_recv_stars[target] * sizeof(struct met_particle_data), MPI_BYTE, target, TAG_METDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(keyBuf + offset_stars[target], count_stars[target] * sizeof(peanokey), MPI_BYTE, target,
                             TAG_KEY_STARS, Key + offset_recv_stars[target],
                             count_recv_stars[target] * sizeof(peanokey), MPI_BYTE, target, TAG_KEY_STARS,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
#endif // SFR

#ifdef BLACK_HOLES
            if (count_BHs[target] > 0 || count_recv_BHs[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset_BHs[target], count_BHs[target] * sizeof(struct particle_data), MPI_BYTE,
                             target, TAG_PDATA_BHS, P + offset_recv_BHs[target],
                             count_recv_BHs[target] * sizeof(struct particle_data), MPI_BYTE, target, TAG_PDATA_BHS,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

#ifdef SFR
                MPI_Sendrecv(BHBuf + offset_BHs[target] - offset_stars[NTask - 1] - count_stars[NTask - 1],
                             count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                             BHP + N_BHs + offset_recv_BHs[target] - offset_recv_stars[NTask - 1] -
                                 count_recv_stars[NTask - 1],
                             count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else // SFR
                MPI_Sendrecv(BHBuf + offset_BHs[target], count_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE,
                             target, TAG_BHDATA, BHP + N_BHs + offset_recv_BHs[target],
                             count_recv_BHs[target] * sizeof(struct bh_particle_data), MPI_BYTE, target, TAG_BHDATA,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif // SFR

                MPI_Sendrecv(keyBuf + offset_BHs[target], count_BHs[target] * sizeof(peanokey), MPI_BYTE, target,
                             TAG_KEY_BHS, Key + offset_recv_BHs[target], count_recv_BHs[target] * sizeof(peanokey),
                             MPI_BYTE, target, TAG_KEY_BHS, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
#endif // BLACK_HOLES

            if (count[target] > 0 || count_recv[target] > 0)
            {
                MPI_Sendrecv(partBuf + offset[target], count[target] * sizeof(struct particle_data), MPI_BYTE, target,
                             TAG_PDATA, P + offset_recv[target], count_recv[target] * sizeof(struct particle_data),
                             MPI_BYTE, target, TAG_PDATA, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Sendrecv(keyBuf + offset[target], count[target] * sizeof(peanokey), MPI_BYTE, target, TAG_KEY,
                             Key + offset_recv[target], count_recv[target] * sizeof(peanokey), MPI_BYTE, target,
                             TAG_KEY, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
#endif // USE_ISEND_IRECV

    NumPart += count_get;
    N_gas += count_get_sph;
#ifdef SFR
    for (int i = 0; i < count_get_stars; i++)
    {
        P[offset_recv_stars[0] + i].pt.MetID = N_stars + i;
        MetP[N_stars + i].PID = offset_recv_stars[0] + i;
    }
    N_stars += count_get_stars;

    ENDRUN_IF(N_stars > All.MaxPartMet,
              "Task %d: too many stars: %d>%d, got %d ... \nTask %d: have %d/%d particles and %d/%d sph particle",
              ThisTask, N_stars, All.MaxPartMet, count_get_stars, ThisTask, NumPart, All.MaxPart, N_gas,
              All.MaxPartSph);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < N_stars; i++)
    {
        ENDRUN_IF((MetP[i].PID >= NumPart || (P[MetP[i].PID].Type & 15) != 4), "Task %d: i=%d/%d, pid=%u, type=%d",
                  ThisTask, i, N_stars, MetP[i].PID, P[MetP[i].PID].Type);
    }
#endif // SFR

#ifdef BLACK_HOLES
    for (int i = 0; i < count_get_BHs; i++)
    {
        P[offset_recv_BHs[0] + i].pt.BHID = N_BHs + i;
        BHP[N_BHs + i].PID = offset_recv_BHs[0] + i;
    }
    N_BHs += count_get_BHs;

    ENDRUN_IF(N_BHs > All.MaxPartBH, "N Black holes is larger than the maximum number of black holes: %d>%d\n", N_BHs,
              All.MaxPartBH);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < N_BHs; i++)
    {
        ENDRUN_IF((BHP[i].PID >= NumPart || (P[BHP[i].PID].Type & 15) != 5), "Task %d: i=%d/%d, pid=%u, type=%d",
                  ThisTask, i, N_BHs, BHP[i].PID, P[BHP[i].PID].Type);
    }
#endif // BLACK_HOLES

    ENDRUN_IF(NumPart > All.MaxPart, "Task=%d NumPart=%d All.MaxPart=%d", ThisTask, NumPart, All.MaxPart);
    ENDRUN_IF(N_gas > All.MaxPartSph,
              "N gas particles is larger then number of SPH particles. N_gas=%d All.MaxPartSph=%d", N_gas,
              All.MaxPartSph);

#ifdef BLACK_HOLES
    myfree(BHBuf);
#endif // BLACK_HOLES

#ifdef SFR
    myfree(metBuf);
#endif // SFR

    myfree(keyBuf);
    myfree(sphBuf);
    myfree(partBuf);

#ifdef BLACK_HOLES
    myfree(offset_recv_BHs);
    myfree(count_recv_BHs);
    myfree(offset_BHs);
    myfree(count_BHs);
#endif // BLACK_HOLES

#if defined(SFR)
    myfree(offset_recv_stars);
    myfree(count_recv_stars);
    myfree(offset_stars);
    myfree(count_stars);
#endif // SFR

    myfree(offset_recv_sph);
    myfree(offset_recv);
    myfree(count_recv_sph);
    myfree(count_recv);

    myfree(offset_sph);
    myfree(offset);
    myfree(count_sph);
    myfree(count);
}

/**
 * @brief Find balanced split of domains based on workload and particle count.
 *
 * This function calculates a balanced split of domains based on the workload and
 * particle count of each domain. It iterates over the domains, computing the
 * combined workload and particle count, and then determines a split that attempts
 * to distribute the workload evenly across the specified number of CPUs.
 *
 * @param ncpu Number of CPUs.
 * @param ndomain Number of domains.
 */
void domain_findSplit_work_balanced(int ncpu, int ndomain)
{
    double min_work = 1.0, max_work = 0.0;

    double work = 0;
    double load = 0;
    double worksph = 0;
    for (int i = 0; i < ndomain; i++)
    {
        work += domainWork[i];
        load += domainCount[i];
        worksph += domainWorkSph[i];
    }

    double fac_work, fac_load, fac_worksph;
    if (worksph > 0)
    {
        /* in this case we give equal weight to gravitational work-load, SPH work load, and particle load.
         */
        fac_work = 0.333 / work;
        fac_load = 0.333 / load;
        fac_worksph = 0.333 / worksph;
    }
    else
    {
        /* in this case we give equal weight to gravitational work-load and particle load.
         * The final pieces should have at most imbalance 2.0 in either of the two
         */
        fac_work = 0.5 / work;
        fac_load = 0.5 / load;
        fac_worksph = 0.0;
    }

    double workavg = 1.0 / ncpu;
    double workavg_before = 0;
    double work_before = 0;

    int start = 0;
    for (int i = 0; i < ncpu; i++)
    {
        work = 0;
        int end = start;

        work += fac_work * domainWork[end] + fac_load * domainCount[end] + fac_worksph * domainWorkSph[end];

        while ((work + work_before < workavg + workavg_before) || (i == ncpu - 1 && end < ndomain - 1))
        {
            if ((ndomain - end) > (ncpu - i))
            {
                end++;
            }
            else
            {
                break;
            }

            work += fac_work * domainWork[end] + fac_load * domainCount[end] + fac_worksph * domainWorkSph[end];
        }

        DomainStartList[i] = start;
        DomainEndList[i] = end;

        work_before += work;
        workavg_before += workavg;
        start = end + 1;

        if (work < min_work)
        {
            min_work = work;
        }
        if (work > max_work)
        {
            max_work = work;
        }
    }

    VERBOSE(2, "Work balance domain: min=%g avg=%g max=%g", min_work, workavg, max_work);
}

/**
 * @brief Struct representing segments assigned to domains.
 *
 * This struct holds information about segments assigned to different domains,
 * including the domain task number, start and end indices of the segment,
 * workload, particle load, active SPH load, and normalized load.
 */
static struct domain_segments_data
{
    int task, start, end;
    double work;
    double load;
    double load_activesph;
    double normalized_load;
} *domainAssign;

/**
 * @brief Entry of a per-criterion task queue, ordered by ascending load `value`.
 *
 * Each of the 3 queues (work / load / active-SPH-load, see domain_assign_load_or_work_balanced)
 * is kept as an std::multiset<queue_entry_data>. This tree gives O(log NTask) insert/erase.
 * `task` breaks ties so equal-value entries stay distinct multiset
 * elements (their relative order among themselves doesn't matter for correctness).
 */
struct queue_entry_data
{
    double value;
    int task;

    bool operator<(const queue_entry_data &other) const
    {
        if (value != other.value)
        {
            return value < other.value;
        }
        return task < other.task;
    }
};

typedef std::multiset<queue_entry_data> queue_set_t;

/**
 * @brief Struct representing a list of tasks.
 *
 * This struct holds information about tasks, including workload, particle load,
 * active SPH load, and count of tasks.
 */
struct tasklist_data
{
    double work;
    double load;
    double load_activesph;
    int count;
} *tasklist;

/**
 * @brief Compare function for sorting domain segments by task number.
 *
 * This function is used as a comparison function for sorting domain segments
 * based on the task number. It compares the task numbers of two domain segments
 * and returns -1 if the task number of the first segment is less than that of
 * the second segment, +1 if it is greater, and 0 if they are equal.
 *
 * @param a Pointer to the first domain_segments_data struct.
 * @param b Pointer to the second domain_segments_data struct.
 * @return -1 if task of 'a' < task of 'b', +1 if task of 'a' > task of 'b', 0 otherwise.
 */
int domain_sort_task(const void *a, const void *b)
{
    if (((struct domain_segments_data *)a)->task < (((struct domain_segments_data *)b)->task))
    {
        return -1;
    }

    if (((struct domain_segments_data *)a)->task > (((struct domain_segments_data *)b)->task))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compare function for sorting domain segments by normalized load.
 *
 * This function is used as a comparison function for sorting domain segments
 * based on their normalized load. It compares the normalized loads of two domain
 * segments and returns -1 if the normalized load of the first segment is greater
 * than that of the second segment, +1 if it is smaller, and 0 if they are equal.
 *
 * @param a Pointer to the first domain_segments_data struct.
 * @param b Pointer to the second domain_segments_data struct.
 * @return -1 if normalized load of 'a' > normalized load of 'b',
 *          +1 if normalized load of 'a' < normalized load of 'b',
 *          0 otherwise.
 */
int domain_sort_load(const void *a, const void *b)
{
    if (((struct domain_segments_data *)a)->normalized_load > (((struct domain_segments_data *)b)->normalized_load))
    {
        return -1;
    }

    if (((struct domain_segments_data *)a)->normalized_load < (((struct domain_segments_data *)b)->normalized_load))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Assign workload or particle load balanced domains.
 *
 * This function assigns workload or particle load balanced domains based on the specified mode
 * and the number of domains. It calculates the target workload balance, target load balance,
 * and target active SPH load balance. Then, it allocates memory for domainAssign and tasklist,
 * initializes tasklist, and computes the total workload, particle load, and active SPH load.
 * Finally, it iterates over the domains to assign workload or particle load balanced domains.
 *
 * This is purely serial, communication-free bookkeeping: it runs identically and redundantly
 * on every MPI rank over data (domainWork/domainCount/domainWorkSph) that was already reduced
 * to a global, identical-on-every-rank state by the caller, which is what lets every rank reach
 * the same decomposition decision without exchanging any further messages. The per-task
 * candidate lookup used in the greedy assignment loop is backed by an ordered std::multiset
 * per balancing criterion (O(log NTask) find/insert/erase).
 *
 * @param mode Mode indicating whether to balance workload or particle load.
 * @param multipledomains Number of domains multiplied by the number of tasks.
 */
void domain_assign_load_or_work_balanced(int mode, int multipledomains)
{
    double target_work_balance, target_load_balance, target_load_activesph_balance;
    double target_max_balance, best_balance;
    double tot_work, tot_load, tot_loadactivesph;

    int best_queue, target;

    domainAssign = (struct domain_segments_data *)mymalloc("domainAssign", multipledomains * NTask *
                                                                               sizeof(struct domain_segments_data));

    tasklist = (struct tasklist_data *)mymalloc("tasklist", NTask * sizeof(struct tasklist_data));

    // reset the running totals kept for every task (work/load/active-SPH-load/number of chunks already assigned to it)
    for (int ta = 0; ta < NTask; ta++)
    {
        tasklist[ta].work = 0;
        tasklist[ta].load = 0;
        tasklist[ta].load_activesph = 0;
        tasklist[ta].count = 0;
    }

    tot_work = 0;
    tot_load = 0;
    tot_loadactivesph = 0;

    // For every domain chunk currently defined by DomainStartList/DomainEndList, sum up its work, particle load and
    // active-SPH load over the sub-domains it spans, and accumulate the global totals used later to normalize balances.
    for (int n = 0; n < multipledomains * NTask; n++)
    {
        domainAssign[n].start = DomainStartList[n];
        domainAssign[n].end = DomainEndList[n];
        domainAssign[n].work = 0;
        domainAssign[n].load = 0;
        domainAssign[n].load_activesph = 0;

        for (int i = DomainStartList[n]; i <= DomainEndList[n]; i++)
        {
            domainAssign[n].work += domainWork[i];
            domainAssign[n].load += domainCount[i];
            domainAssign[n].load_activesph += domainWorkSph[i];
        }

        // mode == 0 means "balance by particle load only"
        if (mode == 0)
        {
            domainAssign[n].work = 1;
        }

        tot_work += domainAssign[n].work;
        tot_load += domainAssign[n].load;
        tot_loadactivesph += domainAssign[n].load_activesph;
    }

    // Precompute a single "normalized load" score per chunk (fraction of total work plus fraction of total active-SPH
    // load) purely to decide the processing order below; the actual balancing decision uses the three separate
    // criteria, not this combined score.
    for (int n = 0; n < multipledomains * NTask; n++)
    {
        domainAssign[n].normalized_load = domainAssign[n].work / (tot_work + 1.0e-30) +
                                          domainAssign[n].load_activesph / (tot_loadactivesph + 1.0e-30);
    }

    // Sort chunks from heaviest to lightest
    STD_SORT(domainAssign, multipledomains * NTask, sizeof(struct domain_segments_data), domain_sort_load);

    // Build three per-criterion task queues (work / load / active-SPH-load),
    // each an ordered std::multiset<queue_entry_data> whose smallest element is always the
    // currently least-loaded task under that metric.
    std::vector<queue_set_t> queue_set(3);
    std::vector<std::vector<queue_set_t::iterator>> queue_iter(3, std::vector<queue_set_t::iterator>(NTask));

    for (int q = 0; q < 3; q++)
    {
        for (int ta = 0; ta < NTask; ta++)
        {
            queue_iter[q][ta] = queue_set[q].insert({0.0, ta});
        }
    }

#ifdef DOMAIN_NUMA
    // we keep track of which computing node got which mega-chunk
    // note a mega-chunk is composed by  (multipledomains * NTask_per_node) chunks
    NumaArchitecture *n_arch = get_numa_architecture();
    int processes_per_node = n_arch->processes_per_node;
    std::map<int, int> chunk_node;
#endif // DOMAIN_NUMA

    // assign each chunk (already sorted heaviest-first) to a task.
    // for every chunk look at the least-loaded task under each of the three criteria (work/load/active-SPH-load), and
    // pick whichever criterion, if used to place this chunk, would leave the *worst* (max) of the three resulting
    // balances the smallest. i.e. minimize the worst-case imbalance rather than optimizing on a single metric in
    // isolation.
    for (int n = 0; n < multipledomains * NTask; n++)
    {
#ifdef DOMAIN_NUMA
        // find the mega_chunk index (i.e. at which computing node belongs this chunk)
        int mega_chunk_index = n / multipledomains / processes_per_node;
#endif // DOMAIN_NUMA
        /* need to decide, which of the tasks that has the lowest load in one of the three quantities is best */
        best_balance = 1.0e30;
        best_queue = 0;
        // for each of the 3 queues, find its least-loaded eligible task -- simply the
        // smallest element of the ordered set, i.e. queue_set[q].begin() -- and evaluate what
        // the resulting balance would be if this chunk were assigned to that task.
        for (int q = 0; q < 3; q++)
        {
            queue_set_t::iterator it = queue_set[q].begin();
#ifdef DOMAIN_NUMA
            // skip this target if current mega-chunk doesn't belong to current computing node
            if (chunk_node.count(mega_chunk_index))
            {
                while (get_numa_of_process_info(it->task).node_idx != chunk_node[mega_chunk_index])
                {
                    ++it;
                }
            }
#endif // DOMAIN_NUMA
            target = it->task;

            // balance each criterion as "this task's load after adding the chunk" over
            // "the global total for that criterion"
            target_work_balance = (domainAssign[n].work + tasklist[target].work) / (tot_work + 1.0e-30);
            target_load_balance = (domainAssign[n].load + tasklist[target].load) / (tot_load + 1.0e-30);
            target_load_activesph_balance =
                (domainAssign[n].load_activesph + tasklist[target].load_activesph) / (tot_loadactivesph + 1.0e-30);

            /* the candidate's quality is judged by its worst (largest) balance across
             * the three criteria, since that is the metric that would actually bottleneck. */
            target_max_balance = target_work_balance;
            if (target_max_balance < target_load_balance)
            {
                target_max_balance = target_load_balance;
            }
            if (target_max_balance < target_load_activesph_balance)
            {
                target_max_balance = target_load_activesph_balance;
            }

            // keep track which criterion's least-loaded task gives the smallest worst-case balance seen so far.
            if (target_max_balance < best_balance)
            {
                best_balance = target_max_balance;
                best_queue = q;
            }
        }

        // Assign piece to task
        {
            queue_set_t::iterator it = queue_set[best_queue].begin();
#ifdef DOMAIN_NUMA
            if (chunk_node.count(mega_chunk_index))
            {
                while (get_numa_of_process_info(it->task).node_idx != chunk_node[mega_chunk_index])
                {
                    ++it;
                }
            }
#endif // DOMAIN_NUMA
            target = it->task;
        }
#ifdef DOMAIN_NUMA
        // assign this megachunk to the targets' computing node
        chunk_node[mega_chunk_index] = get_numa_of_process_info(target).node_idx;
#endif // DOMAIN_NUMA
       // record the decision and update the running totals for the chosen task.
        domainAssign[n].task = target;
        tasklist[target].work += domainAssign[n].work;
        tasklist[target].load += domainAssign[n].load;
        tasklist[target].load_activesph += domainAssign[n].load_activesph;
        tasklist[target].count++;

        // the chosen task's work/load/active-SPH-load just changed, so its position in all 3 ordered sets is now stale.
        // Erase it from each set then either reinsert it at its new sorted position letting std::multiset find the
        // slot, or if this chunk was its last one leave it erased
        for (int q = 0; q < 3; q++)
        {
            /* pick the criterion value relevant to this queue for the reinsertion */
            double value;
            switch (q)
            {
            case 0:
                value = tasklist[target].work;
                break;
            case 1:
                value = tasklist[target].load;
                break;
            case 2:
                value = tasklist[target].load_activesph;
                break;
            default:
                value = 0;
                break;
            }
            // now remove the element target
            queue_set[q].erase(queue_iter[q][target]);
            // reinsert if not finished yet

            if (tasklist[target].count < multipledomains)
            {
                queue_iter[q][target] = queue_set[q].insert({value, target});
            }
        }
    }

    // all chunks are now assigned; re-sort back into task order (as opposed to
    // the load order used for the greedy pass) so DomainStartList/DomainEndList/DomainTask
    // below are written out in a stable, task-ascending layout.
    STD_SORT(domainAssign, multipledomains * NTask, sizeof(struct domain_segments_data), domain_sort_task);

    // write the final decomposition back into the global domain arrays
    for (int n = 0; n < multipledomains * NTask; n++)
    {
        DomainStartList[n] = domainAssign[n].start;
        DomainEndList[n] = domainAssign[n].end;

        for (int i = DomainStartList[n]; i <= DomainEndList[n]; i++)
        {
            DomainTask[i] = domainAssign[n].task;
        }
    }

    myfree(tasklist);
    myfree(domainAssign);
}

/**
 * @brief Find load-balanced split of domains across CPUs.
 *
 * This function calculates a load-balanced split of domains across the specified
 * number of CPUs. It computes the average load per CPU, determines the load
 * balance factors, and iterates over the domains to assign a balanced split.
 *
 * @param ncpu Number of CPUs.
 * @param ndomain Number of domains.
 */
void domain_findSplit_load_balanced(int ncpu, int ndomain)
{
    double loadavg, load_before, loadavg_before, fac_load, fac;
    double min_load = 1.0, max_load = 0.0;
    double loadSph = 0, fac_loadSph = 0;

#ifdef KD_COUNT_STARS_IN_DOMAIN
    double loadStars = 0, fac_loadStars = 0;
#endif // KD_COUNT_STARS_IN_DOMAIN

    double load = 0;
    for (int i = 0; i < ndomain; i++)
    {
        load += domainCount[i];
        loadSph += domainCountSph[i];
#ifdef KD_COUNT_STARS_IN_DOMAIN
        loadStars += domainCountStars[i];
#endif // KD_COUNT_STARS_IN_DOMAIN
    }

    fac = 1;
    if (loadSph > 0)
    {
        fac = fac + 1;
    }
#ifdef KD_COUNT_STARS_IN_DOMAIN
    if (loadStars > 0)
    {
        fac = fac + 1;
    }
#endif // KD_COUNT_STARS_IN_DOMAIN
    fac = 1. / fac;

    fac_load = fac / load;
    if (loadSph > 0)
    {
        fac_loadSph = fac / loadSph;
    }
#ifdef KD_COUNT_STARS_IN_DOMAIN
    if (loadStars > 0)
    {
        fac_loadStars = fac / loadStars;
    }
#endif // KD_COUNT_STARS_IN_DOMAIN

    loadavg = 1.0 / ncpu;
    load_before = loadavg_before = 0;

    int start = 0;
    for (int i = 0; i < ncpu; i++)
    {
        load = 0;
        int end = start;

        load += fac_load * domainCount[end];
        load += fac_loadSph * domainCountSph[end];
#ifdef KD_COUNT_STARS_IN_DOMAIN
        load += fac_loadStars * domainCountStars[end];
#endif // KD_COUNT_STARS_IN_DOMAIN
        while ((load + load_before < loadavg + loadavg_before) || (i == ncpu - 1 && end < ndomain - 1))
        {
            if ((ndomain - end) > (ncpu - i))
            {
                end++;
            }
            else
            {
                break;
            }

            load += fac_load * domainCount[end];
            load += fac_loadSph * domainCountSph[end];
#ifdef KD_COUNT_STARS_IN_DOMAIN
            load += fac_loadStars * domainCountStars[end];
#endif // KD_COUNT_STARS_IN_DOMAIN
        }

        DomainStartList[i] = start;
        DomainEndList[i] = end;

        load_before += load;
        loadavg_before += loadavg;
        start = end + 1;

        if (load < min_load)
        {
            min_load = load;
        }
        if (load > max_load)
        {
            max_load = load;
        }
    }

    VERBOSE(0, "Load balance domain: min=%g avg=%g max=%g", min_load, loadavg, max_load);
}

/**
 * @brief Count particles to send to other CPUs.
 *
 * This function determines how many particles that are currently stored
 * on the local CPU have to be moved off according to the domain
 * decomposition.
 *
 * This function counts the number of stored on the local CPU to be sent
 * to other tasks based on a specified limit for the total size of particles
 * to be sent. It iterates over all particles and determines their destination
 * task, updating the `toGo`, `toGoSph`, `toGoStars`, and `toGoBHs` arrays
 * accordingly.
 *
 * @param nlimit Limit for the total size of particles to be sent.
 * @return Total number of particles to be sent.
 */
int domain_countToGo(size_t nlimit)
{
    int ret, retsum, mpi_overflow_flag;
    size_t mpi_Particle_limit = 2147483647 / sizeof(struct particle_data);
    size_t mpi_SphParticle_limit = 2147483647 / sizeof(struct sph_particle_data);
    size_t package;
#ifdef SFR
    size_t mpi_StarParticle_limit = 2147483647 / sizeof(struct met_particle_data);
#endif // SFR

#ifdef BLACK_HOLES
    size_t mpi_BHParticle_limit = 2147483647 / sizeof(struct bh_particle_data);
#endif // BLACK_HOLES

    for (int n = 0; n < NTask; n++)
    {
        toGo[n] = 0;
        toGoSph[n] = 0;
#if defined(SFR)
        toGoStars[n] = 0;
#endif // SFR

#ifdef BLACK_HOLES
        toGoBHs[n] = 0;
#endif // BLACK_HOLES
    }

    package = (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(peanokey));
#ifdef SFR
    package += sizeof(struct met_particle_data);
#endif // SFR

#ifdef BLACK_HOLES
    package += sizeof(struct bh_particle_data);
#endif // BLACK_HOLES

    ENDRUN_IF(package >= nlimit, "Package size is larger than the limit: %lu>%lu\n", package, nlimit);

    for (int n = 0; n < NumPart && package < nlimit; n++)
    {
#ifdef SUBFIND
        if (GrNr >= 0 && P[n].GrNr != GrNr)
        {
            continue;
        }
#endif // SUBFIND
        mpi_overflow_flag = 0;

        if (P[n].Type & 32)
        {
            int no = 0;

            while (topNodes[no].Daughter >= 0)
            {
                no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size / 8);
            }

            no = topNodes[no].Leaf;

            if (DomainTask[no] != ThisTask)
            {
                toGo[DomainTask[no]] += 1;
                nlimit -= sizeof(struct particle_data) + sizeof(peanokey);

                if (size_t(toGo[DomainTask[no]]) >= mpi_Particle_limit)
                {
                    mpi_overflow_flag = 1;
                }

                if ((P[n].Type & 15) == 0)
                {
                    toGoSph[DomainTask[no]] += 1;
                    nlimit -= sizeof(struct sph_particle_data);

                    if (size_t(toGoSph[DomainTask[no]]) >= mpi_SphParticle_limit)
                    {
                        mpi_overflow_flag = 1;
                    }
                }
#ifdef SFR
                if ((P[n].Type & 15) == 4)
                {
                    toGoStars[DomainTask[no]] += 1;
                    nlimit -= sizeof(struct met_particle_data);

                    if (toGoStars[DomainTask[no]] >= mpi_StarParticle_limit)
                        mpi_overflow_flag = 1;
                }
#endif // SFR

#ifdef BLACK_HOLES
                if ((P[n].Type & 15) == 5)
                {
                    toGoBHs[DomainTask[no]] += 1;
                    nlimit -= sizeof(struct bh_particle_data);

                    if (toGoBHs[DomainTask[no]] >= mpi_BHParticle_limit)
                    {
                        mpi_overflow_flag = 1;
                    }
                }
#endif // BLACK_HOLES
                P[n].Type |= 16; /* flag this particle for export */
                if (mpi_overflow_flag == 1)
                {
                    VERBOSE_ALL(
                        2,
                        "DOMAIN: Task %d: Chopping the send of data to respect the intrinsic limits of MPI send/recv",
                        ThisTask);
                    break;
                }
            }
        }
    }

    MPI_Alltoall(toGo, 1, MPI_INT, toGet, 1, MPI_INT, MYMPI_COMM_WORLD);
    MPI_Alltoall(toGoSph, 1, MPI_INT, toGetSph, 1, MPI_INT, MYMPI_COMM_WORLD);
#if defined(SFR)
    MPI_Alltoall(toGoStars, 1, MPI_INT, toGetStars, 1, MPI_INT, MYMPI_COMM_WORLD);
#endif // SFR

#ifdef BLACK_HOLES
    MPI_Alltoall(toGoBHs, 1, MPI_INT, toGetBHs, 1, MPI_INT, MYMPI_COMM_WORLD);
#endif // BLACK_HOLES

    if (package < nlimit && mpi_overflow_flag == 0)
    {
        ret = 0;
    }
    else
    {
        ret = 1;
    }

    MPI_Allreduce(&ret, &retsum, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    if (retsum)
    {
        /* in this case, we are not guaranteed that the temporary state after
           the partial exchange will actually observe the particle limits on all
           processors... we need to test this explicitly and rework the exchange
           such that this is guaranteed. This is actually a rather non-trivial
           constraint. */

        MPI_Allgather(&NumPart, 1, MPI_INT, list_NumPart, 1, MPI_INT, MYMPI_COMM_WORLD);
        MPI_Allgather(&N_gas, 1, MPI_INT, list_N_gas, 1, MPI_INT, MYMPI_COMM_WORLD);
#if defined(SFR)
        MPI_Allgather(&N_stars, 1, MPI_INT, list_N_stars, 1, MPI_INT, MYMPI_COMM_WORLD);
#endif // SFR

#ifdef BLACK_HOLES
        MPI_Allgather(&N_BHs, 1, MPI_INT, list_N_BHs, 1, MPI_INT, MYMPI_COMM_WORLD);
#endif // BLACK_HOLES

        int flag, flagsum, ntoomany, i, target;
        int count_togo, count_toget, count_togo_sph, count_toget_sph;

#if defined(SFR)
        int ntoomanystars, count_togo_stars, count_toget_stars;
#endif // SFR

#ifdef BLACK_HOLES
        int ntoomanyBHs, count_togo_BHs, count_toget_BHs;
#endif // BLACK_HOLES

        do
        {
            flagsum = 0;

            do
            {
                flag = 0;

                for (int ta = 0; ta < NTask; ta++)
                {
                    if (ta == ThisTask)
                    {
                        count_togo = count_toget = 0;
                        count_togo_sph = count_toget_sph = 0;
#if defined(SFR)
                        count_togo_stars = count_toget_stars = 0;
#endif // SFR

#ifdef BLACK_HOLES
                        count_togo_BHs = count_toget_BHs = 0;
#endif // BLACK_HOLES
                        for (int i = 0; i < NTask; i++)
                        {
                            count_togo += toGo[i];
                            count_toget += toGet[i];
                            count_togo_sph += toGoSph[i];
                            count_toget_sph += toGetSph[i];
#if defined(SFR)
                            count_togo_stars += toGoStars[i];
                            count_toget_stars += toGetStars[i];
#endif // SFR

#ifdef BLACK_HOLES
                            count_togo_BHs += toGoBHs[i];
                            count_toget_BHs += toGetBHs[i];
#endif // BLACK_HOLES
                        }
                    }
                    MPI_Bcast(&count_togo, 1, MPI_INT, ta, MYMPI_COMM_WORLD);
                    MPI_Bcast(&count_toget, 1, MPI_INT, ta, MYMPI_COMM_WORLD);
                    MPI_Bcast(&count_togo_sph, 1, MPI_INT, ta, MYMPI_COMM_WORLD);
                    MPI_Bcast(&count_toget_sph, 1, MPI_INT, ta, MYMPI_COMM_WORLD);

#if defined(SFR)
                    MPI_Bcast(&count_togo_stars, 1, MPI_INT, ta, MYMPI_COMM_WORLD);
                    MPI_Bcast(&count_toget_stars, 1, MPI_INT, ta, MYMPI_COMM_WORLD);
#endif // SFR

#ifdef BLACK_HOLES
                    MPI_Bcast(&count_togo_BHs, 1, MPI_INT, ta, MYMPI_COMM_WORLD);
                    MPI_Bcast(&count_toget_BHs, 1, MPI_INT, ta, MYMPI_COMM_WORLD);
#endif // BLACK_HOLES

                    int ifntoomany;

                    ntoomany = list_N_gas[ta] + count_toget_sph - count_togo_sph - All.MaxPartSph;
                    ifntoomany = (ntoomany > 0);

#ifdef SFR
                    ntoomanystars = list_N_stars[ta] + count_toget_stars - count_togo_stars - All.MaxPartMet;
                    ifntoomany = ifntoomany || (ntoomanystars > 0);
#endif // SFR
#ifdef BLACK_HOLES
                    ntoomanyBHs = list_N_BHs[ta] + count_toget_BHs - count_togo_BHs - All.MaxPartBH;
                    ifntoomany = ifntoomany || (ntoomanyBHs > 0);
#endif // BLACK_HOLES

                    if (ifntoomany)
                    {
                        if (ThisTask == 0)
                        {
                            if (ntoomany > 0)
                            {
                                VERBOSE(
                                    0,
                                    "exchange needs to be modified because I can't receive %d SPH-particles on task=%d",
                                    ntoomany, ta);
                                if (flagsum > 25)
                                {
                                    VERBOSE(2,
                                            "list_N_gas[ta=%d]=%d  count_toget_sph=%d count_togo_sph=%d MaxPartSPH=%d",
                                            ta, list_N_gas[ta], count_toget_sph, count_togo_sph, All.MaxPartSph);
                                }
                            }
#ifdef SFR
                            if (ntoomanystars > 0)
                            {
                                VERBOSE_ALL(0,
                                            "exchange needs to be modified because I can't receive %d STAR-particles "
                                            "on task=%d",
                                            ntoomanystars, ta);
                                if (flagsum > 25)
                                {
                                    VERBOSE_ALL(2,
                                                "list_N_stars[ta=%d]=%d  count_toget_stars=%d count_togo_stars=%d "
                                                "MaxPartStars=%d",
                                                ta, list_N_stars[ta], count_toget_stars, count_togo_stars,
                                                All.MaxPartMet);
                                }
                            }
#endif // SFR

#ifdef BLACK_HOLES
                            if (ntoomanyBHs > 0)
                            {
                                VERBOSE_ALL(
                                    0,
                                    "exchange needs to be modified because I can't receive %d BH-particles on task=%d",
                                    ntoomanyBHs, ta);
                                if (flagsum > 25)
                                {
                                    VERBOSE_ALL(
                                        2, "list_N_BHs[ta=%d]=%d  count_toget_BHs=%d count_togo_BHs=%d MaxPartBH=%d",
                                        ta, list_N_BHs[ta], count_toget_BHs, count_togo_BHs, All.MaxPartBH);
                                }
                            }
#endif // BLACK_HOLES
                        }

                        // Endrun as modify doesnt work with baryons. It always gets stuck in an infinite loop - we dont
                        // have a solution to it.
                        ENDRUN("6142");

                        flag = 1;
                        int i = flagsum % NTask;

                        while (ifntoomany)
                        {
                            if (i == ThisTask)
                            {
                                if (toGoSph[ta] > 0)
                                    if (ntoomany > 0)
                                    {
                                        toGoSph[ta]--;
                                        count_toget_sph--;
                                        count_toget--;
                                        ntoomany--;
                                    }
#if defined(SFR)
                                if (toGoStars[ta] > 0 && ntoomanystars > 0)
                                {
                                    toGoStars[ta]--;
                                    count_toget_stars--;
                                    count_toget--;
                                    ntoomanystars--;
                                }
#endif // SFR

#ifdef BLACK_HOLES
                                if (toGoBHs[ta] > 0 && ntoomanyBHs > 0)
                                {
                                    toGoBHs[ta]--;
                                    count_toget_BHs--;
                                    count_toget--;
                                    ntoomanyBHs--;
                                }
#endif // BLACK_HOLES
                            }

                            MPI_Bcast(&ntoomany, 1, MPI_INT, i, MYMPI_COMM_WORLD);
                            MPI_Bcast(&count_toget, 1, MPI_INT, i, MYMPI_COMM_WORLD);
                            MPI_Bcast(&count_toget_sph, 1, MPI_INT, i, MYMPI_COMM_WORLD);
#if defined(SFR)
                            MPI_Bcast(&count_toget_stars, 1, MPI_INT, i, MYMPI_COMM_WORLD);
#endif // SFR

#ifdef BLACK_HOLES
                            MPI_Bcast(&count_toget_BHs, 1, MPI_INT, i, MYMPI_COMM_WORLD);
#endif // BLACK_HOLES
                            i++;
                            if (i >= NTask)
                            {
                                i = 0;
                            }

                            ifntoomany = (ntoomany > 0);
#if defined(SFR)
                            ifntoomany = ifntoomany || (ntoomanystars > 0);
#endif // SFR

#ifdef BLACK_HOLES
                            ifntoomany = ifntoomany || (ntoomanyBHs > 0);
#endif // BLACK_HOLES
                        }
                    }

                    if ((ntoomany = list_NumPart[ta] + count_toget - count_togo - All.MaxPart) > 0)
                    {
                        if (ThisTask == 0)
                        {
                            VERBOSE(2, "exchange needs to be modified because I can't receive %d particles on task=%d",
                                    ntoomany, ta);
                            if (flagsum > 25)
                            {
                                VERBOSE(2, "list_NumPart[ta=%d]=%d  count_toget=%d count_togo=%d MaxPart=%d", ta,
                                        list_NumPart[ta], count_toget, count_togo, All.MaxPart);
                            }
                        }

                        flag = 1;
                        i = flagsum % NTask;
                        while (ntoomany)
                        {
                            if (i == ThisTask)
                            {
                                if (toGo[ta] > 0)
                                {
                                    toGo[ta]--;
                                    count_toget--;
                                    ntoomany--;
                                }
                            }

                            MPI_Bcast(&ntoomany, 1, MPI_INT, i, MYMPI_COMM_WORLD);
                            MPI_Bcast(&count_toget, 1, MPI_INT, i, MYMPI_COMM_WORLD);

                            i++;
                            if (i >= NTask)
                            {
                                i = 0;
                            }
                        }
                    }
                }
                flagsum += flag;

                if (ThisTask == 0)
                {
                    ENDRUN_IF(flagsum > 100, "Flagsum is larger than 100, flagsum = %d", flagsum);
                }
            } while (flag);

            if (flagsum)
            {
                int *local_toGo, *local_toGoSph;

                local_toGo = (int *)mymalloc("	      local_toGo", NTask * sizeof(int));
                local_toGoSph = (int *)mymalloc("	      local_toGoSph", NTask * sizeof(int));

#if defined(SFR)
                int *local_toGoStars;
                local_toGoStars = (int *)mymalloc("	      local_toGoStars", NTask * sizeof(int));
#endif // SFR

#ifdef BLACK_HOLES
                int *local_toGoBHs;
                local_toGoBHs = (int *)mymalloc("	      local_toGoBHs", NTask * sizeof(int));
#endif // BLACK_HOLES

                for (int n = 0; n < NTask; n++)
                {
                    local_toGo[n] = 0;
                    local_toGoSph[n] = 0;
#if defined(SFR)
                    local_toGoStars[n] = 0;
#endif // SFR

#ifdef BLACK_HOLES
                    local_toGoBHs[n] = 0;
#endif // BLACK_HOLES
                }

                for (int n = 0; n < NumPart; n++)
                {
                    if (P[n].Type & 32)
                    {
                        P[n].Type &= (15 + 32); /* clear 16 */

                        int no = 0;

                        while (topNodes[no].Daughter >= 0)
                            no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size / 8);

                        no = topNodes[no].Leaf;

                        target = DomainTask[no];

                        if ((P[n].Type & 15) == 0)
                        {
                            if (local_toGoSph[target] < toGoSph[target] && local_toGo[target] < toGo[target])
                            {
                                local_toGo[target] += 1;
                                local_toGoSph[target] += 1;
                                P[n].Type |= 16;
                            }
                        }
#if defined(SFR)
                        else if ((P[n].Type & 15) == 4)
                        {
                            if (local_toGoStars[target] < toGoStars[target] && local_toGo[target] < toGo[target])
                            {
                                local_toGo[target] += 1;
                                local_toGoStars[target] += 1;
                                P[n].Type |= 16;
                            }
                        }
#endif // SFR

#ifdef BLACK_HOLES
                        else if ((P[n].Type & 15) == 5)
                        {
                            if (local_toGoBHs[target] < toGoBHs[target] && local_toGo[target] < toGo[target])
                            {
                                local_toGo[target] += 1;
                                local_toGoBHs[target] += 1;
                                P[n].Type |= 16;
                            }
                        }
#endif // BLACK_HOLES
                        else
                        {
                            if (local_toGo[target] < toGo[target])
                            {
                                local_toGo[target] += 1;
                                P[n].Type |= 16;
                            }
                        }
                    }
                }

                for (int n = 0; n < NTask; n++)
                {
                    toGo[n] = local_toGo[n];
                    toGoSph[n] = local_toGoSph[n];
#if defined(SFR)
                    toGoStars[n] = local_toGoStars[n];
#endif // SFR

#ifdef BLACK_HOLES
                    toGoBHs[n] = local_toGoBHs[n];
#endif // BLACK_HOLES
                }

                MPI_Alltoall(toGo, 1, MPI_INT, toGet, 1, MPI_INT, MYMPI_COMM_WORLD);
                MPI_Alltoall(toGoSph, 1, MPI_INT, toGetSph, 1, MPI_INT, MYMPI_COMM_WORLD);

#ifdef BLACK_HOLES
                MPI_Alltoall(toGoBHs, 1, MPI_INT, toGetBHs, 1, MPI_INT, MYMPI_COMM_WORLD);
                myfree(local_toGoBHs);
#endif // BLACK_HOLES

#if defined(SFR)
                MPI_Alltoall(toGoStars, 1, MPI_INT, toGetStars, 1, MPI_INT, MYMPI_COMM_WORLD);
                myfree(local_toGoStars);
#endif // SFR
                myfree(local_toGoSph);
                myfree(local_toGo);
            }
        } while (flagsum);

        return 1;
    }
    else
    {
        return 0;
    }
}

/**
 * Walk the global top tree to establish the number of leaves.
 *
 * This function walks the global top tree recursively to establish the
 * number of leaves it has. These leaves are distributed to different
 * processors.
 *
 * @param no Index of the current top tree node.
 */
void domain_walktoptree(int no)
{
    if (topNodes[no].Daughter == -1)
    {
        topNodes[no].Leaf = NTopleaves;
        NTopleaves++;
    }
    else
    {
        for (int i = 0; i < 8; i++)
        {
            domain_walktoptree(topNodes[no].Daughter + i);
        }
    }
}

/**
 * @brief Compare two peano hilbert keys.
 *
 * This function compares two peano hilbert keys for sorting purposes.
 * It is used as a comparison function for qsort.
 *
 * @param a Pointer to the first peano hilbert data.
 * @param b Pointer to the second peano hilbert data.
 * @return -1 if the key of 'a' is less than the key of 'b',
 *          1 if the key of 'a' is greater than the key of 'b',
 *          0 if the keys are equal.
 */
int domain_compare_key(const void *a, const void *b)
{
    if (((struct peano_hilbert_data *)a)->key < (((struct peano_hilbert_data *)b)->key))
    {
        return -1;
    }

    if (((struct peano_hilbert_data *)a)->key > (((struct peano_hilbert_data *)b)->key))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Check if local refinement is needed for a given node.
 *
 * This function checks if local refinement is needed for a given node in the top tree.
 * It considers various criteria such as count limit, cost limit, and parent node properties.
 * If local refinement is needed, it splits the node into eight sub-nodes and recursively
 * checks each sub-node for local refinement.
 *
 * @param i Index of the current node in the top tree.
 * @param countlimit Threshold for the number of particles in a node to trigger refinement.
 * @param costlimit Threshold for the cost of a node to trigger refinement.
 * @return 1 if local refinement is needed, 0 otherwise.
 */
#ifdef AR_PARALLEL_LOCAL_REFINE
int domain_check_for_local_refine(int i, double countlimit, double costlimit, int layer)
#else // AR_PARALLEL_LOCAL_REFINE
int domain_check_for_local_refine(int i, double countlimit, double costlimit)
#endif // AR_PARALLEL_LOCAL_REFINE
{
    int sub, flag = 0;
#ifdef AR_PARALLEL_LOCAL_REFINE
    int return_code = 0; // in openmp you cannot break/return
#endif // AR_PARALLEL_LOCAL_REFINE
    int parent = topNodes[i].Parent;
    if (parent >= 0)
    {
        if (topNodes[i].Count > 0.8 * topNodes[parent].Count || topNodes[i].Cost > 0.8 * topNodes[parent].Cost)
        {
            flag = 1;
        }
    }

    if ((topNodes[i].Count > countlimit || topNodes[i].Cost > costlimit || flag == 1) && topNodes[i].Size >= 8)
    {
        if (topNodes[i].Size >= 8)
        {
            if ((NTopnodes + 8) < MaxTopNodes)
            {
#ifdef AR_PARALLEL_LOCAL_REFINE
                int _ntopnodes;
#pragma omp atomic capture
                {
                    _ntopnodes = NTopnodes;
                    NTopnodes += 8;
                }
                if ((_ntopnodes + 8) > MaxTopNodes)
                {
                    return 1;
                }
                topNodes[i].Daughter = _ntopnodes;

#else // AR_PARALLEL_LOCAL_REFINE
                topNodes[i].Daughter = NTopnodes;
                NTopnodes += 8;
#endif // AR_PARALLEL_LOCAL_REFINE
                for (int j = 0; j < 8; j++)
                {
                    sub = topNodes[i].Daughter + j;
                    topNodes[sub].Daughter = -1;
                    topNodes[sub].Parent = i;
                    topNodes[sub].Size = (topNodes[i].Size >> 3);
                    topNodes[sub].StartKey = topNodes[i].StartKey + j * topNodes[sub].Size;
                    topNodes[sub].PIndex = topNodes[i].PIndex;
                    topNodes[sub].Count = 0;
                    topNodes[sub].Cost = 0;
                    /*topNodes[sub].SphCost = 0; */ /* added because of arithmetic exception , needed? */
                }

                sub = topNodes[i].Daughter;

                int j = 0;
                for (int p = topNodes[i].PIndex; p < topNodes[i].PIndex + topNodes[i].Count; p++)
                {
                    if (j < 7)
                        while (mp[p].key >= topNodes[sub + 1].StartKey)
                        {
                            j++;
                            sub++;
                            topNodes[sub].PIndex = p;
                            if (j >= 7)
                            {
                                break;
                            }
                        }

                    topNodes[sub].Cost += domain_particle_costfactor(mp[p].index);
                    topNodes[sub].Count++;
                }

#ifdef AR_PARALLEL_LOCAL_REFINE
                int do_omp_parallel = layer < AR_PARALLEL_LOCAL_REFINE;
                int _num_used_threads = 0;
#pragma omp atomic capture
                {
                    NumUsedThreads += 8;
                    _num_used_threads = NumUsedThreads;
                }
                do_omp_parallel = do_omp_parallel && (_num_used_threads < maxThreads);

#pragma omp parallel for private(sub) if (do_omp_parallel)
#endif // AR_PARALLEL_LOCAL_REFINE
                for (int j = 0; j < 8; j++)
                {
                    sub = topNodes[i].Daughter + j;

#ifdef AR_PARALLEL_LOCAL_REFINE
                    if (return_code == 0 && domain_check_for_local_refine(sub, countlimit, costlimit, layer + 1))
                    {
                        return_code = 1;
                    }
#else // AR_PARALLEL_LOCAL_REFINE
                    if (domain_check_for_local_refine(sub, countlimit, costlimit))
                    {
                        return 1;
                    }
#endif // AR_PARALLEL_LOCAL_REFINE
                }
            }
            else
            {
                return 1;
            }
        }
    }
#ifdef AR_PARALLEL_LOCAL_REFINE
    return return_code;
#else // AR_PARALLEL_LOCAL_REFINE
    return 0;
#endif // AR_PARALLEL_LOCAL_REFINE
}

/**
 * @brief Combines domain top trees recursively.
 *
 * This function combines domain top trees recursively using MPI communication.
 * It divides the processors into left and right halves, and each half further divides
 * itself recursively until the number of processors becomes less than or equal to 2.
 * The top trees are exchanged between processors using MPI_Sendrecv and MPI_Send/MPI_Recv.
 * The trees are then swapped between master_left and master_right processors to ensure
 * consistency across all processors.
 * Finally, the combined trees are inserted into the domain structure if space permits.
 *
 * @param start The starting index of the processors involved in the combination.
 * @param ncpu The total number of processors.
 * @return Error flag indicating any errors occurred during the process.
 */
int domain_recursively_combine_topTree(int start, int ncpu)
{
    int i, nleft, nright, errflag = 0;
    int recvTask, ntopnodes_import;
    int master_left, master_right;
    struct local_topnode_data *topNodes_import = 0, *topNodes_temp;

    nleft = ncpu / 2;
    nright = ncpu - nleft;

    if (ncpu > 2)
    {
        errflag += domain_recursively_combine_topTree(start, nleft);
        errflag += domain_recursively_combine_topTree(start + nleft, nright);
    }

    if (ncpu >= 2)
    {
        master_left = start;
        master_right = start + nleft;
        ENDRUN_IF(
            master_left == master_right,
            "Invalid processor partition: master_left (%d) equals master_right (%d). start=%d, ncpu=%d, nleft=%d.",
            master_left, master_right, start, ncpu, nleft);

        if (ThisTask == master_left || ThisTask == master_right)
        {
            if (ThisTask == master_left)
            {
                recvTask = master_right;
            }
            else
            {
                recvTask = master_left;
            }

            /* inform each other about the length of the trees */
            MPI_Sendrecv(&NTopnodes, 1, MPI_INT, recvTask, TAG_GRAV_A, &ntopnodes_import, 1, MPI_INT, recvTask,
                         TAG_GRAV_A, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

            topNodes_import = (struct local_topnode_data *)mymalloc(
                "topNodes_import", std::max(ntopnodes_import, NTopnodes) * sizeof(struct local_topnode_data));

            /* exchange the trees */
            MPI_Sendrecv(topNodes, NTopnodes * sizeof(struct local_topnode_data), MPI_BYTE, recvTask, TAG_GRAV_B,
                         topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE, recvTask,
                         TAG_GRAV_B, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (ThisTask == master_left)
        {
            for (recvTask = master_left + 1; recvTask < master_left + nleft; recvTask++)
            {
                MPI_Send(&ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A, MYMPI_COMM_WORLD);
                MPI_Send(topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE, recvTask,
                         TAG_GRAV_B, MYMPI_COMM_WORLD);
            }
        }

        if (ThisTask == master_right)
        {
            for (recvTask = master_right + 1; recvTask < master_right + nright; recvTask++)
            {
                MPI_Send(&ntopnodes_import, 1, MPI_INT, recvTask, TAG_GRAV_A, MYMPI_COMM_WORLD);
                MPI_Send(topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE, recvTask,
                         TAG_GRAV_B, MYMPI_COMM_WORLD);
            }
        }

        if (ThisTask > master_left && ThisTask < master_left + nleft)
        {
            MPI_Recv(&ntopnodes_import, 1, MPI_INT, master_left, TAG_GRAV_A, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

            topNodes_import = (struct local_topnode_data *)mymalloc(
                "topNodes_import", std::max(ntopnodes_import, NTopnodes) * sizeof(struct local_topnode_data));

            MPI_Recv(topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE, master_left,
                     TAG_GRAV_B, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (ThisTask > master_right && ThisTask < master_right + nright)
        {
            MPI_Recv(&ntopnodes_import, 1, MPI_INT, master_right, TAG_GRAV_A, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

            topNodes_import = (struct local_topnode_data *)mymalloc(
                "topNodes_import", std::max(ntopnodes_import, NTopnodes) * sizeof(struct local_topnode_data));

            MPI_Recv(topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE, master_right,
                     TAG_GRAV_B, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (ThisTask >= master_left && ThisTask < master_left + nleft)
        {
            /* swap the two trees so that result will be equal on all cpus */

            topNodes_temp =
                (struct local_topnode_data *)mymalloc("topNodes_temp", NTopnodes * sizeof(struct local_topnode_data));
            memcpy(topNodes_temp, topNodes, NTopnodes * sizeof(struct local_topnode_data));
            memcpy(topNodes, topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data));
            memcpy(topNodes_import, topNodes_temp, NTopnodes * sizeof(struct local_topnode_data));
            myfree(topNodes_temp);
            i = NTopnodes;
            NTopnodes = ntopnodes_import;
            ntopnodes_import = i;
        }

        if (ThisTask >= start && ThisTask < start + ncpu)
        {
            if (errflag == 0)
            {
                if ((NTopnodes + ntopnodes_import) <= MaxTopNodes)
                {
                    domain_insertnode(topNodes, topNodes_import, 0, 0);
                }
                else
                {
                    errflag += 1;
                }
            }

            myfree(topNodes_import);
        }
    }

    return errflag;
}

#ifdef ALT_QSORT
#define KEY_TYPE struct peano_hilbert_data
#define KEY_BASE_TYPE peanokey
#define KEY_GETVAL(pk) ((pk)->key)
#define KEY_COPY(pk1, pk2)                                                                                             \
    {                                                                                                                  \
        (pk2)->key = (pk1)->key;                                                                                       \
        (pk2)->index = (pk1)->index;                                                                                   \
    }
#define QSORT qsort_domain
#include "../System/myqsort.hpp"
#endif // ALT_QSORT

/**
 * @brief Determine the global top-level tree node.
 *
 * This function constructs the global top-level tree node that is used
 * for the domain decomposition. This is done by considering the string of
 * Peano-Hilbert keys for all particles, which is recursively chopped off
 * in pieces of eight segments until each segment holds at most a certain
 * number of particles.
 *
 * @param SaveKeys Flag to save keys to file.
 * @param multipledomains Flag to enable multiple domains.
 */
#ifdef WRITE_KEY_FILES
int domain_determineTopTree(int SaveKeys, int multipledomains)
#else // WRITE_KEY_FILES
int domain_determineTopTree(void)
#endif // WRITE_KEY_FILES
{
    int ngrp;
    int recvTask, ntopnodes_import, errflag, errsum;
    struct local_topnode_data *topNodes_import, *topNodes_temp;
    double costlimit, countlimit;
    MPI_Status status;
#ifndef WRITE_KEY_FILES
    int multipledomains = Multipledomains;
#endif // WRITE_KEY_FILES

    mp = (struct peano_hilbert_data *)mymalloc("mp", sizeof(struct peano_hilbert_data) * NumPart);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
#ifdef SUBFIND
        if (GrNr >= 0 && P[i].GrNr != GrNr)
        {
            continue;
        }
#endif // SUBFIND
#ifdef WRITE_KEY_FILES
        if (SaveKeys != 0)
        {
            Key[i] = peano_hilbert_key(
                (int)((P[i].Pos[0] - DomainCorner[0]) * DomainFac), (int)((P[i].Pos[1] - DomainCorner[1]) * DomainFac),
                (int)((P[i].Pos[2] - DomainCorner[2]) * DomainFac), BITS_PER_DIMENSION_SAVE_KEYS);
        }
        else
        {
            Key[i] = peano_hilbert_key((int)((P[i].Pos[0] - DomainCorner[0]) * DomainFac),
                                       (int)((P[i].Pos[1] - DomainCorner[1]) * DomainFac),
                                       (int)((P[i].Pos[2] - DomainCorner[2]) * DomainFac), BITS_PER_DIMENSION);
        }
#else // WRITE_KEY_FILES
        Key[i] = peano_hilbert_key((int)((P[i].Pos[0] - DomainCorner[0]) * DomainFac),
                                   (int)((P[i].Pos[1] - DomainCorner[1]) * DomainFac),
                                   (int)((P[i].Pos[2] - DomainCorner[2]) * DomainFac), BITS_PER_DIMENSION);
#endif // WRITE_KEY_FILES

#ifdef SUBFIND_ALTERNATIVE_COLLECTIVE
        P[i].Key = Key[i];
#endif // SUBFIND_ALTERNATIVE_COLLECTIVE
        mp[i].key = Key[i];
        mp[i].index = i;
    }

    int count = NumPart;
#ifdef SUBFIND
    if (GrNr >= 0)
    {
        count = 0;
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].GrNr != GrNr)
            {
                continue;
            }
            mp[count].key = Key[i];
            mp[count].index = i;
            count++;
        }
    }

    ENDRUN_IF(GrNr >= 0 && count != NumPartGroup,
              " GrNr is larger or equal 0, while count is uneqal to the numer of particles in group.");
    ENDRUN_IF(GrNr < 0 && count != NumPart,
              "GrNr is smaller than 0 and count is unequal to number of particles in group");

#endif // SUBFIND

#ifdef _OPENMP
    if (count > OMP_SORT_THRESH)
    {
        serial_sort_omp(mp, count, sizeof(struct peano_hilbert_data), domain_compare_key);
    }
    else
    {
        mysort_domain(mp, count, sizeof(struct peano_hilbert_data));
    }
#else // _OPENMP
    mysort_domain(mp, count, sizeof(struct peano_hilbert_data));
#endif // _OPENMP

    NTopnodes = 1;
    topNodes[0].Daughter = -1;
    topNodes[0].Parent = -1;
#ifdef WRITE_KEY_FILES
    if (SaveKeys != 0)
    {
        topNodes[0].Size = PEANOCELLS_SAVE_KEYS;
    }
    else
    {
        topNodes[0].Size = PEANOCELLS;
    }
#else // WRITE_KEY_FILES
    topNodes[0].Size = PEANOCELLS;
#endif // WRITE_KEY_FILES
    topNodes[0].StartKey = 0;
    topNodes[0].PIndex = 0;
    topNodes[0].Count = count;
    topNodes[0].Cost = gravcost;

    costlimit = totgravcost / (Topnodefactor * multipledomains * NTask);
    countlimit = totpartcount / (Topnodefactor * multipledomains * NTask);
#ifdef AR_PARALLEL_LOCAL_REFINE
    NumUsedThreads = 0;
    errflag = domain_check_for_local_refine(0, countlimit, costlimit, 0);
#else // AR_PARALLEL_LOCAL_REFINE
    errflag = domain_check_for_local_refine(0, countlimit, costlimit);
#endif // AR_PARALLEL_LOCAL_REFINE

    myfree(mp);

    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    if (errsum)
    {
        VERBOSE(2, "We are out of Topnodes. We'll try to repeat with a higher value than All.TopNodeAllocFactor=%g",
                All.TopNodeAllocFactor);
        return errsum;
    }
    /* we now need to exchange tree parts and combine them as needed */

    if (NTask == (1 << PTask)) /* the following algoritm only works for power of 2 */
    {
        for (ngrp = 1, errflag = 0; ngrp < (1 << PTask); ngrp <<= 1)
        {
            int sendTask = ThisTask;
            recvTask = ThisTask ^ ngrp;

            if (recvTask < NTask)
            {
                /* inform each other about the length of the trees */
                MPI_Sendrecv(&NTopnodes, 1, MPI_INT, recvTask, TAG_GRAV_A, &ntopnodes_import, 1, MPI_INT, recvTask,
                             TAG_GRAV_A, MYMPI_COMM_WORLD, &status);

                topNodes_import = (struct local_topnode_data *)mymalloc(
                    "topNodes_import", std::max(ntopnodes_import, NTopnodes) * sizeof(struct local_topnode_data));

                /* exchange the trees */
                MPI_Sendrecv(topNodes, NTopnodes * sizeof(struct local_topnode_data), MPI_BYTE, recvTask, TAG_GRAV_B,
                             topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data), MPI_BYTE, recvTask,
                             TAG_GRAV_B, MYMPI_COMM_WORLD, &status);

                if (sendTask > recvTask) /* swap the two trees so that result will be equal on all cpus */
                {
                    topNodes_temp = (struct local_topnode_data *)mymalloc(
                        "topNodes_temp", NTopnodes * sizeof(struct local_topnode_data));
                    memcpy(topNodes_temp, topNodes, NTopnodes * sizeof(struct local_topnode_data));
                    memcpy(topNodes, topNodes_import, ntopnodes_import * sizeof(struct local_topnode_data));
                    memcpy(topNodes_import, topNodes_temp, NTopnodes * sizeof(struct local_topnode_data));
                    myfree(topNodes_temp);
                    int i = NTopnodes;
                    NTopnodes = ntopnodes_import;
                    ntopnodes_import = i;
                }

                if (errflag == 0)
                {
                    if ((NTopnodes + ntopnodes_import) <= MaxTopNodes)
                    {
                        domain_insertnode(topNodes, topNodes_import, 0, 0);
                    }
                    else
                    {
                        errflag = 1;
                    }
                }

                myfree(topNodes_import);
            }
        }
    }
    else
    {
        errflag = domain_recursively_combine_topTree(0, NTask);
    }

    MPI_Allreduce(&errflag, &errsum, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    if (errsum)
    {
        VERBOSE(1, "can't combine trees due to lack of storage. Will try again.");
        return errsum;
    }

    /* now let's see whether we should still append more nodes, based on the estimated cumulative cost/count in each
     * cell */
    VERBOSE(2, "Before/After=%d", NTopnodes);

    /* count toplevel leaves */
    domain_sumCost();

    ENDRUN_IF(
        NTopleaves < multipledomains * NTask,
        "ERROR: NTopleaves too small, e.g. NTopleaves < MULTIPLEDOMAINS * NTask ( %d < %d * %d ) !!! \n        "
        "increase TOPNODEFACTOR or decrease MULTIPLEDOMAINS or number of MPI ranks to fulfill the above condition!",
        NTopleaves, multipledomains, NTask);
    return 0;
}

/**
 * @brief Calculate and summarize domain costs.
 *
 * This function calculates and summarizes domain costs based on particle properties.
 * It allocates memory for local arrays to store domain work and counts.
 * Each particle's contribution to domain work and counts is computed and accumulated
 * in the local arrays using OpenMP parallelization.
 * The local arrays are then reduced across all MPI processes using MPI_Allreduce.
 * Finally, the results are stored in global arrays for further analysis.
 */
void domain_sumCost(void)
{
    float *local_domainWork;
    float *local_domainWorkSph;
    int *local_domainCount;
    int *local_domainCountSph;

    float max_domainCount = 0, min_domainCount = 1e20;
    float max_domainCountSph = 0, min_domainCountSph = 1e20;
    float max_domainWork = 0, min_domainWork = 1e20;
    float max_domainWorkSph = 0, min_domainWorkSph = 1e20;

    local_domainWork = (float *)mymalloc("local_domainWork", NTopnodes * maxThreads * sizeof(float));
    local_domainWorkSph = (float *)mymalloc("local_domainWorkSph", NTopnodes * maxThreads * sizeof(float));
    local_domainCount = (int *)mymalloc("local_domainCount", NTopnodes * maxThreads * sizeof(int));
    local_domainCountSph = (int *)mymalloc("local_domainCountSph", NTopnodes * maxThreads * sizeof(int));

#if defined(SFR)
    int *local_domainCountStars;
    local_domainCountStars = (int *)mymalloc("local_domainCountStars", NTopnodes * maxThreads * sizeof(int));
#endif // SFR

#ifdef BLACK_HOLES
    int *local_domainCountBHs;
    local_domainCountBHs = (int *)mymalloc("local_domainCountBHs", NTopnodes * maxThreads * sizeof(int));
#endif // BLACK_HOLES

    NTopleaves = 0;
    domain_walktoptree(0); // we recompute NTopleaves
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
    {
        mythreadid = THREAD_ID_OR_ZERO();
        for (int i = 0; i < NTopnodes; i++)
        {
            local_domainWork[i + mythreadid * NTopnodes] = 0;
            local_domainWorkSph[i + mythreadid * NTopnodes] = 0;
            local_domainCount[i + mythreadid * NTopnodes] = 0;
            local_domainCountSph[i + mythreadid * NTopnodes] = 0;
#if defined(SFR)
            local_domainCountStars[i + mythreadid * NTopnodes] = 0;
#endif // SFR

#ifdef BLACK_HOLES
            local_domainCountBHs[i + mythreadid * NTopnodes] = 0;
#endif // BLACK_HOLES
        }
    }
    VERBOSE(2, "NTopleaves= %d  NTopnodes=%d (space for %d)", NTopleaves, NTopnodes, MaxTopNodes);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int n = 0; n < NumPart; n++)
    {
#ifdef SUBFIND
        if (GrNr >= 0 && P[n].GrNr != GrNr)
        {
            continue;
        }
#endif // SUBFIND

        int no = 0;

        while (topNodes[no].Daughter >= 0)
        {
            no = topNodes[no].Daughter + (Key[n] - topNodes[no].StartKey) / (topNodes[no].Size >> 3);
        }

        no = topNodes[no].Leaf;

        local_domainWork[no + mythreadid * NTopnodes] += (float)domain_particle_costfactor(n);

        local_domainCount[no + mythreadid * NTopnodes] += 1;
        if (P[n].Type == 0)
        {
            local_domainCountSph[no + mythreadid * NTopnodes] += 1;
            if (TimeBinActive[P[n].TimeBin] || UseAllParticles)
            {
                local_domainWorkSph[no + mythreadid * NTopnodes] += 1.0;
            }
        }
#if defined(SFR)
        if (P[n].Type == 4)
        {
            local_domainCountStars[no + mythreadid * NTopnodes] += 1;
        }
#endif // SFR

#ifdef BLACK_HOLES
        if (P[n].Type == 5)
        {
            local_domainCountBHs[no + mythreadid * NTopnodes] += 1;
        }
#endif // BLACK_HOLES
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif // _OPENMP
    for (int n = 0; n < NTopnodes; n++)
    {
        for (int t = 1; t < maxThreads; t++)
        {
            local_domainWork[n] += local_domainWork[n + NTopnodes * t];
            local_domainWorkSph[n] += local_domainWorkSph[n + NTopnodes * t];
            local_domainCount[n] += local_domainCount[n + NTopnodes * t];
            local_domainCountSph[n] += local_domainCountSph[n + NTopnodes * t];
#if defined(SFR)
            local_domainCountStars[n] += local_domainCountStars[n + NTopnodes * t];
#endif // SFR

#ifdef BLACK_HOLES
            local_domainCountBHs[n] += local_domainCountBHs[n + NTopnodes * t];
#endif // BLACK_HOLES
        }
    }

    MPI_Allreduce(local_domainWork, domainWork, NTopleaves, MPI_FLOAT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(local_domainWorkSph, domainWorkSph, NTopleaves, MPI_FLOAT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(local_domainCount, domainCount, NTopleaves, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(local_domainCountSph, domainCountSph, NTopleaves, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
#ifdef BLACK_HOLES
    MPI_Allreduce(local_domainCountBHs, domainCountBHs, NTopleaves, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    myfree(local_domainCountBHs);
#endif // BLACK_HOLES
#if defined(SFR)
    MPI_Allreduce(local_domainCountStars, domainCountStars, NTopleaves, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    myfree(local_domainCountStars);
#endif // SFR

    myfree(local_domainCountSph);
    myfree(local_domainCount);
    myfree(local_domainWorkSph);
    myfree(local_domainWork);

    for (int n = 0; n < NTopleaves; n++)
    {
        if (domainCount[n] > max_domainCount)
        {
            max_domainCount = domainCount[n];
        }
        if (domainCount[n] < min_domainCount)
        {
            min_domainCount = domainCount[n];
        }
        if (domainCountSph[n] > max_domainCountSph)
        {
            max_domainCountSph = domainCountSph[n];
        }
        if (domainCountSph[n] < min_domainCountSph)
        {
            min_domainCountSph = domainCountSph[n];
        }

        if (domainWork[n] > max_domainWork)
        {
            max_domainWork = domainWork[n];
        }
        if (domainWork[n] < min_domainWork)
        {
            min_domainWork = domainWork[n];
        }
        if (domainWorkSph[n] > max_domainWorkSph)
        {
            max_domainWorkSph = domainWorkSph[n];
        }
        if (domainWorkSph[n] < min_domainWorkSph)
        {
            min_domainWorkSph = domainWorkSph[n];
        }
    }
    VERBOSE(2, "Balance min/max topleaves (%d): Count=%g/%g, CountSPH=%g/%g, Work=%g/%g, WorkSPH=%g/%g", NTopleaves,
            min_domainCount, max_domainCount, min_domainCountSph, max_domainCountSph, min_domainWork, max_domainWork,
            min_domainWorkSph, max_domainWorkSph);
}

/**
 * @brief Find the extent of the domain.
 *
 * This function determines the spatial extent of the domain by calculating the minimum
 * and maximum coordinates of particles in each dimension. It then reduces these values
 * across all MPI processes using MPI_Allreduce to obtain the global minimum and maximum.
 * The domain length is computed as a percentage increase of the maximum extent.
 * Finally, the center and corner of the domain, as well as the domain length factor,
 * are calculated based on the determined extent.
 */
#ifdef WRITE_KEY_FILES
void domain_findExtent(int SaveKeys)
#else // WRITE_KEY_FILES
void domain_findExtent(void)
#endif // WRITE_KEY_FILES
{
    double len, xmin[3], xmax[3], xmin_glob[3], xmax_glob[3];

    /* determine local extension */
    for (int j = 0; j < 3; j++)
    {
        xmin[j] = MAX_REAL_NUMBER;
        xmax[j] = -MAX_REAL_NUMBER;
    }

#ifdef _OPENMP
    double xmin_tmp[3], xmax_tmp[3];
    for (int j = 0; j < 3; j++)
    {
        xmin_tmp[j] = MAX_REAL_NUMBER;
        xmax_tmp[j] = -MAX_REAL_NUMBER;
    }

#pragma omp parallel shared(xmin_tmp, xmax_tmp) firstprivate(xmin, xmax)
    {
#pragma omp for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
#ifdef SUBFIND
            if (GrNr >= 0 && P[i].GrNr != GrNr)
            {
                continue;
            }
#endif // SUBFIND

            for (int j = 0; j < 3; j++)
            {
                if (xmin[j] > P[i].Pos[j])
                {
                    xmin[j] = P[i].Pos[j];
                }

                if (xmax[j] < P[i].Pos[j])
                {
                    xmax[j] = P[i].Pos[j];
                }
            }
        }
#ifdef _OPENMP
        for (int j = 0; j < 3; j++)
        {
#pragma omp critical
            {
                if (xmin[j] < xmin_tmp[j])
                {
                    xmin_tmp[j] = xmin[j];
                }

                if (xmax[j] > xmax_tmp[j])
                {
                    xmax_tmp[j] = xmax[j];
                }
            }
        }
    }
    for (int j = 0; j < 3; j++)
    {
        xmax[j] = xmax_tmp[j];
        xmin[j] = xmin_tmp[j];
    }
#endif // _OPENMP

    MPI_Allreduce(xmin, xmin_glob, 3, MPI_DOUBLE, MPI_MIN, MYMPI_COMM_WORLD);
    MPI_Allreduce(xmax, xmax_glob, 3, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);

    len = 0;
    for (int j = 0; j < 3; j++)
    {
        if (xmax_glob[j] - xmin_glob[j] > len)
        {
            len = xmax_glob[j] - xmin_glob[j];
        }
    }

    len *= 1.001;

    for (int j = 0; j < 3; j++)
    {
        DomainCenter[j] = 0.5 * (xmin_glob[j] + xmax_glob[j]);
        DomainCorner[j] = 0.5 * (xmin_glob[j] + xmax_glob[j]) - 0.5 * len;
    }

    DomainLen = len;
#ifdef WRITE_KEY_FILES
    if (SaveKeys != 0)
    {
        DomainFac = 1.0 / len * (((peanokey)1) << (BITS_PER_DIMENSION_SAVE_KEYS));
    }
    else
    {
        DomainFac = 1.0 / len * (((peanokey)1) << (BITS_PER_DIMENSION));
    }
#else // WRITE_KEY_FILES
    DomainFac = 1.0 / len * (((peanokey)1) << (BITS_PER_DIMENSION));
#endif // WRITE_KEY_FILES
}

/**
 * @brief Adds cost information to a given node in a top-level tree.
 *
 * This function adds cost information (count, cost, sphcost) to a specified node
 * in a top-level tree. The function distributes the count evenly among the eight
 * subnodes of the parent node and updates their count, cost, and sphcost accordingly.
 * It recursively applies the same process to each subnode if they have further daughters.
 *
 * @param treeA Pointer to the array of local topnode data representing the tree.
 * @param noA Index of the parent node in the tree.
 * @param count Total count to be distributed among subnodes.
 * @param cost Total cost to be distributed among subnodes.
 * @param sphcost Total sphcost to be distributed among subnodes.
 */
void domain_add_cost(struct local_topnode_data *treeA, int noA, long long count, double cost, double sphcost)
{
    long long countB = count / 8;
    long long countA = count - 7 * countB;

    cost = cost / 8;
    sphcost = sphcost / 8;

    for (int i = 0; i < 8; i++)
    {
        int sub = treeA[noA].Daughter + i;

        if (i == 0)
        {
            count = countA;
        }
        else
        {
            count = countB;
        }

        treeA[sub].Count += count;
        treeA[sub].Cost += cost;
        treeA[sub].SphCost += sphcost;

        if (treeA[sub].Daughter >= 0)
        {
            domain_add_cost(treeA, sub, count, cost, sphcost);
        }
    }
}

/**
 * @brief Inserts a node from treeB into treeA.
 *
 * This function inserts a node from treeB into treeA based on their respective
 * sizes and key values. If the size of the node in treeB is smaller than that
 * in treeA, it either creates new subnodes in treeA to accommodate the insertion
 * or recursively inserts the node into an existing subnode. If the sizes match,
 * the counts and costs are added together. If the size of the node in treeB is
 * larger than that in treeA, it raises an error.
 *
 * @param treeA Pointer to the array of local topnode data representing treeA.
 * @param treeB Pointer to the array of local topnode data representing treeB.
 * @param noA Index of the node in treeA where insertion should occur.
 * @param noB Index of the node in treeB to be inserted into treeA.
 */
void domain_insertnode(struct local_topnode_data *treeA, struct local_topnode_data *treeB, int noA, int noB)
{
    int sub;
    long long count, countA, countB;
    double cost, costA, costB;

    if (treeB[noB].Size < treeA[noA].Size)
    {
        if (treeA[noA].Daughter < 0)
        {
            if ((NTopnodes + 8) < MaxTopNodes)
            {
                count = treeA[noA].Count - treeB[treeB[noB].Parent].Count;
                countB = count / 8;
                countA = count - 7 * countB;

                cost = treeA[noA].Cost - treeB[treeB[noB].Parent].Cost;
                costB = cost / 8;
                costA = cost - 7 * costB;

                treeA[noA].Daughter = NTopnodes;
                for (int j = 0; j < 8; j++)
                {
                    if (j == 0)
                    {
                        count = countA;
                        cost = costA;
                    }
                    else
                    {
                        count = countB;
                        cost = costB;
                    }

                    sub = treeA[noA].Daughter + j;
                    topNodes[sub].Size = (treeA[noA].Size >> 3);
                    topNodes[sub].Count = count;
                    topNodes[sub].Cost = cost;
                    topNodes[sub].Daughter = -1;
                    topNodes[sub].Parent = noA;
                    topNodes[sub].StartKey = treeA[noA].StartKey + j * treeA[sub].Size;
                }
                NTopnodes += 8;
            }
            else
            {
                ENDRUN("Problem in domain insertion. NTopnodes=%d, MaxTopNodes=%d", NTopnodes, MaxTopNodes);
            }
        }

        sub = treeA[noA].Daughter + (treeB[noB].StartKey - treeA[noA].StartKey) / (treeA[noA].Size >> 3);
        domain_insertnode(treeA, treeB, sub, noB);
    }
    else if (treeB[noB].Size == treeA[noA].Size)
    {
        treeA[noA].Count += treeB[noB].Count;
        treeA[noA].Cost += treeB[noB].Cost;

        if (treeB[noB].Daughter >= 0)
        {
            for (int j = 0; j < 8; j++)
            {
                sub = treeB[noB].Daughter + j;
                domain_insertnode(treeA, treeB, noA, sub);
            }
        }
        else
        {
            if (treeA[noA].Daughter >= 0)
            {
                domain_add_cost(treeA, noA, treeB[noB].Count, treeB[noB].Cost, treeB[noB].SphCost);
            }
        }
    }
    else
    {
        ENDRUN("Problem in domain insertion. Size of node in treeB is larger than that in treeA.");
    }
}

/**
 * @brief Performs merge sort on an array of peano_hilbert_data using a temporary array.
 *
 * This function implements the merge sort algorithm to sort an array of peano_hilbert_data
 * structures. It recursively divides the array into two halves, sorts each half separately,
 * and then merges them back together in sorted order using a temporary array.
 *
 * @param b Pointer to the array of peano_hilbert_data to be sorted.
 * @param n Size of the array.
 * @param t Temporary array used for merging.
 */
static void msort_domain_with_tmp(struct peano_hilbert_data *b, size_t n, struct peano_hilbert_data *t)
{
    struct peano_hilbert_data *tmp;
    struct peano_hilbert_data *b1, *b2;
    size_t n1, n2;

    if (n <= 1)
    {
        return;
    }

    n1 = n / 2;
    n2 = n - n1;
    b1 = b;
    b2 = b + n1;

    msort_domain_with_tmp(b1, n1, t);
    msort_domain_with_tmp(b2, n2, t);

    tmp = t;

    while (n1 > 0 && n2 > 0)
    {
        if (b1->key <= b2->key)
        {
            --n1;
            *tmp++ = *b1++;
        }
        else
        {
            --n2;
            *tmp++ = *b2++;
        }
    }

    if (n1 > 0)
    {
        memcpy(tmp, b1, n1 * sizeof(struct peano_hilbert_data));
    }

    memcpy(b, t, (n - n2) * sizeof(struct peano_hilbert_data));
}

/**
 * @brief Sorts an array using merge sort algorithm.
 *
 * This function sorts an array of elements using the merge sort algorithm. It allocates
 * memory for a temporary array, sorts the input array using msort_domain_with_tmp function,
 * and then frees the allocated memory.
 *
 * @param b Pointer to the array to be sorted.
 * @param n Number of elements in the array.
 * @param s Size of each element in bytes.
 */
void mysort_domain(void *b, size_t n, size_t s)
{
    const size_t size = n * s;
    struct peano_hilbert_data *tmp;

    tmp = (struct peano_hilbert_data *)mymalloc("tmp", size);

    msort_domain_with_tmp((struct peano_hilbert_data *)b, n, tmp);

    myfree(tmp);
}
