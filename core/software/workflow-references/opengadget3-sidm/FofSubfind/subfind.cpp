/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Substructure finding algorithm (Subfind).
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

#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
#include <hdf5.h>
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)

#include "fof.hpp"
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../CodeBase/domain.hpp"
#include "subfind.hpp"

#ifdef SUBFIND
static struct subfind_id_list
{
    MyIDType ID;
    int GrNr;
    int SubNr;
    float BindingEgy;

#ifdef SUBFIND_SAVE_PARTICLELISTS
    float Pos[3];
    float Vel[3];
    int Type;
    float Mass;
#ifdef SFR
    float StellarAge;
#endif // SFR
#endif // SUBFIND_SAVE_PARTICLELISTS
} *ID_list;

static int Nids;
static int NidsMax;

/**
 * @brief Collects particle indices of a given type into the unbind data array.
 *
 * This function iterates over all particles in parallel using OpenMP.
 * For each particle of type `parttype`, its index is stored in the `d` array.
 * The total number of matching particles is tracked by `n`, which is updated atomically.
 *
 * @param d Pointer to an array of `struct unbind_data` where matching indices will be stored.
 * @param parttype Reference to the particle type to filter for.
 * @param n Reference to an integer used to count how many particles matched the given type.
 *
 * @note This function assumes that `NumPart` and `P` are accessible in the enclosing scope.
 */
void collect_indices(struct unbind_data *d, int &parttype, int &n)
{
#pragma omp parallel for shared(n, d)
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Type == parttype)
        {
            int n_this;
#pragma omp atomic capture
            n_this = n++;
            d[n_this].index = i;
        }
    }
}

/**
 * @brief Perform the substructure finding algorithm (Subfind) on a given set of particles.
 *
 * This function runs subfind, which identifies substructures within larger groups of particles.
 *
 * @param num An integer parameter (only needed when using old CB07 filters).
 */
void subfind(int num)
{
    TIMER_PUSH(1, "SUBFIND");
#ifdef DENSITY_SPLIT_BY_TYPE
    int count[6], countall[6];
    double a3inv;
#endif // DENSITY_SPLIT_BY_TYPE

    grouptasks = (int *)mymalloc("taskspergroup", TotNgroups * sizeof(int));

    VERBOSE(2, "\nWe now execute a parallel version of SUBFIND.");

#ifdef LT_ADD_GAL_TO_SUB
#ifdef LT_METAL_COOLING_WAL
    Redshift = 1.0 / All.Time - 1; /* note that ComovingIntegration is alway on with wal cooling */
    get_cool_redshift(Redshift, &DZ);
#endif // LT_METAL_COOLING_WAL

    VERBOSE(2, "  loading CB07 tables ...");

    int IMFt = 1;

    Filters_Effective_Wavelenght = (float *)mymalloc("Filters_Effective_Wavelenght", sizeof(float) * LT_ADD_GAL_TO_SUB);
    tempiAS = (float *)mymalloc("tempiAS", sizeof(float) * LOG_AGE_BINS);
    CB07 = (float *)mymalloc("CB07", sizeof(float) * N_BINS_METAL * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB);
#ifdef OBSERVER_FRAME
    CB07obs = (float *)mymalloc("CB07obs", sizeof(float) * N_BINS_METAL * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB);
#endif // OBSERVER_FRAME
    if (ThisTask == 0)
#ifndef OLD_FILTERS
        load_CB07_table_new(IMFt);
#else // OLD_FILTERS
        load_CB07_table(IMFt, num);
#endif // OLD_FILTERS

    MPI_Bcast(&bc_use_index[0], sizeof(float) * LT_ADD_GAL_TO_SUB, MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&Filters_Effective_Wavelenght[0], sizeof(float) * LT_ADD_GAL_TO_SUB, MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&tempiAS[0], sizeof(float) * LOG_AGE_BINS, MPI_BYTE, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&CB07[0], sizeof(float) * N_BINS_METAL * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB, MPI_BYTE, 0, MYMPI_COMM_WORLD);
#ifdef OBSERVER_FRAME
    MPI_Bcast(&CB07obs[0], sizeof(float) * N_BINS_METAL * LOG_AGE_BINS * LT_ADD_GAL_TO_SUB, MPI_BYTE, 0,
              MYMPI_COMM_WORLD);
#endif // OBSERVER_FRAME
#endif // LT_ADD_GAL_TO_SUB

#ifdef DENSITY_SPLIT_BY_TYPE
    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1;
    }

    for (int j = 0; j < 6; j++)
    {
        count[j] = 0;
    }

    /* let's count number of particles of selected species */
    TIMER_PUSH(3, "counting_tyes");

#ifndef SUBFIND_DO_NOT_COUNT_IN_PARALLEL
#pragma omp parallel shared(count)
    {
        int count_tmp[6] = {0}; // initialise with 0

#pragma omp for
        for (int i = 0; i < NumPart; i++)
        {
            count_tmp[P[i].Type]++;
        }

        for (int k = 0; k < 6; k++)
        {
#pragma omp atomic
            count[k] += count_tmp[k];
        }
    }
#else // SUBFIND_DO_NOT_COUNT_IN_PARALLEL
    for (int i = 0; i < NumPart; i++)
    {
        count[P[i].Type]++;
    }
#endif // SUBFIND_DO_NOT_COUNT_IN_PARALLEL

    TIMER_POP(3, "counting_tyes");
    MPI_Allreduce(count, countall, 6, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    /* do first loop: basically just defining the hsml for different species */
    for (int j = 0; j < 6; j++)
    {
        if ((1 << j) & (DENSITY_SPLIT_BY_TYPE))
        {
#ifdef BLACK_HOLES
            if (j == 5)
            {
                countall[j] = 0; /* this will prevent that the black holes are treated separately */
            }
#endif // BLACK_HOLES

            force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
            if (countall[j] > sqrt(NTask) * All.DesNumNgb)
            {
                TIMER_PUSH(1, "SUBFIND_TREEBUILD_SPECIES");
                /* build index list of particles of selectes species */
                struct unbind_data *d;
                d = (struct unbind_data *)mymalloc("	      d", count[j] * sizeof(struct unbind_data));
                int n = 0;

                collect_indices(d, j, n);

                ENDRUN_IF(count[j] != n,
                          "count[%d] = %d != %d = n_this !! (set SUBFIND_DO_NOT_COUNT_IN_PARALLEL to avoid this)", j,
                          count[j], n);
                VERBOSE(2, "Tree construction for species %d (%d).", j, countall[j]);

                force_treebuild(count[j], d);

                myfree(d);

                TIMER_POP(1, "SUBFIND_TREEBUILD_SPECIES");
            }
            else
            {
                VERBOSE(2, "Tree construction.");

                TIMER_PUSH(1, "SUBFIND_TREEBUILD");
                force_treebuild(NumPart, NULL);
                TIMER_POP(1, "SUBFIND_TREEBUILD");
            }

            /* let's determine the local densities */
            TIMER_PUSH(1, "SUBFIND_SMOOTHINGLENGTH");
            subfind_setup_smoothinglengths(j);
            TIMER_POP(1, "SUBFIND_SMOOTHINGLENGTH");

            TIMER_PUSH(1, "SUBFIND_DENSITY");
            subfind_density(j);
            TIMER_POP(1, "SUBFIND_DENSITY");

            force_treefree();

            /* let's save density contribution of own species */
            TIMER_PUSH(3, "copying_density");
#pragma omp parallel for
            for (int i = 0; i < NumPart; i++)
            {
                if (P[i].Type == j)
                {
                    P[i].w.density_sum = P[i].u.DM_Density;
                }
            }
            TIMER_POP(3, "copying_density");
        }
    }

    /* do second loop: now calculate all density contributions */
    for (int j = 0; j < 6; j++)
    {
        if ((1 << j) & (DENSITY_SPLIT_BY_TYPE))
        {
            force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
            TIMER_PUSH(1, "SUBFIND_TREEBUILD_SPECIES");

            /* build index list of particles of selectes species */
            struct unbind_data *d;
            d = (struct unbind_data *)mymalloc("	  d", count[j] * sizeof(struct unbind_data));
            int n = 0;

            collect_indices(d, j, n);

            VERBOSE(2, "Tree construction for species %d (%d).", j, countall[j]);

            force_treebuild(count[j], d);

            myfree(d);
            TIMER_POPPUSH(1, "SUBFIND_TREEBUILD_SPECIES", "SUBFIND_DENSITY");

            double density_threshold = sqrt(NTask) * All.DesNumNgb;

            /* let's determine the local densities */
            for (int i = 0; i < 6; i++)
            {
                if ((1 << i) & (DENSITY_SPLIT_BY_TYPE))
                {
                    if (j != i)
                    {
                        if (countall[i] > density_threshold)
                        {
                            VERBOSE(2, "calculating density contribution of species %d to species %d", j, i);
                            subfind_density(-(i + 1));
                        }
                    }
                }
            }
            TIMER_POP(1, "SUBFIND_DENSITY");

            force_treefree();

            /* let's sum up density contribution */
            TIMER_PUSH(3, "summing_density");

#pragma omp parallel for
            for (int i = 0; i < NumPart; i++)
            {
                if ((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE))
                {
                    if (j != P[i].Type)
                    {
                        if (countall[P[i].Type] > density_threshold)
                        {
                            P[i].w.density_sum += P[i].u.DM_Density;
                        }
                    }
                }
            }
            TIMER_POP(3, "summing_density");
        }
    }

    TIMER_PUSH(3, "copying_density_and_evaluating_energy");
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        P[i].u.DM_Density = P[i].w.density_sum;

        if (P[i].Type == 0)
        {
            P[i].w.int_energy = max(
                All.MinEgySpec, (double)(SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].Density * a3inv, GAMMA_MINUS1)));
        }
        else
        {
            P[i].w.int_energy = 0;
        }
    }
    TIMER_POP(3, "copying_density_and_evaluating_energy");

#else
    force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    TIMER_PUSH(1, "SUBFIND_TREEBUILD");
    VERBOSE(2, "Tree construction.");

    force_treebuild(NumPart, NULL);
    TIMER_POPPUSH(1, "SUBFIND_TREEBUILD", "SUBFIND_DMDENSITY");

    /* let's determine the local dark matter densities */
    subfind_setup_smoothinglengths();
    subfind_density();
    TIMER_POP(1, "SUBFIND_DMDENSITY");

    force_treefree();
#endif /* DENSITY_SPLIT_BY_TYPE */

#ifndef SUBFIND_DENSITY_AND_POTENTIAL
#ifdef DO_SAVE_HSML
    if (DumpFlag == 1)
    {
        /* let's save the densities to a file (for making images) */
        subfind_save_densities(num);
    }
#endif // DO_SAVE_HSML
#endif // SUBFIND_DENSITY_AND_POTENTIAL
#ifdef ONLY_PRODUCE_HSML_FILES
    return;
#endif // ONLY_PRODUCE_HSML_FILES

    subfind_get_tasks(); /* finds the tasks for all particles */

    subfind_exchange(); /* distributes gas particles as well if needed */

    subfind_distribute_groups(1);

    STD_SORT(Group, Ngroups, sizeof(group_properties), fof_compare_Group_GrNr);

    TIMER_PUSH(3, "checking_consitency");
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].GrNr > Ncollective && P[i].GrNr <= TotNgroups)
        {
            if (((P[i].GrNr - Ncollective - 1) % (NTask - (Ncollective * ntaskspergroup)) +
                 (Ncollective * ntaskspergroup)) != ThisTask)
            {
                ENDRUN("i=%d %d task=%d type=%d", i, P[i].GrNr, ThisTask, P[i].Type);
            }
        }
    }

    /* lets estimate the maximum number of substructures we need to store on the local CPU */
    TIMER_POPPUSH(3, "checking_consitency", "computing_total_group_length");
    {
        int nlocid = 0;
#pragma omp parallel for reduction(+ : nlocid)
        for (int i = 0; i < Ngroups; i++)
        {
            nlocid += Group[i].Len;
        }

        int loc_MaxNsubgroups = nlocid / All.DesLinkNgb + NTask; /* this is a quite conservative upper limit */

        if (color < Ncollective)
        {
            loc_MaxNsubgroups *= 2.1;
        }

        MPI_Allreduce(&loc_MaxNsubgroups, &MaxNsubgroups, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
    }
    TIMER_POP(3, "computing_total_group_length");

    Nsubgroups = 0;
    SubGroup = (struct subgroup_properties *)mymalloc("SubGroup", MaxNsubgroups * sizeof(struct subgroup_properties));

    TIMER_PUSH(3, "bitshift_subgroup");
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        P[i].SubNr = (1 << 30); /* default */
    }
    TIMER_POP(3, "bitshift_subgroup");
    int oldThisTask = ThisTask;
    int oldNTask = NTask;
    if (Ncollective)
    {
        MPI_Comm_dup(SUB_COMM, &MYMPI_COMM_WORLD);
        ThisTask = SubThisTask;
        NTask = SubNTask;
        for (PTask = 0; NTask > (1 << PTask); PTask++)
            ;
    }

#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        P[i].origindex = i;
        P[i].origintask = ThisTask;
    }

    /* we begin by applying the collective version of subfind to distributed groups */
    if (color >= 0 && color < Ncollective)
    {
        subfind_process_group_collectively(num);
    }
    else
    {
        TIMER_PUSH(1, "SUBFIND_SORTLOCAL");
        STD_SORT(P, NumPart, sizeof(struct particle_data), subfind_compare_P_GrNr_DM_Density);
        TIMER_POP(1, "SUBFIND_SORTLOCAL");

        /* now we have the particles of groups consecutively, but SPH particles are
           not aligned. They can however be accessed via SphP[P[i].originindex] */

        /* let's count how many local particles we have in small groups */
        /*       t0_kd = second(); */
        /*       { */
        /* 	int nlocid = 0; */
        /* #pragma omp parallel for reduction(+:nlocid) */
        /* 	for(int i = 0; i < NumPart; i++) */
        /* 	  if(P[i].GrNr > Ncollective && P[i].GrNr <= Ngroups)	/\* particle is in small group *\/ */
        /* 	    nlocid++; */
        /*       } */
        /*       t1_kd = second(); */

        /*       if(SubThisTask == 0) */
        /* 	printf("EXTRA TIMER SUBFIND:    Counting remaining particles in local group took %g sec\n", timediff(t0_kd,
         * t1_kd)); */

        if (SubThisTask == 0)
        {
            VERBOSE_ALL(2, "constructing tree for serial subfind of local groups");
        }

#ifdef KD_EXTRA_TREEALLOC_FACTOR_SUBFIND
        subfind_loctree_treeallocate((int)(KD_EXTRA_TREEALLOC_FACTOR_SUBFIND * (1 + log10(All.PartAllocFactor)) *
                                           All.TreeAllocFactor * All.MaxPart) +
                                         NTopnodes,
                                     All.MaxPart);
#else // KD_EXTRA_TREEALLOC_FACTOR_SUBFIND
        subfind_loctree_treeallocate(
            (int)(All.TreeAllocFactor * All.MaxPart * (1 + log10(All.PartAllocFactor))) + NTopnodes, All.MaxPart);
#endif // KD_EXTRA_TREEALLOC_FACTOR_SUBFIND

        if (SubThisTask == 0)
        {
            VERBOSE_ALL(2, "Start to do local groups with serial subfind algorithm");
        }

        TIMER_PUSH(1, "SUBFIND_LOCALGROUPS");
        /* we now apply a serial version of subfind to the local groups */
        int local_offset = 0;
        for (int gr = 0; gr < Ngroups; gr++)
        {
            if (Group[gr].GrNr > Ncollective)
            {
                if ((Group[gr].GrNr - Ncollective - 1) % SubNTask == SubThisTask)
                {
                    local_offset = subfind_process_group_serial(num, gr, local_offset);
                }
            }
        }

        //      MPI_Barrier(SUB_COMM);
        TIMER_POP(1, "SUBFIND_LOCALGROUPS");

        subfind_loctree_treefree();

        /* bringing back particles in original positions, such that gas particles are aligned */
        TIMER_PUSH(1, "SUBFIND_UNSORTLOCAL");
        STD_SORT(P, NumPart, sizeof(struct particle_data), subfind_compare_P_origindex);
        TIMER_POP(1, "SUBFIND_UNSORTLOCAL");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* return original communicator and ranks */

    if (Ncollective)
    {
        MPI_Comm_free(&SUB_COMM);
        MPI_Comm_dup(MPI_COMM_WORLD, &MYMPI_COMM_WORLD);
        ThisTask = oldThisTask;
        NTask = oldNTask;
        for (PTask = 0; NTask > (1 << PTask); PTask++)
            ;
    }

    GrNr = -1; /* to ensure that domain decomposition acts normally again */

    /* now determine the remaining spherical overdensity values for the non-local groups */

    domain_free_trick();

    TIMER_PUSH(1, "SUBFIND_MAIN_HALO_PROP");

#ifdef DENSITY_SPLIT_BY_TYPE
    TIMER_PUSH(3, "check_backsend_of_particles");
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].origintask != SubThisTask)
        {
            VERBOSE_ALL(2, "Task %d: Holding particle of task %d !", SubThisTask, P[i].origintask);
        }
        if (P[i].origindex != i)
        {
            VERBOSE_ALL(2, "Task %d: Particles is in wrong position (is=%d, was=%d) !", ThisTask, i, P[i].origindex);
        }
    }
    TIMER_POP(3, "check_backsend_of_particles");
#endif // DENSITY_SPLIT_BY_TYPE

    TIMER_PUSH(3, "memorizing_orig_2_task");
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        P[i].targettask = P[i].origintask2;
    }

    TIMER_POPPUSH(3, "memorizing_orig_2_task", "exchangereturn");
    subfind_exchange(); /* distributes gas particles as well if needed */
    subfind_distribute_groups(2);

    STD_SORT(Group, Ngroups, sizeof(group_properties), fof_compare_Group_GrNr);

    TIMER_POP(3, "exchangereturn");

    All.DoDynamicUpdate = 0;

    domain_Decomposition(1, 0);
    force_treebuild(NumPart, NULL);
    /* compute spherical overdensities for FOF groups */
    TIMER_PUSH(3, "so_computations");
    subfind_overdensity();
    TIMER_POP(3, "so_computations");

    /* determine which halos are contaminated by boundary particles */
    subfind_contamination();

    force_treefree();
    domain_free();
    domain_allocate_trick();
    /* now assemble final output */
    subfind_save_final(num);

    myfree(SubGroup);

#ifdef LT_ADD_GAL_TO_SUB
#ifdef OBSERVER_FRAME
    myfree(CB07obs);
#endif // OBSERVER_FRAME
    myfree(CB07);
    myfree(tempiAS);
    myfree(Filters_Effective_Wavelenght);
#endif // LT_ADD_GAL_TO_SUB

    myfree(grouptasks);

    TIMER_POP(1, "SUBFIND_MAIN_HALO_PROP");
    TIMER_POP(1, "SUBFIND");
}

/**
 * @brief Determines the tasks for collective and parallel processing of FOF halos and particle distribution.
 *
 * This function determines the tasks for collective and parallel processing of FOF halos and particle distribution.
 * It calculates the number of groups that should be processed collectively and the distribution of particles to tasks.
 */
void subfind_get_tasks(void)
{
    int limit_take, nminingrouplocal, nmaxingrouplocal;
    unsigned long ntotingroupcollective;

    /* count how many groups we have that should be done collectively */
    int limit_largest = (int)(0.75 * All.PartAllocFactor * All.TotNumPart / NTask);
#ifdef PLACEHIGHRESREGION
    limit_take = limit_largest;
#else // PLACEHIGHRESREGION
    if (All.PeriodicBoundariesOn == 0)
    {
        limit_take = limit_largest;
    }
    else
    {
        limit_take = (int)(limit_largest / (1 + log10(NTask) / log10(2.0)));
        int alt_limit = (int)(0.8 * All.TotNumPart / NTask);
        if (limit_take <= alt_limit)
        {
            limit_take = alt_limit;
        }
    }
#endif // PLACEHIGHRESREGION
    int ncount = 0;
    int ntotingrouplocal = 0;
    unsigned long ningroupcollective = 0;

#pragma omp parallel for reduction(+ : ncount, ntotingrouplocal, ningroupcollective) if (Ngroups > 50)
    for (int i = 0; i < Ngroups; i++)
    {
        if (Group[i].Len >= limit_take)
        {
            ncount++;
            ningroupcollective += Group[i].Len;
        }
        else
        {
            ntotingrouplocal += Group[i].Len;
        }
    }

    MPI_Allreduce(&ncount, &Ncollective, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&ningroupcollective, &ntotingroupcollective, 1, MPI_UNSIGNED_LONG, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&ntotingrouplocal, &nminingrouplocal, 1, MPI_INT, MPI_MIN, MYMPI_COMM_WORLD);
    MPI_Allreduce(&ntotingrouplocal, &nmaxingrouplocal, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);

    /*  to decide on which task a group should be:
     *  if   GrNr <= Ncollective:  collective groupfinding.
     *  the task where the group info is put is TaskNr = (GrNr - 1) % NTask
     */

    /* now we distribute the particles such that small groups are assigned in
     *  total to certain CPUs, and big groups are left where they are
     */
    int nt = (Group[0].Len / limit_largest) + 1;
    if (nt % 2 != 0)
    {
        nt += 1;
    }
    MPI_Allreduce(&nt, &ntaskspergroup, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);

    if (ThisTask == 0)
    {
        VERBOSE(0, "\n SUBFIND CONFIGURATION REPORT:\n");
        VERBOSE(0, "Number of FOF halos treated with collective SubFind code = %d due to adopted size-limits:\n",
                Ncollective);
        VERBOSE(0, "     N for which halo is processed in parallel:                                 %d\n", limit_take);
        VERBOSE(0, "     Maximum N for a halo which theoretically could be processed in serial mode %d\n",
                limit_largest);
        VERBOSE(0, "     N of largest Group found                                                   %d\n",
                Group[0].Len);
        VERBOSE(0, "the other %d FOF halos are treated in parallel with serial code\n\n", TotNgroups - Ncollective);
        VERBOSE(0, "     Unbalance in total number of particles in FOF halos is %d - %d \n\n", nminingrouplocal,
                nmaxingrouplocal);
        if (Ncollective > 0)
        {
            VERBOSE(0, "     SubFind is running in parallel mode, assuming ");
#ifdef PLACEHIGHRESREGION
            VERBOSE(0, "Zoom-In simulation\n");
            VERBOSE(0, "    to allow SubFind to run in serial mode, increase PartAllocFactor to at least %f\n",
                    (All.PartAllocFactor * Group[0].Len) / limit_largest);
#else // PLACEHIGHRESREGION
            if (All.PeriodicBoundariesOn == 0)
            {
                VERBOSE(0, "Zoom-In simulation\n");
                VERBOSE(0, "    to allow SubFind to run in serial mode, increase PartAllocFactor to at least %f\n",
                        (All.PartAllocFactor * Group[0].Len) / limit_largest);
            }
            else
            {
                VERBOSE(0, "large cosmological box simulation\n");
                VERBOSE(0, "    To switch to Zoom-In simulation mode, set PLACEHIGHRESREGION in Config.sh file\n");
                if (Group[0].Len > limit_largest)
                {
                    VERBOSE(0, "    and additionally increas PartAllocFactor to at least %f\n",
                            (All.PartAllocFactor * Group[0].Len) / limit_largest);
                }
                else
                {
                    VERBOSE(0, "    to allow SubFind to run in serial mode, increase PartAllocFactor to at least %f\n",
                            (All.PartAllocFactor * Group[0].Len) / limit_take);
                }
            }
#endif // PLACEHIGHRESREGION
            VERBOSE(0, "     Using %d tasks per group for the collective subfind and\n", ntaskspergroup);
            VERBOSE(0, "     Therefore %d out of %d total tasks are used for the collective SUBFIND.\n",
                    ntaskspergroup * Ncollective, NTask);
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD);

    ENDRUN_IF(ntaskspergroup * Ncollective > NTask, "Number of tasks per group is larger than total number of tasks");

    grouptasks[0] = 0;
    ncolors = Ncollective + 1;
    color = -1;
    for (int i = 0; i < Ncollective; i++)
    {
        grouptasks[i] = grouptasks[i - 1] + ntaskspergroup;
        if (ThisTask >= grouptasks[i] && ThisTask < grouptasks[i] + ntaskspergroup)
        {
            color = i;
        }
        else if (color < 0)
        {
            color = Ncollective;
        }
    }

    if (Ncollective)
    {
        MPI_Comm_split(MYMPI_COMM_WORLD, color, ThisTask, &SUB_COMM);
        MPI_Comm_rank(SUB_COMM, &SubThisTask);
        MPI_Comm_size(SUB_COMM, &SubNTask);
    }
    else
    {
        SubThisTask = ThisTask;
        SubNTask = NTask;
    }

#pragma omp for
    for (int i = 0; i < NumPart; i++)
    {
        P[i].origintask2 = ThisTask;

        if (P[i].GrNr <= TotNgroups)
        {
            if (P[i].GrNr > Ncollective) /* particle is in small group */
            {
                P[i].targettask = (P[i].GrNr - Ncollective - 1) % (NTask - (Ncollective * ntaskspergroup)) +
                                  (Ncollective * ntaskspergroup);
            }
            else
            {
                P[i].targettask = grouptasks[P[i].GrNr - 1] + i % ntaskspergroup;
            }
        }
        else
        {
            P[i].targettask = ThisTask;
        }
    }
}

/**
 * @brief Saves the final subhalo catalog.
 *
 * This function saves the final subhalo catalog to disk. It prepares the list of particle IDs with assigned group
 * numbers, sorts the groups and subgroups, fills in the `FirstSub` values, and saves the subgroup catalog either
 * collectively or in parallel depending on the configuration.
 *
 * @param num The number identifier of the current snapshot or output.
 */
void subfind_save_final(int num)
{
    TIMER_PUSH(3, "save_final");
    char buf[1000];

    /* prepare list of ids with assigned group numbers */
#ifdef KD_ALTERNATIVE_GROUP_SORT
    TIMER_PUSH(4, "alternative_global_sort_groups");
    sort_groups();
    STD_SORT(Group, Ngroups, sizeof(group_properties), fof_compare_Group_GrNr);
    TIMER_POPPUSH(4, "alternative_global_sort_groups", "alternative_global_sort_subgroups");
    sort_sub_groups();
    STD_SORT(SubGroup, Nsubgroups, sizeof(struct subgroup_properties), subfind_compare_SubGroup_GrNr_SubNr);
    TIMER_POP(4, "alternative_global_sort_subgroups");
#else // KD_ALTERNATIVE_GROUP_SORT
    TIMER_PUSH(4, "global_sort_groups");
    parallel_sort(Group, Ngroups, sizeof(group_properties), fof_compare_Group_GrNr);
    TIMER_POPPUSH(4, "global_sort_groups", "global_sort_subgroups");
    parallel_sort(SubGroup, Nsubgroups, sizeof(struct subgroup_properties), subfind_compare_SubGroup_GrNr_SubNr);
    TIMER_POP(4, "global_sort_subgroups");
#endif // KD_ALTERNATIVE_GROUP_SORT

#ifndef NO_KD_ID_LIST_OPTIMIZATION
    Nids = 0;
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].GrNr <= TotNgroups)
        {
            Nids++;
        }
    }
#ifndef SUBFIND_ID_OVERBOOK_FACTOR
#define SUBFIND_ID_OVERBOOK_FACTOR 30
#endif // SUBFIND_ID_OVERBOOK_FACTOR
    NidsMax = Nids * ntaskspergroup * SUBFIND_ID_OVERBOOK_FACTOR;
    if (NidsMax < All.MaxPart)
    {
        NidsMax = All.MaxPart;
    }
#else // NO_KD_ID_LIST_OPTIMIZATION
    NidsMax = All.MaxPart * ntaskspergroup;
#endif // NO_KD_ID_LIST_OPTIMIZATION
    ID_list = (struct subfind_id_list *)mymalloc("ID_list", sizeof(struct subfind_id_list) * NidsMax);
    Nids = 0;
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].GrNr <= TotNgroups)
        {
            ID_list[Nids].GrNr = P[i].GrNr;
            ID_list[Nids].SubNr = P[i].SubNr;
            ID_list[Nids].BindingEgy = P[i].v.DM_BindingEnergy;
            ID_list[Nids].ID = P[i].ID;
#ifdef SUBFIND_SAVE_PARTICLELISTS
            for (int j = 0; j < 3; j++)
            {
                ID_list[Nids].Pos[j] = P[i].Pos[j];
                ID_list[Nids].Vel[j] = P[i].Vel[j];
            }
            ID_list[Nids].Type = P[i].Type;
            ID_list[Nids].Mass = P[i].Mass;
#ifdef SFR
            if (P[i].Type == 4)
            {
                ID_list[Nids].StellarAge = MPP(i).StellarAge;
            }
            else
            {
                ID_list[Nids].StellarAge = 0;
            }
#endif // SFR
#endif // SUBFIND_SAVE_PARTICLELISTS
            Nids++;
        }
    }

#ifdef KD_ALTERNATIVE_GROUP_SORT
    TIMER_PUSH(4, "alternative_global_sort_ids");
    sort_ids();
    STD_SORT(ID_list, Nids, sizeof(struct subfind_id_list), subfind_compare_ID_list);
    TIMER_POP(4, "alternative_global_sort_ids");
#else // KD_ALTERNATIVE_GROUP_SORT
    TIMER_PUSH(4, "global_sort_ids");
    parallel_sort(ID_list, Nids, sizeof(struct subfind_id_list), subfind_compare_ID_list);
    TIMER_POP(4, "global_sort_ids");
#endif // KD_ALTERNATIVE_GROUP_SORT

    MPI_Allreduce(&Nsubgroups, &TotNsubgroups, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    /* fill in the FirstSub-values */
    int totsubs = 0;
    for (int i = 0; i < Ngroups; i++)
    {
        if (i > 0)
        {
            Group[i].FirstSub = Group[i - 1].FirstSub + Group[i - 1].Nsubs;
        }
        else
        {
            Group[i].FirstSub = 0;
        }
        totsubs += Group[i].Nsubs;
    }

    MPI_Allgather(&totsubs, 1, MPI_INT, Send_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    Send_offset[0] = 0;
    for (int j = 1; j < NTask; j++)
    {
        Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
    }

    for (int i = 0; i < Ngroups; i++)
    {
        Group[i].FirstSub += Send_offset[ThisTask];
    }

#ifdef SUBFIND_COUNT_BIG_HALOS
    Nbiggroups = 0;
    const double threshold = SUBFIND_COUNT_BIG_HALOS;
#pragma omp parallel for reduction(+ : Nbiggroups)
    for (int i = 0; i < Ngroups; i++)
    {
        if (Group[i].M_TopHat200 > threshold)
        {
            Nbiggroups++;
        }
    }
    MPI_Allreduce(&Nbiggroups, &TotNbiggroups, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
#endif // SUBFIND_COUNT_BIG_HALOS

    MPI_Allgather(&Nids, 1, MPI_INT, Send_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    Send_offset[0] = 0;
    for (int j = 1; j < NTask; j++)
    {
        Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
    }

    if (ThisTask == 0)
    {
#ifdef LONG_OUTPUT_NUMBERS
        sprintf(buf, "%s/groups_%04d", All.OutputDir, num);
#else // LONG_OUTPUT_NUMBERS
        sprintf(buf, "%s/groups_%03d", All.OutputDir, num);
#endif // LONG_OUTPUT_NUMBERS
        mkdir(buf, 02755);
    }
    MPI_Barrier(MYMPI_COMM_WORLD);

    ENDRUN_IF(NTask < All.NumFilesWrittenInParallel,
              "Fatal error.\nNumber of processors must be a smaller or equal than `NumFilesWrittenInParallel'.\n");

    TIMER_PUSH(4, "save_Subgroup_catalogues");

#ifdef WRITE_SUB_COLLECTIVELY
    subfind_save_catalogues_collectively(num);
#else // WRITE_SUB_COLLECTIVELY
    int nprocgroup = NTask / All.NumFilesWrittenInParallel;
    if ((NTask % All.NumFilesWrittenInParallel))
    {
        nprocgroup++;
    }
    int masterTask = (ThisTask / nprocgroup) * nprocgroup;
    for (int groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
        if (ThisTask == (masterTask + groupTask)) /* ok, it's this processor's turn */
            subfind_save_local_catalogue(num);
        MPI_Barrier(MYMPI_COMM_WORLD); /* wait inside the group */
    }
#endif // WRITE_SUB_COLLECTIVELY

    TIMER_POP(4, "save_Subgroup_catalogues");
    myfree(ID_list);
    TIMER_POP(3, "save_final");
}

/**
 * @brief Retrieves the entry type of a specified block in a subhalo data structure.
 *
 * This function takes an enumeration representing a block number as input and returns the entry type of that block in a
 * subhalo data structure.
 *
 * @param blocknr An enum value of type `siofields` representing the block number for which the entry type is retrieved.
 *
 * @return An integer representing the entry type of the specified block:
 *         - 0: Indicates that the block does not correspond to a specific entry type.
 *         - 1: Indicates that the block corresponds to a data entry associated with subhalo properties.
 *         - 2: Indicates that the block corresponds to a data entry associated with particle properties.
 *         - 3: Indicates that the block corresponds to a data entry associated with background properties.
 */
int get_sub_entrytype_of_block(enum siofields blocknr)
{
    int type = 0;

    switch (blocknr)
    {
    case SIO_SLEN:
    case SIO_SOFF:
    case SIO_PFOF:
    case SIO_MSUB:
    case SIO_SPOS:
    case SIO_SVEL:
    case SIO_SCM:
    case SIO_SPIN:
    case SIO_DSUB:
    case SIO_VMAX:
    case SIO_RVMAX:
    case SIO_RHMS:
    case SIO_SHMR:
    case SIO_MBID:
    case SIO_GRNR:
    case SIO_SMST:
    case SIO_SLUM:
    case SIO_SLATT:
    case SIO_SLOBS:
    case SIO_DUST:
    case SIO_SAGE:
    case SIO_SZ:
    case SIO_SSFR:
    case SIO_SSYNCHROTRON:
    case SIO_MHI:
    case SIO_SSANG:
    case SIO_SBVAL:
    case SIO_SZGAS:
    case SIO_SZCGAS:
        type = 1;
        break;
    case SIO_PPOS:
    case SIO_PVEL:
    case SIO_PTYP:
    case SIO_PMAS:
    case SIO_PAGE:
    case SIO_PID:
        type = 2;
        break;
    case SIO_BGPOS:
    case SIO_BGMTOP:
    case SIO_BGRTOP:
        type = 3;
        break;
    default:
        type = 0;
        break;
    }

    return type;
}

/**
 * @brief Retrieves the number of values associated with a specified block in a subhalo data structure.
 *
 * This function takes an enumeration representing a block number as input and returns the number of values associated
 * with that block in a subhalo data structure.
 *
 * @param blocknr An enum value of type `siofields` representing the block number for which the number of values is
 * retrieved.
 *
 * @return An integer representing the number of values associated with the specified block.
 */
int get_values_per_sub(enum siofields blocknr)
{
    int n = 1;

    switch (blocknr)
    {
    case SIO_GPOS:
    case SIO_SPOS:
    case SIO_SVEL:
    case SIO_SCM:
    case SIO_SPIN:
    case SIO_PPOS:
    case SIO_PVEL:
    case SIO_BGPOS:
        n = 3;
        break;
    case SIO_MGAS:
    case SIO_MSTR:
    case SIO_TGAS:
    case SIO_LGAS:
    case SIO_YGAS:
    case SIO_SMST:
        n = 6;
        break;
    case SIO_SLUM:
    case SIO_SLATT:
    case SIO_SLOBS:
#ifdef LT_ADD_GAL_TO_SUB
        n = LT_ADD_GAL_TO_SUB;
#else // LT_ADD_GAL_TO_SUB
        n = 1;
#endif // LT_ADD_GAL_TO_SUB
        break;
    case SIO_DUST:
#ifdef DUSTATT
        n = DUSTATT;
#else // DUSTATT
        n = 1;
#endif // DUSTATT
        break;
    default:
        n = 1;
        break;
    }

    return n;
}

#ifdef HAVE_HDF5
/**
 * @brief Retrieves the HDF5 I/O label for a specified block in a data structure.
 *
 * This function takes an enumeration representing a block number as input and returns the corresponding HDF5 I/O label
 * associated with that block.
 *
 * @param blocknr An enum value of type `siofields` representing the block number for which the HDF5 I/O label is
 * retrieved.
 * @param label A character array where the HDF5 I/O label will be stored. It must have a length of at least 10
 * characters to accommodate the label and the null terminator.
 */
void get_IO_Label_HDF5_sub(enum siofields blocknr, char *label)
{
    switch (blocknr)
    {
    case SIO_GLEN:
        strcpy(label, "GroupLen");
        break;
    case SIO_GOFF:
        strcpy(label, "GroupOffset");
        break;
    case SIO_MTOT:
        strcpy(label, "GroupMass");
        break;
    case SIO_GPOS:
        strcpy(label, "GroupPos");
        break;
#ifdef SO_BAR_INFO
    case SIO_MMEA:
        strcpy(label, "MVIR");
        break;
    case SIO_RMEA:
        strcpy(label, "RVIR");
        break;
    case SIO_DMEA:
        strcpy(label, "DVIR");
        break;
    case SIO_MCRI:
        strcpy(label, "M5CC");
        break;
    case SIO_RCRI:
        strcpy(label, "R5CC");
        break;
    case SIO_DCRI:
        strcpy(label, "D5CC");
        break;
    case SIO_MTOP:
        strcpy(label, "M500");
        break;
    case SIO_RTOP:
        strcpy(label, "R500");
        break;
    case SIO_DTOP:
        strcpy(label, "D500");
        break;
#else // SO_BAR_INFO
    case SIO_MMEA:
        strcpy(label, "Group_M_Mean200");
        break;
    case SIO_RMEA:
        strcpy(label, "Group_R_Mean200");
        break;
    case SIO_MCRI:
        strcpy(label, "Group_M_Crit200");
        break;
    case SIO_RCRI:
        strcpy(label, "Group_R_Crit200");
        break;
    case SIO_MTOP:
        strcpy(label, "Group_M_TopHat200");
        break;
    case SIO_RTOP:
        strcpy(label, "Group_R_TopHat200");
        break;
    case SIO_DMEA:
        strcpy(label, "Group_VelDisp_Mean200");
        break;
    case SIO_DCRI:
        strcpy(label, "Group_VelDisp_Crit200");
        break;
    case SIO_DTOP:
        strcpy(label, "Group_VelDisp_TopHat200");
        break;
#endif // SO_BAR_INFO
    case SIO_MGAS:
        strcpy(label, "MGAS");
        break;
    case SIO_MSTR:
        strcpy(label, "MSTR");
        break;
    case SIO_TGAS:
        strcpy(label, "TGAS");
        break;
    case SIO_LGAS:
        strcpy(label, "LGAS");
        break;
    case SIO_NCON:
        strcpy(label, "GroupContaminationCoun");
        break;
    case SIO_MCON:
        strcpy(label, "GroupContaminationMass");
        break;
    case SIO_BGPOS:
        strcpy(label, "BGPO");
        break;
    case SIO_BGMTOP:
        strcpy(label, "BGMA");
        break;
    case SIO_BGRTOP:
        strcpy(label, "BGRA");
        break;
    case SIO_NSUB:
        strcpy(label, "GroupNsubs");
        break;
    case SIO_FSUB:
        strcpy(label, "GroupFirstSub");
        break;
    case SIO_SLEN:
        strcpy(label, "SubhaloLen");
        break;
    case SIO_SOFF:
        strcpy(label, "SubhaloOffset");
        break;
    case SIO_PFOF:
        strcpy(label, "SubhaloParent");
        break;
    case SIO_MSUB:
        strcpy(label, "SubhaloMass");
        break;
    case SIO_SPOS:
        strcpy(label, "SubhaloPos");
        break;
    case SIO_SVEL:
        strcpy(label, "SubhaloVel");
        break;
    case SIO_SCM:
        strcpy(label, "SubhaloCM");
        break;
    case SIO_SPIN:
        strcpy(label, "SubhaloSpin");
        break;
    case SIO_DSUB:
        strcpy(label, "SubhaloVelDisp");
        break;
    case SIO_VMAX:
        strcpy(label, "SubhaloVmax");
        break;
    case SIO_RVMAX:
        strcpy(label, "SubhaloVmaxRad");
        break;
    case SIO_RHMS:
        strcpy(label, "SubhaloHalfmassRad");
        break;
    case SIO_SHMR:
        strcpy(label, "SubhaloStelHalfmRad");
        break;
    case SIO_MBID:
        strcpy(label, "SubhaloIDMostbound");
        break;
    case SIO_GRNR:
        strcpy(label, "SubhaloGrNr");
        break;
    case SIO_SMST:
        strcpy(label, "SMST");
        break;
    case SIO_SLUM:
        strcpy(label, "SLUM");
        break;
    case SIO_SLATT:
        strcpy(label, "SLAT");
        break;
    case SIO_SLOBS:
        strcpy(label, "SLOB");
        break;
    case SIO_DUST:
        strcpy(label, "DUST");
        break;
    case SIO_SAGE:
        strcpy(label, "SAGE");
        break;
    case SIO_SZ:
        strcpy(label, "SZ  ");
        break;
    case SIO_SSFR:
        strcpy(label, "SSFR");
        break;
    case SIO_SSYNCHROTRON:
        strcpy(label, "SSYN");
        break;
    case SIO_MHI:
        strcpy(label, "SMHI");
        break;
    case SIO_SSANG:
        strcpy(label, "SANG");
        break;
    case SIO_SBVAL:
        strcpy(label, "BVAL");
        break;
    case SIO_SZGAS:
        strcpy(label, "ZG");
        break;
    case SIO_SZCGAS:
        strcpy(label, "ZCG");
        break;
    case SIO_PPOS:
        strcpy(label, "PPOS");
        break;
    case SIO_PVEL:
        strcpy(label, "PVEL");
        break;
    case SIO_PTYP:
        strcpy(label, "PTYP");
        break;
    case SIO_PMAS:
        strcpy(label, "PMAS");
        break;
    case SIO_PAGE:
        strcpy(label, "PAGE");
        break;
    case SIO_PID:
        strcpy(label, "PID ");
        break;
    default:
        ENDRUN("Error in SIO labels.");
        break;
    }
}
#endif // HAVE_HDF5

/**
 * @brief Retrieves the I/O label for a specified block in a data structure.
 *
 * This function takes an enumeration representing a block number as input and returns the corresponding I/O label
 * associated with that block.
 *
 * @param blocknr An enum value of type `siofields` representing the block number for which the I/O label is retrieved.
 * @param label A character array where the I/O label will be stored. It must have a length of at least 5 characters to
 * accommodate the label and the null terminator.
 */
void get_IO_Label_sub(enum siofields blocknr, char *label)
{
    switch (blocknr)
    {
    case SIO_GLEN:
        strncpy(label, "GLEN", 4);
        break;
    case SIO_GOFF:
        strncpy(label, "GOFF", 4);
        break;
    case SIO_MTOT:
        strncpy(label, "MTOT", 4);
        break;
    case SIO_GPOS:
        strncpy(label, "GPOS", 4);
        break;
#ifdef SO_BAR_INFO
    case SIO_MMEA:
        strncpy(label, "M200", 4);
        break;
    case SIO_RMEA:
        strncpy(label, "R200", 4);
        break;
    case SIO_DMEA:
        strncpy(label, "D200", 4);
        break;
    case SIO_MCRI:
        strncpy(label, "MCRI", 4);
        break;
    case SIO_RCRI:
        strncpy(label, "RCRI", 4);
        break;
    case SIO_DCRI:
        strncpy(label, "DCRI", 4);
        break;
    case SIO_MTOP:
        strncpy(label, "MVIR", 4);
        break;
    case SIO_RTOP:
        strncpy(label, "RVIR", 4);
        break;
    case SIO_DTOP:
        strncpy(label, "DVIR", 4);
        break;
#else // SO_BAR_INFO
    case SIO_MMEA:
        strncpy(label, "MMEA", 4);
        break;
    case SIO_RMEA:
        strncpy(label, "RMEA", 4);
        break;
    case SIO_MCRI:
        strncpy(label, "MCRI", 4);
        break;
    case SIO_RCRI:
        strncpy(label, "RCRI", 4);
        break;
    case SIO_MTOP:
        strncpy(label, "MTOP", 4);
        break;
    case SIO_RTOP:
        strncpy(label, "RTOP", 4);
        break;
    case SIO_DMEA:
        strncpy(label, "DMEA", 4);
        break;
    case SIO_DCRI:
        strncpy(label, "DCRI", 4);
        break;
    case SIO_DTOP:
        strncpy(label, "DTOP", 4);
        break;
#endif // SO_BAR_INFO
    case SIO_M500:
        strncpy(label, "M500", 4);
        break;
    case SIO_R500:
        strncpy(label, "R500", 4);
        break;
    case SIO_D500:
        strncpy(label, "D500", 4);
        break;
    case SIO_M5CC:
        strncpy(label, "M5CC", 4);
        break;
    case SIO_R5CC:
        strncpy(label, "R5CC", 4);
        break;
    case SIO_D5CC:
        strncpy(label, "D5CC", 4);
        break;
    case SIO_M25K:
        strncpy(label, "M25K", 4);
        break;
    case SIO_R25K:
        strncpy(label, "R25K", 4);
        break;
    case SIO_D25K:
        strncpy(label, "D25K", 4);
        break;
    case SIO_MGAS:
        strncpy(label, "MGAS", 4);
        break;
    case SIO_MSTR:
        strncpy(label, "MSTR", 4);
        break;
    case SIO_TGAS:
        strncpy(label, "TGAS", 4);
        break;
    case SIO_LGAS:
        strncpy(label, "LGAS", 4);
        break;
    case SIO_YGAS:
        strncpy(label, "YGAS", 4);
        break;
    case SIO_NCON:
        strncpy(label, "NCON", 4);
        break;
    case SIO_MCON:
        strncpy(label, "MCON", 4);
        break;
    case SIO_BGPOS:
        strncpy(label, "BGPO", 4);
        break;
    case SIO_BGMTOP:
        strncpy(label, "BGMA", 4);
        break;
    case SIO_BGRTOP:
        strncpy(label, "BGRA", 4);
        break;
    case SIO_NSUB:
        strncpy(label, "NSUB", 4);
        break;
    case SIO_FSUB:
        strncpy(label, "FSUB", 4);
        break;
    case SIO_SLEN:
        strncpy(label, "SLEN", 4);
        break;
    case SIO_SOFF:
        strncpy(label, "SOFF", 4);
        break;
    case SIO_PFOF:
        strncpy(label, "SSUB", 4);
        break;
    case SIO_MSUB:
        strncpy(label, "MSUB", 4);
        break;
    case SIO_SPOS:
        strncpy(label, "SPOS", 4);
        break;
    case SIO_SVEL:
        strncpy(label, "SVEL", 4);
        break;
    case SIO_SCM:
        strncpy(label, "SCM ", 4);
        break;
    case SIO_SPIN:
        strncpy(label, "SPIN", 4);
        break;
    case SIO_DSUB:
        strncpy(label, "DSUB", 4);
        break;
    case SIO_VMAX:
        strncpy(label, "VMAX", 4);
        break;
    case SIO_RVMAX:
        strncpy(label, "RMAX", 4);
        break;
    case SIO_RHMS:
        strncpy(label, "RHMS", 4);
        break;
    case SIO_SHMR:
        strncpy(label, "REFF", 4);
        break;
    case SIO_MBID:
        strncpy(label, "MBID", 4);
        break;
    case SIO_GRNR:
        strncpy(label, "GRNR", 4);
        break;
    case SIO_SMST:
        strncpy(label, "SMST", 4);
        break;
    case SIO_SLUM:
        strncpy(label, "SLUM", 4);
        break;
    case SIO_SLATT:
        strncpy(label, "SLAT", 4);
        break;
    case SIO_SLOBS:
        strncpy(label, "SLOB", 4);
        break;
    case SIO_DUST:
        strncpy(label, "DUST", 4);
        break;
    case SIO_SAGE:
        strncpy(label, "SAGE", 4);
        break;
    case SIO_SZ:
        strncpy(label, "SZ  ", 4);
        break;
    case SIO_SSFR:
        strncpy(label, "SSFR", 4);
        break;
    case SIO_SSYNCHROTRON:
        strncpy(label, "SSYN", 4);
        break;
    case SIO_MHI:
        strncpy(label, "SMHI", 4);
        break;
    case SIO_SSANG:
        strncpy(label, "SANG", 4);
        break;
    case SIO_SBVAL:
        strncpy(label, "BVAL", 4);
        break;
    case SIO_SZGAS:
        strncpy(label, "ZG  ", 4);
        break;
    case SIO_SZCGAS:
        strncpy(label, "ZCG ", 4);
        break;
    case SIO_PPOS:
        strncpy(label, "PPOS", 4);
        break;
    case SIO_PVEL:
        strncpy(label, "PVEL", 4);
        break;
    case SIO_PTYP:
        strncpy(label, "PTYP", 4);
        break;
    case SIO_PMAS:
        strncpy(label, "PMAS", 4);
        break;
    case SIO_PAGE:
        strncpy(label, "PAGE", 4);
        break;
    case SIO_PID:
        strncpy(label, "PID ", 4);
        break;
    default:
        ENDRUN("Error in SIO labels.");
        break;
    }
}

/**
 * @brief Retrieves the data type key for a specified block in a data structure.
 *
 * This function takes an enumeration representing a block number as input and returns the data type key associated with
 * that block in a data structure.
 *
 * @param blocknr An enum value of type `siofields` representing the block number for which the data type key is
 * retrieved.
 * @return Returns an integer value indicating the data type key for the specified block:
 *         - If the block is `SIO_MBID` or `SIO_PID` and `LONGIDS` is defined, returns 2 (native long long).
 *         - If the block is `SIO_MBID` or `SIO_PID` and `LONGIDS` is not defined, returns 0 (native int).
 *         - If the block is one of `SIO_GLEN`, `SIO_GOFF`, `SIO_NCON`, `SIO_NSUB`, `SIO_FSUB`, `SIO_SLEN`, `SIO_SOFF`,
 * `SIO_PFOF`, `SIO_GRNR`, or `SIO_PTYP`, returns 0 (native int).
 *         - For all other blocks, returns 1 (native MyOutputFloat) if `OUTPUT_IN_DOUBLEPRECISION` is not defined;
 * otherwise, returns 3.
 */
int get_datatype_in_sub(enum siofields blocknr)
{
    int typekey;

    switch (blocknr)
    {
    case SIO_MBID:
    case SIO_PID:
#ifdef LONGIDS
        typekey = 2; /* native long long */
#else // LONGIDS
        typekey = 0; /* native int */
#endif // LONGIDS
        break;

    case SIO_GLEN:
    case SIO_GOFF:
    case SIO_NCON:
    case SIO_NSUB:
    case SIO_FSUB:
    case SIO_SLEN:
    case SIO_SOFF:
    case SIO_PFOF:
    case SIO_GRNR:
    case SIO_PTYP:
        typekey = 0; /* native int */
        break;

    default:
#ifdef OUTPUT_IN_DOUBLEPRECISION
        typekey = 3;
#else // OUTPUT_IN_DOUBLEPRECISION
        typekey = 1; /* native MyOutputFloat */
#endif // OUTPUT_IN_DOUBLEPRECISION
        break;
    }

    return typekey;
}

/**
 * @brief Checks if a specified block is present in a data structure.
 *
 * This function takes an enumeration representing a block number as input and checks if the corresponding block is
 * present in a data structure.
 *
 * @param blocknr An enum value of type `siofields` representing the block number to check for presence.
 * @return Returns an integer value indicating whether the specified block is present (1) or not present (0).
 */
int block_in_sub(enum siofields blocknr)
{
    int present = 0;

    switch (blocknr)
    {
    case SIO_GLEN:
    case SIO_GOFF:
    case SIO_MTOT:
    case SIO_GPOS:
    case SIO_MMEA:
    case SIO_RMEA:
    case SIO_MCRI:
    case SIO_RCRI:
    case SIO_MTOP:
    case SIO_RTOP:
#ifdef SO_VEL_DISPERSIONS
    case SIO_DMEA:
    case SIO_DCRI:
    case SIO_DTOP:
#endif // SO_VEL_DISPERSIONS
#ifdef SO_BAR_INFO
    case SIO_M500:
    case SIO_M5CC:
    case SIO_M25K:
    case SIO_R500:
    case SIO_R5CC:
    case SIO_R25K:
#ifdef SO_VEL_DISPERSIONS
    case SIO_D500:
    case SIO_D5CC:
    case SIO_D25K:
#endif // SO_VEL_DISPERSIONS
    case SIO_MGAS:
    case SIO_MSTR:
    case SIO_TGAS:
    case SIO_LGAS:
    case SIO_YGAS:
#endif // SO_BAR_INFO
    case SIO_NCON:
    case SIO_MCON:
#ifdef SUBFIND_COUNT_BIG_HALOS
    case SIO_BGPOS:
    case SIO_BGMTOP:
    case SIO_BGRTOP:
#endif // SUBFIND_COUNT_BIG_HALOS
    case SIO_NSUB:
    case SIO_FSUB:
    case SIO_SLEN:
    case SIO_SOFF:
    case SIO_PFOF:
    case SIO_MSUB:
    case SIO_SPOS:
    case SIO_SVEL:
    case SIO_SCM:
    case SIO_SPIN:
    case SIO_DSUB:
    case SIO_VMAX:
    case SIO_RVMAX:
    case SIO_RHMS:
    case SIO_SHMR:
    case SIO_MBID:
    case SIO_GRNR:
    case SIO_SMST:
    case SIO_SSANG:
    case SIO_SBVAL:
#ifdef LT_ADD_GAL_TO_SUB
    case SIO_SLUM:
    case SIO_SAGE:
    case SIO_SZ:
    case SIO_SSFR:
    case SIO_SZGAS:
    case SIO_SZCGAS:
#ifdef LT_METAL_COOLING_WAL
    case SIO_MHI:
#endif // LT_METAL_COOLING_WAL
#endif // LT_ADD_GAL_TO_SUB
#ifdef LMB_CR_OUTPUT_SYNCHROTRON
    case SIO_SSYNCHROTRON:
#endif // LMB_CR_OUTPUT_SYNCHROTRON
#ifdef DUSTATT
    case SIO_SLATT:
    case SIO_DUST:
#endif // DUSTATT
#ifdef OBSERVER_FRAME
    case SIO_SLOBS:
#endif // OBSERVER_FRAME
#ifdef SUBFIND_SAVE_PARTICLELISTS
    case SIO_PPOS:
    case SIO_PVEL:
    case SIO_PTYP:
    case SIO_PMAS:
#ifdef SFR
    case SIO_PAGE:
#endif // SFR
#endif // SUBFIND_SAVE_PARTICLELISTS
    case SIO_PID:
        present = 1;
        break;
    default:
        present = 0;
        break;
    }

    return present;
}

/**
 * @brief Saves the local catalogue of substructures to a file.
 *
 * This function saves the local catalogue of substructures to a file. It writes various properties of the substructures
 * including their positions, masses, sizes, velocities, parent group indices, and other relevant information.
 *
 * @param num The number associated with the catalogue.
 */
void subfind_save_local_catalogue(int num)
{
    FILE *fd;
    char buf[500], fname[500], label[] = "--------";

    void *IOBuffer;
    MyOutputFloat *fp;
    int *fp_int, nwrite, ndim, bytes_per_blockelement, type, datatype, bytes;
    MyIDType *fp_id;
    enum siofields blocknr;
    unsigned int blksize;

#if defined(SUBFIND_COUNT_BIG_HALOS) || defined(LT_ADD_GAL_TO_SUB)
    int k;
#endif // defined(SUBFIND_COUNT_BIG_HALOS) || defined(LT_ADD_GAL_TO_SUB)

#ifdef SUBFIND_SAVE_PARTICLELISTS
    double a3inv;
#endif // SUBFIND_SAVE_PARTICLELISTS

#ifdef WRITE_SUB_IN_SNAP_FORMAT
    unsigned int nextblock;
#define SKIP                                                                                                           \
    {                                                                                                                  \
        my_fwrite(&blksize, sizeof(int), 1, fd);                                                                       \
    }

#ifdef HAVE_HDF5
    hid_t hdf5_file = 0, hdf5_grp[3], hdf5_headergrp = 0, hdf5_dataspace_memory;
    hid_t hdf5_datatype = 0, hdf5_dataspace_in_file = 0, hdf5_dataset = 0;
    herr_t hdf5_status;
    hsize_t dims[2], count[2], start[2];
    int rank = 0;
#endif // HAVE_HDF5
#endif // WRITE_SUB_IN_SNAP_FORMAT

#ifdef SUBFIND_SAVE_PARTICLELISTS
    if (All.ComovingIntegrationOn)
    {
        a3inv = 1.0 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1.0;
    }
#endif // SUBFIND_SAVE_PARTICLELISTS

#ifdef LONG_OUTPUT_NUMBERS
#ifdef WRITE_SUB_IN_SNAP_FORMAT
#ifdef WRITE_SUB_COLLECTIVELY
    if (All.NumFilesPerSnapshot == 1)
#else // WRITE_SUB_COLLECTIVELY
    if (NTask == 1)
#endif // WRITE_SUB_COLLECTIVELY
    {
        sprintf(fname, "%s/groups_%04d/%s_%04d", All.OutputDir, num, "sub", num);
    }
    else
    {
        sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "sub", num, ThisTask);
    }
#else // WRITE_SUB_IN_SNAP_FORMAT
    sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "subhalo_tab", num, ThisTask);
#endif // WRITE_SUB_IN_SNAP_FORMAT
#else // LONG_OUTPUT_NUMBERS
#ifdef WRITE_SUB_IN_SNAP_FORMAT
#ifdef WRITE_SUB_COLLECTIVELY
    if (All.NumFilesPerSnapshot == 1)
#else // WRITE_SUB_COLLECTIVELY
    if (NTask == 1)
#endif // WRITE_SUB_COLLECTIVELY
    {
        sprintf(fname, "%s/groups_%03d/%s_%03d", All.OutputDir, num, "sub", num);
    }
    else
    {
        sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "sub", num, ThisTask);
    }
#else // WRITE_SUB_IN_SNAP_FORMAT
    sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_tab", num, ThisTask);
#endif // WRITE_SUB_IN_SNAP_FORMAT
#endif // LONG_OUTPUT_NUMBERS
    strcpy(buf, fname);

#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
    if (All.SnapFormat == 3)
    {
        sprintf(buf, "%s.hdf5", fname);
        hdf5_file = H5Fcreate(buf, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        hdf5_headergrp = H5Gcreate2(hdf5_file, "/Header", 0, H5P_DEFAULT, H5P_DEFAULT);
        hdf5_grp[0] = H5Gcreate2(hdf5_file, "/Group", 0, H5P_DEFAULT, H5P_DEFAULT);
        hdf5_grp[1] = H5Gcreate2(hdf5_file, "/Subhalo", 0, H5P_DEFAULT, H5P_DEFAULT);
        hdf5_grp[2] = H5Gcreate2(hdf5_file, "/IDs", 0, H5P_DEFAULT, H5P_DEFAULT);
    }
    else
    {
#endif // defined(HAVE_HDF5)&& defined(WRITE_SUB_IN_SNAP_FORMAT)
        if (!(fd = fopen(buf, "w")))
        {
            ENDRUN("can't open file `%s`\n", buf);
        }
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
    }
#endif // defined(HAVE_HDF5)&& defined(WRITE_SUB_IN_SNAP_FORMAT)

#ifdef WRITE_SUB_IN_SNAP_FORMAT
    for (int i = 0; i < 6; i++)
    {
        header.npart[i] = 0;
        header.npartTotal[i] = 0;
        header.npartTotalHighWord[i] = 0;
        header.mass[i] = 0;
    }
    header.npart[0] = Ngroups;
    header.npartTotal[0] = TotNgroups;
    header.npart[1] = Nsubgroups;
    header.npartTotal[1] = TotNsubgroups;
    header.npart[2] = Nids;
    header.npartTotal[2] = (unsigned int)TotNids;
    header.npartTotalHighWord[2] = (unsigned int)(TotNids >> 32);
#ifdef SUBFIND_COUNT_BIG_HALOS
    header.npart[3] = Nbiggroups;
    header.npartTotal[3] = TotNbiggroups;
#endif // SUBFIND_COUNT_BIG_HALOS

    header.time = All.Time;
    if (All.ComovingIntegrationOn)
    {
        header.redshift = 1.0 / All.Time - 1;
    }
    else
    {
        header.redshift = 0;
    }

#ifdef COOLING
    header.flag_cooling = 1;
#endif // COOLING

#ifdef SFR
    header.flag_sfr = 1;
    header.flag_feedback = 1;
    header.flag_stellarage = 1;
#if !defined(LT_STELLAREVOLUTION)
    header.flag_metals = 1;
#endif // !LT_STELLAREVOLUTION
// #ifdef LT_STELLAREVOLUTION
//   header.flag_stellarevolution = 2;
// #endif // LT_STELLAREVOLUTION
#endif // SFR

    header.num_files = NTask;
    header.BoxSize = All.BoxSize;
    header.Omega0 = All.Omega0;
    header.OmegaLambda = All.OmegaLambda;
    header.HubbleParam = All.HubbleParam;

#ifdef LT_ADD_GAL_TO_SUB
    for (int k = 0; k < LT_ADD_GAL_TO_SUB; k++)
    {
        header.names[k][0] = bc_name[k][0];
        header.names[k][1] = bc_name[k][1];
    }
    for (int k = LT_ADD_GAL_TO_SUB; k < 15; k++)
    {
        header.names[k][0] = ' ';
        header.names[k][1] = ' ';
    }
#endif // LT_ADD_GAL_TO_SUB

#ifdef WRITE_SUB_IN_SNAP_FORMAT
    if (All.SnapFormat == 2)
    {
        blksize = sizeof(int) + 4 * sizeof(char);
        SKIP;
        my_fwrite((void *)"HEAD", sizeof(char), 4, fd);
        nextblock = sizeof(header) + 2 * sizeof(int);
        my_fwrite(&nextblock, sizeof(int), 1, fd);
        SKIP;
    }
#ifdef HAVE_HDF5
    if (All.SnapFormat == 3)
    {
        write_header_attributes_in_hdf5(hdf5_headergrp);
    }

    else
    {
#endif // HAVE_HDF5
#endif // WRITE_SUB_IN_SNAP_FORMAT
        blksize = sizeof(header);
        SKIP;
        my_fwrite(&header, sizeof(header), 1, fd);
        SKIP;
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
    }
#endif // defined(HAVE_HDF5) &&  defined(WRITE_SUB_IN_SNAP_FORMAT)

#ifdef WRITE_SUB_IN_SNAP_FORMAT
    if (All.SnapFormat == 2)
    {
        if (!(InfoBlock = (struct info_block *)mymalloc("InfoBlock", bytes = sizeof(struct info_block) * 1000)))
        {
            ENDRUN("failed to allocate memory for `InfoBlock' (%g MB).\n", bytes / (1024.0 * 1024.0));
        }

        int n_info = 0;
        for (int bnr = 0; bnr < 1000; bnr++)
        {
            blocknr = (enum siofields)bnr;

            if (blocknr == SIO_LASTENTRY)
            {
                break;
            }

            if (block_in_sub(blocknr))
            {
                type = get_sub_entrytype_of_block(blocknr);

                for (int i = 0; i < 6; i++)
                {
                    InfoBlock[n_info].is_present[i] = 0;
                }
                InfoBlock[n_info].is_present[type] = 1;

                InfoBlock[n_info].ndim = get_values_per_sub(blocknr);

                get_IO_Label_sub(blocknr, label);
                for (int i = 0; i < 4; i++)
                {
                    InfoBlock[n_info].label[i] = label[i];
                }

                datatype = get_datatype_in_sub(blocknr);
                switch (datatype)
                {
                case 0:
                    if (InfoBlock[n_info].ndim <= 1)
                    {
                        strncpy(InfoBlock[n_info].type, "LONG    ", 8);
                    }
                    else
                    {
                        strncpy(InfoBlock[n_info].type, "LONGN   ", 8);
                    }
                    break;
                case 1:
                    if (InfoBlock[n_info].ndim <= 1)
                    {
                        strncpy(InfoBlock[n_info].type, "FLOAT   ", 8);
                    }
                    else
                    {
                        strncpy(InfoBlock[n_info].type, "FLOATN  ", 8);
                    }
                    break;
                case 2:
                    if (InfoBlock[n_info].ndim <= 1)
                    {
                        strncpy(InfoBlock[n_info].type, "LLONG   ", 8);
                    }
                    else
                    {
                        strncpy(InfoBlock[n_info].type, "LLONGN  ", 8);
                    }
                    break;
                case 3:
                    if (InfoBlock[n_info].ndim <= 1)
                    {
                        strncpy(InfoBlock[n_info].type, "DOUBLE  ", 8);
                    }
                    else
                    {
                        strncpy(InfoBlock[n_info].type, "DOUBLEN ", 8);
                    }
                    break;
                default:
                    ENDRUN_MAIN("Error in datatype.");
                    break;
                }
                n_info++;
            }
        }

        blksize = sizeof(int) + 4 * sizeof(char);
        SKIP;
        my_fwrite((void *)"INFO", sizeof(char), 4, fd);
        nextblock = n_info * sizeof(struct info_block) + 2 * sizeof(int);
        my_fwrite(&nextblock, sizeof(int), 1, fd);
        SKIP;
        blksize = n_info * sizeof(struct info_block);
        SKIP;
        my_fwrite(InfoBlock, sizeof(struct info_block), n_info, fd);
        SKIP;

        myfree(InfoBlock);
    }
#endif // WRITE_SUB_IN_SNAP_FORMAT

#else // WRITE_SUB_IN_SNAP_FORMAT
    my_fwrite(&Ngroups, sizeof(int), 1, fd);
    my_fwrite(&TotNgroups, sizeof(int), 1, fd);
    my_fwrite(&Nids, sizeof(int), 1, fd);
    my_fwrite(&TotNids, sizeof(long long), 1, fd);
    my_fwrite(&NTask, sizeof(int), 1, fd);
    my_fwrite(&Nsubgroups, sizeof(int), 1, fd);
    my_fwrite(&TotNsubgroups, sizeof(int), 1, fd);
#endif // WRITE_SUB_IN_SNAP_FORMAT

    for (int bnr = 0; bnr < 1000; bnr++)
    {
        blocknr = (enum siofields)bnr;

        if (blocknr == SIO_LASTENTRY)
        {
            break;
        }

        if (block_in_sub(blocknr))
        {
            type = get_sub_entrytype_of_block(blocknr);
            switch (type)
            {
            case 0:
                nwrite = Ngroups;
                break;
            case 1:
                nwrite = Nsubgroups;
                break;
            case 2:
                nwrite = Nids;
                break;
#ifdef SUBFIND_COUNT_BIG_HALOS
            case 3:
                nwrite = Nbiggroups;
                break;
#endif // SUBFIND_COUNT_BIG_HALOS
            }

            ndim = get_values_per_sub(blocknr);

#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
            if (All.SnapFormat == 3)
            {
                dims[0] = nwrite;
                dims[1] = ndim;

                if (dims[1] == 1)
                    rank = 1;
                else
                    rank = 2;

                switch (get_datatype_in_sub(blocknr))
                {
                case 0:
                    hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT);
                    break;
                case 1:
                    hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
                    break;
                case 2:
                    hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT64);
                    break;
                case 3:
                    hdf5_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
                    break;
                default:
                    ENDRUN("Error in datatype.");
                    break;
                }
                get_IO_Label_HDF5_sub(blocknr, buf);
                hdf5_dataspace_in_file = H5Screate_simple(rank, dims, NULL);
                hdf5_dataset = H5Dcreate(hdf5_grp[type], buf, hdf5_datatype, hdf5_dataspace_in_file, H5P_DEFAULT,
                                         H5P_DEFAULT, H5P_DEFAULT);
            }
#endif // defined(HAVE_HDF5) &&  defined(WRITE_SUB_IN_SNAP_FORMAT)

            datatype = get_datatype_in_sub(blocknr);
            switch (datatype)
            {
            case 0:
                bytes_per_blockelement = 4;
                break;
            case 1:
                bytes_per_blockelement = 4;
                break;
            case 2:
                bytes_per_blockelement = 8;
                break;
            case 3:
                bytes_per_blockelement = 8;
                break;
            default:
                ENDRUN("Error in datatype.");
                break;
            }

#ifdef WRITE_SUB_IN_SNAP_FORMAT
            get_IO_Label_sub(blocknr, label);

            if (nwrite > 0)
            {
                if (All.SnapFormat == 2)
                {
                    blksize = sizeof(int) + 4 * sizeof(char);
                    SKIP;
                    my_fwrite((void *)label, sizeof(char), 4, fd);
                    nextblock = nwrite * bytes_per_blockelement * ndim + 2 * sizeof(int);
                    my_fwrite(&nextblock, sizeof(int), 1, fd);
                    SKIP;
                }
                blksize = nwrite * bytes_per_blockelement * ndim;
                if (All.SnapFormat != 3)
                {
                    SKIP;
                }
            }
            else
                blksize = 0;
#else // WRITE_SUB_IN_SNAP_FORMAT
            blksize = nwrite * bytes_per_blockelement * ndim;
#endif // WRITE_SUB_IN_SNAP_FORMAT

            VERBOSE(0, "Writing block %d (%c%c%c%c), n=%d, ptype=%d, dtype=%d, ndim=%d, bpb=%d bytes=%ud\n", bnr,
                    label[0], label[1], label[2], label[3], nwrite, type, datatype, ndim, bytes_per_blockelement,
                    blksize);

            if (!(IOBuffer = mymalloc("IOBuffer", bytes = blksize)))
            {
                ENDRUN("failed to allocate memory for `IOBuffer' (%g MB).\n", bytes / (1024.0 * 1024.0));
            }

            fp = (MyOutputFloat *)IOBuffer;
            fp_int = (int *)IOBuffer;
            fp_id = (MyIDType *)IOBuffer;

            switch (blocknr)
            {
            case SIO_GLEN:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = Group[i].Len;
                }
                break;
            case SIO_GOFF:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = Group[i].Offset;
                }
                break;
            case SIO_MTOT:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].Mass;
                }
                break;
            case SIO_GPOS:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)Group[i].Pos[j];
#ifdef DS_SHIFT_BOX
                        fp[i * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
                    }
                }
                break;
            case SIO_MMEA:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].M_Mean200;
                }
                break;
            case SIO_RMEA:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].R_Mean200;
                }
                break;
            case SIO_MCRI:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].M_Crit200;
                }
                break;
            case SIO_RCRI:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].R_Crit200;
                }
                break;
            case SIO_MTOP:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].M_TopHat200;
                }
                break;
            case SIO_RTOP:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].R_TopHat200;
                }
                break;
#ifdef SO_VEL_DISPERSIONS
            case SIO_DMEA:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].VelDisp_Mean200;
                }
                break;
            case SIO_DCRI:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].VelDisp_Crit200;
                }
                break;
            case SIO_DTOP:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].VelDisp_TopHat200;
                }
                break;
#endif // SO_VEL_DISPERSIONS
#ifdef SO_BAR_INFO
            case SIO_M500:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].M_Mean500;
                }
                break;
            case SIO_R500:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].R_Mean500;
                }
                break;
            case SIO_M5CC:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].M_Crit500;
                }
                break;
            case SIO_R5CC:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].R_Crit500;
                }
                break;
            case SIO_M25K:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].M_Crit2500;
                }
                break;
            case SIO_R25K:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].R_Crit2500;
                }
                break;
            case SIO_MGAS:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)Group[i].gas_mass[j];
                    }
                }
                break;
            case SIO_MSTR:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)Group[i].star_mass[j];
                    }
                }
                break;
            case SIO_TGAS:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)Group[i].temp[j];
                    }
                }
                break;
            case SIO_LGAS:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)Group[i].xlum[j];
                    }
                }
                break;
#endif // SO_BAR_INFO
            case SIO_NCON:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = Group[i].ContaminationLen;
                }
                break;
            case SIO_MCON:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)Group[i].ContaminationMass;
                }
                break;
#ifdef SUBFIND_COUNT_BIG_HALOS
            case SIO_BGPOS:
                k = 0;
                for (int i = 0; i < Ngroups; i++)
                    if (Group[i].M_TopHat200 > SUBFIND_COUNT_BIG_HALOS)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            fp[k * 3 + j] = (MyOutputFloat)Group[i].Pos[j];
#ifdef DS_SHIFT_BOX
                            fp[k * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
                        }
                        k++;
                    }
                ENDRUN_IF(k != nwrite, "Error as k uneaqual to nwrite");
                break;
            case SIO_BGMTOP:
                k = 0;
                for (int i = 0; i < Ngroups; i++)
                    if (Group[i].M_TopHat200 > SUBFIND_COUNT_BIG_HALOS)
                    {
                        fp[k] = (MyOutputFloat)Group[i].M_TopHat200;
                        k++;
                    }
                ENDRUN_IF(k != nwrite, "Error as k uneaqual to nwrite");
                break;
            case SIO_BGRTOP:
                k = 0;
                for (int i = 0; i < Ngroups; i++)
                {
                    if (Group[i].M_TopHat200 > SUBFIND_COUNT_BIG_HALOS)
                    {
                        fp[k] = (MyOutputFloat)Group[i].R_TopHat200;
                        k++;
                    }
                }
                ENDRUN_IF(k != nwrite, "Error as k uneaqual to nwrite");
                break;
#endif // SUBFIND_COUNT_BIG_HALOS
            case SIO_NSUB:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = Group[i].Nsubs;
                }
                break;
            case SIO_FSUB:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = Group[i].FirstSub;
                }
                break;
            case SIO_SLEN:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = SubGroup[i].Len;
                }
                break;
            case SIO_SOFF:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = SubGroup[i].Offset;
                }
                break;
            case SIO_PFOF:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = SubGroup[i].SubParent;
                }
                break;
            case SIO_MSUB:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].Mass;
                }
                break;
            case SIO_SPOS:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)SubGroup[i].Pos[j];
#ifdef DS_SHIFT_BOX
                        fp[i * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
                    }
                }
                break;
            case SIO_SVEL:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)SubGroup[i].Vel[j];
                    }
                }
                break;
            case SIO_SCM:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)SubGroup[i].CM[j];
                    }
                }
                break;
            case SIO_SPIN:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)SubGroup[i].Spin[j];
                    }
                }
                break;
            case SIO_DSUB:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubVelDisp;
                }
                break;
            case SIO_VMAX:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubVmax;
                }
                break;
            case SIO_RVMAX:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubVmaxRad;
                }
                break;
            case SIO_RHMS:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubHalfMass;
                }
                break;
            case SIO_SHMR:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubStelHalfMass;
                }
                break;
            case SIO_MBID:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_id[i] = SubGroup[i].SubMostBoundID;
                }
                break;
            case SIO_GRNR:
                for (int i = 0; i < nwrite; i++)
                {
                    fp_int[i] = SubGroup[i].GrNr;
                }
                break;
            case SIO_SMST:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < 6; j++)
                    {
                        fp[i * 6 + j] = (MyOutputFloat)SubGroup[i].MassTab[j];
                    }
                }
                break;
            case SIO_SSANG:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubStelAngMom;
                }
                break;
            case SIO_SBVAL:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubBval;
                }
                break;
#ifdef LT_ADD_GAL_TO_SUB
            case SIO_SLUM:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
                    {
                        fp[i * LT_ADD_GAL_TO_SUB + j] = (MyOutputFloat)SubGroup[i].SubLum[j];
                    }
                }
                break;
#ifdef DUSTATT
            case SIO_SLATT:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
                    {
                        fp[i * LT_ADD_GAL_TO_SUB + j] = (MyOutputFloat)SubGroup[i].SubLumAtt[j];
                    }
                }
                break;
            case SIO_DUST:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < DUSTATT; j++)
                    {
                        fp[i * DUSTATT + j] = (MyOutputFloat)SubGroup[i].DustProfile[j];
                    }
                }
                break;
#endif // DUSTATT
#ifdef OBSERVER_FRAME
            case SIO_SLOBS:
                for (int i = 0; i < nwrite; i++)
                {
                    for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
                    {
                        fp[i * LT_ADD_GAL_TO_SUB + j] = (MyOutputFloat)SubGroup[i].SubLumObs[j];
                    }
                }
                break;
#endif // OBSERVER_FRAME
            case SIO_SAGE:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubMeanAge;
                }
                break;
            case SIO_SZ:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubMeanZStar;
                }
                break;
            case SIO_SSFR:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubSFR;
                }
                break;
            case SIO_MHI:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubHIMass;
                }
                break;
            case SIO_SZGAS:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubMeanZGas;
                }
                break;
            case SIO_SZCGAS:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)SubGroup[i].SubMeanZColdGas;
                }
                break;
#endif // LT_ADD_GAL_TO_SUB
#ifdef LMB_CR_OUTPUT_SYNCHROTRON
            case SIO_SSYNCHROTRON:
                for (int i = 0; i < nwrite; i++)
                {
                    fp[i] = (MyOutputFloat)log10(SubGroup[i].SubSynchrotron);
                }
                break;
#endif // LMB_CR_OUTPUT_SYNCHROTRON
#ifdef SUBFIND_SAVE_PARTICLELISTS
            case SIO_PPOS:
#ifndef WRITE_SUB_IN_SNAP_FORMAT /* open new file in case of old format */
                fclose(fd);
#ifdef LONG_OUTPUT_NUMBERS
                sprintf(buf, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "subhalo_posvel", num, ThisTask);
#else // LONG_OUTPUT_NUMBERS
            sprintf(buf, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_posvel", num, ThisTask);
#endif // LONG_OUTPUT_NUMBERS
                if (!(fd = fopen(buf, "w")))
                {
                    ENDRUN("can't open file `%s`\n", buf);
                }
                my_fwrite(&Ngroups, sizeof(int), 1, fd);
                my_fwrite(&TotNgroups, sizeof(int), 1, fd);
                my_fwrite(&Nids, sizeof(int), 1, fd);
                my_fwrite(&TotNids, sizeof(long long), 1, fd);
                my_fwrite(&NTask, sizeof(int), 1, fd);
                my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd);
                my_fwrite(&All.Time, sizeof(double), 1, fd);
#endif // WRITE_SUB_IN_SNAP_FORMAT
                for (int i = 0; i < nwrite; i++)
                    for (int j = 0; j < 3; j++)
                    {
                        fp[i * 3 + j] = (MyOutputFloat)ID_list[i].Pos[j];
#ifdef DS_SHIFT_BOX
                        fp[i * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
                    }
                break;
            case SIO_PVEL:
                for (int i = 0; i < nwrite; i++)
                    for (int j = 0; j < 3; j++)
                        fp[i * 3 + j] = (MyOutputFloat)(ID_list[i].Vel[j] * sqrt(a3inv));
                break;
            case SIO_PTYP:
                for (int i = 0; i < nwrite; i++)
                    fp_int[i] = ID_list[i].Type;
                break;
            case SIO_PMAS:
                for (int i = 0; i < nwrite; i++)
                    fp[i] = (MyOutputFloat)ID_list[i].Mass;
                break;
#ifdef SFR
            case SIO_PAGE:
                for (int i = 0; i < nwrite; i++)
                    fp[i] = (MyOutputFloat)ID_list[i].StellarAge;
                break;
#endif // SFR
#endif // SUBFIND_SAVE_PARTICLELISTS
            case SIO_PID:
#ifndef WRITE_SUB_IN_SNAP_FORMAT /* open new file in case of old format */
                fclose(fd);
#ifdef LONG_OUTPUT_NUMBERS
                sprintf(buf, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "subhalo_ids", num, ThisTask);
#else // LONG_OUTPUT_NUMBERS
                sprintf(buf, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_ids", num, ThisTask);
#endif // LONG_OUTPUT_NUMBERS
                if (!(fd = fopen(buf, "w")))
                {
                    ENDRUN("can't open file `%s`\n", buf);
                }
                my_fwrite(&Ngroups, sizeof(int), 1, fd);
                my_fwrite(&TotNgroups, sizeof(int), 1, fd);
                my_fwrite(&Nids, sizeof(int), 1, fd);
                my_fwrite(&TotNids, sizeof(long long), 1, fd);
                my_fwrite(&NTask, sizeof(int), 1, fd);
                my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd);
#endif // WRITE_SUB_IN_SNAP_FORMAT
                for (int i = 0; i < nwrite; i++)
                {
                    fp_id[i] = ID_list[i].ID;
                }
                break;
            default:
                break;
            } /* closing switch(blocknr) */

            if (blksize > 0)
            {
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                if (All.SnapFormat != 3)
                {
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                    my_fwrite(IOBuffer, blksize, sizeof(char), fd);
#ifdef WRITE_SUB_IN_SNAP_FORMAT
                    SKIP;
#endif // WRITE_SUB_IN_SNAP_FORMAT
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                }
                else if (All.SnapFormat == 3)
                {
                    start[0] = 0;
                    start[1] = 0;

                    count[0] = nwrite;
                    count[1] = ndim;

                    H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

                    dims[0] = nwrite;
                    dims[1] = ndim;
                    hdf5_dataspace_memory = H5Screate_simple(rank, dims, NULL);

                    hdf5_status = H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory, hdf5_dataspace_in_file,
                                           H5P_DEFAULT, IOBuffer);

                    H5Sclose(hdf5_dataspace_memory);
                }
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
            }
            myfree(IOBuffer);

        } /* closing if block exist */

    } /* closing for bnr loop */
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
    if (All.SnapFormat == 3)
    {
        for (int type = 0; type < 3; type++)
            H5Gclose(hdf5_grp[type]);
        H5Gclose(hdf5_headergrp);
        H5Fclose(hdf5_file);
    }
    else
    {
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
        fclose(fd);
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
    }
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
}

/**
 * @brief Save subhalo catalogues collectively.
 *
 * This function saves subhalo catalogues collectively in parallel. Each MPI task writes its data to separate files
 * based on color assignment.
 *
 * @param num The snapshot number.
 */
void subfind_save_catalogues_collectively(int num)
{
    int nwriting = All.NumFilesPerSnapshot / All.NumFilesWrittenInParallel;
    ncolors = All.NumFilesPerSnapshot;
    color = floor(ThisTask * ncolors / NTask);

    MPI_Comm_split(MYMPI_COMM_WORLD, color, ThisTask, &SUB_COMM);
    MPI_Comm_rank(SUB_COMM, &SubThisTask);
    MPI_Comm_size(SUB_COMM, &SubNTask);

    int writing_group = floor(color * nwriting / ncolors);

    FILE *fd;
    char buf[500], fname[500], label[] = "--------";

    void *IOBuffer;
    int nwrite, nwrite_tot, nwrite_in_buffer, ndim, bytes_per_blockelement, type, datatype, bytes, task;
    int mpi_nwrite, tmp_nwrite, buffer_offset;
    enum siofields blocknr;
    unsigned int blksize, blksize_tot, buffer_size, tmp_blksize;

    int *Sub_nwrite;
    int WriteTaskNgroups, WriteTaskNsubgroups, WriteTaskNids;
#ifdef SUBFIND_COUNT_BIG_HALOS
    int WriteTaskNbiggroups;
#endif // SUBFIND_COUNT_BIG_HALOS
    Sub_nwrite = (int *)mymalloc("Sub_nwrite", SubNTask * sizeof(int));

    buffer_size = All.BufferSize * 1024 * 1024;

    if (!(IOBuffer = (void *)mymalloc("IOBuffer", bytes = buffer_size)))
    {
        ENDRUN("failed to allocate memory for `IOBuffer' (%g MB).\n", bytes / (1024.0 * 1024.0));
    };

#if defined(SUBFIND_COUNT_BIG_HALOS) || defined(LT_ADD_GAL_TO_SUB)
        // int k = 0;
#endif // defined(SUBFIND_COUNT_BIG_HALOS) || defined(LT_ADD_GAL_TO_SUB)

#ifdef WRITE_SUB_IN_SNAP_FORMAT
    unsigned int nextblock;
#ifdef HAVE_HDF5
    hid_t hdf5_file = 0, hdf5_grp[3], hdf5_headergrp = 0, hdf5_dataspace_memory;
    hid_t hdf5_datatype = 0, hdf5_dataspace_in_file = 0, hdf5_dataset = 0;
    herr_t hdf5_status;
    hsize_t dims[2], count[2], start[2];
    int rank = 0;
#endif // HAVE_HDF5
#endif // WRITE_SUB_IN_SNAP_FORMAT

    for (int ngr = 0; ngr < nwriting; ngr++)
    {
        if (writing_group == ngr)
        {
#ifdef LONG_OUTPUT_NUMBERS
#ifdef WRITE_SUB_IN_SNAP_FORMAT
#ifdef WRITE_SUB_COLLECTIVELY
            if (All.NumFilesPerSnapshot == 1)
#else // WRITE_SUB_COLLECTIVELY
            if (NTask == 1)
#endif // WRITE_SUB_COLLECTIVELY
            {
                sprintf(fname, "%s/groups_%04d/%s_%04d", All.OutputDir, num, "sub", num);
            }
            else
            {
                sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "sub", num, color);
            }
#else // WRITE_SUB_IN_SNAP_FORMAT
            sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "subhalo_tab", num, color);
#endif // WRITE_SUB_IN_SNAP_FORMAT
#else // LONG_OUTPUT_NUMBERS
#ifdef WRITE_SUB_IN_SNAP_FORMAT
#ifdef WRITE_SUB_COLLECTIVELY
            if (All.NumFilesPerSnapshot == 1)
#else // WRITE_SUB_COLLECTIVELY
            if (NTask == 1)
#endif // WRITE_SUB_COLLECTIVELY
            {
                sprintf(fname, "%s/groups_%03d/%s_%03d", All.OutputDir, num, "sub", num);
            }
            else
            {
                sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "sub", num, color);
            }
#else // WRITE_SUB_IN_SNAP_FORMAT
            sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_tab", num, color);
#endif // WRITE_SUB_IN_SNAP_FORMAT
#endif // LONG_OUTPUT_NUMBERS
            strcpy(buf, fname);

            if (SubThisTask == 0)
            {
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                if (All.SnapFormat == 3)
                {
                    sprintf(buf, "%s.hdf5", fname);
                    hdf5_file = H5Fcreate(buf, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                    hdf5_headergrp = H5Gcreate2(hdf5_file, "/Header", 0, H5P_DEFAULT, H5P_DEFAULT);

                    hdf5_grp[0] = H5Gcreate2(hdf5_file, "/Group", 0, H5P_DEFAULT, H5P_DEFAULT);
                    hdf5_grp[1] = H5Gcreate2(hdf5_file, "/Subhalo", 0, H5P_DEFAULT, H5P_DEFAULT);
                    hdf5_grp[2] = H5Gcreate2(hdf5_file, "/IDs", 0, H5P_DEFAULT, H5P_DEFAULT);
                }
                else
                {
#endif // defined(HAVE_HDF5)&& defined(WRITE_SUB_IN_SNAP_FORMAT)
                    if (!(fd = fopen(buf, "w")))
                    {
                        ENDRUN("can't open file `%s`\n", buf);
                    }

#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                }
#endif // defined(HAVE_HDF5)&& defined(WRITE_SUB_IN_SNAP_FORMAT)
            }

            /* total numbers in the sub communicator */
            MPI_Allreduce(&Ngroups, &WriteTaskNgroups, 1, MPI_INT, MPI_SUM, SUB_COMM);
            MPI_Allreduce(&Nsubgroups, &WriteTaskNsubgroups, 1, MPI_INT, MPI_SUM, SUB_COMM);
            MPI_Allreduce(&Nids, &WriteTaskNids, 1, MPI_INT, MPI_SUM, SUB_COMM);
#ifdef SUBFIND_COUNT_BIG_HALOS
            MPI_Allreduce(&Nbiggroups, &WriteTaskNbiggroups, 1, MPI_INT, MPI_SUM, SUB_COMM);
#endif // SUBFIND_COUNT_BIG_HALOS

#ifdef WRITE_SUB_IN_SNAP_FORMAT
            for (int i = 0; i < 6; i++)
            {
                header.npart[i] = 0;
                header.npartTotal[i] = 0;
                header.npartTotalHighWord[i] = 0;
                header.mass[i] = 0;
            }
            header.npart[0] = WriteTaskNgroups;
            header.npartTotal[0] = TotNgroups;
            header.npart[1] = WriteTaskNsubgroups;
            header.npartTotal[1] = TotNsubgroups;
            header.npart[2] = WriteTaskNids;
            header.npartTotal[2] = (unsigned int)TotNids;
            header.npartTotalHighWord[2] = (unsigned int)(TotNids >> 32);
#ifdef SUBFIND_COUNT_BIG_HALOS
            header.npart[3] = WriteTaskNbiggroups;
            header.npartTotal[3] = TotNbiggroups;
#endif // SUBFIND_COUNT_BIG_HALOS

            header.time = All.Time;
            if (All.ComovingIntegrationOn)
            {
                header.redshift = 1.0 / All.Time - 1;
            }
            else
            {
                header.redshift = 0;
            }

#ifdef COOLING
            header.flag_cooling = 1;
#endif // COOLING

#ifdef SFR
            header.flag_sfr = 1;
            header.flag_feedback = 1;
            header.flag_stellarage = 1;
#if !defined(LT_STELLAREVOLUTION)
            header.flag_metals = 1;
#endif // !LT_STELLAREVOLUTION
#ifdef LT_STELLAREVOLUTION
            // header.flag_stellarevolution = 2;
#endif // LT_STELLAREVOLUTION
#endif // SFR

            header.num_files = ncolors;
            header.BoxSize = All.BoxSize;
            header.Omega0 = All.Omega0;
            header.OmegaLambda = All.OmegaLambda;
            header.HubbleParam = All.HubbleParam;

#ifdef LT_ADD_GAL_TO_SUB
            for (int k = 0; k < LT_ADD_GAL_TO_SUB; k++)
            {
                header.names[k][0] = bc_name[k][0];
                header.names[k][1] = bc_name[k][1];
            }
            for (int k = LT_ADD_GAL_TO_SUB; k < 15; k++)
            {
                header.names[k][0] = ' ';
                header.names[k][1] = ' ';
            }
#endif // LT_ADD_GAL_TO_SUB

#ifdef WRITE_SUB_IN_SNAP_FORMAT
            if (All.SnapFormat == 2 && SubThisTask == 0)
            {
                blksize = sizeof(int) + 4 * sizeof(char);
                my_fwrite(&blksize, sizeof(int), 1, fd);
                my_fwrite((void *)"HEAD", sizeof(char), 4, fd);
                nextblock = sizeof(header) + 2 * sizeof(int);
                my_fwrite(&nextblock, sizeof(int), 1, fd);
                my_fwrite(&blksize, sizeof(int), 1, fd);
            }
#ifdef HAVE_HDF5
            if (All.SnapFormat == 3 && SubThisTask == 0)
            {
                write_header_attributes_in_hdf5(hdf5_headergrp);
            }
            else
            {
#endif // HAVE_HDF5
#endif // WRITE_SUB_IN_SNAP_FORMAT
                if (SubThisTask == 0)
                {
                    blksize = sizeof(header);
                    my_fwrite(&blksize, sizeof(int), 1, fd);
                    my_fwrite(&header, sizeof(header), 1, fd);
                    my_fwrite(&blksize, sizeof(int), 1, fd);
                }
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
            }
#endif //  defined(HAVE_HDF5) &&  defined(WRITE_SUB_IN_SNAP_FORMAT)

#ifdef WRITE_SUB_IN_SNAP_FORMAT
            if (All.SnapFormat == 2)
            {
                if (!(InfoBlock = (struct info_block *)mymalloc("InfoBlock", bytes = sizeof(struct info_block) * 1000)))
                {
                    ENDRUN("failed to allocate memory for `InfoBlock' (%g MB).\n", bytes / (1024.0 * 1024.0));
                }

                int n_info = 0;
                for (int bnr = 0; bnr < 1000; bnr++)
                {
                    blocknr = (enum siofields)bnr;

                    if (blocknr == SIO_LASTENTRY)
                    {
                        break;
                    }

                    if (block_in_sub(blocknr))
                    {
                        type = get_sub_entrytype_of_block(blocknr);

                        for (int i = 0; i < 6; i++)
                        {
                            InfoBlock[n_info].is_present[i] = 0;
                        }
                        InfoBlock[n_info].is_present[type] = 1;

                        InfoBlock[n_info].ndim = get_values_per_sub(blocknr);

                        get_IO_Label_sub(blocknr, label);
                        for (int i = 0; i < 4; i++)
                        {
                            InfoBlock[n_info].label[i] = label[i];
                        }

                        datatype = get_datatype_in_sub(blocknr);
                        switch (datatype)
                        {
                        case 0:
                            if (InfoBlock[n_info].ndim <= 1)
                                strncpy(InfoBlock[n_info].type, "LONG    ", 8);
                            else
                                strncpy(InfoBlock[n_info].type, "LONGN   ", 8);
                            break;
                        case 1:
                            if (InfoBlock[n_info].ndim <= 1)
                                strncpy(InfoBlock[n_info].type, "FLOAT   ", 8);
                            else
                                strncpy(InfoBlock[n_info].type, "FLOATN  ", 8);
                            break;
                        case 2:
                            if (InfoBlock[n_info].ndim <= 1)
                                strncpy(InfoBlock[n_info].type, "LLONG   ", 8);
                            else
                                strncpy(InfoBlock[n_info].type, "LLONGN  ", 8);
                            break;
                        case 3:
                            if (InfoBlock[n_info].ndim <= 1)
                                strncpy(InfoBlock[n_info].type, "DOUBLE  ", 8);
                            else
                                strncpy(InfoBlock[n_info].type, "DOUBLEN ", 8);
                            break;
                        default:
                            ENDRUN("Error in datatype.");
                            break;
                        }
                        n_info++;
                    }
                }

                if (SubThisTask == 0)
                {
                    blksize = sizeof(int) + 4 * sizeof(char);
                    my_fwrite(&blksize, sizeof(int), 1, fd);
                    my_fwrite((void *)"INFO", sizeof(char), 4, fd);
                    nextblock = n_info * sizeof(struct info_block) + 2 * sizeof(int);
                    my_fwrite(&nextblock, sizeof(int), 1, fd);
                    my_fwrite(&blksize, sizeof(int), 1, fd);
                    blksize = n_info * sizeof(struct info_block);
                    my_fwrite(&blksize, sizeof(int), 1, fd);
                    my_fwrite(InfoBlock, sizeof(struct info_block), n_info, fd);
                    my_fwrite(&blksize, sizeof(int), 1, fd);
                }
                myfree(InfoBlock);
            }
#endif // WRITE_SUB_IN_SNAP_FORMAT

#else // WRITE_SUB_IN_SNAP_FORMAT
            if (SubThisTask == 0)
            {
                my_fwrite(&WriteTaskNgroups, sizeof(int), 1, fd);
                my_fwrite(&TotNgroups, sizeof(int), 1, fd);
                my_fwrite(&WriteTaskNids, sizeof(int), 1, fd);
                my_fwrite(&TotNids, sizeof(long long), 1, fd);
                my_fwrite(&ncolors, sizeof(int), 1, fd);
                my_fwrite(&WriteTaskNsubgroups, sizeof(int), 1, fd);
                my_fwrite(&TotNsubgroups, sizeof(int), 1, fd);
            }
#endif // WRITE_SUB_IN_SNAP_FORMAT

            for (int bnr = 0; bnr < 1000; bnr++)
            {
                blocknr = (enum siofields)bnr;

                if (blocknr == SIO_LASTENTRY)
                {
                    break;
                }

                if (block_in_sub(blocknr))
                {
                    type = get_sub_entrytype_of_block(blocknr);
                    switch (type)
                    {
                    case 0:
                        nwrite = Ngroups;
                        nwrite_tot = WriteTaskNgroups;
                        break;
                    case 1:
                        nwrite = Nsubgroups;
                        nwrite_tot = WriteTaskNsubgroups;
                        break;
                    case 2:
                        nwrite = Nids;
                        nwrite_tot = WriteTaskNids;
                        break;
#ifdef SUBFIND_COUNT_BIG_HALOS
                    case 3:
                        nwrite = Nbiggroups;
                        nwrite_tot = WriteTaskNbiggroups;
                        break;
#endif // SUBFIND_COUNT_BIG_HALOS
                    }

                    ndim = get_values_per_sub(blocknr);

#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                    if (All.SnapFormat == 3 && SubThisTask == 0)
                    {
                        dims[0] = nwrite_tot;
                        dims[1] = ndim;

                        if (dims[1] == 1)
                        {
                            rank = 1;
                        }
                        else
                        {
                            rank = 2;
                        }

                        switch (get_datatype_in_sub(blocknr))
                        {
                        case 0:
                            hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT);
                            break;
                        case 1:
                            hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
                            break;
                        case 2:
                            hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT64);
                            break;
                        case 3:
                            hdf5_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
                            break;
                        default:
                            ENDRUN("Error in datatype of blocknr.");
                            break;
                        }
                        get_IO_Label_HDF5_sub(blocknr, buf);
                        hdf5_dataspace_in_file = H5Screate_simple(rank, dims, NULL);
                        hdf5_dataset = H5Dcreate(hdf5_grp[type], buf, hdf5_datatype, hdf5_dataspace_in_file,
                                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    }
#endif // defined(HAVE_HDF5) &&  defined(WRITE_SUB_IN_SNAP_FORMAT)

                    datatype = get_datatype_in_sub(blocknr);
                    switch (datatype)
                    {
                    case 0:
                        bytes_per_blockelement = 4;
                        break;
                    case 1:
                        bytes_per_blockelement = 4;
                        break;
                    case 2:
                        bytes_per_blockelement = 8;
                        break;
                    case 3:
                        bytes_per_blockelement = 8;
                        break;
                    default:
                        ENDRUN("Error in datatype.");
                        break;
                    }

#ifndef WRITE_SUB_IN_SNAP_FORMAT /* open new file in case of old format */
                    if (blocknr == SIO_PPOS)
                    {
                        if (SubThisTask == 0)
                        {
                            fclose(fd);
                        }
#ifdef LONG_OUTPUT_NUMBERS
                        sprintf(buf, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "subhalo_posvel", num, color);
#else // LONG_OUTPUT_NUMBERS
                        sprintf(buf, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_posvel", num, color);
#endif // LONG_OUTPUT_NUMBERS

                        if (SubThisTask == 0)
                        {
                            if (!(fd = fopen(buf, "w")))
                            {
                                ENDRUN("can't open file `%s`\n", buf);
                            }

                            my_fwrite(&WriteTaskNgroups, sizeof(int), 1, fd);
                            my_fwrite(&TotNgroups, sizeof(int), 1, fd);
                            my_fwrite(&WriteTaskNids, sizeof(int), 1, fd);
                            my_fwrite(&TotNids, sizeof(long long), 1, fd);
                            my_fwrite(&ncolors, sizeof(int), 1, fd);
                            my_fwrite(&Send_offset[color * SubNTask], sizeof(int), 1, fd);
                            my_fwrite(&All.Time, sizeof(double), 1, fd);
                        }
                    }

                    if (blocknr == SIO_PID)
                    {
                        if (SubThisTask == 0)
                            fclose(fd);
#ifdef LONG_OUTPUT_NUMBERS
                        sprintf(buf, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "subhalo_ids", num, color);
#else // LONG_OUTPUT_NUMBERS
                        sprintf(buf, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "subhalo_ids", num, color);
#endif // LONG_OUTPUT_NUMBERS
                        if (SubThisTask == 0)
                        {
                            if (!(fd = fopen(buf, "w")))
                            {
                                ENDRUN("can't open file `%s`\n", buf);
                            }

                            my_fwrite(&WriteTaskNgroups, sizeof(int), 1, fd);
                            my_fwrite(&TotNgroups, sizeof(int), 1, fd);
                            my_fwrite(&WriteTaskNids, sizeof(int), 1, fd);
                            my_fwrite(&TotNids, sizeof(long long), 1, fd);
                            my_fwrite(&ncolors, sizeof(int), 1, fd);
                            my_fwrite(&Send_offset[color * SubNTask], sizeof(int), 1, fd);
                        }
                    }
#endif // WRITE_SUB_IN_SNAP_FORMAT

#ifdef WRITE_SUB_IN_SNAP_FORMAT
                    get_IO_Label_sub(blocknr, label);

                    if (All.SnapFormat == 2 && SubThisTask == 0)
                    {
                        blksize = sizeof(int) + 4 * sizeof(char);
                        my_fwrite(&blksize, sizeof(int), 1, fd);
                        my_fwrite((void *)label, sizeof(char), 4, fd);
                        nextblock = nwrite_tot * bytes_per_blockelement * ndim + 2 * sizeof(int);
                        my_fwrite(&nextblock, sizeof(int), 1, fd);
                        my_fwrite(&blksize, sizeof(int), 1, fd);
                    }
                    blksize_tot = nwrite_tot * bytes_per_blockelement * ndim;
                    blksize = nwrite * bytes_per_blockelement * ndim;

                    if (All.SnapFormat != 3 && SubThisTask == 0)
                    {
                        my_fwrite(&blksize_tot, sizeof(int), 1, fd);
                    }
#else // WRITE_SUB_IN_SNAP_FORMAT
                    blksize = nwrite * bytes_per_blockelement * ndim;
                    blksize_tot = nwrite_tot * bytes_per_blockelement * ndim;
#endif // WRITE_SUB_IN_SNAP_FORMAT

                    nwrite_in_buffer = buffer_size / (bytes_per_blockelement * ndim);

                    MPI_Allgather(&nwrite, 1, MPI_INT, Sub_nwrite, 1, MPI_INT, SUB_COMM);

                    VERBOSE(
                        2,
                        "Color %d writing block %d (%c%c%c%c), n=%d, ptype=%d, dtype=%d, ndim=%d, bpb=%d, bytes=%u\n",
                        color, bnr, label[0], label[1], label[2], label[3], nwrite_tot, type, datatype, ndim,
                        bytes_per_blockelement, blksize_tot);

#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                    int mpi_nwrite_done = 0;
#endif // #if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)

                    for (task = 0, buffer_offset = 0; task < SubNTask; task++)
                    {
                        tmp_nwrite = Sub_nwrite[task];

                        while (tmp_nwrite > 0)
                        {
                            mpi_nwrite = tmp_nwrite;

                            if (mpi_nwrite > nwrite_in_buffer)
                            {
                                mpi_nwrite = nwrite_in_buffer;
                            }

                            if (SubThisTask == task)
                            {
                                fill_subfind_io_buffer(IOBuffer, bnr, &buffer_offset, mpi_nwrite);
                            }

                            tmp_blksize = mpi_nwrite * bytes_per_blockelement * ndim;

                            if (SubThisTask == 0 && task != 0)
                            {
                                MPI_Recv(IOBuffer, tmp_blksize, MPI_BYTE, task, TAG_SUBFIND_IO, SUB_COMM,
                                         MPI_STATUS_IGNORE);
                            }

                            if (SubThisTask != 0 && task == SubThisTask)
                            {
                                MPI_Ssend(IOBuffer, tmp_blksize, MPI_BYTE, 0, TAG_SUBFIND_IO, SUB_COMM);
                            }

                            if (SubThisTask == 0)
                            {
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                                if (All.SnapFormat != 3)
                                {
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                                    my_fwrite(IOBuffer, sizeof(char), tmp_blksize, fd);

#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                                }
                                else if (All.SnapFormat == 3)
                                {
                                    start[0] = mpi_nwrite_done;
                                    start[1] = 0;

                                    count[0] = mpi_nwrite;
                                    count[1] = ndim;

                                    H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count,
                                                        NULL);

                                    dims[0] = mpi_nwrite;
                                    dims[1] = ndim;
                                    hdf5_dataspace_memory = H5Screate_simple(rank, dims, NULL);

                                    hdf5_status = H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory,
                                                           hdf5_dataspace_in_file, H5P_DEFAULT, IOBuffer);

                                    H5Sclose(hdf5_dataspace_memory);
                                }
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                            }
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                            mpi_nwrite_done += mpi_nwrite;
#endif // #if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                            tmp_nwrite -= nwrite_in_buffer;
                        }
                    }

#ifdef WRITE_SUB_IN_SNAP_FORMAT
                    if (SubThisTask == 0)
                    {
                        if (All.SnapFormat != 3)
                        {
                            my_fwrite(&blksize_tot, sizeof(int), 1, fd);
                        }
                    }
#endif // WRITE_SUB_IN_SNAP_FORMAT
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                    if (All.SnapFormat == 3 && SubThisTask == 0)
                    {
                        H5Dclose(hdf5_dataset);
                        H5Sclose(hdf5_dataspace_in_file);
                        H5Tclose(hdf5_datatype);
                    }
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                } /* closing if block exist */
            } /* closing for bnr loop */

            if (SubThisTask == 0)
            {
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                if (All.SnapFormat == 3 && SubThisTask == 0)
                {
                    for (int type = 0; type < 3; type++)
                    {
                        H5Gclose(hdf5_grp[type]);
                    }
                    H5Gclose(hdf5_headergrp);
                    H5Fclose(hdf5_file);
                }
                else
                {
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                    fclose(fd);
#if defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
                }
#endif // defined(HAVE_HDF5) && defined(WRITE_SUB_IN_SNAP_FORMAT)
            }

        } /* closing if-this-writing-group's-turn loop */

        MPI_Barrier(MYMPI_COMM_WORLD);
    } /* closing writing group loop */

    myfree(IOBuffer);
    myfree(Sub_nwrite);

    MPI_Barrier(MYMPI_COMM_WORLD);
    MPI_Comm_free(&SUB_COMM);
}

/**
 * @brief Fills the I/O buffer with group and subgroup data for output.
 *
 * This function populates the provided I/O buffer with data from the Group and SubGroup structures.
 * The specific type of data written to the buffer is determined by the `bnr` parameter, which specifies
 * the data block. The function writes `nwrite` entries starting from the provided `offset`.
 *
 * @param IOBuffer Pointer to the buffer where the data will be written.
 * @param bnr Integer representing the data block type.
 * @param offset Pointer to the starting offset within the Group/SubGroup arrays.
 * @param nwrite Number of entries to write to the buffer.
 */
void fill_subfind_io_buffer(void *IOBuffer, int bnr, int *offset, int nwrite)
{
#if defined(SUBFIND_COUNT_BIG_HALOS) || defined(LT_ADD_GAL_TO_SUB)
    int k;
#endif // defined(SUBFIND_COUNT_BIG_HALOS) || defined(LT_ADD_GAL_TO_SUB)

    enum siofields blocknr;

    MyOutputFloat *fp;
    int *fp_int;
    MyIDType *fp_id;

#ifdef SUBFIND_SAVE_PARTICLELISTS
    double sqra3inv;

    if (All.ComovingIntegrationOn)
    {
        sqra3inv = 1 / (All.Time * All.Time * All.Time);
        sqra3inv = sqrt(sqra3inv);
    }
    else
    {
        sqra3inv = 1;
    }
#endif // SUBFIND_SAVE_PARTICLELISTS

    fp = (MyOutputFloat *)IOBuffer;
    fp_int = (int *)IOBuffer;
    fp_id = (MyIDType *)IOBuffer;

    int gr_index = *offset;

    blocknr = (enum siofields)bnr;

    switch (blocknr)
    {
    case SIO_GLEN:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = Group[gr_index++].Len;
        }
        break;
    case SIO_GOFF:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = Group[gr_index++].Offset;
        }
        break;
    case SIO_MTOT:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].Mass;
        }
        break;
    case SIO_GPOS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                fp[i * 3 + j] = (MyOutputFloat)Group[gr_index].Pos[j];
#ifdef DS_SHIFT_BOX
                fp[i * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
            }
            gr_index++;
        }
        break;
    case SIO_MMEA:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].M_Mean200;
        }
        break;
    case SIO_RMEA:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].R_Mean200;
        }
        break;
    case SIO_MCRI:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].M_Crit200;
        }
        break;
    case SIO_RCRI:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].R_Crit200;
        }
        break;
    case SIO_MTOP:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].M_TopHat200;
        }
        break;
    case SIO_RTOP:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].R_TopHat200;
        }
        break;
#ifdef SO_VEL_DISPERSIONS
    case SIO_DMEA:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].VelDisp_Mean200;
        }
        break;
    case SIO_DCRI:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].VelDisp_Crit200;
        }
        break;
    case SIO_DTOP:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].VelDisp_TopHat200;
        }
        break;
#endif // SO_VEL_DISPERSIONS
#ifdef SO_BAR_INFO
    case SIO_M500:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].M_Mean500;
        }
        break;
    case SIO_R500:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].R_Mean500;
        }
        break;
    case SIO_M5CC:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].M_Crit500;
        }
        break;
    case SIO_R5CC:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].R_Crit500;
        }
        break;
    case SIO_M25K:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].M_Crit2500;
        }
        break;
    case SIO_R25K:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].R_Crit2500;
        }
        break;
#ifdef SO_VEL_DISPERSIONS
    case SIO_D500:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].VelDisp_Mean500;
        }
        break;
    case SIO_D5CC:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].VelDisp_Crit500;
        }
        break;
    case SIO_D25K:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].VelDisp_Crit2500;
        }
        break;
#endif // SO_VEL_DISPERSIONS
    case SIO_MGAS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                fp[i * 6 + j] = (MyOutputFloat)Group[gr_index].gas_mass[j];
            }
            gr_index++;
        }
        break;
    case SIO_MSTR:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                fp[i * 6 + j] = (MyOutputFloat)Group[gr_index].star_mass[j];
            }
            gr_index++;
        }
        break;
    case SIO_TGAS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                fp[i * 6 + j] = (MyOutputFloat)Group[gr_index].temp[j];
            }
            gr_index++;
        }
        break;
    case SIO_LGAS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                fp[i * 6 + j] = (MyOutputFloat)Group[gr_index].xlum[j];
            }
            gr_index++;
        }
        break;
    case SIO_YGAS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                fp[i * 6 + j] = (MyOutputFloat)Group[gr_index].ygas[j];
            }
            gr_index++;
        }
        break;
#endif // SO_BAR_INFO
    case SIO_NCON:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = Group[gr_index++].ContaminationLen;
        }
        break;
    case SIO_MCON:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)Group[gr_index++].ContaminationMass;
        }
        break;
#ifdef SUBFIND_COUNT_BIG_HALOS
    case SIO_BGPOS:
        k = 0;
        for (int i = 0; i < Ngroups; i++)
        {
            if (Group[gr_index].M_TopHat200 > SUBFIND_COUNT_BIG_HALOS)
            {
                for (int j = 0; j < 3; j++)
                {
                    fp[k * 3 + j] = (MyOutputFloat)Group[gr_index].Pos[j];
#ifdef DS_SHIFT_BOX
                    fp[k * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
                }
                k++;
            }
            gr_index++;
        }
        ENDRUN_IF(k != nwrite, "Error as k uneaqual to nwrite");
        break;
    case SIO_BGMTOP:
        k = 0;
        for (int i = 0; i < Ngroups; i++)
        {
            if (Group[gr_index].M_TopHat200 > SUBFIND_COUNT_BIG_HALOS)
            {
                fp[k] = (MyOutputFloat)Group[gr_index].M_TopHat200;
                k++;
            }
            gr_index++;
        }
        ENDRUN_IF(k != nwrite, "Error as k uneaqual to nwrite");
        break;
    case SIO_BGRTOP:
        k = 0;
        for (int i = 0; i < Ngroups; i++)
        {
            if (Group[gr_index].M_TopHat200 > SUBFIND_COUNT_BIG_HALOS)
            {
                fp[k] = (MyOutputFloat)Group[gr_index].R_TopHat200;
                k++;
            }
            gr_index++;
        }
        ENDRUN_IF(k != nwrite, "Error as k uneaqual to nwrite");
        break;
#endif // SUBFIND_COUNT_BIG_HALOS
    case SIO_NSUB:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = Group[gr_index++].Nsubs;
        }
        break;
    case SIO_FSUB:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = Group[gr_index++].FirstSub;
        }
        break;
    case SIO_SLEN:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = SubGroup[gr_index++].Len;
        }
        break;
    case SIO_SOFF:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = SubGroup[gr_index++].Offset;
        }
        break;
    case SIO_PFOF:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = SubGroup[gr_index++].SubParent;
        }
        break;
    case SIO_MSUB:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].Mass;
        }
        break;
    case SIO_SPOS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                fp[i * 3 + j] = (MyOutputFloat)SubGroup[gr_index].Pos[j];
#ifdef DS_SHIFT_BOX
                fp[i * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
            }
            gr_index++;
        }
        break;
    case SIO_SVEL:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                fp[i * 3 + j] = (MyOutputFloat)SubGroup[gr_index].Vel[j];
            }
            gr_index++;
        }
        break;
    case SIO_SCM:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                fp[i * 3 + j] = (MyOutputFloat)SubGroup[gr_index].CM[j];
            }
            gr_index++;
        }
        break;
    case SIO_SPIN:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                fp[i * 3 + j] = (MyOutputFloat)SubGroup[gr_index].Spin[j];
            }
            gr_index++;
        }
        break;
    case SIO_DSUB:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubVelDisp;
        }
        break;
    case SIO_VMAX:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubVmax;
        }
        break;
    case SIO_RVMAX:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubVmaxRad;
        }
        break;
    case SIO_RHMS:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubHalfMass;
        }
        break;
    case SIO_SHMR:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubStelHalfMass;
        }
        break;
    case SIO_MBID:
        for (int i = 0; i < nwrite; i++)
        {
            fp_id[i] = SubGroup[gr_index++].SubMostBoundID;
        }
        break;
    case SIO_GRNR:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = SubGroup[gr_index++].GrNr;
        }
        break;
    case SIO_SMST:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                fp[i * 6 + j] = (MyOutputFloat)SubGroup[gr_index].MassTab[j];
            }
            gr_index++;
        }
        break;
    case SIO_SSANG:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubStelAngMom;
        }
        break;
    case SIO_SBVAL:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubBval;
        }
        break;
#ifdef LT_ADD_GAL_TO_SUB
    case SIO_SLUM:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
            {
                fp[i * LT_ADD_GAL_TO_SUB + j] = (MyOutputFloat)SubGroup[gr_index].SubLum[j];
            }
            gr_index++;
        }
        break;
#ifdef DUSTATT
    case SIO_SLATT:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
            {
                fp[i * LT_ADD_GAL_TO_SUB + j] = (MyOutputFloat)SubGroup[gr_index].SubLumAtt[j];
            }
            gr_index++;
        }
        break;
    case SIO_DUST:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < DUSTATT; j++)
            {
                fp[i * DUSTATT + j] = (MyOutputFloat)SubGroup[gr_index].DustProfile[j];
            }
            gr_index++;
        }
        break;
#endif // DUSTATT
#ifdef OBSERVER_FRAME
    case SIO_SLOBS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < LT_ADD_GAL_TO_SUB; j++)
            {
                fp[i * LT_ADD_GAL_TO_SUB + j] = (MyOutputFloat)SubGroup[gr_index].SubLumObs[j];
            }
            gr_index++;
        }
        break;
#endif // OBSERVER_FRAME
    case SIO_SAGE:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubMeanAge;
        }
        break;
    case SIO_SZ:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubMeanZStar;
        }
        break;
    case SIO_SSFR:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubSFR;
        }
        break;
    case SIO_MHI:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[gr_index++].SubHIMass;
        }
        break;
    case SIO_SZGAS:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[i].SubMeanZGas;
        }
        break;
    case SIO_SZCGAS:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)SubGroup[i].SubMeanZColdGas;
        }
        break;
#endif // LT_ADD_GAL_TO_SUB
#ifdef LMB_CR_OUTPUT_SYNCHROTRON
    case SIO_SSYNCHROTRON:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)log10(SubGroup[gr_index++].SubSynchrotron);
        }
        break;
#endif // LMB_CR_OUTPUT_SYNCHROTRON
#ifdef SUBFIND_SAVE_PARTICLELISTS
    case SIO_PPOS:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                fp[i * 3 + j] = (MyOutputFloat)ID_list[gr_index].Pos[j];
#ifdef DS_SHIFT_BOX
                fp[i * 3 + j] -= All.BoxSize / 2.;
#endif // DS_SHIFT_BOX
            }
            gr_index++;
        }
        break;
    case SIO_PVEL:
        for (int i = 0; i < nwrite; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                fp[i * 3 + j] = (MyOutputFloat)(ID_list[gr_index].Vel[j] * sqra3inv);
            }
            gr_index++;
        }
        break;
    case SIO_PTYP:
        for (int i = 0; i < nwrite; i++)
        {
            fp_int[i] = ID_list[gr_index++].Type;
        }
        break;
    case SIO_PMAS:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)ID_list[gr_index++].Mass;
        }
        break;
#ifdef SFR
    case SIO_PAGE:
        for (int i = 0; i < nwrite; i++)
        {
            fp[i] = (MyOutputFloat)ID_list[gr_index++].StellarAge;
        }
        break;
#endif // SFR
#endif // SUBFIND_SAVE_PARTICLELISTS
    case SIO_PID:
        for (int i = 0; i < nwrite; i++)
            fp_id[i] = ID_list[gr_index++].ID;
        break;
    default:
        break;
    } /* closing switch(blocknr) */
    *offset = gr_index;
}

#ifdef KD_ALTERNATIVE_GROUP_SORT
/**
 * @brief Determines the task responsible for a given group.
 *
 * This function calculates which MPI task is responsible for processing a given group
 * based on the group's index and the total number of groups. It uses a weighted distribution
 * approach to assign groups to tasks more evenly, taking into account the total number of groups
 * and tasks.
 *
 * @param i The index of the group.
 * @param totngroups The total number of groups.
 * @return The task number responsible for the group.
 */
int GroupToTask(int i, int totngroups)
{
    int ret;
    int nGroupsPerTask = totngroups / NTask + 1;

    if (totngroups > 0)
    {
        double x1 = (double)i;
        double x2 = -1;
        if (totngroups > 1)
        {
            x2 = sqrt(x1 / (totngroups - 1)) * (totngroups - 1);
        }
        else
        {
            x2 = 0;
        }

        double w1 = 5 + ((double)NTask) / totngroups;
        double w2 = ((double)totngroups) / NTask;
        double w = w1 + w2;
        w1 /= w;
        w2 /= w;
        ret = (w1 * x1 + w2 * x2) / nGroupsPerTask;
    }
    else
    {
        ret = 0;
    }

    if (ret >= NTask)
    {
        ret = NTask - 1;
    }

    return ret;
}

/**
 * @brief Sorts and redistributes group properties among tasks.
 *
 * This function sorts group properties by their group number (GrNr) and redistributes them
 * among MPI tasks based on the group number. It first counts how many groups need to be sent
 * to each task, then redistributes the groups accordingly. The function ensures that the groups
 * are evenly distributed across tasks for further processing.
 */
void sort_groups()
{
    int nexport = 0, nimport = 0, target, recvTask;
    group_properties *send_Group;

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Ngroups; i++)
    {
        target = GroupToTask(Group[i].GrNr, TotNgroups);
        if (target != ThisTask)
        {
            Send_count[target]++;
        }
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    Recv_offset[0] = Send_offset[0] = 0;
    for (int i = 0; i < NTask; i++)
    {
        nimport += Recv_count[i];
        nexport += Send_count[i];

        if (i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    send_Group = (group_properties *)mymalloc("send_Group", nexport * sizeof(group_properties));

    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Ngroups; i++)
    {
        target = GroupToTask(Group[i].GrNr, TotNgroups);
        if (target != ThisTask)
        {
            send_Group[Send_offset[target] + Send_count[target]] = Group[i];
            Send_count[target]++;

            Group[i] = Group[Ngroups - 1];
            Ngroups--;
            i--;
        }
    }

    ENDRUN_IF(Ngroups + nimport >= MaxNgroups, "Task %d: NgroupThis = %d, NgroupMax = %d, Ntotgroups=%d \n", ThisTask,
              Ngroups + nimport, MaxNgroups, TotNgroups);

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                // create individual tag for each pair of tasks
                int Tag = TAG_ALTSORT_GRP + std::min(ThisTask, recvTask) * NTask + std::max(ThisTask, recvTask);
                /* get the group info */
                MPI_Sendrecv(&send_Group[Send_offset[recvTask]], Send_count[recvTask] * sizeof(group_properties),
                             MPI_BYTE, recvTask, Tag, &Group[Ngroups + Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(group_properties), MPI_BYTE, recvTask, Tag, MYMPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
            }
        }
    }

    Ngroups += nimport;

    myfree(send_Group);
}

/**
 * @brief Sorts and redistributes subgroup properties among tasks.
 *
 * This function sorts subgroup properties by their group number (GrNr) and redistributes them
 * among MPI tasks based on the group number. It first counts how many subgroups need to be sent
 * to each task, then redistributes the subgroups accordingly. The function ensures that the subgroups
 * are evenly distributed across tasks for further processing.
 */
void sort_sub_groups()
{
    int nexport = 0, nimport = 0, target, recvTask;
    struct subgroup_properties *send_SubGroup;

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Nsubgroups; i++)
    {
        target = GroupToTask(SubGroup[i].GrNr, TotNgroups);
        if (target != ThisTask)
            Send_count[target]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    Recv_offset[0] = Send_offset[0] = 0;
    for (int i = 0; i < NTask; i++)
    {
        nimport += Recv_count[i];
        nexport += Send_count[i];

        if (i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    send_SubGroup =
        (struct subgroup_properties *)mymalloc("send_SubGroup", nexport * sizeof(struct subgroup_properties));

    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Nsubgroups; i++)
    {
        target = GroupToTask(SubGroup[i].GrNr, TotNgroups);
        if (target != ThisTask)
        {
            send_SubGroup[Send_offset[target] + Send_count[target]] = SubGroup[i];
            Send_count[target]++;

            SubGroup[i] = SubGroup[Nsubgroups - 1];
            Nsubgroups--;
            i--;
        }
    }

    ENDRUN_IF(Nsubgroups + nimport >= MaxNsubgroups, "Task %d: NsubgroupThis = %d, NsubgroupMax = %d\n", ThisTask,
              Nsubgroups + nimport, MaxNsubgroups);

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                // create individual tag for each pair of tasks
                int Tag = TAG_ALTSORT_SUB + std::min(ThisTask, recvTask) * NTask + std::max(ThisTask, recvTask);
                /* get the group info */
                MPI_Sendrecv(&send_SubGroup[Send_offset[recvTask]],
                             Send_count[recvTask] * sizeof(struct subgroup_properties), MPI_BYTE, recvTask, Tag,
                             &SubGroup[Nsubgroups + Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(struct subgroup_properties), MPI_BYTE, recvTask, Tag,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    Nsubgroups += nimport;
    myfree(send_SubGroup);
}

/**
 * @brief Sorts and redistributes subfind ID lists among tasks.
 *
 * This function sorts subfind ID lists by their group number (GrNr) and redistributes them
 * among MPI tasks based on the group number. It first counts how many IDs need to be sent
 * to each task, then redistributes the IDs accordingly. The function ensures that the IDs
 * are evenly distributed across tasks for further processing.
 */
void sort_ids()
{
    int nexport = 0, nimport = 0, target, recvTask;
    struct subfind_id_list *send_IDs;

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Nids; i++)
    {
        target = GroupToTask(ID_list[i].GrNr, TotNgroups);
        if (target != ThisTask)
        {
            Send_count[target]++;
        }
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    Recv_offset[0] = Send_offset[0] = 0;
    for (int i = 0; i < NTask; i++)
    {
        nimport += Recv_count[i];
        nexport += Send_count[i];

        if (i > 0)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
            Recv_offset[i] = Recv_offset[i - 1] + Recv_count[i - 1];
        }
    }

    send_IDs = (struct subfind_id_list *)mymalloc("send_IDs", nexport * sizeof(struct subfind_id_list));

    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

#ifndef NO_KD_ID_LIST_OPTIMIZATION
    int Nids_merk = Nids;
#endif // NO_KD_ID_LIST_OPTIMIZATION

    for (int i = 0; i < Nids; i++)
    {
        target = GroupToTask(ID_list[i].GrNr, TotNgroups);
        if (target != ThisTask)
        {
            send_IDs[Send_offset[target] + Send_count[target]] = ID_list[i];
            Send_count[target]++;

            ID_list[i] = ID_list[Nids - 1];
            Nids--;
            i--;
        }
    }

    if (Nids + nimport >= NidsMax && NidsMax > 0)
    {
#ifndef NO_KD_ID_LIST_OPTIMIZATION
        if (NidsMax == All.MaxPart)
        {
            VERBOSE_ALL(0,
                        "Task %d: ERROR: NidsThis = %d > NidsMax = %d = All.MaxPart!!\n You should set "
                        "SUBFIND_ID_OVERBOOK_FACTOR to > %g)\n",
                        ThisTask, Nids + nimport, NidsMax, ((float)(Nids + nimport)) / (Nids_merk * ntaskspergroup));
        }
        else
        {
            VERBOSE_ALL(0,
                        "Task %d: ERROR: NidsThis = %d > NidsMax = %d !!\n You should increase "
                        "SUBFIND_ID_OVERBOOK_FACTOR (currently %g) accordingly\n",
                        ThisTask, Nids + nimport, NidsMax, (float)SUBFIND_ID_OVERBOOK_FACTOR);
        }
#else // NO_KD_ID_LIST_OPTIMIZATION
        VERBOSE_ALL(0,
                    "Task %d: ERROR: NidsThis = %d > NidsMax = %d !!\n This should not happen !!\n You need to "
                    "re-implement SUBFIND_ID_OVERBOOK_FACTOR\n",
                    ThisTask, Nids + nimport, NidsMax);
#endif // NO_KD_ID_LIST_OPTIMIZATION
        ENDRUN("Error in sort_ids - check previous verbose output.");
    }

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                // create individual tag for each pair of tasks
                int Tag = TAG_ALTSORT_IDS + std::min(ThisTask, recvTask) * NTask + std::max(ThisTask, recvTask);
                /* get the group info */
                MPI_Sendrecv(&send_IDs[Send_offset[recvTask]], Send_count[recvTask] * sizeof(struct subfind_id_list),
                             MPI_BYTE, recvTask, Tag, &ID_list[Nids + Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(struct subfind_id_list), MPI_BYTE, recvTask, Tag,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    Nids += nimport;

    myfree(send_IDs);
}
#endif // KD_ALTERNATIVE_GROUP_SORT

/**
 * @brief Compares the GrNr, SubNr, and BindingEgy fields of two subfind_id_list structures.
 *
 * This function compares two subfind ID lists based on their group number (GrNr) first.
 * If the group numbers are equal, it then compares based on their subgroup number (SubNr).
 * If the subgroup numbers are also equal, it finally compares based on their binding energy (BindingEgy).
 *
 * @param a Pointer to the first subfind_id_list structure.
 * @param b Pointer to the second subfind_id_list structure.
 * @return An integer of -1, or 1 if the GrNr of the first particle
 *         is smaller or larger compared to the second particle.
 *         If GrNr values are equal, an integer of -1, or 1 if the SubNr of the first particle
 *         is smaller or larger compared to the second particle.
 *         If SubNrs are equal then an integer of -1, 0 or 1 f the first index
 *         is smaller, equal or larger than the BindingEgy of the second particle.
 */
int subfind_compare_ID_list(const void *a, const void *b)
{
    if (((struct subfind_id_list *)a)->GrNr < ((struct subfind_id_list *)b)->GrNr)
    {
        return -1;
    }

    if (((struct subfind_id_list *)a)->GrNr > ((struct subfind_id_list *)b)->GrNr)
    {
        return +1;
    }

    if (((struct subfind_id_list *)a)->SubNr < ((struct subfind_id_list *)b)->SubNr)
    {
        return -1;
    }

    if (((struct subfind_id_list *)a)->SubNr > ((struct subfind_id_list *)b)->SubNr)
    {
        return +1;
    }

    if (((struct subfind_id_list *)a)->BindingEgy < ((struct subfind_id_list *)b)->BindingEgy)
    {
        return -1;
    }

    if (((struct subfind_id_list *)a)->BindingEgy > ((struct subfind_id_list *)b)->BindingEgy)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares the GrNr and SubNr fields of two subgroup_properties structures.
 *
 * This function compares two subgroups based on their group number (GrNr) first.
 * If the group numbers are equal, it then compares based on their subgroup number (SubNr).
 *
 * @param a Pointer to the first subgroup_properties structure.
 * @param b Pointer to the second subgroup_properties structure.
 * @return An integer of -1, or 1 if the GrNr of the first particle
 *         is smaller or larger compared to the second particle.
 *         If GrNr values are equal, the comparison is done based on SubNr, then it will return
 *         a -1,0 or 1 if the SubNr of the first index
 *         is smaller, equal or larger than the origindex of the second particle.
 */
int subfind_compare_SubGroup_GrNr_SubNr(const void *a, const void *b)
{
    if (((struct subgroup_properties *)a)->GrNr < ((struct subgroup_properties *)b)->GrNr)
    {
        return -1;
    }

    if (((struct subgroup_properties *)a)->GrNr > ((struct subgroup_properties *)b)->GrNr)
    {
        return +1;
    }

    if (((struct subgroup_properties *)a)->SubNr < ((struct subgroup_properties *)b)->SubNr)
    {
        return -1;
    }

    if (((struct subgroup_properties *)a)->SubNr > ((struct subgroup_properties *)b)->SubNr)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares the GrNr and DM_Density fields of two particle_data structures.
 *
 * This function compares two particles based on their group number (GrNr) first.
 * If the group numbers are equal, it then compares based on their dark matter density (DM_Density).
 * It then returns an integer based on the comparison of the two particles.
 *
 * @param a Pointer to the first particle_data structure.
 * @param b Pointer to the second particle_data structure.
 * @return An integer of -1, or 1 if the GrNr of the first particle
 *         is smaller or larger compared to the second particle.
 *         If GrNr values are equal, the comparison is done based on DM_Density, then it will return
 *         a -1,0 or 1 if the DM_Density of the first index
 *         is smaller, equal or larger than the origindex of the second particle.
 */
int subfind_compare_P_GrNr_DM_Density(const void *a, const void *b)
{
    if (((struct particle_data *)a)->GrNr < (((struct particle_data *)b)->GrNr))
    {
        return -1;
    }

    if (((struct particle_data *)a)->GrNr > (((struct particle_data *)b)->GrNr))
    {
        return +1;
    }

    if (((struct particle_data *)a)->u.DM_Density > (((struct particle_data *)b)->u.DM_Density))
    {
        return -1;
    }

    if (((struct particle_data *)a)->u.DM_Density < (((struct particle_data *)b)->u.DM_Density))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares the origindex fields of two particle_data structures.
 *
 * This function compares the origindex of two particles and returns an integer depending
 * if it the index of the first particle is smaller, equal or larger than the index of the second particle.
 *
 * @param a Pointer to the first particle_data structure.
 * @param b Pointer to the second particle_data structure.
 * @return An integer -1, 0 or 1 if the origindex of the first index
 *         is smaller, equal or larger than the origindex of the second particle.
 */
int subfind_compare_P_origindex(const void *a, const void *b)
{
    if (((struct particle_data *)a)->origindex < (((struct particle_data *)b)->origindex))
    {
        return -1;
    }

    if (((struct particle_data *)a)->origindex > (((struct particle_data *)b)->origindex))
    {
        return +1;
    }

    return 0;
}
#endif // SUBFIND
