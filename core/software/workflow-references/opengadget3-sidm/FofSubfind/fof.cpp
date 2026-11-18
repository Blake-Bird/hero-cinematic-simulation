/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Parallel FoF group finder.
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
#include <inttypes.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/communication.hpp"

#ifdef FOF
#ifdef HAVE_HDF5
#include <hdf5.h>
#endif // HAVE_HDF5

#include "fof.hpp"

#ifdef SUBFIND
#include "subfind.hpp"
#endif // SUBFIND

int NAllocatedGroups; // we use it for debugging purposes

int Ngroups, TotNgroups;
long long TotNids;

group_properties *Group;

/**
 * @brief Structure to store input data for the Friends-of-Friends algorithm.
 */
static struct fofdata_in
{
    MyLongDouble Pos[3];
    MyFloat Hsml;
    MyIDType MinID;
    MyIDType MinIDTask;
    int NodeList[NODELISTLENGTH];
} *FoFDataIn, *FoFDataGet;

/**
 * @brief Structure to store output data for the Friends-of-Friends algorithm.
 */
static struct fofdata_out
{
    MyFloat Distance;
    MyIDType MinID;
    MyIDType MinIDTask;
} *FoFDataResult, *FoFDataOut;

/**
 * @brief Structure to store information about particles for the Friends-of-Friends algorithm.
 */
static struct fof_particle_list
{
    MyIDType MinID;
    MyIDType MinIDTask;
    int Pindex;
} *FOF_PList;

static fof_group_list *FOF_GList;

static fof_id_list *ID_list;

static double LinkL, LinkLDM;
static int NgroupsExt, Nids;

static int MyFOF_PRIMARY_LINK_TYPES;
static int MyFOF_SECONDARY_LINK_TYPES;
static int MyFOF_GROUP_MIN_LEN;

static MyIDType *Head, *Len, *Next, *Tail, *MinID, *MinIDTask;
static char *NonlocalFlag;

static float *fof_nearest_distance;
static float *fof_nearest_hsml;

#ifdef AR_GREEN_TREE_FOF
#include "../GreenTree_AR/ar_fof.cpp"
#endif // AR_GREEN_TREE_FOF

#ifdef AR_GREEN_TREE_FOF
/**
 * @brief Find the nearest dark matter particle for evaluation in the FOF algorithm using OpenMP.
 *
 * This function is responsible for finding the nearest dark matter particle for evaluation
 * in the Friends-of-Friends (FOF) algorithm using OpenMP for parallelism. It searches for
 * neighboring particles within a specified smoothing length (h) around the target particle's
 * position. The nearest dark matter particle is determined based on the minimum distance
 * squared (r2max).
 *
 * @param target        The index of the target particle.
 * @param mode          The mode of operation: 0 for primary particle, 1 for data exchange.
 * @param exportflag    Pointer to the array of export flags for inter-process communication.
 * @param exportnodecount Pointer to the array of counts for exported nodes to other tasks.
 * @param exportindex   Pointer to the array of indices for exported particles to other tasks.
 * @param ngblist       Pointer to the array of neighboring particle indices.
 *
 * @return 0 upon successful completion, -1 if an error occurs during neighbor search.
 */
int fof_find_nearest_dmparticle_evaluate_openmp(int target, int mode, int *exportflag, int *exportnodecount,
                                                int *exportindex, int *ngblist)
{
    int j, n, index, listindex = 0;
    int startnode, numngb_inbox;
    double h, r2max;
    double dx, dy, dz, r2;
    MyLongDouble *pos;
    MyLongDouble xtmp;

    if (mode == 0)
    {
        pos = P[target].Pos;
        h = fof_nearest_hsml[target];
    }
    else
    {
        pos = FoFDataGet[target].Pos;
        h = FoFDataGet[target].Hsml;
    }

    index = -1;
    r2max = 1.0e30;

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = FoFDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            numngb_inbox =
                ngb_treefind_fof_nearest_openmp(pos, h, target, &startnode, mode, exportflag, exportnodecount,
                                                exportindex, ngblist, MyFOF_PRIMARY_LINK_TYPES);

            if (numngb_inbox < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
#ifdef AR_GREEN_TREE_FOF
                j = ngblist[n];
#else // AR_GREEN_TREE_FOF
                j = Ngblist[n];
#endif // AR_GREEN_TREE_FOF

                dx = NGB_PERIODIC_LONG_X(pos[0] - P[j].Pos[0]);
                dy = NGB_PERIODIC_LONG_Y(pos[1] - P[j].Pos[1]);
                dz = NGB_PERIODIC_LONG_Z(pos[2] - P[j].Pos[2]);
                r2 = dx * dx + dy * dy + dz * dz;
                //              if(r2<h*h)        printf("[DEBUG] ThisTsk%d, trget%d, ngb%d NumPart%d
                //              numngb_inbox%d\n",ThisTask, target, j,NumPart,numngb_inbox);
                if (r2 < r2max && r2 < h * h)
                {
                    index = j;
                    r2max = r2;
                }
            }
        }
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = FoFDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        if (index >= 0)
        {
            fof_nearest_distance[target] = sqrt(r2max);
            MinID[target] = MinID[Head[index]];
            MinIDTask[target] = MinIDTask[Head[index]];
        }
    }
    else
    {
        if (index >= 0)
        {
            FoFDataResult[target].Distance = sqrt(r2max);
            FoFDataResult[target].MinID = MinID[Head[index]];
            FoFDataResult[target].MinIDTask = MinIDTask[Head[index]];
        }
        else
        {
            FoFDataResult[target].Distance = 2.0e30;
        }
    }
    return 0;
}

/**
 * @brief Find the nearest dark matter particle for evaluation in the FOF algorithm.
 *
 * This function is responsible for finding the nearest dark matter particle for evaluation
 * in the Friends-of-Friends (FOF) algorithm. It searches for neighboring particles within
 * a specified smoothing length (h) around the target particle's position. The nearest
 * dark matter particle is determined based on the minimum distance squared (r2max).
 *
 * @param target The index of the target particle.
 * @param mode   The mode of operation: 0 for primary particle, 1 for data exchange.
 * @param nexport Pointer to the array of export flags for inter-process communication.
 * @param nsend_local Pointer to the array of counts for local data to be sent to other tasks.
 *
 * @return 0 upon successful completion, -1 if an error occurs during neighbor search.
 */
int fof_find_nearest_dmparticle_evaluate_orig(int target, int mode, int *nexport, int *nsend_local)
{
    int j, n, index, listindex = 0;
    int startnode, numngb_inbox;
    double h, r2max;
    double dx, dy, dz, r2;
    MyLongDouble *pos;
    MyLongDouble xtmp;

    if (mode == 0)
    {
        pos = P[target].Pos;
        h = fof_nearest_hsml[target];
    }
    else
    {
        pos = FoFDataGet[target].Pos;
        h = FoFDataGet[target].Hsml;
    }

    index = -1;
    r2max = 1.0e30;

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = FoFDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            numngb_inbox = ngb_treefind_fof_nearest(pos, h, target, &startnode, mode, nexport, nsend_local,
                                                    MyFOF_PRIMARY_LINK_TYPES);

            if (numngb_inbox < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
                j = Ngblist[n];
                dx = NGB_PERIODIC_LONG_X(pos[0] - P[j].Pos[0]);
                dy = NGB_PERIODIC_LONG_Y(pos[1] - P[j].Pos[1]);
                dz = NGB_PERIODIC_LONG_Z(pos[2] - P[j].Pos[2]);
                r2 = dx * dx + dy * dy + dz * dz;
                // if(r2<h*h)        printf("[DEBUG] ThisTsk%d, trget%d, ngb%d\n",ThisTask, target, j);
                if (r2 < r2max && r2 < h * h)
                {
                    index = j;
                    r2max = r2;
                }
            }
        }
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = FoFDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        if (index >= 0)
        {
            fof_nearest_distance[target] = sqrt(r2max);
            MinID[target] = MinID[Head[index]];
            MinIDTask[target] = MinIDTask[Head[index]];
        }
    }
    else
    {
        if (index >= 0)
        {
            FoFDataResult[target].Distance = sqrt(r2max);
            FoFDataResult[target].MinID = MinID[Head[index]];
            FoFDataResult[target].MinIDTask = MinIDTask[Head[index]];
        }
        else
        {
            FoFDataResult[target].Distance = 2.0e30;
        }
    }
    return 0;
}

/**
 * @brief Evaluate primary particles for the AR FOF algorithm using OpenMP parallelism.
 *
 * This function is responsible for evaluating primary particles for the AR (Adaptive
 * Refinement) FOF (Friends-of-Friends) algorithm using OpenMP parallelism. It iterates
 * over particles, finds their nearest dark matter particle, and fills the export buffer
 * with relevant information for each particle. The evaluation is performed within a loop
 * until all particles have been processed or the export buffer is filled.
 *
 * @param p Pointer to the thread ID indicating the thread's index.
 *
 * @return NULL upon completion of particle evaluation.
 */
void *ar_fof_evaluate_primary_openmp(void *p)
{
    int thread_id = *(int *)p;
    int i, j;
    int *exportflag, *exportnodecount, *exportindex, *ngblist;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    /* Note: exportflag is local to each thread */
    for (int j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }

    while (1)
    {
        int exitFlag = 0;
        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            if (BufferFullFlag != 0 || NextParticle >= NumPart)
            {
                exitFlag = 1;
            }
            else
            {
#ifdef _OPENMP
                do
                {
                    i = NextParticle;
                    NextParticle++; // = NextActiveParticle[NextParticle];
                    if (NextParticle >= NumPart)
                    {
                        break;
                    }
                    ProcessedFlag[i] = 1;

                } while (!fof_isactive(i));
                ProcessedFlag[i] = 0;
#else // _OPENMP
                i = NextParticle;
                ProcessedFlag[i] = 0;
                NextParticle++; // = NextActiveParticle[NextParticle];
#endif // _OPENMP
            }
        }
        UNLOCK_NEXPORT;
        if (exitFlag)
        {
            break;
        }
        //    printf("[DEBUG] I: ThisTsk%d, trget%d\n",ThisTask, i);

        if (fof_isactive(i))
        {
            if (fof_find_nearest_dmparticle_evaluate_openmp(i, 0, exportflag, exportnodecount, exportindex, ngblist) <
                0)
            {
                break; /* export buffer has filled up */
            }
        }

        ProcessedFlag[i] = 1; /* particle successfully finished */
    }

    return NULL;
}
#endif // AR_GREEN_TREE_FOF

/**
 * @brief Performs Friends-of-Friends (FOF) group finding and associated tasks.
 *
 * This function orchestrates the Friends-of-Friends (FOF) group finding algorithm along
 * with several related tasks such as computing group properties, exchanging group data,
 * finishing group properties, and potentially making black holes. It also saves group
 * information and may invoke further substructure finding.
 *
 * @param num Integer specifying the particle type or task to perform (e.g., star particles
 * for black hole seeding).
 */
void fof_fof(int num)
{
    TIMER_PUSH(1, "FOF");
    int i, ndm, start, lenloc, largestgroup, n;
    double mass, masstot, rhodm;
    struct unbind_data *d;
    long long ndmtot;

#ifdef SUBFIND_DENSITY_AND_POTENTIAL
    VERBOSE(3, "Starting SUBFIND_DENSITY_AND_POTENTIAL...");
    subfind(num);
    strcat(All.SnapshotFileBase, "_rho_and_pot");
    savepositions(RestartSnapNum, -1);
    ENDRUN("SUBFIND_DENSITY_AND_POTENTIAL done.");
#endif // SUBFIND_DENSITY_AND_POTENTIAL

#ifdef SUBFIND_READ_FOF
    read_fof(num);
#endif // SUBFIND_READ_FOF

#ifdef SUBFIND_RESHUFFLE_CATALOGUE
    force_treefree();

    read_subfind_ids();

    ENDRUN_IF(All.TotN_gas > 0, "The option SUBFIND_RESHUFFLE_CATALOGUE does not work with gas particles yet.");

    //  parallel_sort(P, NumPart, sizeof(struct particle_data), io_compare_P_GrNr_ID);
    parallel_sort_special_P_GrNr_ID();

    strcat(All.SnapshotFileBase, "_subidorder");
    savepositions(RestartSnapNum, -1);
    ENDRUN("SUBFIND_RESHUFFLE_CATALOGUE done.");
#endif // SUBFIND_RESHUFFLE_CATALOGUE

    if (num < -1) /* Run on stars (for BH seeding) ! */
    {
        MyFOF_PRIMARY_LINK_TYPES = 16;
        MyFOF_SECONDARY_LINK_TYPES = 1 + 2 + 4 + 32;
        MyFOF_GROUP_MIN_LEN = 10;
    }
    else
    {
        MyFOF_PRIMARY_LINK_TYPES = FOF_PRIMARY_LINK_TYPES;
        MyFOF_SECONDARY_LINK_TYPES = FOF_SECONDARY_LINK_TYPES;
        MyFOF_GROUP_MIN_LEN = FOF_GROUP_MIN_LEN;
    }

    VERBOSE(2, "\nBegin to compute FoF group catalogues...  (presently allocated=%g MB)",
            AllocatedBytes / (1024.0 * 1024.0));

    domain_Decomposition(1, 0);

    force_treefree();

#ifdef ONLY_PRODUCE_HSML_FILES
    subfind(num);
    ENDRUN("SUBFIND produced HSML files.");
#endif // ONLY_PRODUCE_HSML_FILES
    ndm = 0;
    mass = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : ndm, mass)
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        if (num == -2) /* Run on stars (for BH seeding) ! */
        {
            if (P[i].Type == 1)
                ndm++;
        }
        else if (((1 << P[i].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
        {
            ndm++;
        }

        if (((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE)))
        {
            mass += P[i].Mass;
        }
    }

    sumup_large_ints(1, &ndm, &ndmtot);
    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    if (All.TotN_gas)
    {
        rhodm = (All.Omega0 - All.OmegaBaryon) * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);
    }
    else
    {
        rhodm = All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);
    }

    LinkL = LinkLDM = LINKLENGTH * pow(masstot / ndmtot / rhodm, 1.0 / 3);

    if (num == -2) /* Run on stars (for BH seeding) ! */
    {
        LinkL /= 7;
    }

    if (ThisTask == 0)
    {
        VERBOSE(2, "\nComoving linking length: %g (%g)   ", LinkL, LinkLDM);
        if (num == -2) /* Run on stars (for BH seeding) ! */
        {
            VERBOSE(2, "\n      linking on star particles for BH seeding (prim=%d, sec=%d, Nmin=%d) ...",
                    MyFOF_PRIMARY_LINK_TYPES, MyFOF_SECONDARY_LINK_TYPES, MyFOF_GROUP_MIN_LEN);
        }
        VERBOSE(2, "(presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));
    }

#ifdef NO_KD_FOF_MEM_TRICK
    FOF_PList = (struct fof_particle_list *)mymalloc(
        "FOF_PList", NumPart * sizemax(sizeof(struct fof_particle_list), 3 * sizeof(MyIDType)));
#else // NO_KD_FOF_MEM_TRICK
    FOF_PList = (struct fof_particle_list *)mymalloc_movable(
        &FOF_PList, "FOF_PList", NumPart * sizemax(sizeof(struct fof_particle_list), 3 * sizeof(MyIDType)));
#endif // NO_KD_FOF_MEM_TRICK

    MinID = (MyIDType *)FOF_PList;
    MinIDTask = MinID + NumPart;
    Head = MinIDTask + NumPart;
    Len = (MyIDType *)mymalloc("Len", NumPart * sizeof(MyIDType));
    Next = (MyIDType *)mymalloc("Next", NumPart * sizeof(MyIDType));
    Tail = (MyIDType *)mymalloc("Tail", NumPart * sizeof(MyIDType));

    VERBOSE(2, "Tree construction.");

    /* build index list of particles of selected primary species */
    d = (struct unbind_data *)mymalloc("d", NumPart * sizeof(struct unbind_data));
    n = 0;
    for (int i = 0; i < NumPart; i++)
    {
        if (((1 << P[i].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
        {
            d[n++].index = i;
        }
    }

    force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    force_treebuild(n, d);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        Head[i] = Tail[i] = i;
        Len[i] = 1;
        Next[i] = -1;
        MinID[i] = P[i].ID;
        MinIDTask[i] = ThisTask;
    }

    fof_find_groups();
    fof_find_nearest_dmparticle();

    TIMER_PUSH(1, "FOF_COM_LOC_DAT");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        Next[i] = MinID[Head[i]];
        Tail[i] = MinIDTask[Head[i]];

        if (Tail[i] >= (MyIDType)NTask) /* it appears that the Intel C 9.1 on Itanium2 produces incorrect code if
                       this if-statemet is omitted. Apparently, the compiler then joins the two loops,
                       but this is here not permitted because storage for FOF_PList actually overlaps
                       (on purpose) with MinID/MinIDTask/Head */
        {
            ENDRUN("Oh no: ThisTask=%d i=%d Head[i]=%d  NumPart=%d MinIDTask[Head[i]]=%d", ThisTask, i, (int)Head[i],
                   NumPart, (int)MinIDTask[Head[i]]);
        }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        FOF_PList[i].MinID = Next[i];
        FOF_PList[i].MinIDTask = Tail[i];
        FOF_PList[i].Pindex = i;
    }

    force_treefree();

    myfree(d);

    myfree(Tail);
    myfree(Next);
    myfree(Len);

#ifdef NO_KD_FOF_MEM_TRICK
    FOF_GList = (fof_group_list *)mymalloc("FOF_GList", sizeof(fof_group_list) * NumPart);
#else // NO_KD_FOF_MEM_TRICK
    FOF_GList = (fof_group_list *)mymalloc_movable(&FOF_GList, "FOF_GList", sizeof(fof_group_list) * NumPart);
#endif // NO_KD_FOF_MEM_TRICK

    fof_compile_catalogue();

    MPI_Allreduce(&Ngroups, &TotNgroups, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    sumup_large_ints(1, &Nids, &TotNids);

    if (TotNgroups > 0)
    {
        int largestloc = 0;

        for (i = 0; i < NgroupsExt; i++)
        {
            if (FOF_GList[i].LocCount + FOF_GList[i].ExtCount > largestloc)
            {
                largestloc = FOF_GList[i].LocCount + FOF_GList[i].ExtCount;
            }
        }
        MPI_Allreduce(&largestloc, &largestgroup, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
    }
    else
    {
        largestgroup = 0;
    }

    if (ThisTask == 0)
    {
        VERBOSE(2, "\nTotal number of groups with at least %d particles: %d", MyFOF_GROUP_MIN_LEN, TotNgroups);
        if (TotNgroups > 0)
        {
            VERBOSE(2, "Largest group has %d particles.", largestgroup);
            VERBOSE(2, "Total number of particles in groups: %d%09d", (int)(TotNids / 1000000000),
                    (int)(TotNids % 1000000000));
        }
    }

    TIMER_POPPUSH(1, "FOF_COM_LOC_DAT", "FOF_GROUP_PROP");

#ifdef KD_ALTERNATIVE_GROUP_SORT
    MaxNgroups = 2 * max(NgroupsExt, TotNgroups / NTask + NTask);
#ifdef NO_KD_FOF_MEM_TRICK
    NAllocatedGroups = MaxNgroups;
    Group = (group_properties *)mymalloc("Group", sizeof(group_properties) * MaxNgroups);
#else // NO_KD_FOF_MEM_TRICK
    NAllocatedGroups = MaxNgroups;
    Group = (group_properties *)mymalloc_movable(&Group, "Group", sizeof(group_properties) * MaxNgroups);
#endif // NO_KD_FOF_MEM_TRICK
#else // KD_ALTERNATIVE_GROUP_SORT
#ifdef NO_KD_FOF_MEM_TRICK
    NAllocatedGroups = max(NgroupsExt, TotNgroups / NTask + 1);
    Group = (group_properties *)mymalloc("Group", sizeof(group_properties) * max(NgroupsExt, TotNgroups / NTask + 1));
#else // NO_KD_FOF_MEM_TRICK
    NAllocatedGroups = max(NgroupsExt, TotNgroups / NTask + 1);
    Group = (group_properties *)mymalloc_movable(&Group, "Group",
                                                 sizeof(group_properties) * max(NgroupsExt, TotNgroups / NTask + 1));
#endif // NO_KD_FOF_MEM_TRICK
#endif // KD_ALTERNATIVE_GROUP_SORT

    VERBOSE(2, "group properties are now allocated.. (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));

    start = 0;
    for (int i = 0; i < NgroupsExt; i++)
    {
        while (FOF_PList[start].MinID < FOF_GList[i].MinID)
        {
            start++;
            ENDRUN_IF(start > NumPart, "Start is larger than total number of particles.");
        }

        ENDRUN_IF(FOF_PList[start].MinID != FOF_GList[i].MinID, "FOF_PList[start].MinID != FOF_GList[i].MinID");

        for (lenloc = 0; start + lenloc < NumPart;)
        {
            if (FOF_PList[start + lenloc].MinID == FOF_GList[i].MinID)
            {
                lenloc++;
            }
            else
            {
                break;
            }
        }

        Group[i].MinID = FOF_GList[i].MinID;
        Group[i].MinIDTask = FOF_GList[i].MinIDTask;

        fof_compute_group_properties(i, start, lenloc);

        start += lenloc;
    }

    fof_exchange_group_data();

    fof_finish_group_properties();

    TIMER_POP(1, "FOF_GROUP_PROP");

#if defined(BLACK_HOLES) || defined(LT_STELLAREVOLUTION)

#ifdef BLACK_HOLES
    if (num < 0)
    {
        TIMER_PUSH(1, "FOF_MAKE_BH");
        fof_make_black_holes();
        TIMER_POP(1, "FOF_MAKE_BH");
    }
#endif // BLACK_HOLES

    TIMER_PUSH(3, "check_particles");
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Type > 5 || P[i].Type < 0)
        {
            ENDRUN("task=%d:  P[i=%d].Type=%d", ThisTask, i, P[i].Type);
        }
#ifdef LT_STELLAREVOLUTION
        if (P[i].Type == 4)
            if (MetP[P[i].pt.MetID].PID != i)
            {
                ENDRUN("task=%d:  error in cross-indexes for star-particle %d ID %llu MetID %d PID[MetID] %d", ThisTask,
                       i, (unsigned long long)P[i].ID, P[i].pt.MetID, MetP[P[i].pt.MetID].PID);
            }
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
        if (P[i].Type == 5)
        {
            if (BHP[P[i].pt.BHID].PID != i)
            {
                ENDRUN("task=%d:  error in cross-indexes for bh-particle %d ID %llu", ThisTask, i,
                       (unsigned long long)P[i].ID);
            }
        }
#endif // BLACK_HOLES
    }
    TIMER_POP(3, "check_particles");
#endif // defined(BLACK_HOLES) || defined(LT_STELLAREVOLUTION)

    if (num >= 0)
    {
        fof_save_groups(num);
#ifdef SUBFIND
#ifndef NO_KD_FOF_MEM_TRICK
        myfree_movable(FOF_GList);
        myfree_movable(FOF_PList);
#endif // NO_KD_FOF_MEM_TRICK
        if (DumpFlag != 2)
        {
            subfind(num);
        }
#endif // SUBFIND
    }
    else
    {
#ifdef DISKPOT
        fof_save_groups(num);
#endif // DISKPOT
#ifndef NO_KD_FOF_MEM_TRICK
        myfree_movable(FOF_GList);
        myfree_movable(FOF_PList);
#endif // NO_KD_FOF_MEM_TRICK
    }

    myfree(Group);

#ifdef NO_KD_FOF_MEM_TRICK
    myfree(FOF_GList);
    myfree(FOF_PList);
#endif // NO_KD_FOF_MEM_TRICK

    VERBOSE(2, "Finished computing FoF groups.  (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

#ifdef SUBFIND
    domain_Decomposition(1, 0);
#else // SUBFIND
    force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
#endif // SUBFIND

    VERBOSE(2, "Tree construction.");
    force_treebuild(NumPart, NULL);

    TreeReconstructFlag = 0;
    TIMER_POP(1, "FOF");
}

/**
 * @brief Finds groups of particles using the Friends-of-Friends (FOF) algorithm.
 *
 * This function implements the Friends-of-Friends (FOF) algorithm to find groups of particles
 * that are gravitationally linked. It first links only among local particles, then performs
 * linking across processors. It iteratively evaluates particle links until no new links are found.
 */
void fof_find_groups(void)
{
    TIMER_PUSH(1, "FOF_FIND_GROUPS");
    int ndone_flag, link_count, dummy, nprocessed;
    int ndone, place, nexport, nimport, link_across;
    int npart, marked;
    long long totmarked, totnpart;
    long long link_across_tot, ntot;
    MyIDType *MinIDOld;
    char *FoFDataOut, *FoFDataResult, *MarkedFlag, *ChangedFlag;

    VERBOSE(3, "Start linking particles (presently allocated=%g MB)\n", AllocatedBytes / (1024.0 * 1024.0));

    /* allocate buffers to arrange communication */

    Ngblist = (int *)mymalloc("Ngblist", NumPart * sizeof(int));

    All.BunchSize = (int)((All.BufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + sizeof(struct data_nodelist) + 2 * sizeof(struct fofdata_in)));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    NonlocalFlag = (char *)mymalloc("NonlocalFlag", NumPart * sizeof(char));
    MarkedFlag = (char *)mymalloc("MarkedFlag", NumPart * sizeof(char));
    ChangedFlag = (char *)mymalloc("ChangedFlag", NumPart * sizeof(char));
    MinIDOld = (MyIDType *)mymalloc("MinIDOld", NumPart * sizeof(MyIDType));

    TIMER_PUSH(3, "linking_local");

    /* first, link only among local particles */
    marked = 0, npart = 0;
    for (int i = 0; i < NumPart; i++)
    {
        if (((1 << P[i].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
        {
            fof_find_dmparticles_evaluate(i, -1, &dummy, &dummy);

            npart++;

            if (NonlocalFlag[i])
            {
                marked++;
            }
        }
    }

    sumup_large_ints(1, &marked, &totmarked);
    sumup_large_ints(1, &npart, &totnpart);

    VERBOSE(3, "links on local processor done.\nMarked=%d%09d out of the %d%09d primaries which are linked\n",
            (int)(totmarked / 1000000000), (int)(totmarked % 1000000000), (int)(totnpart / 1000000000),
            (int)(totnpart % 1000000000));

    VERBOSE(3, "linking across processors (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));

    TIMER_POP(3, "linking_local");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        MinIDOld[i] = MinID[Head[i]];
        MarkedFlag[i] = 1;
    }

    do
    {
        TIMER_PUSH(3, "cross_links");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            ChangedFlag[i] = MarkedFlag[i];
            MarkedFlag[i] = 0;
        }

        int i = 0; /* begin with this index */
        link_across = 0;
        nprocessed = 0;

        do
        {
            for (int j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
                Exportflag[j] = -1;
            }

            /* do local particles and prepare export list */
            for (nexport = 0; i < NumPart; i++)
            {
                if (((1 << P[i].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
                {
                    if (NonlocalFlag[i] && ChangedFlag[i])
                    {
                        if (fof_find_dmparticles_evaluate(i, 0, &nexport, Send_count) < 0)
                        {
                            break;
                        }

                        nprocessed++;
                    }
                }
            }

            MYSORT_DATAINDEX(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);

            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
            nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
            for (int j = 0; j < NTask; j++)
            {
                nimport += Recv_count[j];

                if (j > 0)
                {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }

            FoFDataGet = (struct fofdata_in *)mymalloc("FoFDataGet", nimport * sizeof(struct fofdata_in));
            FoFDataIn = (struct fofdata_in *)mymalloc("FoFDataIn", nexport * sizeof(struct fofdata_in));

            /* prepare particle data for export */
            for (int j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;

                FoFDataIn[j].Pos[0] = P[place].Pos[0];
                FoFDataIn[j].Pos[1] = P[place].Pos[1];
                FoFDataIn[j].Pos[2] = P[place].Pos[2];
                FoFDataIn[j].MinID = MinID[Head[place]];
                FoFDataIn[j].MinIDTask = MinIDTask[Head[place]];

                memcpy(FoFDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                       NODELISTLENGTH * sizeof(int));
            }

            /* exchange particle data */
            TIMER_PUSH(4, "fof_comm");
            templated_exchange_data_Sendrecv(FoFDataIn, FoFDataGet, TAG_DENS_A, Send_offset, Send_count, Recv_offset,
                                             Recv_count);
            TIMER_POP(4, "fof_comm");

            myfree(FoFDataIn);
            FoFDataResult = (char *)mymalloc("FoFDataResult", nimport * sizeof(char));
            FoFDataOut = (char *)mymalloc("FoFDataOut", nexport * sizeof(char));

            /* now do the particles that were sent to us */
            for (int j = 0; j < nimport; j++)
            {
                link_count = fof_find_dmparticles_evaluate(j, 1, &dummy, &dummy);
                link_across += link_count;
                if (link_count)
                {
                    FoFDataResult[j] = 1;
                }
                else
                {
                    FoFDataResult[j] = 0;
                }
            }

            /* exchange data */
            TIMER_PUSH(4, "fof_comm");
            templated_exchange_data_Sendrecv(FoFDataResult, FoFDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POP(4, "fof_comm");

            /* need to mark the particle if it induced a link */
            for (int j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;
                if (FoFDataOut[j])
                {
                    MarkedFlag[place] = 1;
                }
            }

            myfree(FoFDataOut);
            myfree(FoFDataResult);
            myfree(FoFDataGet);

            if (i >= NumPart)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }

            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        } while (ndone < NTask);

        sumup_large_ints(1, &link_across, &link_across_tot);
        sumup_large_ints(1, &nprocessed, &ntot);

        VERBOSE(3, "have done %d%09d cross links (processed %d%09d)", (int)(link_across_tot / 1000000000),
                (int)(link_across_tot % 1000000000), (int)(ntot / 1000000000), (int)(ntot % 1000000000));

        TIMER_POP(3, "cross_links");

        /* let's check out which particles have changed their MinID */
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            if (NonlocalFlag[i])
            {
                if (MinID[Head[i]] != MinIDOld[i])
                {
                    MarkedFlag[i] = 1;
                }

                MinIDOld[i] = MinID[Head[i]];
            }
        }

    } while (link_across_tot > 0);

    myfree(MinIDOld);
    myfree(ChangedFlag);
    myfree(MarkedFlag);
    myfree(NonlocalFlag);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);

    VERBOSE(3, "Local groups found.\n");

    TIMER_POP(1, "FOF_FIND_GROUPS");
}

/**
 * @brief Evaluates the Friends-of-Friends (FOF) algorithm to find dark matter particles.
 *
 * This function evaluates the FOF algorithm to find dark matter particles linked by gravity.
 * It iterates through the particles and their neighbors to establish links between particles
 * based on a specified linking length. It updates the linked particles' information, such as
 * their group ID, length, and task ID. This function can operate in two modes: local evaluation
 * (mode 0) and communication mode (mode 1). In local evaluation mode, it operates on local
 * particle data, whereas in communication mode, it operates on received data for communication
 * between tasks. Additionally, it updates the non-local flag for each particle to indicate
 * whether it has non-local neighbors.
 *
 * @param target The index of the target particle.
 * @param mode The mode of operation: 0 for local evaluation, -1 for initializing non-local flag,
 *             and 1 for communication mode.
 * @param nexport Pointer to the number of particles to be exported.
 * @param nsend_local Pointer to the number of particles to be sent locally.
 * @return The number of links established between particles.
 */
int fof_find_dmparticles_evaluate(int target, int mode, int *nexport, int *nsend_local)
{
    int j, links, p, s, ss, listindex = 0;
    int startnode, numngb_inbox;
    MyLongDouble *pos;

    links = 0;

    if (mode == 0 || mode == -1)
    {
        pos = P[target].Pos;
    }
    else
    {
        pos = FoFDataGet[target].Pos;
    }

    if (mode == 0 || mode == -1)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = FoFDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            if (mode == -1)
            {
                *nexport = 0;
            }

            numngb_inbox = ngb_treefind_fof_primary(pos, LinkL, target, &startnode, mode, nexport, nsend_local,
                                                    MyFOF_PRIMARY_LINK_TYPES);

            if (numngb_inbox < 0)
            {
                return -1;
            }

            if (mode == -1)
            {
                if (*nexport == 0)
                {
                    NonlocalFlag[target] = 0;
                }
                else
                {
                    NonlocalFlag[target] = 1;
                }
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
                j = Ngblist[n];

                if (mode == 0 || mode == -1)
                {
                    if (Head[target] != Head[j]) /* only if not yet linked */
                    {

                        ENDRUN_IF(mode == 0, "Mode is 0.");

                        if (Len[Head[target]] > Len[Head[j]]) /* p group is longer */
                        {
                            p = target;
                            s = j;
                        }
                        else
                        {
                            p = j;
                            s = target;
                        }
                        Next[Tail[Head[p]]] = Head[s];

                        Tail[Head[p]] = Tail[Head[s]];

                        Len[Head[p]] += Len[Head[s]];

                        ss = Head[s];
                        do
                        {
                            Head[ss] = Head[p];
                        } while ((ss = Next[ss]) >= 0);

                        if (MinID[Head[s]] < MinID[Head[p]])
                        {
                            MinID[Head[p]] = MinID[Head[s]];
                            MinIDTask[Head[p]] = MinIDTask[Head[s]];
                        }
                    }
                }
                else /* mode is 1 */
                {
                    if (MinID[Head[j]] > FoFDataGet[target].MinID)
                    {
                        MinID[Head[j]] = FoFDataGet[target].MinID;
                        MinIDTask[Head[j]] = FoFDataGet[target].MinIDTask;
                        links++;
                    }
                }
            }
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = FoFDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    return links;
}

/**
 * @brief Compiles the FOF (Friends-of-Friends) catalogue.
 *
 * This function compiles the FOF catalogue based on the list of FOF particles (`FOF_PList`).
 * It first sorts the FOF particle list according to the minimum ID (`MinID`) of the corresponding groups.
 * Then, it assigns the appropriate local and external counts to each group in the FOF group list (`FOF_GList`).
 * Duplicate groups are eliminated, and the remaining groups are sorted according to the task.
 * Next, it determines the number of groups to be sent to each task for MPI communication.
 * Groups are exchanged between tasks using MPI_Alltoall, and the received groups are merged with local groups.
 * Finally, groups that are too small are eliminated, and the local group count and total particle count are updated.
 */
void fof_compile_catalogue(void)
{
    int start, nimport, ngrp;
    fof_group_list *get_FOF_GList;

    /* sort according to MinID */
    STD_SORT(FOF_PList, NumPart, sizeof(struct fof_particle_list), fof_compare_FOF_PList_MinID);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        FOF_GList[i].MinID = FOF_PList[i].MinID;
        FOF_GList[i].MinIDTask = FOF_PList[i].MinIDTask;
        if (FOF_GList[i].MinIDTask == (MyIDType)ThisTask)
        {
            FOF_GList[i].LocCount = 1;
            FOF_GList[i].ExtCount = 0;
#ifdef DENSITY_SPLIT_BY_TYPE
            if (((1 << P[FOF_PList[i].Pindex].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
            {
                FOF_GList[i].LocDMCount = 1;
            }
            else
            {
                FOF_GList[i].LocDMCount = 0;
            }
            FOF_GList[i].ExtDMCount = 0;
#endif // DENSITY_SPLIT_BY_TYPE
        }
        else
        {
            FOF_GList[i].LocCount = 0;
            FOF_GList[i].ExtCount = 1;
#ifdef DENSITY_SPLIT_BY_TYPE
            FOF_GList[i].LocDMCount = 0;
            if (((1 << P[FOF_PList[i].Pindex].Type) & (MyFOF_PRIMARY_LINK_TYPES)))
            {
                FOF_GList[i].ExtDMCount = 1;
            }
            else
            {
                FOF_GList[i].ExtDMCount = 0;
            }
#endif // DENSITY_SPLIT_BY_TYPE
        }
    }

    /* eliminate duplicates in FOF_GList with respect to MinID */
    if (NumPart)
    {
        NgroupsExt = 1;
    }
    else
    {
        NgroupsExt = 0;
    }

    start = 0;
    for (int i = 1; i < NumPart; i++)
    {
        if (FOF_GList[i].MinID == FOF_GList[start].MinID)
        {
            FOF_GList[start].LocCount += FOF_GList[i].LocCount;
            FOF_GList[start].ExtCount += FOF_GList[i].ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
            FOF_GList[start].LocDMCount += FOF_GList[i].LocDMCount;
            FOF_GList[start].ExtDMCount += FOF_GList[i].ExtDMCount;
#endif // DENSITY_SPLIT_BY_TYPE
        }
        else
        {
            start = NgroupsExt;
            FOF_GList[start] = FOF_GList[i];
            NgroupsExt++;
        }
    }

    /* sort the remaining ones according to task */
    STD_SORT(FOF_GList, NgroupsExt, sizeof(fof_group_list), fof_compare_FOF_GList_MinIDTask);

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }
    for (int i = 0; i < NgroupsExt; i++)
    {
        Send_count[FOF_GList[i].MinIDTask]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
    for (int j = 0; j < NTask; j++)
    {
        if (j == ThisTask) /* we will not exchange the ones that are local */
        {
            Recv_count[j] = 0;
        }
        nimport += Recv_count[j];

        if (j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

    get_FOF_GList = (fof_group_list *)mymalloc("get_FOF_GList", nimport * sizeof(fof_group_list));

    for (ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                /* get the group info */
                MPI_Sendrecv(&FOF_GList[Send_offset[recvTask]], Send_count[recvTask] * sizeof(fof_group_list), MPI_BYTE,
                             recvTask, TAG_DENS_A, &get_FOF_GList[Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(fof_group_list), MPI_BYTE, recvTask, TAG_DENS_A,
                             MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    for (int i = 0; i < nimport; i++)
    {
        get_FOF_GList[i].MinIDTask = i;
    }

    /* sort the groups according to MinID */
    STD_SORT(FOF_GList, NgroupsExt, sizeof(fof_group_list), fof_compare_FOF_GList_MinID);
    STD_SORT(get_FOF_GList, nimport, sizeof(fof_group_list), fof_compare_FOF_GList_MinID);

    /* merge the imported ones with the local ones */
    start = 0;
    for (int i = 0; i < nimport; i++)
    {
        while (FOF_GList[start].MinID < get_FOF_GList[i].MinID)
        {
            start++;
            ENDRUN_IF(start >= NgroupsExt, "Start is larger than total number of groups.");
        }

        ENDRUN_IF(get_FOF_GList[i].LocCount != 0, "Local count is not zero.");
        ENDRUN_IF(FOF_GList[start].MinIDTask != (MyIDType)ThisTask, "MinIDTask is not equal to ThisTask.");

        FOF_GList[start].ExtCount += get_FOF_GList[i].ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
        FOF_GList[start].ExtDMCount += get_FOF_GList[i].ExtDMCount;
#endif // DENSITY_SPLIT_BY_TYPE
    }

    /* copy the size information back into the list, to inform the others */
    start = 0;
    for (int i = 0; i < nimport; i++)
    {
        while (FOF_GList[start].MinID < get_FOF_GList[i].MinID)
        {
            start++;
            ENDRUN_IF(start >= NgroupsExt, "Start is larger then total number of groups.");
        }

        get_FOF_GList[i].ExtCount = FOF_GList[start].ExtCount;
        get_FOF_GList[i].LocCount = FOF_GList[start].LocCount;
#ifdef DENSITY_SPLIT_BY_TYPE
        get_FOF_GList[i].ExtDMCount = FOF_GList[start].ExtDMCount;
        get_FOF_GList[i].LocDMCount = FOF_GList[start].LocDMCount;
#endif // DENSITY_SPLIT_BY_TYPE
    }

    /* sort the imported/exported list according to MinIDTask */
    STD_SORT(get_FOF_GList, nimport, sizeof(fof_group_list), fof_compare_FOF_GList_MinIDTask);
    STD_SORT(FOF_GList, NgroupsExt, sizeof(fof_group_list), fof_compare_FOF_GList_MinIDTask);

    for (int i = 0; i < nimport; i++)
    {
        get_FOF_GList[i].MinIDTask = ThisTask;
    }

    TIMER_PUSH(4, "fof_comm");
    templated_exchange_data_Sendrecv(get_FOF_GList, FOF_GList, TAG_DENS_A, Recv_offset, Recv_count, Send_offset,
                                     Send_count);
    TIMER_POP(4, "fof_comm");

    myfree(get_FOF_GList);

    /* eliminate all groups that are too small, and count local groups */
    Ngroups = 0, Nids = 0;
    for (int i = 0; i < NgroupsExt; i++)
    {
#ifdef DENSITY_SPLIT_BY_TYPE
        if (FOF_GList[i].LocDMCount + FOF_GList[i].ExtDMCount < MyFOF_GROUP_MIN_LEN)
#else // DENSITY_SPLIT_BY_TYPE
        if (FOF_GList[i].LocCount + FOF_GList[i].ExtCount < MyFOF_GROUP_MIN_LEN)
#endif // DENSITY_SPLIT_BY_TYPE
        {
            FOF_GList[i] = FOF_GList[NgroupsExt - 1];
            NgroupsExt--;
            i--;
        }
        else
        {
            if (FOF_GList[i].MinIDTask == (MyIDType)ThisTask)
            {
                Ngroups++;
                Nids += FOF_GList[i].LocCount + FOF_GList[i].ExtCount;
            }
        }
    }
    /* sort the group list according to MinID */
    STD_SORT(FOF_GList, NgroupsExt, sizeof(fof_group_list), fof_compare_FOF_GList_MinID);
}

/**
 * @brief Outputs properties of a group and its member particles.
 *
 * This function outputs properties of a group, such as its center of mass, velocity, mass, and
 * potentially the maximum density (if black hole related properties are enabled). Additionally,
 * it outputs properties of the particles belonging to the group, including their type, position,
 * velocity, mass, and potential.
 *
 * @param gr The index of the group whose properties are to be output.
 */
void fof_output_group_properties(int gr)
{
    VERBOSE_ALL(2, "FOF_G_BHSEED %d: x y z vx vy vz m minpot", ThisTask);
    MyFloat maxdens = 0;
#ifdef BLACK_HOLES
    maxdens = Group[gr].MaxDens;
#endif // BLACK_HOLES
    VERBOSE_ALL(2, "FOF_G_BHSEED %d: %f %f %f %f %f %f %f %f", ThisTask, Group[gr].CM[0], Group[gr].CM[1],
                Group[gr].CM[2], Group[gr].Vel[0], Group[gr].Vel[1], Group[gr].Vel[2], Group[gr].Mass, maxdens);

    VERBOSE_ALL(2, "FOF_P_BHSEED %d: type x y z vx vy vz m pot", ThisTask);

    for (int i = 0; i < NumPart; i++)
    {
        if (FOF_PList[i].MinID == FOF_GList[gr].MinID)
        {
            int index = FOF_PList[i].Pindex;
            MyFloat pot = 0.0;
#ifdef EVALPOTENTIAL
            pot = P[index].Potential;
#endif // EVALPOTENTIAL
            VERBOSE_ALL(2, "FOF_P_BHSEED %d: %d %f %f %f %f %f %f %f %f", ThisTask, P[index].Type, P[index].Pos[0],
                        P[index].Pos[1], P[index].Pos[2], P[index].Vel[0], P[index].Vel[1], P[index].Vel[2],
                        P[index].Mass, pot);
        }
    }
}

/**
 * @brief Computes properties of a group.
 *
 * This function calculates various properties of a group, such as its length, mass, center of mass,
 * velocity, etc., based on the particles within the group. It iterates over the particles within
 * the group, updating the group's properties accordingly. Properties related to specific physics,
 * such as star formation rate (SFR), black hole (BH) mass and accretion rate, and synchrotron power,
 * are also computed if the respective compilation flags are enabled.
 *
 * @param gr The index of the group whose properties are to be computed.
 * @param start The starting index of the particle list corresponding to the group.
 * @param len The length of the particle list corresponding to the group.
 */
void fof_compute_group_properties(int gr, int start, int len)
{
    int index;
    double xyz[3];

    Group[gr].Len = 0;
    Group[gr].Mass = 0;

#ifdef SFR
    Group[gr].Sfr = 0;
#endif // SFR

#ifdef BLACK_HOLES
    Group[gr].BH_Mass = 0;
    Group[gr].BH_Mdot = 0;
    Group[gr].index_maxdens = Group[gr].task_maxdens = -1;
    if (All.BlackHoleSeedStarMassFraction > 0)
    {
        Group[gr].MaxDens = 1.5e30;
    }
    else
    {
        Group[gr].MaxDens = 0;
    }
#endif // BLACK_HOLES

#ifdef LMB_CR_OUTPUT_SYNCHROTRON
    for (int i_nu = 0; i_nu < 8; i_nu++)
    {
        Group[gr].Synchrotron[i_nu] = 0;
    }
#endif // LMB_CR_OUTPUT_SYNCHROTRON
    for (int k = 0; k < 3; k++)
    {
        Group[gr].CM[k] = 0;
        Group[gr].Vel[k] = 0;
        Group[gr].FirstPos[k] = P[FOF_PList[start].Pindex].Pos[k];
    }

#ifndef FOF_EXTENDED_PROPERTIES
    for (int k = 0; k < 6; k++)
    {
        Group[gr].LenType[k] = 0;
        Group[gr].MassType[k] = 0;
    }
#endif // FOF_EXTENDED_PROPERTIES

    for (int k = 0; k < len; k++)
    {
        index = FOF_PList[start + k].Pindex;

        Group[gr].Len++;
        Group[gr].Mass += P[index].Mass;
#ifndef FOF_EXTENDED_PROPERTIES
        Group[gr].LenType[P[index].Type]++;
        Group[gr].MassType[P[index].Type] += P[index].Mass;
#endif // FOF_EXTENDED_PROPERTIES

#ifdef SFR
        if (P[index].Type == 0)
        {
            Group[gr].Sfr += SphP[index].Sfr;
        }
#endif // SFR

#ifdef BLACK_HOLES
        if (P[index].Type == 5)
        {
            Group[gr].BH_Mdot += BPP(index).BH_Mdot;
            Group[gr].BH_Mass += BPP(index).BH_Mass;
        }
        if (P[index].Type == 0)
        {
#ifdef WINDS
            if (SphP[index].DelayTime <= 0)
            {
#endif // WINDS
                if (All.BlackHoleSeedStarMassFraction > 0)
                {
                    if (P[index].Potential < Group[gr].MaxDens)
                    {
                        Group[gr].MaxDens = P[index].Potential;
                        Group[gr].index_maxdens = index;
                        Group[gr].task_maxdens = ThisTask;
                    }
                }
                else
                {
                    if (SphP[index].Density > Group[gr].MaxDens)
                    {
                        Group[gr].MaxDens = SphP[index].Density;
                        Group[gr].index_maxdens = index;
                        Group[gr].task_maxdens = ThisTask;
                    }
                }
#ifdef WINDS
            }
#endif // WINDS
        }
#endif // BLACK_HOLES

#ifdef LMB_CR_OUTPUT_SYNCHROTRON

        if (P[index].Type == 0)
        {
            MyAtLeastDouble cosmo_a = 1.0;

            if (All.ComovingIntegrationOn)
            {
                cosmo_a = All.Time;
            }

            // compute density and mass in physical cgs units
            MyAtLeastDouble density = SphP[index].Density / (cosmo_a * cosmo_a * cosmo_a) * All.HubbleParam *
                                      All.HubbleParam * All.UnitDensity_in_cgs;
            MyAtLeastDouble mass = P[index].Mass / All.HubbleParam * All.UnitMass_in_g;

            // store synchrotron power in [ W / Hz ]
            for (int i_nu = 0; i_nu < 8; i_nu++)
            {
                Group[gr].Synchrotron[i_nu] += mass / density * SphP[index].SynchEmissivity[i_nu] * 1.e-7;
            }
        }

#endif // LMB_CR_OUTPUT_SYNCHROTRON

        for (int j = 0; j < 3; j++)
        {
            xyz[j] = P[index].Pos[j];
#ifdef PERIODIC
            xyz[j] = fof_periodic(xyz[j] - Group[gr].FirstPos[j]);
#endif // PERIODIC
            Group[gr].CM[j] += P[index].Mass * xyz[j];
            Group[gr].Vel[j] += P[index].Mass * P[index].Vel[j];
        }
    }
}

/**
 * @brief Exchanges group data between tasks.
 *
 * This function sorts the groups according to the minimum ID task, counts the number of groups
 * to be sent to each task, performs an all-to-all communication to exchange group data,
 * and updates various properties of the groups based on the received data. It ensures consistency
 * by sorting the groups again and adding the imported group data to the main ones.
 */
void fof_exchange_group_data(void)
{
    group_properties *get_Group;
    int nimport, start;
    double xyz[3];

    /* sort the groups according to task */
    STD_SORT(Group, NgroupsExt, sizeof(group_properties), fof_compare_Group_MinIDTask);

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }
    for (int i = 0; i < NgroupsExt; i++)
    {
        Send_count[FOF_GList[i].MinIDTask]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
    for (int j = 0; j < NTask; j++)
    {
        if (j == ThisTask) /* we will not exchange the ones that are local */
        {
            Recv_count[j] = 0;
        }
        nimport += Recv_count[j];

        if (j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

    get_Group = (group_properties *)mymalloc("get_Group", sizeof(group_properties) * nimport);

    TIMER_PUSH(4, "fof_comm");
    templated_exchange_data_Sendrecv(Group, get_Group, TAG_DENS_A, Send_offset, Send_count, Recv_offset, Recv_count);
    TIMER_POP(4, "fof_comm");

    /* sort the groups again according to MinID */
    STD_SORT(Group, NgroupsExt, sizeof(group_properties), fof_compare_Group_MinID);
    STD_SORT(get_Group, nimport, sizeof(group_properties), fof_compare_Group_MinID);

    /* now add in the partial imported group data to the main ones */
    start = 0;
    for (int i = 0; i < nimport; i++)
    {
        while (Group[start].MinID < get_Group[i].MinID)
        {
            start++;
            ENDRUN_IF(start >= NgroupsExt, "Start is larger than total number of groups.");
        }

        Group[start].Len += get_Group[i].Len;
        Group[start].Mass += get_Group[i].Mass;

#ifndef FOF_EXTENDED_PROPERTIES
        for (int j = 0; j < 6; j++)
        {
            Group[start].LenType[j] += get_Group[i].LenType[j];
            Group[start].MassType[j] += get_Group[i].MassType[j];
        }
#endif // FOF_EXTENDED_PROPERTIES

#ifdef SFR
        Group[start].Sfr += get_Group[i].Sfr;
#endif // SFR

#ifdef BLACK_HOLES
        Group[start].BH_Mdot += get_Group[i].BH_Mdot;
        Group[start].BH_Mass += get_Group[i].BH_Mass;

        if ((All.BlackHoleSeedStarMassFraction > 0 && get_Group[i].MaxDens < Group[start].MaxDens) ||
            (All.BlackHoleSeedStarMassFraction == 0 && get_Group[i].MaxDens > Group[start].MaxDens))
        {
            Group[start].MaxDens = get_Group[i].MaxDens;
            Group[start].index_maxdens = get_Group[i].index_maxdens;
            Group[start].task_maxdens = get_Group[i].task_maxdens;
        }
#endif // BLACK_HOLES

#ifdef LMB_CR_OUTPUT_SYNCHROTRON
        for (int i_nu = 0; i_nu < 8; i_nu++)
        {
            Group[start].Synchrotron[i_nu] += get_Group[i].Synchrotron[i_nu];
        }
#endif // LMB_CR_OUTPUT_SYNCHROTRON

        for (int j = 0; j < 3; j++)
        {
            xyz[j] = get_Group[i].CM[j] / get_Group[i].Mass + get_Group[i].FirstPos[j];
#ifdef PERIODIC
            xyz[j] = fof_periodic(xyz[j] - Group[start].FirstPos[j]);
#endif // PERIODIC
            Group[start].CM[j] += get_Group[i].Mass * xyz[j];
            Group[start].Vel[j] += get_Group[i].Vel[j];
        }
    }

    myfree(get_Group);
}

#ifdef BLACK_HOLES
/**
 * @brief Exchanges binding group data between tasks.
 *
 * This function sorts the groups according to the minimum ID task, counts the number of groups
 * to be sent to each task, performs an all-to-all communication to exchange group data,
 * and updates the maximum density information for groups. It ensures consistency by sorting
 * the groups again and adding the imported group data to the main ones.
 */
void fof_exchange_group_data_binding(void)
{
    group_properties *get_Group;

    /* sort the groups according to task */
    STD_SORT(Group, NgroupsExt, sizeof(group_properties), fof_compare_Group_MinIDTask);

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }
    for (int i = 0; i < NgroupsExt; i++)
    {
        Send_count[FOF_GList[i].MinIDTask]++;
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    Recv_offset[0] = 0, Send_offset[0] = 0;
    int nimport = 0;
    for (int j = 0; j < NTask; j++)
    {
        if (j == ThisTask) /* we will not exchange the ones that are local */
        {
            Recv_count[j] = 0;
        }
        nimport += Recv_count[j];

        if (j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

    get_Group = (group_properties *)mymalloc("get_Group", sizeof(group_properties) * nimport);

    TIMER_PUSH(4, "fof_comm");
    templated_exchange_data_Sendrecv(Group, get_Group, TAG_DENS_A, Send_offset, Send_count, Recv_offset, Recv_count);
    TIMER_POP(4, "fof_comm");

    /* sort the groups again according to MinID */
    STD_SORT(Group, NgroupsExt, sizeof(group_properties), fof_compare_Group_MinID);
    STD_SORT(get_Group, nimport, sizeof(group_properties), fof_compare_Group_MinID);

    /* now add in the partial imported group data to the main ones */
    int start = 0;
    for (int i = 0; i < nimport; i++)
    {
        while (Group[start].MinID < get_Group[i].MinID)
        {
            start++;
            ENDRUN_IF(start >= NgroupsExt, "Start is larger than total number of groups.");
        }

        if (get_Group[i].MaxDens < Group[start].MaxDens)
        {
            Group[start].MaxDens = get_Group[i].MaxDens;
            Group[start].index_maxdens = get_Group[i].index_maxdens;
            Group[start].task_maxdens = get_Group[i].task_maxdens;
        }
    }

    myfree(get_Group);
}
#endif // BLACK_HOLES

/**
 * @brief Finalizes group properties after FoF calculation.
 *
 * This function completes the computation of group properties, such as the center of mass and velocity,
 * for locally assigned groups. It then eliminates non-local groups and ensures consistency in the group array.
 */
void fof_finish_group_properties(void)
{
    double cm[3];

#ifdef _OPENMP
#pragma omp parallel for private(cm)
#endif // _OPENMP
    for (int i = 0; i < NgroupsExt; i++)
    {
        if (Group[i].MinIDTask == (MyIDType)ThisTask)
        {
            for (int j = 0; j < 3; j++)
            {
                Group[i].Vel[j] /= Group[i].Mass;

                cm[j] = Group[i].CM[j] / Group[i].Mass;
#ifdef PERIODIC
                cm[j] = fof_periodic_wrap(cm[j] + Group[i].FirstPos[j]);
#endif // PERIODIC
                Group[i].CM[j] = cm[j];
            }
        }
    }

    int ngr = NgroupsExt;
    /* eliminate the non-local groups */
    for (int i = 0; i < ngr; i++)
    {
        if (Group[i].MinIDTask != (MyIDType)ThisTask)
        {
            Group[i] = Group[ngr - 1];
            i--;
            ngr--;
        }
    }

    ENDRUN_IF(ngr != Ngroups, "Number of groups is not equal to the expected number of groups.");

    STD_SORT(Group, Ngroups, sizeof(group_properties), fof_compare_Group_MinID);
}

#ifdef FOF_EXTENDED_PROPERTIES
/**
 * @brief Determines the target tasks for particles and groups in the FoF calculation.
 *
 * This function assigns each particle to its target task based on its group number,
 * and stores the original task for each group.
 */
void fof_get_tasks(void)
{
    VERBOSE(2, "Determining target tasks.");
    // Assign target tasks for particles
    for (int i = 0; i < NumPart; i++)
    {
        P[i].origintask = ThisTask;

        if (P[i].GrNr <= TotNgroups)
        {
            P[i].targettask = (P[i].GrNr - 1) % NTask;
        }
        else
        {
            P[i].targettask = ThisTask;
        }
    }
    // Store the task where each group came from
    for (int i = 0; i < Ngroups; i++) /* store the task where the group came from */
    {
        Group[i].Origintask = ThisTask;
    }
}

/**
 * @brief Structure representing a particle in a sorted list for FoF calculations.
 */
typedef struct
{
    MyFloat r;
    MyFloat mass;
    int grnr;
} fof_sort_list;

/**
 * @brief Comparison function for sorting a list of fof_sort_list structures.
 *
 * This function is used to compare two fof_sort_list structures based on their grnr and r fields.
 *
 * @param a Pointer to the first fof_sort_list structure.
 * @param b Pointer to the second fof_sort_list structure.
 * @return int Returns -1 if the grnr of the first structure is less than the grnr of the second structure,
 *             +1 if the grnr of the first structure is greater than the grnr of the second structure,
 *             -1 if the grnr of both structures are equal and the r of the first structure is less than the r of the
 * second structure, +1 if the grnr of both structures are equal and the r of the first structure is greater than the r
 * of the second structure, and 0 if both structures are considered equal.
 */
int fof_compare_sort_list(const void *a, const void *b)
{
    if (((fof_sort_list *)a)->grnr < ((fof_sort_list *)b)->grnr)
    {
        return -1;
    }

    if (((fof_sort_list *)a)->grnr > ((fof_sort_list *)b)->grnr)
    {
        return +1;
    }

    if (((fof_sort_list *)a)->r < ((fof_sort_list *)b)->r)
    {
        return -1;
    }

    if (((fof_sort_list *)a)->r > ((fof_sort_list *)b)->r)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Comparison function for sorting groups by their group number (GrNr).
 *
 * This function is used to compare two group_properties structures based on their GrNr field. It is designed
 * to be used with sorting functions like qsort.
 *
 * @param a Pointer to the first group_properties structure.
 * @param b Pointer to the second group_properties structure.
 * @return int Returns -1 if the GrNr of the first group is less than the GrNr of the second group,
 *             +1 if the GrNr of the first group is greater than the GrNr of the second group,
 *             and 0 if the GrNr of both groups are equal.
 */
int fof_compare_grnr(const void *a, const void *b)
{
    if (((group_properties *)a)->GrNr < ((group_properties *)b)->GrNr)
    {
        return -1;
    }

    if (((group_properties *)a)->GrNr > ((group_properties *)b)->GrNr)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Computes the center of each Friends-of-Friends (FoF) group using the shrinking sphere method.
 *
 * This function iteratively determines the center of each FoF group by calculating the distance to the outermost
 * particle and shrinking the sphere until the number of particles within the sphere falls below a given threshold.
 *
 * @param particle_list Array of integers indicating the group index for each particle.
 *                      The value is -1 for particles not in any group.
 */
void fof_shrinking_sphere_center(int *particle_list)
{
    float max_rad_sq[Ngroups];
    double *new_group_pos;
    double *new_group_mass;
    int num_part_in_max_rad[Ngroups];
    int groups_done = 0;

    double last_time, cur_time;
    last_time = report_time();

    int particle_threshold = 10;

    new_group_pos = (double *)mymalloc("new_group_pos", sizeof(double) * 3 * Ngroups);
    new_group_mass = (double *)mymalloc("new_group_mass", sizeof(double) * Ngroups);

    VERBOSE(2, "Searching group center.");

    /* initialize stuff */
    for (int g = 0; g < Ngroups; g++)
    {
        num_part_in_max_rad[g] = 0;
        max_rad_sq[g] = 0;

        for (int k = 0; k < 3; k++)
        {
            Group[g].Pos[k] = Group[g].CM[k];
            new_group_pos[3 * g + k] = 0;
        }
        new_group_mass[g] = 0;
    }

    /* determine distance to outermost particle for each group */
    for (int j = 0; j < NumPart; j++)
    {
        int g = particle_list[j];
        if (g < 0)
        {
            continue;
        }

        if (P[j].GrNr != Group[g].GrNr)
        {
            ENDRUN_MAIN("something wrong with particle list.");
        }

        double dist_sq = 0;
        for (int k = 0; k < 3; k++)
        {
#ifdef PERIODIC
            double d = fof_periodic(Group[g].Pos[k] - P[j].Pos[k]);
#else // PERIODIC
            double d = Group[g].Pos[k] - P[j].Pos[k];
#endif // PERIODIC
            dist_sq += d * d;
        }

        if (dist_sq > max_rad_sq[g])
        {
            max_rad_sq[g] = dist_sq;
        }
    }

    /* iterate to find the center */
    while (groups_done < Ngroups)
    {
        for (int j = 0; j < NumPart; j++)
        {
            int g = particle_list[j];

            if ((g < 0) || (num_part_in_max_rad[g] < 0)) /* particle is not in group or group is already done */
            {
                continue;
            }

            double dist_sq = 0;
            for (int k = 0; k < 3; k++)
            {
#ifdef PERIODIC
                double d = fof_periodic(Group[g].Pos[k] - P[j].Pos[k]);
#else // PERIODIC
                double d = Group[g].Pos[k] - P[j].Pos[k];
#endif // PERIODIC
                dist_sq += d * d;
            }

            if (dist_sq > max_rad_sq[g]) /* particle is outside the considered sphere */
            {
                continue;
            }

            num_part_in_max_rad[g]++;

            for (int k = 0; k < 3; k++)
            {
#ifdef PERIODIC
                d = fof_periodic(P[j].Pos[k] - Group[g].Pos[k]);
#else // PERIODIC
                d = P[j].Pos[k] - Group[g].Pos[k];
#endif // PERIODIC
                new_group_pos[3 * g + k] += d * P[j].Mass;
            }
            new_group_mass[g] += P[j].Mass;
        }

        for (int g = 0; g < Ngroups; g++)
        {
            if (num_part_in_max_rad[g] <
                particle_threshold) /* the number of particles decreased below the limit, let's keep the  old value */
            {
                num_part_in_max_rad[g] = -1;
                groups_done++;
                continue;
            }

            num_part_in_max_rad[g] = 0;
            max_rad_sq[g] *= 0.9 * 0.9;

            for (int k = 0; k < 3; k++)
            {
                Group[g].Pos[k] += new_group_pos[3 * g + k] / new_group_mass[g];
                new_group_pos[3 * g + k] = 0;
            }
            new_group_mass[g] = 0;
        }

        /* debug */
        VERBOSE(2, "%i groups done.", groups_done);
    }

    myfree(new_group_mass);
    myfree(new_group_pos);

    cur_time = report_time();
    VERBOSE(2, "Shrinking sphere took %g s", cur_time - last_time);
}

/**
 * @brief Computes the tensor of inertia (ToI) and the maximum circular velocity (Vmax) for each Friends-of-Friends
 * (FoF) group.
 *
 * This function calculates various properties for each FoF group, including the tensor of inertia (ToI), angular
 * momentum, velocity dispersion, and maximum circular velocity (Vmax). It uses the positions and velocities of
 * particles belonging to the groups to compute these properties.
 *
 * @param particle_list Array of integers indicating the group index for each particle.
 *                      The value is -1 for particles not in any group.
 * @param n_part_in_group The number of particles in groups.
 */
void fof_ToI_and_Vmax(int *particle_list, int n_part_in_group)
{
    int i, j, k, g, pig;
    int i1, i2;
    int num_part_per_group[Ngroups];
    double xyz[3], vel_diff[3];
    double dist_sq;
    int particle_offset;
    double vmax, rmax;

    double vel_to_peculiar, H_of_a, atime; /* needed to estimate peculiar velocities */

    if (All.ComovingIntegrationOn)
    {
        H_of_a = hubble_function(All.Time);
        vel_to_peculiar = 1.0 / All.Time;
        atime = All.Time;
    }
    else
    {
        H_of_a = 0;
        vel_to_peculiar = atime = 1;
    }

    VERBOSE(2, "calculating ToI.");

    fof_sort_list sort_list[n_part_in_group]; /* list for all particles which are member of a group */

    /* check whether the number of particles per group is consistent with group length */
    for (int i = 0; i < Ngroups; i++)
    {
        num_part_per_group[i] = 0;
    }

    for (int i = 0; i < NumPart; i++)
    {
        if (particle_list[i] >= 0)
        {
            num_part_per_group[particle_list[i]]++;

            if (Group[particle_list[i]].GrNr != P[i].GrNr)
            {
                ENDRUN_MAIN("something wrong with communication.");
            }
        }
    }

    for (int i = 0; i < Ngroups; i++)
    {
        if (Group[i].Len != num_part_per_group[i])
        {
            VERBOSE_ALL(2, "Group %i, len %i, number of particles %i on task %i", i, Group[i].Len,
                        num_part_per_group[i], ThisTask);
            ENDRUN_MAIN("wrong number of particles moved");
        }
    }

    /* add the contribution of each particle to the ToI of the corresponding group */
    for (int i = 0; i < Ngroups; i++) /* initialize */
    {
        Group[i].VelDisp = 0;
        for (int k = 0; k < 9; k++)
        {
            Group[i].ToI[k] = 0;
        }
        for (int k = 0; k < 3; k++)
        {
            Group[i].AngMom[k] = 0;
        }
    }
    pig = 0;
    for (int j = 0; j < NumPart; j++)
    {
        g = particle_list[j];
        if (g < 0)
        {
            continue;
        }

        dist_sq = 0;
        for (int i = 0; i < 3; i++)
        {
#ifdef PERIODIC
            xyz[i] = fof_periodic(P[j].Pos[i] - Group[g].Pos[i]);

            /* debug */
            /*if(xyz[i] == 0)
               {
               printf("ThisTask %i, P[%i].Pos[%i] %g, Group[%i].CM[%i] %g\n", ThisTask, j, i, P[j].Pos[i], g, i,
               Group[g].Pos[i]); fflush(stdout);
               } */
#else // PERIODIC
            xyz[i] = P[j].Pos[i] - Group[g].Pos[i];
#endif // PERIODIC
            dist_sq += xyz[i] * xyz[i]; /* calculate the distance to the Group CM; needed for Rmax and Vmax */

            /* also calculate the group velocity dispersion here (using peculiar velocities) */
            vel_diff[i] = vel_to_peculiar * (P[j].Vel[i] - Group[g].Vel[i]) + H_of_a * atime * xyz[i];
            Group[g].VelDisp += P[j].Mass * vel_diff[i] * vel_diff[i];
        }

        if (pig >= n_part_in_group)
        {
            ENDRUN_MAIN("pig too large");
        }

        sort_list[pig].r = sqrt(dist_sq);
        sort_list[pig].mass = P[j].Mass;
        sort_list[pig].grnr = Group[g].GrNr;

        for (int i = 0; i < 3; i++)
        {
            i1 = (i + 1) % 3;
            i2 = (i + 2) % 3;
            Group[g].AngMom[i] += (xyz[i1] * vel_diff[i2] - xyz[i2] * vel_diff[i1]) * P[j].Mass;

            Group[g].ToI[3 * i + i] += P[j].Mass * (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
            for (int k = 0; k < 3; k++)
            {
                Group[g].ToI[3 * i + k] -= P[j].Mass * xyz[i] * xyz[k];
            }
        }
        pig++;
    }

    /* calculate Vmax and Rmax */
    qsort(sort_list, n_part_in_group, sizeof(fof_sort_list),
          fof_compare_sort_list); /* sort the particle list by GrNr and distance to the group center */
    qsort(Group, Ngroups, sizeof(group_properties), fof_compare_grnr); /* sort the group list by GrNr */

    /* debug : check if sorting was successful */
    for (int i = 1; i < n_part_in_group; i++)
    {
        if (sort_list[i - 1].grnr > sort_list[i].grnr)
        {
            ENDRUN_MAIN("sorting went wrong with sorting.");
        }

        if ((sort_list[i - 1].grnr == sort_list[i].grnr) && (sort_list[i - 1].r > sort_list[i].r))
        {
            ENDRUN_MAIN("sorting went wrong with sorting.");
        }
    }

    for (int g = 1; g < Ngroups; g++)
    {
        if (Group[g - 1].GrNr >= Group[g].GrNr)
        {
            ENDRUN_MAIN("something wrong with group order");
        }
    }
    /* end debug */

    particle_offset = 0;
    for (int g = 0; g < Ngroups; g++)
    {
        vmax = rmax = 0;

        /* debug */
        for (i = particle_offset; i < particle_offset + Group[g].Len; i++)
            if ((sort_list[i].grnr != Group[g].GrNr) || (i >= n_part_in_group))
                ENDRUN_MAIN("something wrong with offset.");
        /* end debug */

        /* compute cumulative mass */
        for (i = particle_offset + 1; i < particle_offset + Group[g].Len; i++)
        {
            sort_list[i].mass = sort_list[i - 1].mass + sort_list[i].mass;
        }

        /* find the maximum of the circular velocity profile */
        for (i = particle_offset + Group[g].Len - 1; i > particle_offset + 5; i--)
        {
            if (sort_list[i].mass / sort_list[i].r > vmax)
            {
                vmax = sort_list[i].mass / sort_list[i].r;
                rmax = sort_list[i].r;
            }
        }
        Group[g].Vmax = sqrt(All.GravityConstantInternal * vmax);
        Group[g].Rmax = rmax;

        particle_offset += Group[g].Len;

        /* finalize the velocity dispersion */
        Group[g].VelDisp /= 3; /*convert from 3D to 1D dispersion */
        Group[g].VelDisp /= Group[g].Mass;
        Group[g].VelDisp = sqrt(Group[g].VelDisp);
    }
}

/**
 * @brief Distributes Friends-of-Friends (FoF) group properties among tasks.
 *
 * This function distributes FoF group properties among different MPI tasks based on a specified mode.
 * It ensures that each group is sent to the appropriate task and receives groups from other tasks as needed.
 *
 * The function performs the following steps:
 * - Initializes send and receive counts for each task.
 * - Determines the target task for each group based on the specified mode.
 * - Counts the number of groups to be sent to each task and updates the send and receive offsets.
 * - Allocates memory for the groups to be sent.
 * - Prepares the groups for sending and removes them from the local list if they are sent.
 * - Uses MPI to send and receive group data between tasks.
 * - Updates the number of groups after receiving new groups.
 * - Frees the allocated memory.
 *
 * @param mode The mode for determining target task:
 *             - 1: Target task is determined by the group number modulo the total number of tasks.
 *             - 2: Target task is the original task assigned to the group.
 *
 */
void fof_distribute_groups(int mode)
{
    int i, nexport = 0, nimport = 0, target = -1, ngrp;
    group_properties *send_Group;

    /* count how many we have of each task */
    for (int i = 0; i < NTask; i++)
    {
        Send_count[i] = 0;
    }

    for (int i = 0; i < Ngroups; i++)
    {
        if (mode == 1)
        {
            target = (Group[i].GrNr - 1) % NTask;
        }
        if (mode == 2)
        {
            target = Group[i].Origintask;
        }
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
        if (mode == 1)
        {
            target = (Group[i].GrNr - 1) % NTask;
        }
        if (mode == 2)
        {
            target = Group[i].Origintask;
        }
        if (target != ThisTask)
        {
            send_Group[Send_offset[target] + Send_count[target]] = Group[i];
            Send_count[target]++;

            Group[i] = Group[Ngroups - 1];
            Ngroups--;
            i--;
        }
    }

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
            {
                // create individual tag for each pair of tasks
                int Tag = TAG_DENS_A + std::min(ThisTask, recvTask) * NTask + std::max(ThisTask, recvTask);
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
 * @brief Compute extended properties for Friends-of-Friends (FoF) groups.
 *
 * This function calculates additional properties for FoF groups such as the shrinking sphere center
 * and tensor of inertia (ToI) and Vmax. It involves distributing particles and groups across tasks,
 * identifying group membership for each particle, and then computing the properties.
 *
 * The function performs the following steps:
 * - Initializes timers and logs the start of extended property computation.
 * - Distributes tasks, particles, and groups among processors.
 * - Identifies the group each particle belongs to and maintains a list of particles in groups.
 * - Computes the shrinking sphere center and other extended properties (ToI and Vmax) for each group.
 * - Frees the allocated memory for the particle list.
 * - Redistributes particles and groups after computation.
 * - Re-sorts the groups according to group number.
 * - Logs the completion of extended property computation and stops the timers.
 */
void fof_compute_extended_properties(void)
{
    TIMER_PUSH(1, "FOF_COMP_EXT_PROP");
    int *particle_list;
    int n_part_in_group = 0;

    VERBOSE(3, "Computing extended FoF properties.");

    fof_get_tasks();
    subfind_distribute_particles(1);
    fof_distribute_groups(1);

    /* find the group each particle belongs to */
    particle_list = (int *)mymalloc("particle_list", sizeof(int) * NumPart);

    for (int j = 0; j < NumPart; j++)
    {
        particle_list[j] = -1;
        if (P[j].GrNr > TotNgroups)
        {
            continue;
        }

        for (int i = 0; i < Ngroups; i++)
        {
            if (P[j].GrNr == Group[i].GrNr)
            {
                particle_list[j] = i;
                n_part_in_group++;
                break;
            }
        }
    }

    fof_shrinking_sphere_center(particle_list);
    fof_ToI_and_Vmax(particle_list, n_part_in_group);

    myfree(particle_list);

    subfind_distribute_particles(2);
    fof_distribute_groups(2);

    /* sort the groups according to group-number again since we messed up the order before */
#ifdef ALTERNATIVE_PSORT
    fof_sort_Group_GrNr(Group, Ngroups);
#else // ALTERNATIVE_PSORT
    parallel_sort(Group, Ngroups, sizeof(group_properties), fof_compare_Group_GrNr);
#endif // ALTERNATIVE_PSORT

    VERBOSE(4, "Computing extended FoF properties done. ");
    TIMER_POP(1, "FOF_COMP_EXT_PROP");
}
#endif // FOF_EXTENDED_PROPERTIES

/**
 * @brief Save the global group catalogue by aggregating local group data.
 *
 * This function collects and processes local group data from all tasks, assigns global group numbers,
 * and writes the group catalogues to files. It ensures the groups are sorted and the group properties
 * are correctly assigned and written.
 *
 * The function performs the following steps:
 * - Starts a global sorting of group catalogues.
 * - Assigns group numbers and updates local and external group counts.
 * - Sorts the group list by local count, task, and MinID using a parallel sort.
 * - Assigns group numbers to the groups and ensures unique numbering across tasks.
 * - Gathers the total number of groups across all tasks and checks for consistency.
 * - Restores the group list to its original order.
 * - Assigns group numbers to the group properties array.
 * - Sorts the groups according to the assigned group numbers.
 * - Computes the offset values for each group.
 * - Prepares a list of particle IDs with assigned group numbers.
 * - Optionally computes extended properties and saves the group catalogue to files.
 *
 * @param num The snapshot number used to determine the output filenames.
 */
void fof_save_groups(int num)
{
    int i, start, lenloc, ngr, totlen;
    long long totNids;
    char buf[500];
    VERBOSE(3, "start global sorting of group catalogues.");
    TIMER_PUSH(3, "fof_save_groups");
    TIMER_PUSH(3, "global_sort");

    /* assign group numbers (at this point, both Group and FOF_GList are sorted by MinID) */
    for (int i = 0; i < NgroupsExt; i++)
    {
        FOF_GList[i].LocCount += FOF_GList[i].ExtCount; /* total length */
        FOF_GList[i].ExtCount = ThisTask; /* original task */
#ifdef DENSITY_SPLIT_BY_TYPE
        FOF_GList[i].LocDMCount += FOF_GList[i].ExtDMCount; /* total length */
        FOF_GList[i].ExtDMCount = ThisTask; /* not longer needed/used (hopefully) */
#endif // DENSITY_SPLIT_BY_TYPE
    }

#ifdef ALTERNATIVE_PSORT
    fof_sort_FOF_GList_LocCountTaskDiffMinID(FOF_GList, NgroupsExt);
#else // ALTERNATIVE_PSORT
    parallel_sort(FOF_GList, NgroupsExt, sizeof(fof_group_list), fof_compare_FOF_GList_LocCountTaskDiffMinID);
#endif // ALTERNATIVE_PSORT

    ngr = 0;
    for (int i = 0; i < NgroupsExt; i++)
    {
        if ((MyIDType)FOF_GList[i].ExtCount == FOF_GList[i].MinIDTask)
            ngr++;

        FOF_GList[i].GrNr = ngr;
    }

    MPI_Allgather(&ngr, 1, MPI_INT, Send_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    Send_offset[0] = 0;
    for (int j = 1; j < NTask; j++)
    {
        Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
    }

    for (int i = 0; i < NgroupsExt; i++)
    {
        FOF_GList[i].GrNr += Send_offset[ThisTask];
    }

    MPI_Allreduce(&ngr, &i, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    if (i != TotNgroups)
    {
        ENDRUN("i=%d", i);
    }

    /* bring the group list back into the original order */
#ifdef ALTERNATIVE_PSORT
    fof_sort_FOF_GList_ExtCountMinID(FOF_GList, NgroupsExt);
#else // ALTERNATIVE_PSORT
    parallel_sort(FOF_GList, NgroupsExt, sizeof(fof_group_list), fof_compare_FOF_GList_ExtCountMinID);
#endif // ALTERNATIVE_PSORT

    /* Assign the group numbers to the group properties array */
    start = 0;
    for (int i = 0; i < Ngroups; i++)
    {
        while (FOF_GList[start].MinID < Group[i].MinID)
        {
            start++;
            ENDRUN_IF(start >= NgroupsExt, "Start is greater than the number of groups.");
        }
        Group[i].GrNr = FOF_GList[start].GrNr;
    }

    /* sort the groups according to group-number */
#ifdef ALTERNATIVE_PSORT
    fof_sort_Group_GrNr(Group, Ngroups);
#else // ALTERNATIVE_PSORT
    parallel_sort(Group, Ngroups, sizeof(group_properties), fof_compare_Group_GrNr);
#endif // ALTERNATIVE_PSORT

    /* fill in the offset-values */
    totlen = 0;
    for (int i = 0; i < Ngroups; i++)
    {
        if (i > 0)
        {
            Group[i].Offset = Group[i - 1].Offset + Group[i - 1].Len;
        }
        else
        {
            Group[i].Offset = 0;
        }
        totlen += Group[i].Len;
    }

    MPI_Allgather(&totlen, 1, MPI_INT, Send_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    unsigned int *uoffset = (unsigned int *)mymalloc("uoffset", NTask * sizeof(unsigned int));

    uoffset[0] = 0;
    for (int j = 1; j < NTask; j++)
    {
        uoffset[j] = uoffset[j - 1] + Send_count[j - 1];
    }

    for (int i = 0; i < Ngroups; i++)
    {
        Group[i].Offset += uoffset[ThisTask];
    }

    myfree(uoffset);

    /* prepare list of ids with assigned group numbers */

    ID_list = (fof_id_list *)mymalloc("ID_list", sizeof(fof_id_list) * NumPart);

#if defined(SUBFIND) || defined(DISKPOT) || defined(FOF_EXTENDED_PROPERTIES)
    for (i = 0; i < NumPart; i++)
    {
        P[i].GrNr = TotNgroups + 1; /* will mark particles that are not in any group */
    }
#endif // defined(SUBFIND) || defined(DISKPOT) || defined(FOF_EXTENDED_PROPERTIES)

    start = 0, Nids = 0;
    for (int i = 0; i < NgroupsExt; i++)
    {
        while (FOF_PList[start].MinID < FOF_GList[i].MinID)
        {
            start++;
            ENDRUN_IF(start > NumPart, "Start is greater than the number of particles.");
        }

        ENDRUN_IF(FOF_PList[start].MinID != FOF_GList[i].MinID, "MinID does not match.");

        for (lenloc = 0; start + lenloc < NumPart;)
        {
            if (FOF_PList[start + lenloc].MinID == FOF_GList[i].MinID)
            {
                ID_list[Nids].GrNr = FOF_GList[i].GrNr;
                ID_list[Nids].ID = P[FOF_PList[start + lenloc].Pindex].ID;
#if defined(SUBFIND) || defined(DISKPOT) || defined(FOF_EXTENDED_PROPERTIES)
                P[FOF_PList[start + lenloc].Pindex].GrNr = FOF_GList[i].GrNr;
#endif // defined(SUBFIND) || defined(DISKPOT) || defined(FOF_EXTENDED_PROPERTIES)
                Nids++;
                lenloc++;
            }
            else
            {
                break;
            }
        }

        start += lenloc;
    }

#ifdef DISKPOT
    if (num < 0)
    {
        myfree(ID_list);
        return;
    }
#endif // DISKPOT

#ifdef FOF_EXTENDED_PROPERTIES
    /*
     * Compute the extended properties before saving.
     * Done here, because it needs the group number for each particle
     */
    fof_compute_extended_properties();
#endif // FOF_EXTENDED_PROPERTIES

    sumup_large_ints(1, &Nids, &totNids);

    MPI_Allgather(&Nids, 1, MPI_INT, Send_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    Send_offset[0] = 0;
    for (int j = 1; j < NTask; j++)
    {
        Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
    }

    if (totNids != TotNids)
    {
        ENDRUN("Task=%d Nids=%d totNids=%d TotNids=%d\n", ThisTask, Nids, (int)totNids, (int)TotNids);
    }

    /* sort the particle IDs according to group-number */
#ifdef ALTERNATIVE_PSORT
    fof_sort_ID_list_GrNrID(ID_list, Nids);
#else // ALTERNATIVE_PSORT
    parallel_sort(ID_list, Nids, sizeof(fof_id_list), fof_compare_ID_list_GrNrID);
#endif // ALTERNATIVE_PSORT

    TIMER_POP(3, "global_sort");
    VERBOSE(3, "Group catalogues globally sorted. Starting saving of group catalogue.");
    TIMER_PUSH(3, "saving_group_catalogue");

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
              "Fatal error. Number of processors must be a smaller or equal than `NumFilesWrittenInParallel'.");

#if defined(DO_FOF_OUTPUT) || !defined(SUBFIND)
    int nprocgroup = NTask / All.NumFilesWrittenInParallel;
    if ((NTask % All.NumFilesWrittenInParallel))
    {
        nprocgroup++;
    }
    int masterTask = (ThisTask / nprocgroup) * nprocgroup;
    for (int groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
        if (ThisTask == (masterTask + groupTask)) /* ok, it's this processor's turn */
        {
            fof_save_local_catalogue(num);
        }
        MPI_Barrier(MYMPI_COMM_WORLD); /* wait inside the group */
    }
#endif // defined(DO_FOF_OUTPUT) || !defined(SUBFIND)

    myfree(ID_list);

    TIMER_POP(3, "saving_group_catalogue");
    TIMER_POP(3, "fof_save_groups");
}

/**
 * @brief Save the local catalogue of groups to files.
 *
 * This function writes various properties of groups, such as length, mass, center of mass, velocity, and more, to files
 * in a specified output directory. The filenames and some properties written depend on preprocessor macros.
 *
 * The function performs the following steps:
 * - Opens a file for writing the group properties.
 * - Writes the basic information about the groups (e.g., number of groups, total number of groups, number of IDs).
 * - Allocates memory for and writes group properties such as:
 *     - Length of each group.
 *     - Offset into the ID list for each group.
 *     - Mass of each group.
 *     - Center of mass (CM) of each group.
 *     - Velocity of each group.
 * - Writes extended properties if `FOF_EXTENDED_PROPERTIES` is not defined:
 *     - Length and mass for each particle type in each group.
 *     - Velocity dispersion of each group.
 *     - Tensor of inertia of each group.
 *     - Maximum radius (Rmax) and maximum velocity (Vmax) of each group.
 *     - Position and angular momentum of each group.
 * - Conditionally writes additional properties based on defined macros:
 *     - `SFR` (Star Formation Rate)
 *     - `BLACK_HOLES` (Black Hole mass and accretion rate)
 *     - `LMB_CR_OUTPUT_SYNCHROTRON` (Synchrotron emission)
 * - Writes the IDs of particles in each group to a separate file if `FOF_WRITE_IDS` is defined.
 *
 * @param num The snapshot number.
 */
void fof_save_local_catalogue(int num)
{
    FILE *fd;
    float *mass, *cm, *vel;
    char fname[500];
    int *len;
    MyIDType *ids;

#ifdef LONG_OUTPUT_NUMBERS
    sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "group_tab", num, ThisTask);
#else // LONG_OUTPUT_NUMBERS
    sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, ThisTask);
#endif // LONG_OUTPUT_NUMBERS
    if (!(fd = fopen(fname, "w")))
    {
        ENDRUN("can't open file `%s`\n", fname);
    }

    my_fwrite(&Ngroups, sizeof(int), 1, fd);
    my_fwrite(&TotNgroups, sizeof(int), 1, fd);
    my_fwrite(&Nids, sizeof(int), 1, fd);
    my_fwrite(&TotNids, sizeof(long long), 1, fd);
    my_fwrite(&NTask, sizeof(int), 1, fd);

    /* group len */
    len = (int *)mymalloc("len", Ngroups * sizeof(int));
    for (int i = 0; i < Ngroups; i++)
    {
        len[i] = Group[i].Len;
    }
    my_fwrite(len, Ngroups, sizeof(int), fd);
    myfree(len);

    /* offset into id-list */
    len = (int *)mymalloc("len", Ngroups * sizeof(int));
    for (int i = 0; i < Ngroups; i++)
    {
        len[i] = Group[i].Offset;
    }
    my_fwrite(len, Ngroups, sizeof(int), fd);
    myfree(len);

    /* mass */
    mass = (float *)mymalloc("mass", Ngroups * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        mass[i] = Group[i].Mass;
    }
    my_fwrite(mass, Ngroups, sizeof(float), fd);
    myfree(mass);

    /* CM */
    cm = (float *)mymalloc("cm", Ngroups * 3 * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cm[i * 3 + j] = Group[i].CM[j];
        }
    }
    my_fwrite(cm, Ngroups, 3 * sizeof(float), fd);
    myfree(cm);

    /* vel */
    vel = (float *)mymalloc("vel", Ngroups * 3 * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            vel[i * 3 + j] = Group[i].Vel[j];
        }
    }
    my_fwrite(vel, Ngroups, 3 * sizeof(float), fd);
    myfree(vel);

#ifndef FOF_EXTENDED_PROPERTIES
    /* group len for each type */
    len = (int *)mymalloc("len", Ngroups * 6 * sizeof(int));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            len[i * 6 + j] = Group[i].LenType[j];
        }
    }
    my_fwrite(len, Ngroups, 6 * sizeof(int), fd);
    myfree(len);

    /* group mass for each type */
    mass = (float *)mymalloc("mass", Ngroups * 6 * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            mass[i * 6 + j] = Group[i].MassType[j];
        }
    }
    my_fwrite(mass, Ngroups, 6 * sizeof(float), fd);
    myfree(mass);

#else // FOF_EXTENDED_PROPERTIES

    float *veldisp, *tensor, *rmax, *vmax, *pos, *angmom;
    /* group velocity dispersion */
    veldisp = (float *)mymalloc("veldisp", Ngroups * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        veldisp[i] = Group[i].VelDisp;
    }
    my_fwrite(veldisp, Ngroups, sizeof(float), fd);
    myfree(veldisp);

    /* tensor of inertia */
    tensor = (float *)mymalloc("ToI", Ngroups * 9 * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            tensor[i * 9 + j] = Group[i].ToI[j];
        }
    }
    my_fwrite(tensor, Ngroups, 9 * sizeof(float), fd);
    myfree(tensor);

    /* Rmax and Vmax */
    rmax = (float *)mymalloc("Rmax", Ngroups * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        rmax[i] = Group[i].Rmax;
    }
    my_fwrite(rmax, Ngroups, sizeof(float), fd);
    myfree(rmax);

    vmax = (float *)mymalloc("Vmax", Ngroups * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        vmax[i] = Group[i].Vmax;
    }
    my_fwrite(vmax, Ngroups, sizeof(float), fd);
    myfree(vmax);

    /* pos */
    pos = (float *)mymalloc("pos", Ngroups * 3 * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            pos[i * 3 + j] = Group[i].Pos[j];
        }
    }
    my_fwrite(pos, Ngroups, 3 * sizeof(float), fd);
    myfree(pos);

    /* angular momentum */
    angmom = (float *)mymalloc("angmom", Ngroups * 3 * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            angmom[i * 3 + j] = Group[i].AngMom[j];
        }
    }
    my_fwrite(angmom, Ngroups, 3 * sizeof(float), fd);
    myfree(angmom);
#endif // FOF_EXTENDED_PROPERTIES

#ifdef SFR
    /* sfr */
    mass = (float *)mymalloc("mass", Ngroups * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        mass[i] = Group[i].Sfr;
    }
    my_fwrite(mass, Ngroups, sizeof(float), fd);
    myfree(mass);
#endif // SFR

#ifdef BLACK_HOLES
    /* BH_Mass */
    mass = (float *)mymalloc("mass", Ngroups * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        mass[i] = Group[i].BH_Mass;
    }
    my_fwrite(mass, Ngroups, sizeof(float), fd);
    myfree(mass);

    /* BH_Mdot */
    mass = (float *)mymalloc("mass", Ngroups * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        mass[i] = Group[i].BH_Mdot;
    }
    my_fwrite(mass, Ngroups, sizeof(float), fd);
    myfree(mass);
#endif // BLACK_HOLES

#ifdef LMB_CR_OUTPUT_SYNCHROTRON
    /* synchrotron emission */
    mass = (float *)mymalloc("mass", Ngroups * 8 * sizeof(float));
    for (int i = 0; i < Ngroups; i++)
    {
        for (int i_nu = 0; i_nu < 8; i_nu++)
        {
            mass[i * 8 + i_nu] = Group[i].Synchrotron[i_nu];
        }
    }
    my_fwrite(mass, Ngroups, sizeof(float), fd);
    myfree(mass);
#endif // LMB_CR_OUTPUT_SYNCHROTRON

    fclose(fd);

    ids = (MyIDType *)ID_list;
    for (int i = 0; i < Nids; i++)
    {
        ids[i] = ID_list[i].ID;
    }
#ifdef FOF_WRITE_IDS

#ifdef LONG_OUTPUT_NUMBERS
    sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "group_ids", num, ThisTask);
#else // LONG_OUTPUT_NUMBERS
    sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_ids", num, ThisTask);
#endif // LONG_OUTPUT_NUMBERS
    if (!(fd = fopen(fname, "w")))
    {
        ENDRUN("can't open file `%s`", fname);
    }

    my_fwrite(&Ngroups, sizeof(int), 1, fd);
    my_fwrite(&TotNgroups, sizeof(int), 1, fd);
    my_fwrite(&Nids, sizeof(int), 1, fd);
    my_fwrite(&TotNids, sizeof(long long), 1, fd);
    my_fwrite(&NTask, sizeof(int), 1, fd);
    my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd); /* this is the number of IDs in previous files */
    my_fwrite(ids, sizeof(MyIDType), Nids, fd);
    fclose(fd);
#endif // FOF_WRITE_IDS
}

/**
 * @brief Find the nearest dark matter particle for each particle in the simulation.
 *
 * This function searches for the nearest dark matter particle to each particle in the simulation,
 * using a tree-based search method. It allocates memory for the necessary arrays, performs the search
 * iteratively, and handles communication between tasks in a parallel environment. The search is repeated
 * for particles where the nearest neighbor could not be found within a specified distance.
 */
void fof_find_nearest_dmparticle(void)
{
    TIMER_PUSH(1, "FOF_NEAR_DM");
    int dummy, ndone, ndone_flag, place, nexport, nimport, npleft, iter;
    long long ntot;

    VERBOSE(2, "Start finding nearest dm-particle (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));

    fof_nearest_distance = (float *)mymalloc("fof_nearest_distance", sizeof(float) * NumPart);
    fof_nearest_hsml = (float *)mymalloc("fof_nearest_hsml", sizeof(float) * NumPart);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int n = 0; n < NumPart; n++)
    {
        if (((1 << P[n].Type) & (MyFOF_SECONDARY_LINK_TYPES)))
        {
            fof_nearest_distance[n] = 1.0e30;
            if (P[n].Type == 0)
            {
                if (P[n].Hsml / 5 > 2.0 * LinkLDM)
                {
                    fof_nearest_hsml[n] = 2.0 * LinkLDM;
                }
                else
                {
                    fof_nearest_hsml[n] = P[n].Hsml / 5;
                }
            }
            else
            {
                fof_nearest_hsml[n] = 2.0 * LinkLDM;
            }
        }
    }

    /* allocate buffers to arrange communication */

#ifdef AR_GREEN_TREE_FOF
    int NTaskTimesNumPart;
    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer;", NTaskTimesNumPart * sizeof(int));
#else // AR_GREEN_TREE_FOF
    Ngblist = (int *)mymalloc("Ngblist", NumPart * sizeof(int));
#endif // AR_GREEN_TREE_FOF

    All.BunchSize =
        (int)((All.BufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct fofdata_in) +
               sizeof(struct fofdata_out) + sizemax(sizeof(struct fofdata_in), sizeof(struct fofdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    iter = 0;
    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {
        int i = 0; /* beginn with this index */
#ifdef AR_GREEN_TREE_FOF
        int DoPrimaryOnGPU = 0;
        int should_drift = 1;
        int group_particles = 0;
        NextParticle = 0;
        int save_NextParticle;
#endif // AR_GREEN_TREE_FOF
        do
        {

#ifdef AR_GREEN_TREE_FOF
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextParticle = NextParticle;
#endif // AR_GREEN_TREE_FOF
            for (int j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
                Exportflag[j] = -1;
            }

#ifdef AR_GREEN_TREE_FOF
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                ar_fof_evaluate_primary_openmp(&thread_id);
            }
            //            ar_green_fof_evaluate_primary(&ThisThread, DoPrimaryOnGPU,  should_drift,group_particles); /*
            //            do local particles and prepare export list */

            nexport = Nexport;

            int ENDRUN_on_error = !group_particles;
            int is_send_receive_ok = prepare_sendrec_offset_send(save_NextParticle, ENDRUN_on_error);
            if (!is_send_receive_ok)
            {
                group_particles = 0;
                continue;
            }

            // printf("ThisTsk%d, poi NextParticle=%d NumPrt=%d\n",ThisTask,NextParticle,NumPart);
            i = NextParticle; // we fake we did  for loop
            nexport = Nexport;
            nimport = Nimport;

#else // AR_GREEN_TREE_FOF
            /* do local particles and prepare export list */
            for (nexport = 0; i < NumPart; i++)
            {
                if (((1 << P[i].Type) & (MyFOF_SECONDARY_LINK_TYPES)))
                {
                    if (fof_nearest_distance[i] > 1.0e29)
                    {

                        if (fof_find_nearest_dmparticle_evaluate(i, 0, &nexport, Send_count) < 0)
                            break;
                    }
                }
            }
            MYSORT_DATAINDEX(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);

            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
            nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
            for (int j = 0; j < NTask; j++)
            {
                nimport += Recv_count[j];

                if (j > 0)
                {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }
#endif // AR_GREEN_TREE_FOF

            FoFDataGet = (struct fofdata_in *)mymalloc("FoFDataGet", nimport * sizeof(struct fofdata_in));
            FoFDataIn = (struct fofdata_in *)mymalloc("FoFDataIn", nexport * sizeof(struct fofdata_in));

            VERBOSE(2, "still finding nearest... (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));

            for (int j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;

                FoFDataIn[j].Pos[0] = P[place].Pos[0];
                FoFDataIn[j].Pos[1] = P[place].Pos[1];
                FoFDataIn[j].Pos[2] = P[place].Pos[2];
                FoFDataIn[j].Hsml = fof_nearest_hsml[place];

                memcpy(FoFDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                       NODELISTLENGTH * sizeof(int));
            }

            /* exchange particle data */
            TIMER_PUSH(4, "fof_comm");
            templated_exchange_data_Sendrecv(FoFDataIn, FoFDataGet, TAG_DENS_A, Send_offset, Send_count, Recv_offset,
                                             Recv_count);
            TIMER_POP(4, "fof_comm");

            myfree(FoFDataIn);
            FoFDataResult = (struct fofdata_out *)mymalloc("FoFDataResult", nimport * sizeof(struct fofdata_out));
            FoFDataOut = (struct fofdata_out *)mymalloc("FoFDataOut", nexport * sizeof(struct fofdata_out));
#ifdef AR_GREEN_TREE_FOF
            // #pragma omp parallel
            //             ar_green_fof_evaluate_secondary(&ThisThread, should_drift);
            //         printf("ThisTask%d II nimport%d\n", ThisTask, nimport);
            for (int j = 0; j < nimport; j++)
            {
                fof_find_nearest_dmparticle_evaluate_orig(j, 1, &dummy, &dummy);
            }
#else // AR_GREEN_TREE_FOF
            for (int j = 0; j < nimport; j++)
            {
                fof_find_nearest_dmparticle_evaluate(j, 1, &dummy, &dummy);
            }
#endif // AR_GREEN_TREE_FOF

            TIMER_PUSH(4, "fof_comm");
            templated_exchange_data_Sendrecv(FoFDataResult, FoFDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POP(4, "fof_comm");

            for (int j = 0; j < nexport; j++)
            {
                place = DataIndexTable[j].Index;

                if (FoFDataOut[j].Distance < fof_nearest_distance[place])
                {
                    fof_nearest_distance[place] = FoFDataOut[j].Distance;
                    MinID[place] = FoFDataOut[j].MinID;
                    MinIDTask[place] = FoFDataOut[j].MinIDTask;
                }
            }

            if (i >= NumPart)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }

            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

            myfree(FoFDataOut);
            myfree(FoFDataResult);
            myfree(FoFDataGet);
        } while (ndone < NTask);

        /* do final operations on results */
        npleft = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : npleft)
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            if (((1 << P[i].Type) & (MyFOF_SECONDARY_LINK_TYPES)))
            {
                if (fof_nearest_distance[i] > 1.0e29)
                {
                    if (fof_nearest_hsml[i] < 4 * LinkLDM) /* we only search out to a maximum distance */
                    {
                        /* need to redo this particle */
                        npleft++;
                        fof_nearest_hsml[i] *= 2.0;
                        if (iter >= MAXITER - 10)
                        {
                            VERBOSE_ALL(2, "i=%d task=%d ID=%llu Hsml=%g  pos=(%g|%g|%g)", i, ThisTask,
                                        (unsigned long long)P[i].ID, fof_nearest_hsml[i], P[i].Pos[0], P[i].Pos[1],
                                        P[i].Pos[2]);
                        }
                    }
                    else
                    {
                        fof_nearest_distance[i] = 0; /* we not continue to search for this particle */
                    }
                }
            }
        }

        sumup_large_ints(1, &npleft, &ntot);

        if (ntot > 0)
        {
            iter++;
            if (iter > 0 && ThisTask == 0)
            {
                VERBOSE_ALL(2, "fof-nearest iteration %d: need to repeat for %lld particles.\n", iter, ntot);
            }

            ENDRUN_IF(iter > MAXITER, "failed to converge in fof-nearest\n");
        }
    } while (ntot > 0);

    myfree(DataNodeList);
    myfree(DataIndexTable);
#ifdef AR_GREEN_TREE_FOF
    myfree(RecyclingBuffer);
#endif // AR_GREEN_TREE_FOF
    myfree(Ngblist);

    myfree(fof_nearest_hsml);
    myfree(fof_nearest_distance);

    VERBOSE(2, "done finding nearest dm-particle.");
    TIMER_POP(1, "FOF_NEAR_DM");
}

#ifdef AR_GREEN_TREE_FOF
/**
 * @brief Evaluate the nearest dark matter particle for a given target in the AR_GREEN_TREE_FOF mode.
 *
 * This function finds the nearest dark matter particle to a specified target particle. It searches
 * through a list of neighbor particles and updates the nearest particle if a closer one is found.
 *
 * @param target The target particle index.
 * @param mode The mode of operation (0 or 1).
 * @param ngblist List of neighbor particles.
 * @param extended_numngb_inbox Number of neighbors in the extended inbox.
 * @param i_startnode Start node index for the search.
 * @param node The current node index.
 * @param i_node Node index for the iteration.
 * @return int Returns 0 on success, -1 on failure.
 */
int fof_find_nearest_dmparticle_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox,
                                         int *i_startnode, int node, int i_node)
#else // AR_GREEN_TREE_FOF
/**
 * @brief Evaluate the nearest dark matter particle for a given target.
 *
 * This function finds the nearest dark matter particle to a specified target particle. It searches
 * through a list of neighbor particles and updates the nearest particle if a closer one is found.
 *
 * @param target The target particle index.
 * @param mode The mode of operation (0 or 1).
 * @param nexport Pointer to the number of particles to export.
 * @param nsend_local Pointer to the number of particles to send locally.
 * @return int Returns 0 on success, -1 on failure.
 */
int fof_find_nearest_dmparticle_evaluate(int target, int mode, int *nexport, int *nsend_local)
#endif // AR_GREEN_TREE_FOF
{
    int j, index, listindex = 0;
    int startnode, numngb_inbox;
    double h, r2max;
    double dx, dy, dz, r2;
    MyLongDouble *pos;
    MyLongDouble xtmp;

    if (mode == 0)
    {
        pos = P[target].Pos;
        h = fof_nearest_hsml[target];
    }
    else
    {
        pos = FoFDataGet[target].Pos;
        h = FoFDataGet[target].Hsml;
    }

    index = -1;
    r2max = 1.0e30;

#ifndef AR_GREEN_TREE_FOF
    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = FoFDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }
#else // AR_GREEN_TREE_FOF
    startnode = node;
#endif // AR_GREEN_TREE_FOF

#ifdef AR_GREEN_TREE_FOF
    if (i_node > 0)
    {
        r2max = FoFDataResult[target].Distance * FoFDataResult[target].Distance;
    }
#endif // AR_GREEN_TREE_FOF

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
#ifdef AR_GREEN_TREE_FOF
            numngb_inbox = extended_numngb_inbox;
            startnode = *i_startnode;
#else // AR_GREEN_TREE_FOF
            numngb_inbox = ngb_treefind_fof_nearest(pos, h, target, &startnode, mode, nexport, nsend_local,
                                                    MyFOF_PRIMARY_LINK_TYPES);
#endif // AR_GREEN_TREE_FOF
            if (numngb_inbox < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb_inbox; n++)
            {
#ifdef AR_GREEN_TREE_FOF
                j = ngblist[n];
#else // AR_GREEN_TREE_FOF
                j = Ngblist[n];
#endif // AR_GREEN_TREE_FOF
                dx = NGB_PERIODIC_LONG_X(pos[0] - P[j].Pos[0]);
                dy = NGB_PERIODIC_LONG_Y(pos[1] - P[j].Pos[1]);
                dz = NGB_PERIODIC_LONG_Z(pos[2] - P[j].Pos[2]);
                r2 = dx * dx + dy * dy + dz * dz;
                //        if(r2<h*h)        printf("[DEBUG] ThisTsk%d, trget%d, ngb%d\n",ThisTask, target, j);
                if (r2 < r2max && r2 < h * h)
                {
                    index = j;
                    r2max = r2;
                }
            }
        }
#ifndef AR_GREEN_TREE_FOF // we open stuff outside
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = FoFDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
#endif // AR_GREEN_TREE_FOF
    }

    if (mode == 0)
    {
        if (index >= 0)
        {
            fof_nearest_distance[target] = sqrt(r2max);
            MinID[target] = MinID[Head[index]];
            MinIDTask[target] = MinIDTask[Head[index]];
            //    printf("[DEBUG] THistsk%d trget%d\n", ThisTask, target);
        }
    }
    else
    {
        if (index >= 0)
        {
            FoFDataResult[target].Distance = sqrt(r2max);
            FoFDataResult[target].MinID = MinID[Head[index]];
            FoFDataResult[target].MinIDTask = MinIDTask[Head[index]];
        }
        else
        {
            FoFDataResult[target].Distance = 2.0e30;
        }
    }
    return 0;
}

#ifdef BLACK_HOLES
/**
 * @brief Seed black holes in dark matter halos.
 *
 * This function seeds black holes in dark matter halos based on specific criteria.
 * It assigns particles as black holes and sets their properties such as mass and spin.
 *
 * The seeding criteria depend on various parameters such as the mass fraction of star particles,
 * gas fraction, and specific conditions related to the dark matter halo properties.
 */
void fof_make_black_holes(void)
{
    int ntot;
    int nexport, nimport;
    int *import_indices, *export_indices;
    float *import_fofmass, *export_fofmass;
    float *import_vx, *export_vx;
    float *import_vy, *export_vy;
    float *import_vz, *export_vz;

    double massDMpart;

    if (All.MassTable[1] > 0)
    {
        massDMpart = All.MassTable[1];
    }
    else
    {
        massDMpart = All.massDMpart;
    }

    for (int n = 0; n < NTask; n++)
    {
        Send_count[n] = 0;
    }

    for (int i = 0; i < Ngroups; i++)
    {
        int doit = 0;
        if (All.BlackHoleSeedStarMassFraction == 0)
        {
            if (Group[i].LenType[1] * massDMpart >=
                (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.BlackHoleFoFMinMassForNewSeed)
            {
                doit = 1;
            }
        }
        else
        {
            if (Group[i].MassType[4] > All.BlackHoleSeedStarMassFraction * All.BlackHoleFoFMinMassForNewSeed &&
                Group[i].MaxDens < 1e30)
            {
                if (Group[i].LenType[2] == 0 &&
                    Group[i].MassType[4] / Group[i].MassType[1] > All.BlackHoleSeedDMFraction &&
                    Group[i].MassType[0] / Group[i].MassType[4] > All.BlackHoleSeedGasFraction)
                {
                    doit = 1;
                }
                else
                {
                    if (All.BlackHoleDetails >= 2)
                    {
                        fprintf(FdBlackHolesDetails,
                                "BHSEED: Seeding BH in group at (%e,%e,%e) not done, f_dm=%e (need >%e) and f_gas=%e "
                                "(need >%e) and contamination=%d (need 0)\n",
                                Group[i].CM[0], Group[i].CM[1], Group[i].CM[2],
                                Group[i].MassType[4] / Group[i].MassType[1], All.BlackHoleSeedDMFraction,
                                Group[i].MassType[0] / Group[i].MassType[4], All.BlackHoleSeedGasFraction,
                                Group[i].LenType[2]);
                    }
                }
            }
        }

        if (Group[i].LenType[5] == 0 && doit == 1)
        {
            if (All.BlackHoleDetails >= 2)
            {
                fprintf(FdBlackHolesDetails,
                        "BHSEED: Seeding BH in group at (%e,%e,%e) DONE, f_dm=%e (need >%e) and f_gas=%e (need >%e)\n",
                        Group[i].CM[0], Group[i].CM[1], Group[i].CM[2], Group[i].MassType[4] / Group[i].MassType[1],
                        All.BlackHoleSeedDMFraction, Group[i].MassType[0] / Group[i].MassType[4],
                        All.BlackHoleSeedGasFraction);
            }

            if (Group[i].index_maxdens >= 0)
            {
                Send_count[Group[i].task_maxdens]++;
            }
        }
    }

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    nimport = nexport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
    for (int j = 0; j < NTask; j++)
    {
        nexport += Send_count[j];
        nimport += Recv_count[j];

        if (j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

    import_indices = (int *)mymalloc("import_indices", nimport * sizeof(int));
    export_indices = (int *)mymalloc("export_indices", nexport * sizeof(int));
    if (All.BlackHoleSeedStarMassFraction > 0)
    {
        import_fofmass = (float *)mymalloc("import_fofmass", nimport * sizeof(float));
        export_fofmass = (float *)mymalloc("export_fofmass", nexport * sizeof(float));
        import_vx = (float *)mymalloc("import_vx", nimport * sizeof(float));
        export_vx = (float *)mymalloc("export_vx", nexport * sizeof(float));
        import_vy = (float *)mymalloc("import_vy", nimport * sizeof(float));
        export_vy = (float *)mymalloc("export_vy", nexport * sizeof(float));
        import_vz = (float *)mymalloc("import_vz", nimport * sizeof(float));
        export_vz = (float *)mymalloc("export_vz", nexport * sizeof(float));
    }

    for (int n = 0; n < NTask; n++)
    {
        Send_count[n] = 0;
    }

    for (int i = 0; i < Ngroups; i++)
    {
        int doit = 0;
        if (All.BlackHoleSeedStarMassFraction == 0)
        {
            if (Group[i].LenType[1] * massDMpart >=
                (All.Omega0 - All.OmegaBaryon) / All.Omega0 * All.BlackHoleFoFMinMassForNewSeed)
            {
                doit = 1;
            }
        }
        else
        {
            if (Group[i].MassType[4] > All.BlackHoleSeedStarMassFraction * All.BlackHoleFoFMinMassForNewSeed &&
                Group[i].LenType[2] == 0 && Group[i].MassType[4] / Group[i].MassType[1] > 0.05 &&
                Group[i].MassType[0] / Group[i].MassType[4] > All.BlackHoleSeedGasFraction && Group[i].MaxDens < 1e30)
            {
                doit = 1;
            }
        }

        if (Group[i].LenType[5] == 0 && doit == 1)
        {
            if (All.BlackHoleSeedStarMassFraction > 0)
            {
                if (Group[i].index_maxdens >= 0)
                {
                    export_fofmass[Send_offset[Group[i].task_maxdens] + Send_count[Group[i].task_maxdens]] =
                        Group[i].MassType[4];
                    export_vx[Send_offset[Group[i].task_maxdens] + Send_count[Group[i].task_maxdens]] = Group[i].Vel[0];
                    export_vy[Send_offset[Group[i].task_maxdens] + Send_count[Group[i].task_maxdens]] = Group[i].Vel[1];
                    export_vz[Send_offset[Group[i].task_maxdens] + Send_count[Group[i].task_maxdens]] = Group[i].Vel[2];
                }
            }
            if (Group[i].index_maxdens >= 0)
            {
                export_indices[Send_offset[Group[i].task_maxdens] + Send_count[Group[i].task_maxdens]++] =
                    Group[i].index_maxdens;
            }
        }
    }

    memcpy(&import_indices[Recv_offset[ThisTask]], &export_indices[Send_offset[ThisTask]],
           Send_count[ThisTask] * sizeof(int));

    if (All.BlackHoleSeedStarMassFraction > 0)
    {
        memcpy(&import_fofmass[Recv_offset[ThisTask]], &export_fofmass[Send_offset[ThisTask]],
               Send_count[ThisTask] * sizeof(float));
        memcpy(&import_vx[Recv_offset[ThisTask]], &export_vx[Send_offset[ThisTask]],
               Send_count[ThisTask] * sizeof(float));
        memcpy(&import_vy[Recv_offset[ThisTask]], &export_vy[Send_offset[ThisTask]],
               Send_count[ThisTask] * sizeof(float));
        memcpy(&import_vz[Recv_offset[ThisTask]], &export_vz[Send_offset[ThisTask]],
               Send_count[ThisTask] * sizeof(float));
    }

    for (int level = 1; level < (1 << PTask); level++)
    {
        int recvTask = ThisTask ^ level;

        if (recvTask < NTask)
            MPI_Sendrecv(&export_indices[Send_offset[recvTask]], Send_count[recvTask] * sizeof(int), MPI_BYTE, recvTask,
                         TAG_FOF_E, &import_indices[Recv_offset[recvTask]], Recv_count[recvTask] * sizeof(int),
                         MPI_BYTE, recvTask, TAG_FOF_E, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (All.BlackHoleSeedStarMassFraction > 0)
        {
            if (recvTask < NTask)
            {
                MPI_Sendrecv(&export_fofmass[Send_offset[recvTask]], Send_count[recvTask] * sizeof(float), MPI_BYTE,
                             recvTask, TAG_FOF_M, &import_fofmass[Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(float), MPI_BYTE, recvTask, TAG_FOF_M, MYMPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                MPI_Sendrecv(&export_vx[Send_offset[recvTask]], Send_count[recvTask] * sizeof(float), MPI_BYTE,
                             recvTask, TAG_FOF_VX, &import_vx[Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(float), MPI_BYTE, recvTask, TAG_FOF_VX, MYMPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                MPI_Sendrecv(&export_vy[Send_offset[recvTask]], Send_count[recvTask] * sizeof(float), MPI_BYTE,
                             recvTask, TAG_FOF_VY, &import_vy[Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(float), MPI_BYTE, recvTask, TAG_FOF_VY, MYMPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                MPI_Sendrecv(&export_vz[Send_offset[recvTask]], Send_count[recvTask] * sizeof(float), MPI_BYTE,
                             recvTask, TAG_FOF_VZ, &import_vz[Recv_offset[recvTask]],
                             Recv_count[recvTask] * sizeof(float), MPI_BYTE, recvTask, TAG_FOF_VZ, MYMPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Allreduce(&nimport, &ntot, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    VERBOSE(2, "\nMaking %d new black hole particles\n", ntot);

    All.TotBHs += ntot;

    for (int n = 0; n < nimport; n++)
    {
        ENDRUN_IF(P[import_indices[n]].Type != 0, "Type uneqal to 0.");

#ifdef LT_STELLAREVOLUTION
        if (P[import_indices[n]].Type == 4)
        {
            int i = P[import_indices[n]].pt.MetID;
            MetP[i] = MetP[N_stars - 1];
            P[MetP[i].PID].pt.MetID = i;
            N_stars--;
        }
#endif // LT_STELLAREVOLUTION

        P[import_indices[n]].Type = 5; /* make it a black hole particle */

#ifndef DO_NOT_USE_LARGE_DYNAMICAL_MASS
        P[import_indices[n]].Mass = massDMpart;
#endif // DO_NOT_USE_LARGE_DYNAMICAL_MASS
        /*
           printf("Task %d: Seeding BH is at at (%e,%e,%e) with pot=%e\n",ThisTask,P[import_indices[n]].Pos[0],
           P[import_indices[n]].Pos[1],P[import_indices[n]].Pos[2],P[import_indices[n]].Potential);
         */
        ENDRUN_IF(N_BHs + 1 > All.MaxPartBH,
                  "On Task=%d with NumPart=%d I tried to make too many BH. Sorry, no space left...(All.MaxPartBH=%d)",
                  ThisTask, NumPart, All.MaxPartBH);

        P[import_indices[n]].pt.BHID = N_BHs;
        BHP[N_BHs].PID = import_indices[n];
        N_BHs++;

#ifdef SFR
        BPP(import_indices[n]).StellarAge = All.Time;
#endif // SFR
        if (All.BlackHoleSeedStarMassFraction > 0)
        {
            BPP(import_indices[n]).BH_Mass = All.SeedBlackHoleMass * import_fofmass[n] /
                                             (All.BlackHoleSeedStarMassFraction * All.BlackHoleFoFMinMassForNewSeed);
        }
        else
        {
            BPP(import_indices[n]).BH_Mass = All.SeedBlackHoleMass;
        }

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
        BPP(import_indices[n]).BH_AccrCounter = 0;
        BPP(import_indices[n]).BH_Mass_Sum = 0;
        BPP(import_indices[n]).BH_accr_mass_Sum = 0;
        BPP(import_indices[n]).BH_dt_Sum = 0;
        for (int k = 0; k < 3; k++)
        {
            BPP(import_indices[n]).Jgas_in_Kernel_Sum[k] = 0;
        }
        MyAtLeastDouble a = 0;
        MyAtLeastDouble risco = 0;
        a = All.SeedBlackHoleSpin;
        MyAtLeastDouble BH_Spin_mod = a * All.GravityConstantInternal * (pow(BPP(import_indices[n]).BH_Mass, 2)) /
                                      (C_LIGHT / All.UnitVelocity_in_cm_per_s);
        BPP(import_indices[n]).BH_Spin[0] = BH_Spin_mod * 0;
        BPP(import_indices[n]).BH_Spin[1] = BH_Spin_mod * 0;
        BPP(import_indices[n]).BH_Spin[2] = BH_Spin_mod * 1;

        MyAtLeastDouble z1 = 1. + pow((1. - pow(a, 2.)), 1. / 3.) * (pow((1. + a), 1. / 3.) + pow((1. - a), 1. / 3.));
        MyAtLeastDouble z2 = pow((3 * pow(a, 2.) + pow(z1, 2.)), 1. / 2.);
        if (a <= 0.)
        {
            risco = 3. + z2 + pow(((3. - z1) * (3. + z1 + 2 * z2)), 1. / 2.);
        }
        else
        {
            risco = 3. + z2 - pow(((3. - z1) * (3. + z1 + 2 * z2)), 1. / 2.);
        }
        BPP(import_indices[n]).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION

#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
        /* Initialize cone-axis direction for AGN wind kick ... RANDOM direction for every BH created. */

        double theta_Dir, phi_Dir;

        theta_Dir = acos(2 * get_random_number(P[import_indices[n]].ID + 1) - 1);
        phi_Dir = 2 * M_PI * get_random_number(P[import_indices[n]].ID + 2);

        BPP(import_indices[n]).UnitVector_AGN_Wind[0] = sin(theta_Dir) * cos(phi_Dir);
        BPP(import_indices[n]).UnitVector_AGN_Wind[1] = sin(theta_Dir) * sin(phi_Dir);
        BPP(import_indices[n]).UnitVector_AGN_Wind[2] = cos(theta_Dir);

        fprintf(FdBlackHolesDetails,
                "PB_BH_KINETIC_FEEDBACK New BH: ID=%ld, BH_Mass=%e, "
                "UnitVector_AGN_Wind=%e  %e  %e \n",
                P[import_indices[n]].ID, (float)BPP(import_indices[n]).BH_Mass,
                (float)BPP(import_indices[n]).UnitVector_AGN_Wind[0],
                (float)BPP(import_indices[n]).UnitVector_AGN_Wind[1],
                (float)BPP(import_indices[n]).UnitVector_AGN_Wind[2]);
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE

        BPP(import_indices[n]).BH_Mdot = 0;
        BPP(import_indices[n]).BH_CountProgs = 1;

        Particles_converted++;
        TimeBinCountSph[P[import_indices[n]].TimeBin]--;
    }

    if (All.BlackHoleSeedStarMassFraction > 0)
    {
        myfree(export_vz);
        myfree(import_vz);
        myfree(export_vy);
        myfree(import_vy);
        myfree(export_vx);
        myfree(import_vx);
        myfree(export_fofmass);
        myfree(import_fofmass);
    }
    myfree(export_indices);
    myfree(import_indices);
}
#endif // BLACK_HOLES

/**
 * @brief Applies periodic boundary conditions to a value.
 *
 * This function applies periodic boundary conditions to a given value based on the periodic box size
 * defined by the variable All.BoxSize.
 *
 * @param x The value to which periodic boundary conditions are applied.
 * @return The value with periodic boundary conditions applied.
 */
double fof_periodic(double x)
{
    if (x >= 0.5 * All.BoxSize)
    {
        x -= All.BoxSize;
    }
    if (x < -0.5 * All.BoxSize)
    {
        x += All.BoxSize;
    }
    return x;
}

/**
 * @brief Wraps a value into the periodic box.
 *
 * This function wraps a given value into the periodic box defined by the variable All.BoxSize.
 *
 * @param x The value to be wrapped.
 * @return The wrapped value within the periodic box.
 */
double fof_periodic_wrap(double x)
{
    while (x >= All.BoxSize)
    {
        x -= All.BoxSize;
    }
    while (x < 0)
    {
        x += All.BoxSize;
    }
    return x;
}

/**
 * @brief Compares two elements of type fof_particle_list based on MinID.
 *
 * This comparison function is used for sorting elements of type fof_particle_list based on their MinID values.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the MinID of the first argument is considered to be
 * less than, equal to, or greater than the MinID of the second, respectively.
 */
int fof_compare_FOF_PList_MinID(const void *a, const void *b)
{
    if (((struct fof_particle_list *)a)->MinID < ((struct fof_particle_list *)b)->MinID)
    {
        return -1;
    }

    if (((struct fof_particle_list *)a)->MinID > ((struct fof_particle_list *)b)->MinID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type fof_group_list based on MinID.
 *
 * This comparison function is used for sorting elements of type fof_group_list based on their MinID values.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the MinID of the first argument is considered to be
 * less than, equal to, or greater than the MinID of the second, respectively.
 */
int fof_compare_FOF_GList_MinID(const void *a, const void *b)
{
    if (((fof_group_list *)a)->MinID < ((fof_group_list *)b)->MinID)
    {
        return -1;
    }

    if (((fof_group_list *)a)->MinID > ((fof_group_list *)b)->MinID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type fof_group_list based on MinIDTask.
 *
 * This comparison function is used for sorting elements of type fof_group_list based on their MinIDTask values.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the MinIDTask of the first argument is considered to
 * be less than, equal to, or greater than the MinIDTask of the second, respectively.
 */
int fof_compare_FOF_GList_MinIDTask(const void *a, const void *b)
{
    if (((fof_group_list *)a)->MinIDTask < ((fof_group_list *)b)->MinIDTask)
    {
        return -1;
    }

    if (((fof_group_list *)a)->MinIDTask > ((fof_group_list *)b)->MinIDTask)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type fof_group_list based on LocCount, MinID, and ExtCount - MinIDTask.
 *
 * This comparison function is used for sorting elements of type fof_group_list.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the LocCount, MinID, and ExtCount - MinIDTask of the
 * first argument are considered to be less than, equal to, or greater than the corresponding values of the second,
 * respectively.
 */
int fof_compare_FOF_GList_LocCountTaskDiffMinID(const void *a, const void *b)
{
    if (((fof_group_list *)a)->LocCount > ((fof_group_list *)b)->LocCount)
    {
        return -1;
    }

    if (((fof_group_list *)a)->LocCount < ((fof_group_list *)b)->LocCount)
    {
        return +1;
    }

    if (((fof_group_list *)a)->MinID < ((fof_group_list *)b)->MinID)
    {
        return -1;
    }

    if (((fof_group_list *)a)->MinID > ((fof_group_list *)b)->MinID)
    {
        return +1;
    }

    if ((((fof_group_list *)a)->ExtCount - ((fof_group_list *)a)->MinIDTask) <
        (((fof_group_list *)b)->ExtCount - ((fof_group_list *)b)->MinIDTask))
    {
        return -1;
    }

    if ((((fof_group_list *)a)->ExtCount - ((fof_group_list *)a)->MinIDTask) >
        (((fof_group_list *)b)->ExtCount - ((fof_group_list *)b)->MinIDTask))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type fof_group_list based on ExtCount and MinID.
 *
 * This comparison function is used for sorting elements of type fof_group_list.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the ExtCount and MinID of the first argument are
 * considered to be less than, equal to, or greater than the ExtCount and MinID of the second, respectively.
 */
int fof_compare_FOF_GList_ExtCountMinID(const void *a, const void *b)
{
    if (((fof_group_list *)a)->ExtCount < ((fof_group_list *)b)->ExtCount)
    {
        return -1;
    }

    if (((fof_group_list *)a)->ExtCount > ((fof_group_list *)b)->ExtCount)
    {
        return +1;
    }

    if (((fof_group_list *)a)->MinID < ((fof_group_list *)b)->MinID)
    {
        return -1;
    }

    if (((fof_group_list *)a)->MinID > ((fof_group_list *)b)->MinID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type group_properties based on MinID.
 *
 * This comparison function is used for sorting elements of type group_properties.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the MinID of the first argument is considered to be
 * less than, equal to, or greater than the MinID of the second, respectively.
 */
int fof_compare_Group_MinID(const void *a, const void *b)
{
    if (((group_properties *)a)->MinID < ((group_properties *)b)->MinID)
    {
        return -1;
    }

    if (((group_properties *)a)->MinID > ((group_properties *)b)->MinID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type group_properties based on GrNr.
 *
 * This comparison function is used for sorting elements of type group_properties.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the GrNr of the first argument is considered to be
 * less than, equal to, or greater than the GrNr of the second, respectively.
 */
int fof_compare_Group_GrNr(const void *a, const void *b)
{
    if (((group_properties *)a)->GrNr < ((group_properties *)b)->GrNr)
    {
        return -1;
    }

    if (((group_properties *)a)->GrNr > ((group_properties *)b)->GrNr)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type group_properties based on MinIDTask.
 *
 * This comparison function is used for sorting elements of type group_properties.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the MinIDTask of the first argument is considered to
 * be less than, equal to, or greater than the MinIDTask of the second, respectively.
 */
int fof_compare_Group_MinIDTask(const void *a, const void *b)
{
    if (((group_properties *)a)->MinIDTask < ((group_properties *)b)->MinIDTask)
    {
        return -1;
    }

    if (((group_properties *)a)->MinIDTask > ((group_properties *)b)->MinIDTask)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two elements of type fof_id_list based on GrNr and ID.
 *
 * This comparison function is used for sorting elements of type fof_id_list.
 *
 * @param a Pointer to the first element to compare.
 * @param b Pointer to the second element to compare.
 * @return An integer less than, equal to, or greater than zero if the first argument is considered to be less than,
 * equal to, or greater than the second, respectively.
 */
int fof_compare_ID_list_GrNrID(const void *a, const void *b)
{
    if (((fof_id_list *)a)->GrNr < ((fof_id_list *)b)->GrNr)
    {
        return -1;
    }

    if (((fof_id_list *)a)->GrNr > ((fof_id_list *)b)->GrNr)
    {
        return +1;
    }

    if (((fof_id_list *)a)->ID < ((fof_id_list *)b)->ID)
    {
        return -1;
    }

    if (((fof_id_list *)a)->ID > ((fof_id_list *)b)->ID)
    {
        return +1;
    }

    return 0;
}

#ifdef SUBFIND_READ_FOF /* read already existing FOF instead of recomputing it */
/**
 * @brief Reads the Friends-of-Friends (FoF) group catalogues for a given snapshot number.
 *
 * This function reads FoF group catalogues from files and assigns group numbers to particles.
 *
 * @param num The snapshot number.
 */
void read_fof(int num)
{
    FILE *fd;
    char fname[500];
    float *mass, *cm;
    int i, j, ntask, *len, count;
    MyIDType *ids;
    int *list_of_ngroups, *list_of_nids, *list_of_allgrouplen;
    int *recvoffset;
    int grnr, ngrp, imax1, imax2;
    int nprocgroup, masterTask, groupTask, nid_previous;
    int fof_compare_P_SubNr(const void *a, const void *b);

    VERBOSE(2, "\nTrying to read preexisting FoF group catalogues...  (presently allocated=%g MB)",
            AllocatedBytes / (1024.0 * 1024.0));

    domain_Decomposition(1, 0);

    force_treefree();

    /* start reading of group catalogue */
    if (ThisTask == 0)
    {
#ifdef LONG_OUTPUT_NUMBERS
        sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "group_tab", num, 0);
#else // LONG_OUTPUT_NUMBERS
        sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, 0);
#endif // LONG_OUTPUT_NUMBERS
        if (!(fd = fopen(fname, "r")))
        {
            ENDRUN_MAIN("can't read file `%s`", fname);
        }

        my_fread(&Ngroups, sizeof(int), 1, fd);
        my_fread(&TotNgroups, sizeof(int), 1, fd);
        my_fread(&Nids, sizeof(int), 1, fd);
        my_fread(&TotNids, sizeof(long long), 1, fd);
        my_fread(&ntask, sizeof(int), 1, fd);
        fclose(fd);
    }

    MPI_Bcast(&ntask, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&TotNgroups, 1, MPI_INT, 0, MYMPI_COMM_WORLD);
    MPI_Bcast(&TotNids, sizeof(long long), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    TIMER_PUSH(3, "reading");
    if (NTask != ntask)
    {
        VERBOSE(2, "number of files (%d) in group catalogues does not match MPI-Tasks, I'm working around this.\n",
                ntask);

#ifdef NO_KD_FOF_MEM_TRICK
        NAllocatedGroups = (TotNgroups / NTask + NTask);
        Group = (group_properties *)mymalloc("Group", sizeof(group_properties) * (TotNgroups / NTask + NTask));
#else // NO_KD_FOF_MEM_TRICK
        NAllocatedGroups = (TotNgroups / NTask + NTask);
        Group = (group_properties *)mymalloc(&Group, "Group", sizeof(group_properties) * (TotNgroups / NTask + NTask));
#endif // NO_KD_FOF_MEM_TRICK
        ID_list = mymalloc("ID_list", (TotNids / NTask + NTask) * sizeof(fof_id_list));

        int filenr, target, ngroups, nids, nsend, stored;

        int *ngroup_to_get = mymalloc("ngroup_to_get", NTask * sizeof(NTask));
        int *nids_to_get = mymalloc("nids_to_get", NTask * sizeof(NTask));
        int *ngroup_obtained = mymalloc("ngroup_obtained", NTask * sizeof(NTask));
        int *nids_obtained = mymalloc("nids_obtained", NTask * sizeof(NTask));

        for (int i = 0; i < NTask; i++)
        {
            ngroup_obtained[i] = nids_obtained[i] = 0;
        }

        for (int i = 0; i < NTask - 1; i++)
        {
            ngroup_to_get[i] = TotNgroups / NTask;
        }
        ngroup_to_get[NTask - 1] = TotNgroups - (NTask - 1) * (TotNgroups / NTask);

        for (int i = 0; i < NTask - 1; i++)
        {
            nids_to_get[i] = (int)(TotNids / NTask);
        }
        nids_to_get[NTask - 1] = (int)(TotNids - (NTask - 1) * (TotNids / NTask));

        Ngroups = ngroup_to_get[ThisTask];
        Nids = nids_to_get[ThisTask];

        if (ThisTask == 0)
        {
            for (int filenr = 0; filenr < ntask; filenr++)
            {
#ifdef LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "group_tab", num, filenr);
#else // LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, filenr);
#endif // LONG_OUTPUT_NUMBERS
                if (!(fd = fopen(fname, "r")))
                {
                    ENDRUN_MAIN("can't read file `%s`\n", fname);
                }

                VERBOSE(2, "reading '%s'\n", fname);

                my_fread(&ngroups, sizeof(int), 1, fd);
                my_fread(&TotNgroups, sizeof(int), 1, fd);
                my_fread(&nids, sizeof(int), 1, fd);
                my_fread(&TotNids, sizeof(long long), 1, fd);
                my_fread(&ntask, sizeof(int), 1, fd);

                group_properties *tmpGroup =
                    (group_properties *)mymalloc("tmpGroup", sizeof(group_properties) * ngroups);

                /* group len */
                len = mymalloc("len", ngroups * sizeof(int));
                my_fread(len, ngroups, sizeof(int), fd);
                for (int i = 0; i < ngroups; i++)
                {
                    tmpGroup[i].Len = len[i];
                }
                myfree(len);

                /* offset into id-list */
                len = mymalloc("len", ngroups * sizeof(int));
                my_fread(len, ngroups, sizeof(int), fd);
                for (int i = 0; i < ngroups; i++)
                {
                    tmpGroup[i].Offset = len[i];
                }
                myfree(len);

                /* mass */
                mass = mymalloc("mass", ngroups * sizeof(float));
                my_fread(mass, ngroups, sizeof(float), fd);
                for (int i = 0; i < ngroups; i++)
                {
                    tmpGroup[i].Mass = mass[i];
                }
                myfree(mass);

                /* CM */
                cm = mymalloc("cm", ngroups * 3 * sizeof(float));
                my_fread(cm, ngroups, 3 * sizeof(float), fd);
                for (int i = 0; i < ngroups; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        tmpGroup[i].CM[j] = cm[i * 3 + j];
                    }
                }
                myfree(cm);

                fclose(fd);

                target = 0;
                stored = 0;
                while (ngroups > 0)
                {
                    while (ngroup_to_get[target] == 0)
                    {
                        target++;
                    }

                    if (ngroups > ngroup_to_get[target])
                    {
                        nsend = ngroup_to_get[target];
                    }
                    else
                    {
                        nsend = ngroups;
                    }

                    if (target == 0)
                    {
                        memcpy(&Group[ngroup_obtained[target]], &tmpGroup[stored], nsend * sizeof(group_properties));
                    }
                    else
                    {
                        MPI_Send(&nsend, 1, MPI_INT, target, TAG_N, MYMPI_COMM_WORLD);
                        MPI_Send(&tmpGroup[stored], nsend * sizeof(group_properties), MPI_BYTE, target, TAG_PDATA,
                                 MYMPI_COMM_WORLD);
                    }

                    ngroup_to_get[target] -= nsend;
                    ngroup_obtained[target] += nsend;
                    ngroups -= nsend;
                    stored += nsend;
                }
                myfree(tmpGroup);
            }

            /**** now ids ****/
            for (int filenr = 0; filenr < ntask; filenr++)
            {
#ifdef LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "group_ids", num, filenr);
#else // LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_ids", num, filenr);
#endif // LONG_OUTPUT_NUMBERS
                if (!(fd = fopen(fname, "r")))
                {
                    ENDRUN(0, "can't read file `%s`", fname);
                }

                VERBOSE(2, "reading '%s'", fname);

                my_fread(&ngroups, sizeof(int), 1, fd);
                my_fread(&TotNgroups, sizeof(int), 1, fd);
                my_fread(&nids, sizeof(int), 1, fd);
                my_fread(&TotNids, sizeof(long long), 1, fd);
                my_fread(&ntask, sizeof(int), 1, fd);
                my_fread(&nid_previous, sizeof(int), 1, fd); /* this is the number of IDs in previous files */

                fof_id_list *tmpID_list = mymalloc("tmpID_list", nids * sizeof(fof_id_list));

                ids = mymalloc("ids", nids * sizeof(MyIDType));

                my_fread(ids, sizeof(MyIDType), nids, fd);

                for (int i = 0; i < nids; i++)
                {
                    tmpID_list[i].ID = ids[i];
                }

                myfree(ids);

                fclose(fd);

                target = 0;
                stored = 0;
                while (nids > 0)
                {
                    while (nids_to_get[target] == 0)
                    {
                        target++;
                    }

                    if (nids > nids_to_get[target])
                    {
                        nsend = nids_to_get[target];
                    }
                    else
                    {
                        nsend = nids;
                    }

                    if (target == 0)
                    {
                        memcpy(&ID_list[nids_obtained[target]], &tmpID_list[stored], nsend * sizeof(fof_id_list));
                    }
                    else
                    {
                        MPI_Send(&nsend, 1, MPI_INT, target, TAG_HEADER, MYMPI_COMM_WORLD);
                        MPI_Send(&tmpID_list[stored], nsend * sizeof(fof_id_list), MPI_BYTE, target, TAG_SPHDATA,
                                 MYMPI_COMM_WORLD);
                    }

                    nids_to_get[target] -= nsend;
                    nids_obtained[target] += nsend;
                    nids -= nsend;
                    stored += nsend;
                }

                myfree(tmpID_list);
            }
        }
        else
        {
            while (ngroup_to_get[ThisTask])
            {
                MPI_Recv(&nsend, 1, MPI_INT, 0, TAG_N, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&Group[ngroup_obtained[ThisTask]], nsend * sizeof(group_properties), MPI_BYTE, 0, TAG_PDATA,
                         MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                ngroup_to_get[ThisTask] -= nsend;
                ngroup_obtained[ThisTask] += nsend;
            }

            while (nids_to_get[ThisTask])
            {
                MPI_Recv(&nsend, 1, MPI_INT, 0, TAG_HEADER, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&ID_list[nids_obtained[ThisTask]], nsend * sizeof(fof_id_list), MPI_BYTE, 0, TAG_SPHDATA,
                         MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                nids_to_get[ThisTask] -= nsend;
                nids_obtained[ThisTask] += nsend;
            }
        }
        myfree(nids_obtained);
        myfree(ngroup_obtained);
        myfree(nids_to_get);
        myfree(ngroup_to_get);
    }

    else
    {
        /* read routine can constinue in parallel */
        nprocgroup = NTask / All.NumFilesWrittenInParallel;
        if ((NTask % All.NumFilesWrittenInParallel))
        {
            nprocgroup++;
        }
        masterTask = (ThisTask / nprocgroup) * nprocgroup;
        for (int groupTask = 0; groupTask < nprocgroup; groupTask++)
        {
            if (ThisTask == (masterTask + groupTask)) /* ok, it's this processor's turn */
            {
#ifdef LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "group_tab", num, ThisTask);
#else // LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_tab", num, ThisTask);
#endif // LONG_OUTPUT_NUMBERS
                if (!(fd = fopen(fname, "r")))
                {
                    ENDRUN_MAIN("can't read file `%s`", fname);
                }

                VERBOSE(2, "reading '%s'", fname);

                my_fread(&Ngroups, sizeof(int), 1, fd);
                my_fread(&TotNgroups, sizeof(int), 1, fd);
                my_fread(&Nids, sizeof(int), 1, fd);
                my_fread(&TotNids, sizeof(long long), 1, fd);
                my_fread(&ntask, sizeof(int), 1, fd);
                if (NTask != ntask)
                {
                    ENDRUN("Number of files in group catalogues needs to match MPI-Tasks");
                }

                VERBOSE(2, "TotNgroups=%d", TotNgroups);
                NAllocatedGroups = (TotNgroups / NTask + NTask);
                Group = (group_properties *)mymalloc("Group",
                                                     sizeof(group_properties) * max(Ngroups, TotNgroups / NTask + 1));

                /* group len */
                len = mymalloc("len", Ngroups * sizeof(int));
                my_fread(len, Ngroups, sizeof(int), fd);
                for (int i = 0; i < Ngroups; i++)
                {
                    Group[i].Len = len[i];
                }
                myfree(len);

                /* offset into id-list */
                len = mymalloc("len", Ngroups * sizeof(int));
                my_fread(len, Ngroups, sizeof(int), fd);
                for (int i = 0; i < Ngroups; i++)
                {
                    Group[i].Offset = len[i];
                }
                myfree(len);

                /* mass */
                mass = mymalloc("mass", Ngroups * sizeof(float));
                my_fread(mass, Ngroups, sizeof(float), fd);
                for (int i = 0; i < Ngroups; i++)
                {
                    Group[i].Mass = mass[i];
                }
                myfree(mass);

                /* CM */
                cm = mymalloc("cm", Ngroups * 3 * sizeof(float));
                my_fread(cm, Ngroups, 3 * sizeof(float), fd);
                for (int i = 0; i < Ngroups; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        Group[i].CM[j] = cm[i * 3 + j];
                    }
                }
                myfree(cm);

                fclose(fd);

                VERBOSE(2, "reading '%s'", fname);

#ifdef LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%04d/%s_%04d.%d", All.OutputDir, num, "group_ids", num, ThisTask);
#else // LONG_OUTPUT_NUMBERS
                sprintf(fname, "%s/groups_%03d/%s_%03d.%d", All.OutputDir, num, "group_ids", num, ThisTask);
#endif // LONG_OUTPUT_NUMBERS
                if (!(fd = fopen(fname, "r")))
                {
                    ENDRUN("Can't read file `%s`", fname);
                }

                VERBOSE(2, "reading '%s'", fname);

                my_fread(&Ngroups, sizeof(int), 1, fd);
                my_fread(&TotNgroups, sizeof(int), 1, fd);
                my_fread(&Nids, sizeof(int), 1, fd);
                my_fread(&TotNids, sizeof(long long), 1, fd);
                my_fread(&ntask, sizeof(int), 1, fd);
                my_fread(&nid_previous, sizeof(int), 1, fd); /* this is the number of IDs in previous files */

                ID_list = mymalloc("ID_list", Nids * sizeof(fof_id_list));
                ids = mymalloc("ids", Nids * sizeof(MyIDType));

                my_fread(ids, sizeof(MyIDType), Nids, fd);

                for (int i = 0; i < Nids; i++)
                {
                    ID_list[i].ID = ids[i];
                }

                myfree(ids);

                fclose(fd);
            }

            MPI_Barrier(MYMPI_COMM_WORLD); /* wait inside the group */
        }
    }
    TIMER_POPPUSH(3, "reading", "assigning");

    /* now need to assign group numbers */
    list_of_ngroups = mymalloc("list_of_ngroups", NTask * sizeof(int));
    list_of_nids = mymalloc("list_of_nids", NTask * sizeof(int));

    MPI_Allgather(&Ngroups, 1, MPI_INT, list_of_ngroups, 1, MPI_INT, MYMPI_COMM_WORLD);
    MPI_Allgather(&Nids, 1, MPI_INT, list_of_nids, 1, MPI_INT, MYMPI_COMM_WORLD);

    list_of_allgrouplen = mymalloc("list_of_allgrouplen", TotNgroups * sizeof(int));

    recvoffset = mymalloc("recvoffset", NTask * sizeof(int));
    recvoffset[0] = 0;
    for (int i = 1; i < NTask; i++)
    {
        recvoffset[i] = recvoffset[i - 1] + list_of_ngroups[i - 1];
    }
    len = mymalloc("len", Ngroups * sizeof(int));
    for (int i = 0; i < Ngroups; i++)
    {
        len[i] = Group[i].Len;
    }
    MPI_Allgatherv(len, Ngroups, MPI_INT, list_of_allgrouplen, list_of_ngroups, recvoffset, MPI_INT, MYMPI_COMM_WORLD);
    myfree(len);
    myfree(recvoffset);

    /* do a check */
    long long totlen = 0;
    for (int i = 0; i < TotNgroups; i++)
    {
        totlen += list_of_allgrouplen[i];
    }

    ENDRUN_IF(totlen != TotNids, "Error total number of IDs in groups does not match total number of IDs");

    for (int i = 0, count = 0; i < ThisTask; i++)
    {
        count += list_of_ngroups[i];
    }

    for (int i = 0; i < Ngroups; i++)
    {
        Group[i].GrNr = 1 + count + i;
    }

    /* fix Group.Offset (may have overflown) */
    if (Ngroups > 0)
    {
        count = 0, Group[0].Offset = 0;
        for (int i = 0; i < ThisTask; i++)
        {
            for (int j = 0; j < list_of_ngroups[i]; j++)
            {
                Group[0].Offset += list_of_allgrouplen[count++];
            }
        }
    }

    for (int i = 1; i < Ngroups; i++)
    {
        Group[i].Offset = Group[i - 1].Offset + Group[i - 1].Len;
    }

    long long *idoffset = mymalloc("idoffset", NTask * sizeof(long long));

    idoffset[0] = 0;
    for (int i = 1; i < NTask; i++)
    {
        idoffset[i] = idoffset[i - 1] + list_of_nids[i - 1];
    }

    grnr = 1, totlen = 0, count = 0;
    for (int i = 0; i < TotNgroups; totlen += list_of_allgrouplen[i++], grnr++)
    {
        if (totlen + list_of_allgrouplen[i] - 1 >= idoffset[ThisTask] && totlen < idoffset[ThisTask] + Nids)
        {
            for (int j = 0; j < list_of_allgrouplen[i]; j++)
            {
                if ((totlen + j) >= idoffset[ThisTask] && (totlen + j) < (idoffset[ThisTask] + Nids))
                {
                    ID_list[(totlen + j) - idoffset[ThisTask]].GrNr = grnr;
                    count++;
                }
            }
        }
    }

    ENDRUN_IF(count != Nids, "Number of IDs does not match counted number.");

    myfree(idoffset);

    TIMER_POPPUSH(3, "assigning", "sorting");

    for (int i = 0; i < NumPart; i++)
    {
        P[i].SubNr = i;
    }

    STD_SORT(P, NumPart, sizeof(struct particle_data), io_compare_P_ID);
    STD_SORT(ID_list, Nids, sizeof(fof_id_list), fof_compare_ID_list_ID);

    for (int i = 0; i < NumPart; i++)
    {
        P[i].GrNr = TotNgroups + 1; /* will mark particles that are not in any group */
    }

    TIMER_POPPUSH(3, "sorting", "assigning_GrNr_to_P");

    static fof_id_list *recv_ID_list;
    int matches = 0;

    /* exchange  data */
    for (int ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
        int sendTask = ThisTask;
        int recvTask = ThisTask ^ ngrp;

        if (recvTask < NTask)
        {
            if (list_of_nids[sendTask] > 0 || list_of_nids[recvTask] > 0)
            {
                if (ngrp == 0)
                {
                    recv_ID_list = ID_list;
                }
                else
                {
                    recv_ID_list = mymalloc("recv_ID_list", list_of_nids[recvTask] * sizeof(fof_id_list));

                    /* get the particles */
                    MPI_Sendrecv(ID_list, Nids * sizeof(fof_id_list), MPI_BYTE, recvTask, TAG_DENS_A, recv_ID_list,
                                 list_of_nids[recvTask] * sizeof(fof_id_list), MPI_BYTE, recvTask, TAG_DENS_A,
                                 MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                j = 0;
                for (int i = 0; i < list_of_nids[recvTask]; i++)
                {
                    while (j < NumPart - 1 && P[j].ID < recv_ID_list[i].ID)
                    {
                        j++;
                    }

                    if (recv_ID_list[i].ID == P[j].ID)
                    {
                        P[j].GrNr = recv_ID_list[i].GrNr;
                        matches++;
                    }
                }

                if (ngrp != 0)
                {
                    myfree(recv_ID_list);
                }
            }
        }
    }

    sumup_large_ints(1, &matches, &totlen);

    ENDRUN_IF(totlen != TotNids, "Number of total IDs does not match counted number.");

    TIMER_POP(3, "assigning_GrNr_to_P");

    MPI_Barrier(MYMPI_COMM_WORLD);

    myfree(list_of_allgrouplen);
    myfree(list_of_nids);
    myfree(list_of_ngroups);

    /* restore peano-hilbert order */
    STD_SORT(P, NumPart, sizeof(struct particle_data), fof_compare_P_SubNr);

    subfind(num);
    ENDRUN("FoF read done");
}

/**
 * @brief Compares two elements of a list of FOF IDs based on their IDs.
 *
 * This function compares two elements of a list of FOF IDs based on their ID values.
 *
 * @param a Pointer to the first fof_id_list structure.
 * @param b Pointer to the second fof_id_list structure.
 * @return -1 if the ID of the first element is smaller than that of the second element,
 *         +1 if the ID of the first element is larger than that of the second element,
 *         0 if both IDs are equal.
 */
int fof_compare_ID_list_ID(const void *a, const void *b)
{
    if (((fof_id_list *)a)->ID < ((fof_id_list *)b)->ID)
    {
        return -1;
    }

    if (((fof_id_list *)a)->ID > ((fof_id_list *)b)->ID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two particles based on their SubNr values.
 *
 * This function compares two particle_data structures based on their SubNr values.
 *
 * @param a Pointer to the first particle_data structure.
 * @param b Pointer to the second particle_data structure.
 * @return -1 if the SubNr of the first particle is smaller than that of the second particle,
 *         +1 if the SubNr of the first particle is larger than that of the second particle,
 *         0 if both SubNrs are equal.
 */
int fof_compare_P_SubNr(const void *a, const void *b)
{
    if (((struct particle_data *)a)->SubNr < (((struct particle_data *)b)->SubNr))
    {
        return -1;
    }

    if (((struct particle_data *)a)->SubNr > (((struct particle_data *)b)->SubNr))
    {
        return +1;
    }

    return 0;
}

#endif // SUBFIND_READ_FOF
#endif // FOF
