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

/*! This is the main routine for the domain decomposition.  It acts as a
 *  driver routine that allocates various temporary buffers, maps the
 *  particles back onto the periodic box if needed, and then does the
 *  domain decomposition, and a final Peano-Hilbert order of all particles
 *  as a tuning measure.
 */
#ifdef WRITE_KEY_FILES
static int domain_recompose_inner(int SaveKeys);
static int ar_domain_determineTopTree(int SaveKeys, int multipledomains);
#else // WRITE_KEY_FILES
static int domain_recompose_inner(void);
static int ar_domain_determineTopTree(void);
#endif // WRITE_KEY_FILES

#define AR_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define AR_MIN(x, y) (((x) < (y)) ? (x) : (y))

struct start_key
{
    int chunkId;
    int cpu;
    int len;
    double cost;
    int delta;
    int key;
};

#ifdef RECOMPOSE_DOMAIN_DIFFUSE
/**
 * @brief Compares two `start_key` structures based on their `key` values.
 *
 * This function is used to compare two `start_key` structures to determine their ordering based on the `key` field.
 * It returns an integer indicating the relative order of the two keys.
 *
 * @param a Pointer to the first `start_key` structure.
 * @param b Pointer to the second `start_key` structure.
 *
 * @return An integer less than, equal to, or greater than zero, if the `key` of the first structure is less than, equal
 * to, or greater than the `key` of the second structure, respectively.
 */
int start_key_compare_key(const void *a, const void *b)
{
    if (((struct start_key *)a)->key < (((struct start_key *)b)->key))
    {
        return -1;
    }

    if (((struct start_key *)a)->key > (((struct start_key *)b)->key))
    {
        return +1;
    }

    return 0;
}
#endif // RECOMPOSE_DOMAIN_DIFFUSE

/**
 * @brief Returns the sign of an integer.
 *
 * This function determines the sign of an integer `x`. It returns -1 if `x` is negative, 1 if `x` is positive, and 0 if
 * `x` is zero.
 *
 * @param x The integer whose sign is to be determined.
 *
 * @return -1 if `x` is negative, 1 if `x` is positive, and 0 if `x` is zero.
 */
static inline int sign(int x)
{
    return (x > 0) - (x < 0);
}

/**
 * @brief Finds the top-level node for a given leaf node.
 *
 * This function traverses up the tree from a given leaf node to find the top-level node (root) of the tree.
 * It adjusts the node index to be relative to the range of top-level nodes.
 *
 * @param l The index of the leaf node.
 *
 * @return The index of the top-level node, adjusted to be relative to the maximum number of particles.
 */
static inline int get_top_node_from_leaf(int l)
{
    int topno = l;
    while (topno >= NTopnodes + All.MaxPart)
    {
        topno = Nodes[topno].u.d.father;
    }
    return topno - All.MaxPart;
}

#define NODECOST(node_i) Nodes[node_i].u.d.mass;

/**
 * @brief Computes the total cost of nodes in a specified range and optionally counts the nodes.
 *
 * This function iterates over a range of nodes, calculates their costs, and accumulates the total cost.
 * If the `n_cunts` pointer is not `NULL`, it increments the count of nodes processed.
 *
 * @param from The starting index (inclusive) of the range of nodes to consider.
 * @param to The ending index (exclusive) of the range of nodes to consider.
 * @param n_cunts Pointer to an integer where the count of nodes processed will be stored. If `NULL`, no count will be
 * kept.
 *
 * @return The total cost of the nodes in the specified range.
 */
static inline double compute_cost(int from, int to, int *n_cunts)
{
    double cost = 0.0;
    for (int i = from; i < to; i++)
    {
        int noi = DomainNodeIndex[i];
        double this_cost = NODECOST(noi);
        if ((!this_cost) > 0.)
        {
            this_cost = 0.0;
        }
        cost += this_cost;
        if (n_cunts != NULL)
        {
            (*n_cunts)++;
        }
    }
    return cost;
}

#ifdef RECOMPOSE_DOMAIN_DIFFUSE

#ifndef RECOMPOSE_DOMAIN_FRACTION_SIGMA
#define RECOMPOSE_DOMAIN_FRACTION_SIGMA 1.0
#warning                                                                                                               \
    "Automatically set RECOMPOSE_DOMAIN_FRACTION_SIGMA=1.0 (which fraction of the computational cost of my domain I am willing to diffuse, here: one standard deviation)"
#endif // RECOMPOSE_DOMAIN_FRACTION_SIGMA

#if defined(RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED)
#elif defined(RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION)
#else // RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED
#define RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION 0.2
#warning                                                                                                               \
    "Automatically set RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION=0.2 (how much fractional standard deviation is eligible to be trasferred to a different domain, here 0.2 times the over-load)"
#endif // RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED

/**
 * @brief Performs domain diffusion to balance the cost across domains.
 *
 * This function calculates the cost for each domain chunk, sorts them, and adjusts the domain distribution to balance
 * the load. It computes statistics such as average cost, standard deviation, and adjusts the start and end points of
 * domains to achieve a more balanced distribution.
 *
 * The function outputs statistics before and after the diffusion process, including average cost, cost spread, and the
 * number of moved blocks. It uses the `compute_cost` function to evaluate costs and may shift domain chunks between
 * CPUs based on the computed costs.
 */
void domain_diffuse()
{
    int m, i;
    double cost;
    double last_cost = -1;
    double avg_cost;
    double avg2_cost;
    double max_cost = -1, min_cost = -1;

    double new_avg_cost = 0.;
    double new_avg2_cost = 0.;
    double new_max_cost = -1;
    double new_min_cost = -1;

    int cpu;
    int topno;
    float n_chunks;
    int from = 0;
    int to = 0;
    int t = 0;
    int moves = 0;
    struct start_key *StartKeys =
        (struct start_key *)mymalloc("ordered_key", sizeof(struct start_key) * Multipledomains * NTask);
    int n_cunts = 0;
    avg_cost = 0.0;
    avg2_cost = 0.0;
    n_chunks = (1.0 * Multipledomains * NTask);
    n_cunts = 0;

    /*
       first we output the average cost and standard deviation of domains
     */
    for (int m = 0; m < Multipledomains * NTask; m++)
    {
        int from = DomainStartList[m];
        int to = DomainEndList[m];
        double cost = compute_cost(from, to, &n_cunts);
        avg_cost += cost;
        avg2_cost += cost * cost;
        max_cost = max_cost < 0 ? cost : (max_cost < cost ? cost : max_cost);
        min_cost = min_cost < 0 ? cost : (min_cost > cost ? cost : min_cost);
    }
    avg_cost /= n_chunks;
    avg2_cost /= n_chunks;
    double sigma = sqrt(avg2_cost - avg_cost * avg_cost);

    VERBOSE(2, "Before Domain Diffuse: <AVG COST>=%f, interval=[%f, %f], <spread>=%f \n", avg_cost, min_cost, max_cost,
            sigma);

    /*
       now we save all domain information in the array StartKeys
     */
    for (int t = 0; t < NTask; t++)
    {
        for (int m = 0; m < Multipledomains; m++)
        {
            int from = DomainStartList[t * Multipledomains + m];
            int to = DomainEndList[t * Multipledomains + m];
            double cost = compute_cost(from, to, NULL);

            StartKeys[t * Multipledomains + m].chunkId = t * Multipledomains + m;
            StartKeys[t * Multipledomains + m].key = from;
            StartKeys[t * Multipledomains + m].cpu = t;
            StartKeys[t * Multipledomains + m].len = to - from;
            StartKeys[t * Multipledomains + m].cost = cost;
        }
    }

    /*
       we should sort domain information with peano key, so we will be able to shift contiguous chunks of domains.
     */
    STD_SORT(StartKeys, Multipledomains * NTask, sizeof(struct start_key), start_key_compare_key);

    /*
       now we loop over the domain information to see what shifts we can make
     */
    int total_delta = 0;
    for (int i = 0; i < NTask * Multipledomains; i++)
    {
        int next_i = (i + 1) % ((int)(NTask * Multipledomains));

        // information about the  current and next domain-chunks:

        double current_cost = StartKeys[i].cost;
        double next_cost = StartKeys[next_i].cost;

        int current_id = StartKeys[i].chunkId;
        int next_id = StartKeys[next_i].chunkId;

        int current_len = StartKeys[i].len;
        int next_len = StartKeys[next_i].len;

        int current_cpu = StartKeys[i].cpu;
        int next_cpu = StartKeys[next_i].cpu;

        int delta_next = 0;

#ifdef RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION
        // average costs of this and next chinks alltogether
        double avg_cost_per_ele = 1.0 * (1. * current_cost + next_cost) / (current_len + next_len);

        // we only try to move a RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION fraction that exceed the average cost of all
        // chunks
        double cost_to_be_shifted = RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION * abs(current_cost - avg_cost);

        // we estimate how many chunks we should move (if they all have approximaetly the same cost)
        int eles_to_be_shifted = cost_to_be_shifted / avg_cost_per_ele - 1;
        if (eles_to_be_shifted < 0)
        {
            eles_to_be_shifted = 0;
        }
#endif // RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION

        // if this chunk cost exceed the average more than  RECOMPOSE_DOMAIN_FRACTION_SIGMA*standard-deviation, then we
        // give blocks away
        if (current_cost - avg_cost > RECOMPOSE_DOMAIN_FRACTION_SIGMA * sigma)
        {
#ifdef RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED
            delta_next = -RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED;
#elif defined(RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION)
            delta_next = -eles_to_be_shifted;
#endif // RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED
        }

        // if this chunk cost is less than the average, then we get a chunk from the next domain
        if (avg_cost - current_cost > RECOMPOSE_DOMAIN_FRACTION_SIGMA * sigma)
        {
#ifdef RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED
            delta_next = RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED;
#elif defined(RECOMPOSE_DOMAIN_SHIFT_EXCEED_FRACTION)
            delta_next = eles_to_be_shifted;
#endif // RECOMPOSE_DOMAIN_FRACTION_SIGMA_FIXED
        }

        // statistics information
        total_delta += abs(delta_next);

        // if we need to move elements, then let's do it!
        if ((delta_next > 0 && next_len > delta_next) || (delta_next < 0 && current_len > -delta_next))
        {

            DomainStartList[next_id] += delta_next;
            DomainEndList[current_id] += delta_next;

            StartKeys[i].len += delta_next;
            StartKeys[next_i].len -= delta_next;

            // update the costs (in our internal variables) after we moved data
            int sign = delta_next > 0 ? -1 : +1;

            StartKeys[i].cost = compute_cost(DomainStartList[current_id], DomainEndList[current_id], NULL);
            StartKeys[next_i].cost = compute_cost(DomainStartList[next_id], DomainEndList[next_id], NULL);

            int first_shifted_ele =
                delta_next > 0 ? DomainStartList[next_id] - delta_next : DomainStartList[next_id] - delta_next - 1;
            int new_cpu = delta_next > 0 ? current_cpu : next_cpu;
            int inc_counter = delta_next > 0 ? 1 : -1;

            // update DomainTask accordingly to our change in cpus
            for (int m = 0; m < inc_counter * delta_next; m++)
            {
                int n = first_shifted_ele + m * inc_counter;

                DomainTask[n] = new_cpu;
            }
        }
    }

    /*
       lets check the avg cost and spread after the diffusion
     */

    new_avg_cost = 0.0;
    new_avg2_cost = 0.0;
    n_chunks = (1.0 * Multipledomains * NTask);
    n_cunts = 0;

    for (int m = 0; m < Multipledomains * NTask; m++)
    {
        int from = DomainStartList[m];
        int to = DomainEndList[m];
        double cost = compute_cost(from, to, &n_cunts);
        new_avg_cost += cost;
        new_avg2_cost += cost * cost;
        new_max_cost = new_max_cost < 0 ? cost : (new_max_cost < cost ? cost : new_max_cost);
        new_min_cost = new_min_cost < 0 ? cost : (new_min_cost > cost ? cost : new_min_cost);
    }
    new_avg_cost /= n_chunks;
    new_avg2_cost /= n_chunks;

    int accepted = sqrt(new_avg2_cost - new_avg_cost * new_avg_cost) < sqrt(avg2_cost - avg_cost * avg_cost);
    VERBOSE(2, "After Domain Diffusion  <AVG COST>=%f, interval=[%f, %f], <spread>=%f, moved blocks=%d; %s\n",
            new_avg_cost, new_min_cost, new_max_cost, sqrt(new_avg2_cost - new_avg_cost * new_avg_cost), total_delta,
            accepted ? "Good" : "Bad");

    // TODO: store data only if diffusion is good.

    myfree(StartKeys);
}

#endif // RECOMPOSE_DOMAIN_DIFFUSE

/**
 * @brief Reconstructs the domain decomposition and reorders particles.
 *
 * This function performs a series of operations to reconstruct the domain decomposition, which includes:
 * - Drifting particles to their current positions.
 * - Rearranging the particle sequence if necessary.
 * - Applying periodic boundary conditions if enabled.
 * - Allocating and freeing memory for temporary storage used in the domain decomposition process.
 * - Recomputing domain structures and the Peano-Hilbert order.
 * - Checking consistency of particle data.
 * - Handling specific cases for different particle types (e.g., stars, black holes).
 */
void domain_recomposition()
{

    int numpart_min;
    MPI_Allreduce(&NumPart, &numpart_min, 1, MPI_INT, MPI_MIN, MYMPI_COMM_WORLD);
    if (numpart_min <= 0)
    {
        VERBOSE(0,
                "Some tasks have NumPart = %d, some recomposition buffers fail if a task have zero numpart, we skip.",
                numpart_min);
        return;
    }

#ifdef TIMER_DOMAIN
    TIMER_PUSH_CLASS_MAINONLY(DomainRecompositionTimer, 0, "DOMAIN_RECOMPOSITION");
#endif // TIMER_DOMAIN
    TIMER_PUSH(1, "DOMAIN_RECOMPOSITION");

    int UseAllTimeBins = 0;
#ifdef WRITE_KEY_FILES
    int SaveKeys = 0;
#endif // WRITE_KEY_FILES
    int i, ret, retsum, diff, highest_bin_to_include;
    size_t bytes, all_bytes;

    VERBOSE(2, "Starting domain_recomposition");
    UseAllParticles = UseAllTimeBins;

#ifdef WRITE_KEY_FILES
    peanokey old_key;
    int key_count, previous_keys;
#endif // WRITE_KEY_FILES

    TIMER_PUSH(1, "DOMAIN");
// drift everything! because why not!
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Ti_current != All.Ti_Current)
        {
            drift_particle(i, All.Ti_Current);
        }
    }

    /* at this point the DD frees the tree and the domain here, but not the recomposition of the domain */

#if defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)
    rearrange_particle_sequence();
#endif // defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

#ifdef PERIODIC
    do_box_wrapping();
#endif // PERIODIC

// general checks
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        ENDRUN_IF(P[i].Type > 5 || P[i].Type < 0, "task=%d:  P[i=%d].Type=%d\n", ThisTask, i, P[i].Type);

#ifdef SFR
        ENDRUN_IF(P[i].Type == 4 && MetP[P[i].pt.MetID].PID != i,
                  "task=%d:  error in cross-indexes for star-particle %d ID %llu\n", ThisTask, i,
                  (unsigned long long)P[i].ID);
#endif // SFR

#ifdef BLACK_HOLES
        ENDRUN_IF(P[i].Type == 5 && BHP[P[i].pt.BHID].PID != i,
                  "task=%d:  error in cross-indexes for bh-particle %d ID %llu\n", ThisTask, i,
                  (unsigned long long)P[i].ID);
#endif // BLACK_HOLES
    }

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

    for (i = 1, TakeLevel = 0, diff = abs(All.LevelToTimeBin[0] - highest_bin_to_include); i < Gravcostlevels; i++)
    {
        if (diff > abs(All.LevelToTimeBin[i] - highest_bin_to_include))
        {
            TakeLevel = i;
            diff = abs(All.LevelToTimeBin[i] - highest_bin_to_include);
        }
    }

    VERBOSE(2, "domain decomposition... LevelToTimeBin[TakeLevel=%d]=%d  (presently allocated=%g MB)\n", TakeLevel,
            All.LevelToTimeBin[TakeLevel], AllocatedBytes / (1024.0 * 1024.0));

    TIMER_PUSH(1, "DOMAIN_DECOMPOSITION");
    // do /*domain is already allocated so we need no iterations */
    {
        //  domain_allocate(); /*domain is already allocated*/

        all_bytes = 0;

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

        VERBOSE(2, "use of %g MB of temporary storage for domain decomposition... (presently allocated=%g MB)\n",
                all_bytes / (1024.0 * 1024.0), AllocatedBytes / (1024.0 * 1024.0));

        maxLoad = (int)(All.MaxPart * REDUC_FAC);
        maxLoadsph = (int)(All.MaxPartSph * REDUC_FAC);
#ifdef SFR
        maxLoadstars = (int)(All.MaxPartMet * REDUC_FAC);
#endif // SFR

#ifdef BLACK_HOLES
        maxLoadBHs = (int)(All.MaxPartBH * REDUC_FAC);
#endif // BLACK_HOLES

        report_memory_usage(&HighMark_domain, "RECOMPOSE");

#ifdef WRITE_KEY_FILES
        ret = domain_recompose_inner(SaveKeys);
#else // WRITE_KEY_FILES
        ret = domain_recompose_inner();
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
            // myfree(Key);
            // domain_free();

            VERBOSE(2, "Increasing TopNodeAllocFactor=%g  ", All.TopNodeAllocFactor);

            // All.TopNodeAllocFactor *= 1.3;

            VERBOSE(2, "new value=%g\n", All.TopNodeAllocFactor);

            ENDRUN_IF(All.TopNodeAllocFactor > 1000, "something seems to be going seriously wrong here. Stopping.\n");
        }
    }
    // while(retsum);
    TIMER_POP(1, "DOMAIN_DECOMPOSITION");
    TIMER_POP(1, "DOMAIN");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        ENDRUN_IF(P[i].Type > 5 || P[i].Type < 0, "task=%d:  P[i=%d].Type=%d\n", ThisTask, i, P[i].Type);

#ifdef SFR
        ENDRUN_IF(P[i].Type == 4 && MetP[P[i].pt.MetID].PID != i,
                  "task=%d:  error in cross-indexes for star-particle %d ID %llu\n", ThisTask, i,
                  (unsigned long long)P[i].ID);
#endif // SFR

#ifdef BLACK_HOLES
        ENDRUN_IF(P[i].Type == 5 && BHP[P[i].pt.BHID].PID != i,
                  "task=%d:  error in cross-indexes for bh-particle %d ID %llu\n", ThisTask, i,
                  (unsigned long long)P[i].ID);
#endif // BLACK_HOLES
    }

#ifdef SUBFIND
    if (GrNr < 0) /* we don't do it when SUBFIND is executed for a certain group */
#endif // SUBFIND
    {
        peano_hilbert_order();
    }

#ifdef WRITE_KEY_FILES
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
                        ENDRUN_IF(key_count > NumPart, "Keycount : %d > %d !!\n", key_count, NumPart);

                        NKeys[k] = key_count - previous_keys;
                        previous_keys += NKeys[k];
                    }
                }
            }
        }
#ifdef SUBFIND
    }
#endif // SUBFIND
#endif // WRITE_KEY_FILES

    myfree(Key);

    // memmove(TopNodes + NTopnodes, DomainTask, NTopnodes * sizeof(int));

    force_treefree();
    force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);

    reconstruct_timebins();
    TreeReconstructFlag = 1; /* ensures that new tree will be constructed */

    TIMER_POP(1, "DOMAIN_RECOMPOSITION");

#ifdef TIMER_DOMAIN
    TIMER_POP_CLASS_MAINONLY(DomainRecompositionTimer, 0, "DOMAIN_RECOMPOSITION");
#endif // TIMER_DOMAIN
}

/**
 * @brief Performs the actual domain decomposition for all particle types.
 *
 * This function balances the workload for each domain based on particle cost factors (`P[i]-GravCost` values)
 * and ensures that the decomposition respects the maximum allowed memory imbalance, as specified by `PartAllocFactor`.
 *
 * It executes the following steps:
 * - Computes the local particle counts and cost factors.
 * - Determines global domain grid dimensions.
 * - Flags particles that need to be exchanged between domains.
 * - Manages particle exchanges between tasks to achieve balanced domain distribution.
 *
 * @param SaveKeys If defined, determines whether to save keys for domain decomposition.
 * @return Returns 0 on success, or 1 if an error occurs during domain decomposition.
 */
#ifdef WRITE_KEY_FILES
static int domain_recompose_inner(int SaveKeys)
#else // WRITE_KEY_FILES
static int domain_recompose_inner(void)
#endif // WRITE_KEY_FILES
{
#ifdef WRITE_KEY_FILES
    int multipledomains = Multipledomains;

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
#endif
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
    sumup_large_ints(6, NtypeLocal, Ntype);

    totpartcount = 0;
    for (int i = 0; i < 6; i++)
    {
        totpartcount += Ntype[i];
    }

    MPI_Allreduce(&gravcost, &totgravcost, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&sphcost, &totsphcost, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    /* determine global dimensions of domain grid */
#ifdef WRITE_KEY_FILES
    domain_findExtent(SaveKeys);
    if (ar_domain_determineTopTree(SaveKeys, multipledomains))
    {
        return 1;
    }
#else // WRITE_KEY_FILES
    domain_findExtent();
    if (ar_domain_determineTopTree())
    {
        return 1;
    }
#endif // WRITE_KEY_FILES

    /* flag the particles that need to be exported */
    int to_exchange = 0;
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
            to_exchange++;
        }
    }
    //      printf("Task%d to_xcehance=%d\n",ThisTask,to_exchange);

    int iter = 0, ret;
    size_t exchange_limit;

    do
    {

        exchange_limit = FreeBytes - NTask * (24 * sizeof(int) + 22 * sizeof(MPI_Request));
#ifdef MPISENDRECV_CHECKSUM
        exchange_limit -= All.BufferSize * 1024 * 1024;
#endif // MPISENDRECV_CHECKSUM
        ENDRUN_IF(exchange_limit <= 0, "task=%d: exchange_limit=%d\n", ThisTask, (int)exchange_limit);

        /* determine for each cpu how many particles have to be shifted to other cpus */
        ret = domain_countToGo(exchange_limit);

        long long sumtogo = 0;
        for (int i = 0; i < NTask; i++)
        {
            sumtogo += toGo[i];
        }

        sumup_longs(1, &sumtogo, &sumtogo);

        VERBOSE(2, "[AR ]iter=%d exchange of %d%09d particles (ret=%d)\n", iter, (int)(sumtogo / 1000000000),
                (int)(sumtogo % 1000000000), ret);

        domain_exchange();

        iter++;
    } while (ret > 0);

    return 0;
}

/**
 * @brief Constructs the global top-level tree node for domain decomposition.
 *
 * This function builds a global top-level tree used for domain decomposition by considering the Peano-Hilbert keys
 * of all particles. The process involves recursively partitioning the particle set into segments of eight until
 * each segment contains a manageable number of particles.
 *
 * The steps performed by this function include:
 * - Calculating Peano-Hilbert keys for each particle based on its position.
 * - Sorting particles according to their keys.
 * - Copying necessary data for the top-level tree nodes.
 * - Ensuring that the number of top-level leaves meets the required threshold.
 *
 * @param SaveKeys If defined, specifies whether to save keys for domain decomposition.
 * @param multipledomains The number of domains to use for decomposition, relevant if `WRITE_KEY_FILES` is defined.
 *
 * @return Returns 0 on success, or 1 if an error occurs during the construction of the top-level tree.
 */
#ifdef WRITE_KEY_FILES
static int ar_domain_determineTopTree(int SaveKeys, int multipledomains)
#else // WRITE_KEY_FILES
static int ar_domain_determineTopTree(void)
#endif // WRITE_KEY_FILES
{
    // MPI_Status status;
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

    ENDRUN_IF(GrNr >= 0 && count != NumPartGroup, "GrNr >= 0 && count != NumPartGroup");
    ENDRUN_IF(GrNr < 0 && count != NumPart, "GrNr >= 0 && count != NumPartGroup");
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

    /* copy what we need for the topnodes */
    VERBOSE(2, "NTOPNODES=%d\n", NTopnodes);
    for (int i = 0; i < NTopnodes; i++)
    {
        topNodes[i].StartKey = TopNodes[i].StartKey;
        topNodes[i].Size = TopNodes[i].Size;
        topNodes[i].Daughter = TopNodes[i].Daughter;
        topNodes[i].Leaf = TopNodes[i].Leaf;
    }

    myfree(mp);

    VERBOSE(2, "Before=%d\n", NTopnodes);
    VERBOSE(2, "After=%d\n", NTopnodes);

    /* count toplevel leaves */
    domain_sumCost();

    ENDRUN_IF(NTopleaves < multipledomains * NTask, "NTopleaves=%d < multipledomains * NTask = %d * %d = %d",
              NTopleaves, multipledomains, NTask, multipledomains * NTask);

    return 0;
}
