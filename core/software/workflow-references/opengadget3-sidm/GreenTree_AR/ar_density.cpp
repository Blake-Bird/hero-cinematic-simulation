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

/**
 * @brief Evaluates the primary density calculation for a group of particles.
 *
 * This function evaluates the density of a group of particles using a tree-based neighbor search.
 * It processes particles in groups, handling the computation in parallel across multiple threads.
 * The function supports operations on GPU.
 *
 * @param p Pointer to the thread ID or other thread-specific parameters.
 * @param is_primary_on_gpu Flag indicating whether the primary operation is performed on the GPU.
 * @param should_drift Flag indicating whether to account for drift in particle positions.
 * @param group_particles Flag indicating whether particles should be grouped for density evaluation.
 *
 * @return A pointer (typically NULL) indicating the completion of the density evaluation.
 */
void *ar_green_density_evaluate_primary(void *p, int is_primary_on_gpu, int should_drift, int group_particles)
{
    int thread_id = *(int *)p;
    int i, j;
    int *exportflag, *exportnodecount, *exportindex, *ngblist, *recyclingbuffer;
    int mode = 0;
    if (is_primary_on_gpu)
        mode = 3;
    int factor = 1;
    factor = 4;

    ngblist = Ngblist + thread_id * factor * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    recyclingbuffer = RecyclingBuffer + thread_id * factor * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    int *group_ids;
    MyLongDouble xtmp;
    int k;
    group_ids = recyclingbuffer;

    /* Note: exportflag is local to each thread */
    for (j = 0; j < NTask; j++)
        exportflag[j] = -1;
    int max_group_len = -1;
    int skipped_active = 0;
    int processed_particles = 0;
    //  printf("Task=%d:thread=%d going to densi primary\n",ThisTask,thread_id);
    while (1)
    {
        int break_em_all = 0;
        int group_len = 0;
        int exitFlag = 0, first_group_particle = -1;
        MyLongDouble dx, dy, dz, dist2, cluster_dist_2 = -1.;
        MyFloat max_r2, max_h, max_r1, max_h_p_r;
#pragma omp critical(_nexport_)

        {
            if (BufferFullFlag != 0 || NextParticle < 0)
            {
                exitFlag = 1;
            }
            else
            {
                while (1)
                {
                    if (NextParticle < 0)
                    {
                        if (first_group_particle == -1)
                            exitFlag = 1;
                        break;
                    }
                    if (!density_isactive(NextParticle))
                    {
                        ProcessedFlag[NextParticle] = 1;
                        NextParticle = NextActiveParticle[NextParticle];
                        continue;
                    }

                    if (first_group_particle == -1)
                    {
                        max_r2 = 0.;
                        first_group_particle = NextParticle;
                        ProcessedFlag[NextParticle] = 0;
                        max_h = P[NextParticle].Hsml;
                        group_ids[group_len] = first_group_particle;
                        group_len++;
                        processed_particles++;
                        cluster_dist_2 = P[first_group_particle].Hsml * AR_GREEN_TREE_HSML_FRACTION;
                        cluster_dist_2 *= cluster_dist_2;
                        NextParticle = NextActiveParticle[NextParticle];
                        if (HasFisrtParticleBeenProcessed == 0)
                        {
                            HasFisrtParticleBeenProcessed = 1;
                            break;
                        }

                        if (!group_particles)
                            break;

#ifdef AR_GREENTREE_LIMIT_GROUPING
                        double hsml_ratio = P[NextParticle].Hsml / P[first_group_particle].Hsml;
                        if (hsml_ratio > 1.05 || hsml_ratio < .95)
                            break;
#endif

#ifdef AR_ONE_PARTICLE_PER_GROUP_I
                        break;
#endif
#if defined(LT_STELLAREVOLUTION)
                        if (P[first_group_particle].Type == 5)
                            break;
#endif
                        continue;
                    }
#ifdef AR_GREEN_TREE_MAX_GROUP_SIZE
                    if (group_len >= AR_GREEN_TREE_MAX_GROUP_SIZE)
                        break;
#endif
#if defined(LT_STELLAREVOLUTION)
                    if (P[NextParticle].Type == 5)
                        break;
#endif

                    dx = NGB_PERIODIC_LONG_X(P[first_group_particle].Pos[0] - P[NextParticle].Pos[0]);
                    dy = NGB_PERIODIC_LONG_Y(P[first_group_particle].Pos[1] - P[NextParticle].Pos[1]);
                    dz = NGB_PERIODIC_LONG_Z(P[first_group_particle].Pos[2] - P[NextParticle].Pos[2]);
                    dist2 = dx * dx + dy * dy + dz * dz;

                    if (dist2 <= cluster_dist_2)

                    {

                        if (dist2 > max_r2)
                            max_r2 = dist2;
                        if (P[NextParticle].Hsml > max_h)
                            max_h = P[NextParticle].Hsml;
                        group_ids[group_len] = NextParticle;
                        group_len++;

                        processed_particles++;
                        ProcessedFlag[NextParticle] = 0;
                        NextParticle = NextActiveParticle[NextParticle];
                    }
                    else
                        break;
                }
            }
        }
        UNLOCK_NEXPORT;

        if (max_group_len < 0 || max_group_len < group_len)
            max_group_len = group_len;
        max_r1 = sqrt(max_r2);
        max_h_p_r = max_r1 + max_h;

        if (exitFlag)
        {
            break;
        }
        if (group_len > 0)
        {
            int i_target;
            int i_startnode;
            int node = All.MaxPart;
            int i_node = 0;
            int n_targets = group_len;
            int *targets;
            targets = group_ids;
            int extended_numngb_inbox = -2;

            ENDRUN_IF(n_targets > ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)),
                      "[AR] [DIE] [n_grouped_particles=%d] , max=numpart/density_less_ngb_factor=%d \n", n_targets,
                      ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)));

            for (i_target = 0; i_target < n_targets; i_target++)
            {
                int target = targets[i_target];
                int itarget = -1;
                int i = target;
                ENDRUN_IF(!density_isactive(target), "Target is not active");
                //     printf("task %d, evaluate target %d=%d maxhmsml=%f, maxdist=%f
                //     radius=%f\n",ThisTask,i_target,target,max_hsml,max_dist,search_radius);
                if (extended_numngb_inbox == -2)
                {
                    itarget = target;
                    i_startnode = node;
                    extended_numngb_inbox = ar_green_ngb_treefind_variable_threads(
                        P[target].Pos, &max_h_p_r, &i_startnode, mode, exportflag, exportnodecount, exportindex,
                        ngblist, targets, n_targets, should_drift);

                    ENDRUN_IF(
                        extended_numngb_inbox > ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)),
                        "[AR] [DIE] [n_extended_neighbourlist_size=%d] , max=numpart/density_less_ngb_factor=%d \n",
                        extended_numngb_inbox, ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)));
                }

                if (extended_numngb_inbox == -1)
                {
                    return NULL;
                }

                if (density_evaluate(target, mode, ngblist, extended_numngb_inbox, &i_startnode, node, i_node, NULL) <
                    0)
                {
                    return NULL;
                }

                ProcessedFlag[i] = 1; /* particle successfully finished */
            }
        }
        continue;
    }
    //  printf("Task%d - 1fase fine. max_group_len=%d, skipped_active=%d
    //  processed=%d\n",ThisTask,max_group_len,skipped_active,processed_particles);

    return NULL;
}

/**
 * @brief Evaluates the secondary phase of density calculations for a group of particles.
 *
 * This function processes particle groups during the secondary phase of density evaluation, using a
 * tree-based neighbor search. It operates in parallel across multiple threads and handles
 * particles that share similar node lists. The function supports handling of drift during computation.
 *
 * @param p Pointer to the thread ID or other thread-specific parameters.
 * @param should_drift Flag indicating whether to account for drift in particle positions.
 *
 * @return pointer (typically NULL) indicating the completion of the density evaluation.
 */
void *ar_green_density_evaluate_secondary(void *p, int should_drift)
{
    int thread_id = *(int *)p;

    int j, dummy, *ngblist, *recyclingbuffer, processed_particles = 0;
    //  printf("Task=%d:thread=%d OMG mom!, going to density secondary\n",ThisTask,thread_id);
    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    recyclingbuffer = RecyclingBuffer + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    int *group_ids;
    group_ids = recyclingbuffer;
    int exitFlag = 0;
    int i_node, max_group_len = -1;
    MyLongDouble xtmp;
    while (1)
    {
        MyLongDouble dx, dy, dz, dist2, cluster_dist_2 = -1.;
        int break_em_all = 0;
        int n_nodes = 0;
        int k;
        int same_nodes = 1;
        int group_len = 0;
        int exitFlag = 0, first_group_particle = -1;
        MyFloat max_h;
        MyFloat max_r2;
        MyFloat max_r1;
        MyFloat max_r_p_h;

        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif
        {

            struct densdata_in *D;

            while (1)
            {

                if (NextJ >= Nimport)
                {
                    if (first_group_particle == -1)
                        exitFlag = 1;
                    break;
                }

                if (first_group_particle == -1)
                {
                    first_group_particle = NextJ;
                    group_len++;
                    processed_particles++;
                    NextJ++;
                    n_nodes = 0;

                    D = &DensDataGet[first_group_particle];

                    while (DataNodeListGet[first_group_particle].NodeList[n_nodes] >= 0)
                    {
                        n_nodes++;
                        if (n_nodes == NODELISTLENGTH)
                            break;
                    }
                    max_h = D->Hsml;
                    max_r2 = 0.;
                    cluster_dist_2 = D->Hsml * AR_GREEN_TREE_HSML_FRACTION;

                    cluster_dist_2 *= cluster_dist_2;

#if defined(LT_STELLAREVOLUTION)
                    if (D->Type == 5)
                        break;
#endif
                    continue;
                }
#if defined(LT_STELLAREVOLUTION)

                if (DensDataGet[NextJ].Type == 5)
                    break;

#endif
                same_nodes = 1;
                // surprise!in the secondary phase, group them if they share the same exchanged nodelist!
                i_node = 0;
                while (DataNodeListGet[NextJ].NodeList[i_node] >= 0)
                {
                    if (DataNodeListGet[first_group_particle].NodeList[i_node] !=
                        DataNodeListGet[NextJ].NodeList[i_node])
                    {
                        same_nodes = 0;
                        break;
                    }
                    i_node++;

                    if (i_node > n_nodes)
                    {
                        same_nodes = 0;
                        break;
                    }
                    if (i_node == NODELISTLENGTH)
                        break;
                }
                if (i_node != n_nodes)
                {
                    same_nodes = 0;
                }

                struct densdata_in *N = &DensDataGet[NextJ];

#ifdef AR_ONE_PARTICLE_PER_GROUP_II
                break;
#endif
                if (same_nodes != 1)
                    break;
                dx = NGB_PERIODIC_LONG_X(D->Pos[0] - N->Pos[0]);
                dy = NGB_PERIODIC_LONG_Y(D->Pos[1] - N->Pos[1]);
                dz = NGB_PERIODIC_LONG_Z(D->Pos[2] - N->Pos[2]);
                dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 <= cluster_dist_2 && same_nodes == 1)
                {
                    if (dist2 > max_r2)
                        max_r2 = dist2;
                    if (N->Hsml > max_h)
                        max_h = N->Hsml;
                    group_len++;
                    processed_particles++;
                    NextJ++;
                }
                else
                    break;
            }
        }
        UNLOCK_NEXPORT;
        if (exitFlag == 1)
            break;
        // printf("Task%d - grupd %d !\n",ThisTask,NextParticle);
        max_r1 = sqrt(max_r2);
        max_r_p_h = max_r1 + max_h;
        if (max_group_len < 0 || max_group_len < group_len)
            max_group_len = group_len;

        for (k = 0; k < group_len;)
        {

            group_ids[k] = first_group_particle;

            first_group_particle++;
            k++;
        }

        int *targets;
        targets = group_ids;
        int n_targets = group_len;
        int i_node = 0;
        int i_target;
        int i_startnode;
        int *nodes = DataNodeListGet[targets[0]].NodeList;

        while (nodes[i_node] >= 0)
        {
            int extended_numngb_inbox = -2;
            int node = nodes[i_node];
            if (node < 0)
            {
                break;
            }
            ENDRUN_IF(node >= All.MaxPart + MaxNodes, "node=%d, AllMaxPArt=%d, MaxNodes=%d, All.MaxPart + MaxNodes=%d",
                      node, All.MaxPart, MaxNodes, All.MaxPart + MaxNodes);

            node = Nodes[node].u.d.nextnode;
            for (i_target = 0; i_target < n_targets; i_target++)
            {
                int target = targets[i_target];
                if (extended_numngb_inbox == -2)
                {

                    struct densdata_in *T = &DensDataGet[target];

                    i_startnode = node;
                    extended_numngb_inbox =
                        ar_green_ngb_treefind_variable_threads(T->Pos, &max_r_p_h, &i_startnode, 1, &dummy, &dummy,
                                                               &dummy, ngblist, targets, n_targets, should_drift);
                }
                if (extended_numngb_inbox == -1)
                    return NULL;

                if (density_evaluate(target, 1, ngblist, extended_numngb_inbox, &i_startnode, node, i_node, NULL) < 0)
                    return NULL;

            } // end initial for loop over targets
            if (++i_node >= NODELISTLENGTH)
                break;
        }
    }
    //  printf("Task=%d:thread=%d OMG mom!, DONE dens secondary maxgrouplen=%d\n",ThisTask,thread_id,max_group_len);
    return NULL;
}
