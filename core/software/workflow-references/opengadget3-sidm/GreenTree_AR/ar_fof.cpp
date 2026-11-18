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
 * @brief Checks if a particle is active for friends-of-friends (FOF) processing.
 *
 * Determines if a particle is active for FOF linking based on its type and nearest distance.
 * A particle is considered active if its type matches the specified secondary link types
 * and its nearest distance exceeds a predefined threshold.
 *
 * @param i Index of the particle to check.
 * @return int Returns 1 if the particle is active, 0 otherwise.
 */
inline static int fof_isactive(int i)
{
    return (((1 << P[i].Type) & (MyFOF_SECONDARY_LINK_TYPES)) && fof_nearest_distance[i] > 1.0e29);
}

#define UNLOCK_NEXPORT
#define LOCK_NEXPORT

/**
 * @brief Evaluates the primary Friends-of-Friends (FOF) linking process for a thread.
 *
 * This function handles the primary phase of the FOF linking algorithm for a single thread,
 * processing particles to determine their grouping based on spatial proximity. It supports
 * optional GPU processing, drift correction, and particle grouping. The function iteratively
 * groups nearby particles and evaluates their nearest dark matter (DM) particle neighbors.
 *
 * @param p Pointer to the thread ID.
 * @param is_primary_on_gpu Flag indicating if the primary process should be executed on the GPU.
 * @param should_drift Flag indicating if drift correction should be applied.
 * @param group_particles Flag indicating if particles should be grouped together.
 * @return void* Returns NULL upon completion.
 */
void *ar_green_fof_evaluate_primary(void *p, int is_primary_on_gpu, int should_drift, int group_particles)
{
    int thread_id = *(int *)p;
    int i, j;
    int *exportflag, *exportnodecount, *exportindex, *ngblist, *recyclingbuffer;
    int mode = 0;
    if (is_primary_on_gpu)
    {
        mode = 3;
    }

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    recyclingbuffer = RecyclingBuffer + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    int *group_ids;
    MyLongDouble xtmp;
    int k;
    group_ids = recyclingbuffer;

    /* Note: exportflag is local to each thread */
    for (j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }
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
            if (BufferFullFlag != 0 || NextParticle >= NumPart)
            {
                exitFlag = 1;
            }
            else
            {
                while (1)
                {
                    if (NextParticle >= NumPart)
                    {
                        if (first_group_particle == -1)
                        {
                            exitFlag = 1;
                        }
                        break;
                    }

                    if (!fof_isactive(NextParticle))
                    {
                        ProcessedFlag[NextParticle] = 1;
                        NextParticle++; // = NextActiveParticle[NextParticle];
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
                        NextParticle++; // = NextActiveParticle[NextParticle];

                        if (HasFisrtParticleBeenProcessed == 0)
                        {
                            HasFisrtParticleBeenProcessed = 1;
                            break;
                        }

                        if (!group_particles)
                        {
                            break;
                        }

#ifdef AR_ONE_PARTICLE_PER_GROUP_I
                        break;
#endif // AR_ONE_PARTICLE_PER_GROUP_I
       // #if defined(LT_STELLAREVOLUTION)
       //                   if(P[first_group_particle].Type==5)break;
       // #endif
                        continue;
                    }
#ifdef AR_GREEN_TREE_MAX_GROUP_SIZE
                    if (group_len >= AR_GREEN_TREE_MAX_GROUP_SIZE)
                    {
                        break;
                    }
#endif // AR_GREEN_TREE_MAX_GROUP_SIZE
#if defined(LT_STELLAREVOLUTION)
                    if (P[NextParticle].Type == 5)
                    {
                        break;
                    }
#endif // LT_STELLAREVOLUTION

                    dx = NGB_PERIODIC_LONG_X(P[first_group_particle].Pos[0] - P[NextParticle].Pos[0]);
                    dy = NGB_PERIODIC_LONG_Y(P[first_group_particle].Pos[1] - P[NextParticle].Pos[1]);
                    dz = NGB_PERIODIC_LONG_Z(P[first_group_particle].Pos[2] - P[NextParticle].Pos[2]);
                    dist2 = dx * dx + dy * dy + dz * dz;

                    if (dist2 <= cluster_dist_2)
                    {
                        if (dist2 > max_r2)
                        {
                            max_r2 = dist2;
                        }
                        if (P[NextParticle].Hsml > max_h)
                        {
                            max_h = P[NextParticle].Hsml;
                        }
                        group_ids[group_len] = NextParticle;
                        group_len++;

                        processed_particles++;
                        ProcessedFlag[NextParticle] = 0;
                        NextParticle++; // = NextActiveParticle[NextParticle];
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }

        if (max_group_len < 0 || max_group_len < group_len)
        {
            max_group_len = group_len;
        }
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
                ENDRUN_IF(!fof_isactive(target), "target is not active");
                //     printf("task %d, evaluate target %d=%d maxhmsml=%f, maxdist=%f
                //     radius=%f\n",ThisTask,i_target,target,max_hsml,max_dist,search_radius);
                if (extended_numngb_inbox == -2)
                {
                    itarget = target;
                    i_startnode = node;
                    extended_numngb_inbox = ar_green_ngb_treefind_fof_variable_threads(
                        P[target].Pos, &max_h_p_r, &i_startnode, mode, exportflag, exportnodecount, exportindex,
                        ngblist, targets, n_targets, should_drift, MyFOF_PRIMARY_LINK_TYPES);

                    ENDRUN_IF(
                        extended_numngb_inbox >= ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)),
                        "[AR] [DIE] [n_extended_neighbourlist_size=%d] , max=numpart/density_less_ngb_factor=%d \n",
                        extended_numngb_inbox, ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)));
                }

                if (extended_numngb_inbox == -1)
                {
                    return NULL;
                }

                VERBOSE_ALL(2, "ThisTask%d elboro%d\n", ThisTask, i);
                if (fof_find_nearest_dmparticle_evaluate(target, mode, ngblist, extended_numngb_inbox, &i_startnode,
                                                         node, i_node) < 0)
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
 * @brief Evaluates the secondary Friends-of-Friends (FOF) linking process for a thread.
 *
 * This function handles the secondary phase of the FOF linking algorithm for a single thread.
 * It groups particles based on their proximity and shared node lists, then evaluates their
 * nearest dark matter (DM) particle neighbors. This process continues iteratively until all
 * particles are processed.
 *
 * @param p Pointer to the thread ID.
 * @param should_drift Flag indicating if drift correction should be applied.
 * @return void* Returns NULL upon completion.
 */
void *ar_green_fof_evaluate_secondary(void *p, int should_drift)
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

#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            struct fofdata_in *D;
            while (1)
            {
                if (NextJ >= Nimport)
                {
                    if (first_group_particle == -1)
                    {
                        exitFlag = 1;
                    }
                    break;
                }

                if (first_group_particle == -1)
                {
                    first_group_particle = NextJ;
                    group_len++;
                    processed_particles++;
                    NextJ++;
                    n_nodes = 0;

                    D = &FoFDataGet[first_group_particle];

                    while (FoFDataGet[first_group_particle].NodeList[n_nodes] >= 0)
                    {
                        n_nodes++;
                        if (n_nodes == NODELISTLENGTH)
                        {
                            break;
                        }
                    }
                    max_h = D->Hsml;
                    max_r2 = 0.;
                    cluster_dist_2 = D->Hsml * AR_GREEN_TREE_HSML_FRACTION;
                    cluster_dist_2 *= cluster_dist_2;

#if defined(LT_STELLAREVOLUTION)
//          if(D->Type==5) break;
#endif // LT_STELLAREVOLUTION
                    continue;
                }

                same_nodes = 1;
                // surprise!in the secondary phase, group them if they share the same exchanged nodelist!
                i_node = 0;
                while (FoFDataGet[NextJ].NodeList[i_node] >= 0)
                {
                    if (FoFDataGet[first_group_particle].NodeList[i_node] != FoFDataGet[NextJ].NodeList[i_node])
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

                struct fofdata_in *N = &FoFDataGet[NextJ];

#ifdef AR_ONE_PARTICLE_PER_GROUP_II
                break;
#endif // AR_ONE_PARTICLE_PER_GROUP_II
                if (same_nodes != 1)
                {
                    break;
                }
                dx = NGB_PERIODIC_LONG_X(D->Pos[0] - N->Pos[0]);
                dy = NGB_PERIODIC_LONG_Y(D->Pos[1] - N->Pos[1]);
                dz = NGB_PERIODIC_LONG_Z(D->Pos[2] - N->Pos[2]);
                dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 <= cluster_dist_2 && same_nodes == 1)
                {
                    if (dist2 > max_r2)
                    {
                        max_r2 = dist2;
                    }
                    if (N->Hsml > max_h)
                    {
                        max_h = N->Hsml;
                    }
                    group_len++;
                    processed_particles++;
                    NextJ++;
                }
                else
                {
                    break;
                }
            }
        }

        if (exitFlag == 1)
        {
            break;
        }
        // printf("Task%d - grupd %d !\n",ThisTask,NextParticle);
        max_r1 = sqrt(max_r2);
        max_r_p_h = max_r1 + max_h;
        if (max_group_len < 0 || max_group_len < group_len)
        {
            max_group_len = group_len;
        }

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
        int *nodes = FoFDataGet[targets[0]].NodeList;

        while (nodes[i_node] >= 0)
        {
            int extended_numngb_inbox = -2;
            int node = nodes[i_node];
            if (node < 0)
            {
                break;
            }
            if (i_node >= NODELISTLENGTH)
            {
                break;
            }

            ENDRUN_IF(node >= All.MaxPart + MaxNodes, "666");

            node = Nodes[node].u.d.nextnode;
            for (i_target = 0; i_target < n_targets; i_target++)
            {
                int target = targets[i_target];
                if (extended_numngb_inbox == -2)
                {
                    struct fofdata_in *T = &FoFDataGet[target];
                    i_startnode = node;
                    extended_numngb_inbox = ar_green_ngb_treefind_fof_variable_threads(
                        T->Pos, &max_r_p_h, &i_startnode, 1, &dummy, &dummy, &dummy, ngblist, targets, n_targets,
                        should_drift, MyFOF_PRIMARY_LINK_TYPES);
                }
                if (extended_numngb_inbox == -1)
                {
                    return NULL;
                }

                if (fof_find_nearest_dmparticle_evaluate(target, 1, ngblist, extended_numngb_inbox, &i_startnode, node,
                                                         i_node) < 0)
                {
                    return NULL;
                }
            } // end initial for loop over targets
            i_node++;
        }
    }
    //  printf("Task=%d:thread=%d OMG mom!, DONE dens secondary maxgrouplen=%d\n",ThisTask,thread_id,max_group_len);
    return NULL;
}
