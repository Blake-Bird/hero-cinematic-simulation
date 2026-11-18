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
 * @brief Evaluates the primary hydrodynamic interaction process.
 *
 * This function processes particles in the hydrodynamic phase, grouping them based on their
 * proximity and evaluating hydrodynamic interactions. The evaluation considers various factors,
 * including particle smoothing lengths and positions. The function is executed by individual threads,
 * allowing parallel processing.
 *
 * @param p Pointer to the thread ID.
 * @param is_primary_on_gpu Flag indicating if the primary evaluation is performed on the GPU.
 * @param should_drift Flag indicating if drift correction should be applied.
 * @param group_particles Flag indicating if particles should be grouped during evaluation.
 * @return void* Returns NULL upon completion.
 */
void *ar_green_hydro_evaluate_primary(void *p, int is_primary_on_gpu, int should_drift, int group_particles)
{
    int thread_id = *(int *)p;
    int i, j;
    int *exportflag, *exportnodecount, *exportindex, *ngblist, *recyclingbuffer;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    recyclingbuffer = RecyclingBuffer + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    int *group_ids;
    MyLongDouble xtmp;
    int k;
    group_ids = recyclingbuffer;

    int mode = 0;
    if (is_primary_on_gpu)
    {
        mode = 3;
    }

    /* Note: exportflag is local to each thread */
    for (j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }
    int max_group_len = -1;
    int skipped_active = 0;
    int processed_particles = 0;
    //  printf("Task=%d:thread=%d going to hydro primary\n",ThisTask,thread_id);
    while (1)
    {
        int break_em_all = 0;

        int group_len = 0;
        int exitFlag = 0, first_group_particle = -1;
        MyLongDouble dx, dy, dz, dist2, cluster_dist_2 = -1.;
        MyFloat max_r2, max_h, max_r1, max_h_p_r;
        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
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
                        {
                            exitFlag = 1;
                        }
                        break;
                    }
#ifdef BLACK_HOLES
                    if (!(P[NextParticle].Type == 0 && P[NextParticle].Mass > 0))
#else // BLACK_HOLES
                    if (!(P[NextParticle].Type == 0))
#endif // BLACK_HOLES
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
                        continue;
                    }

                    if (!group_particles)
                    {
                        break;
                    }
#ifdef AR_GREEN_TREE_MAX_GROUP_SIZE
                    if (group_len >= AR_GREEN_TREE_MAX_GROUP_SIZE)
                    {
                        break;
                    }
#endif // AR_GREEN_TREE_MAX_GROUP_SIZE

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
                        NextParticle = NextActiveParticle[NextParticle];
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        UNLOCK_NEXPORT;

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
            for (i_target = 0; i_target < n_targets; i_target++)
            {
                int target = targets[i_target];
                int i = target;
                //     printf("task %d, evaluate target %d=%d maxhmsml=%f, maxdist=%f
                //     radius=%f\n",ThisTask,i_target,target,max_hsml,max_dist,search_radius);
#ifdef BLACK_HOLES
                if (!(P[target].Type == 0 && P[target].Mass > 0))
#else // BLACK_HOLES
                if (!(P[target].Type == 0))
#endif // BLACK_HOLES
                {
                    ENDRUN("Error: target is not gas particle and mass is not positive");
                }
                if (extended_numngb_inbox == -2)
                {
                    i_startnode = node;
                    extended_numngb_inbox = ar_green_ngb_treefind_pairs_threads(
                        P[target].Pos, &max_h, &max_r1, &i_startnode, mode, exportflag, exportnodecount, exportindex,
                        ngblist, targets, n_targets, should_drift);
                }

                if (hydro_evaluate(target, mode, ngblist, extended_numngb_inbox, &i_startnode, node, i_node, NULL) < 0)
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
 * @brief Evaluates the secondary hydrodynamic interaction process.
 *
 * This function processes particles in the secondary hydrodynamic phase, grouping them based on their
 * proximity and evaluating hydrodynamic interactions. The function is executed by individual threads,
 * enabling parallel processing of particle interactions, and considers factors such as particle positions
 * and smoothing lengths.
 *
 * @param p Pointer to the thread ID.
 * @param should_drift Flag indicating if drift correction should be applied during the evaluation.
 * @return void* Returns NULL upon completion.
 */
void *ar_green_hydro_evaluate_secondary(void *p, int should_drift)
{
    int thread_id = *(int *)p;

    int j, dummy, *ngblist, *recyclingbuffer, processed_particles = 0;
    //  printf("Task=%d:thread=%d OMG mom!, going to hydro secondary\n",ThisTask,thread_id);
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
        MyFloat max_r2, max_h, max_r1, max_h_p_r;
        int break_em_all = 0;
        int n_nodes = 0;
        int k;
        int same_nodes = 1;
        int group_len = 0;
        int exitFlag = 0, first_group_particle = -1;

        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {

#ifdef ACC_HYDRA
            struct hydrodata_in *H;
#else // ACC_HYDRA
            struct hydrodata_in *H;
#endif // ACC_HYDRA

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

                    H = &HydroDataGet[first_group_particle];

                    max_h = H->Hsml;

                    max_r2 = 0.;
                    group_len++;
                    processed_particles++;
                    NextJ++;
                    n_nodes = 0;

                    while (DataNodeListGet[first_group_particle].NodeList[n_nodes] >= 0)
                    {
                        n_nodes++;
                        if (n_nodes == NODELISTLENGTH)
                        {
                            break;
                        }
                    }

                    cluster_dist_2 = H->Hsml * AR_GREEN_TREE_HSML_FRACTION;
                    cluster_dist_2 *= cluster_dist_2;
                    //      break;
                    continue;
                }

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
                    //      same_nodes=0;           break;
                    if (i_node > n_nodes)
                    {
                        same_nodes = 0;
                        break;
                    }
                    if (i_node == NODELISTLENGTH)
                    {
                        break;
                    }
                }
                if (i_node != n_nodes)
                {
                    same_nodes = 0;
                }

                if (same_nodes != 1)
                {
                    break;
                }

                struct hydrodata_in *N = &HydroDataGet[NextJ];

                dx = NGB_PERIODIC_LONG_X(H->Pos[0] - N->Pos[0]);
                dy = NGB_PERIODIC_LONG_Y(H->Pos[1] - N->Pos[1]);
                dz = NGB_PERIODIC_LONG_Z(H->Pos[2] - N->Pos[2]);
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
        UNLOCK_NEXPORT;
        if (exitFlag == 1)
        {
            break;
        }
        // printf("Task%d - grupd %d !\n",ThisTask,NextParticle);
        if (max_group_len < 0 || max_group_len < group_len)
        {
            max_group_len = group_len;
        }
        max_r1 = sqrt(max_r2);
        max_h_p_r = max_r1 + max_h;
        for (k = 0; k < group_len;)
        {
            group_ids[k] = first_group_particle;
            first_group_particle++;
            k++;
        }

        int *targets;
        targets = group_ids;
        int n_targets = group_len;
        int i_target;
        int i_startnode;
        int *nodes = DataNodeListGet[targets[0]].NodeList;
        i_node = 0;
        while (nodes[i_node] >= 0)
        {
            int extended_numngb_inbox = -2;
            int node;
            node = nodes[i_node];
            if (node < 0)
            {
                break;
            }
            ENDRUN_IF(node >= All.MaxPart + MaxNodes, "Node is not valid.");
            if (Nodes[node].u.d.nextnode >= All.MaxPart + MaxNodes)
            {
                ENDRUN_MAIN("target=%d+%d mode=%d node=%d inode=%d  nextnode=%d>%d+%d=%d", targets[0], n_targets, 1,
                            node, i_node, Nodes[node].u.d.nextnode, All.MaxPart, MaxNodes, All.MaxPart + MaxNodes);
            }
            node = Nodes[node].u.d.nextnode;
            ENDRUN_IF(node >= All.MaxPart + MaxNodes, "Node is not valid");
            //       if(mode==1)node=Nodes[node].u.d.nextnode;
            //      printf("task %d, evaluate node %d=%d\n",ThisTask,i_node,node);
            for (i_target = 0; i_target < n_targets; i_target++)
            {
                int target = targets[i_target];
                //     printf("task %d, evaluate target %d=%d maxhmsml=%f, maxdist=%f
                //     radius=%f\n",ThisTask,i_target,target,max_hsml,max_dist,search_radius);
                if (extended_numngb_inbox == -2)
                {
                    struct hydrodata_in *T = &HydroDataGet[target];

                    i_startnode = node;

                    extended_numngb_inbox =
                        ar_green_ngb_treefind_pairs_threads(T->Pos, &max_h, &max_r1, &i_startnode, 1, &dummy, &dummy,
                                                            &dummy, ngblist, targets, n_targets, should_drift);
                }
                if (extended_numngb_inbox == -1)
                {
                    return NULL;
                }
                if (hydro_evaluate(target, 1, ngblist, extended_numngb_inbox, &i_startnode, node, i_node, NULL) < 0)
                {
                    return NULL;
                }
            }

            if (++i_node >= NODELISTLENGTH)
            {
                break;
            }
        }
        //            printf("Task=%d:thread=%d goin to hydra eval
        //            %d+%d<%d<%d\n",ThisTask,thread_id,group_ids[0],group_len,NextJ,Nimport);
    }
    //  printf("Task=%d:thread=%d OMG mom!, DONE hydro secondary maxgrouplen=%d\n",ThisTask,thread_id,max_group_len);
    return NULL;
}
