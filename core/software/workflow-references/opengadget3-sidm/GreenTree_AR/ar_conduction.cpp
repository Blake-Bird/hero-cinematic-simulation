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

int ar_green_ngb_treefind_pairs_threads(MyLongDouble *, MyFloat *, MyFloat *, int *, int, int *, int *, int *, int *,
                                        int *, int, int);

/**
 * @brief Evaluates primary conduction.
 *
 * This function processes gas particles in groups to perform the primary conduction phase.
 * The function can operate on either CPU or GPU. It handles the
 * evaluation of conduction interactions between particles, considering whether particles should be grouped
 * and drift applied.
 *
 * @param p Pointer to thread parameters containing thread-specific data such as input/output buffers and sums.
 * @param is_primary_on_gpu Flag indicating if the primary conduction is executed on the GPU (1) or CPU (0).
 * @param should_drift Flag indicating whether to apply drift during the conduction evaluation.
 * @param group_particles Flag indicating whether particles should be grouped for processing.
 * @return A void pointer, which is always NULL in this implementation.
 */
void *ar_conduction_evaluate_primary(void *p, int is_primary_on_gpu, int should_drift, int group_particles)
{
    struct thread_params *my_threadparams = (struct thread_params *)p;
    int thread_id = my_threadparams->tid;
    double *in = my_threadparams->in;
    double *out = my_threadparams->out;
    double *sum = my_threadparams->sum;

    int mode = 0;
    if (is_primary_on_gpu)
    {
        mode = 3;
    }

    int i;
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

    /* Note: exportflag is local to each thread */
    for (int j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }
    int max_group_len = -1;
    int skipped_active = 0;
    int processed_particles = 0;
    // printf("Task=%d:thread=%d going to conduction primary\n",ThisTask,thread_id);
    while (1)
    {
        int break_em_all = 0;

        int group_len = 0;
        int exitFlag = 0, first_group_particle = -1;
        MyLongDouble dx, dy, dz, dist2, cluster_dist_2 = -1.;
        MyFloat max_r2, max_h, max_r1, max_h_p_r;
#pragma omp critical(_nexport_)
        {
            //      if(BufferFullFlag != 0 || NextParticle < 0 || NextParticle >= N_gas)
            if (BufferFullFlag != 0 || NextParticle >= N_gas)
            {
                exitFlag = 1;
            }
            else
            {
                while (1)
                {
                    //              if(NextParticle<0 || NextParticle >= N_gas) {
                    if (NextParticle >= N_gas)
                    {
                        if (first_group_particle == -1)
                            exitFlag = 1;
                        break;
                    }
                    if (P[NextParticle].Mass <= 0)
                    {
                        ProcessedFlag[NextParticle] = 1;
                        NextParticle++;
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
                            NextParticle++; // = NextActiveParticle[NextParticle];
                        }
                        else
                            break;
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
                for (i_target = 0; i_target < n_targets; i_target++)
                {
                    int target = targets[i_target];
                    int i = target;
                    //     printf("task %d, evaluate target %d=%d maxhmsml=%f, maxdist=%f
                    //     radius=%f\n",ThisTask,i_target,target,max_hsml,max_dist,search_radius);

                    ENDRUN_IF(!(P[target].Type == 0), "Particle is not of type gas");
                    if (extended_numngb_inbox == -2)
                    {
                        i_startnode = node;
                        extended_numngb_inbox = ar_green_ngb_treefind_pairs_threads(
                            P[target].Pos, &max_h, &max_r1, &i_startnode, mode, exportflag, exportnodecount,
                            exportindex, ngblist, targets, n_targets, should_drift);
                    }

                    if (conduction_evaluate(target, mode, ngblist, extended_numngb_inbox, &i_startnode, node, i_node,
                                            my_threadparams) < 0)
                    {
                        return NULL;
                    }

                    ProcessedFlag[i] = 1; /* particle successfully finished */
                }
            }
            continue;
        }
        // printf("Task=%d:thread=%d finishing onduction primary %d, %d
        // ,%d\n",ThisTask,thread_id,processed_particles,NextParticle,N_gas);
        //   printf("Task%d - 1fase fine. max_group_len=%d, skipped_active=%d
        //   processed=%d\n",ThisTask,max_group_len,skipped_active,processed_particles);
        return NULL;
    }

    /**
     * @brief Evaluate secondary conduction.
     *
     * This function processes particles in groups based on shared node lists and computes the secondary
     * conduction phase for each group.
     *
     * @param p Pointer to thread parameters containing thread-specific data, including output buffers and sums.
     * @param should_drift Flag indicating whether to apply drift during the conduction evaluation.
     * @return A void pointer, which is always NULL
     */
    void *ar_conduction_evaluate_secondary(void *p, int should_drift)
    {
        struct thread_params *my_threadparams = (struct thread_params *)p;
        int thread_id = my_threadparams->tid;

        double *out = my_threadparams->out;
        double *sum = my_threadparams->sum;

        int j, dummy, *ngblist, *recyclingbuffer, processed_particles = 0;
        //  printf("Task=%d:thread=%d OMG mom!, going to hydro secondary\n",ThisTask,thread_id);
        ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
        recyclingbuffer = RecyclingBuffer + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

        int *group_ids;
        group_ids = recyclingbuffer;
        int exitFlag = 0;
        int i_node, max_group_len = -1;
        MyLongDouble xtmp;
        // printf("Task=%d:thread=%d going to conduction secondary\n",ThisTask,thread_id);
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

#pragma omp critical(_nexport_)
            {
#ifdef ACC_CONDUCTION
                struct conductiondata_in *ConDI;
#else // ACC_CONDUCTION
                struct conductiondata_in *ConDI;
#endif // ACC_CONDUCTION

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
                        max_r2 = 0.;
                        group_len++;
                        processed_particles++;
                        NextJ++;
                        n_nodes = 0;

#ifdef ACC_CONDUCTION
                        //      struct base_conductiondata_in *ConDI;
                        ConDI = &ConductionDataGet[first_group_particle];
#else // ACC_CONDUCTION
      //      struct conductiondata_in *ConDI;
                        ConDI = &ConductionDataGet[first_group_particle];
#endif // ACC_CONDUCTION

                        max_h = ConDI->Hsml;

                        while (DataNodeListGet[first_group_particle].NodeList[n_nodes] >= 0)
                        {
                            n_nodes++;
                            if (n_nodes == NODELISTLENGTH)
                            {
                                break;
                            }
                        }

                        cluster_dist_2 = ConDI->Hsml * AR_GREEN_TREE_HSML_FRACTION;
                        cluster_dist_2 *= cluster_dist_2;
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
                        break;
#ifdef ACC_CONDUCTION
                    struct conductiondata_in *N = &ConductionDataGet[NextJ];
#else // ACC_CONDUCTION
                    struct conductiondata_in *N = &ConductionDataGet[NextJ];
#endif // ACC_CONDUCTION
                    dx = NGB_PERIODIC_LONG_X(ConDI->Pos[0] - N->Pos[0]);
                    dy = NGB_PERIODIC_LONG_Y(ConDI->Pos[1] - N->Pos[1]);
                    dz = NGB_PERIODIC_LONG_Z(ConDI->Pos[2] - N->Pos[2]);

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
                ENDRUN_IF(node >= All.MaxPart + MaxNodes, "node is not a valid node");
                if (Nodes[node].u.d.nextnode >= All.MaxPart + MaxNodes)
                {
                    ENDRUN("target=%d+%d mode=%d node=%d inode=%d  nextnode=%d>%d+%d=%d", targets[0], n_targets, 1,
                           node, i_node, Nodes[node].u.d.nextnode, All.MaxPart, MaxNodes, All.MaxPart + MaxNodes);
                }
                node = Nodes[node].u.d.nextnode;
                ENDRUN_IF(node >= All.MaxPart + MaxNodes, "node is not a valid one");
                //       if(mode==1)node=Nodes[node].u.d.nextnode;
                //      printf("task %d, evaluate node %d=%d\n",ThisTask,i_node,node);
                for (i_target = 0; i_target < n_targets; i_target++)
                {
                    int target = targets[i_target];
                    //     printf("task %d, evaluate target %d=%d maxhmsml=%f, maxdist=%f
                    //     radius=%f\n",ThisTask,i_target,target,max_hsml,max_dist,search_radius);
                    if (extended_numngb_inbox == -2)
                    {

#ifdef ACC_CONDUCTION
                        struct conductiondata_in *CDT = &ConductionDataGet[target];
#else // ACC_CONDUCTION
                        struct conductiondata_in *CDT = &ConductionDataGet[target];
#endif // ACC_CONDUCTION

                        i_startnode = node;
                        extended_numngb_inbox = extended_numngb_inbox = ar_green_ngb_treefind_pairs_threads(
                            CDT->Pos, &max_h, &max_r1, &i_startnode, 1, &dummy, &dummy, &dummy, ngblist, targets,
                            n_targets, should_drift);
                    }
                    if (extended_numngb_inbox == -1)
                    {
                        return NULL;
                    }
                    if (conduction_evaluate(target, 1, ngblist, extended_numngb_inbox, &i_startnode, node, i_node,
                                            my_threadparams) < 0)
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
        // printf("Task=%d:thread=%d done conduction secondary\n",ThisTask,thread_id);

        //  printf("Task=%d:thread=%d OMG mom!, DONE hydro secondary
        //  maxgrouplen=%d\n",ThisTask,thread_id,max_group_len);
        return NULL;
    }
