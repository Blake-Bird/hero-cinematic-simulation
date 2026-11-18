/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the GreenTree algorithm.
 *
 * @author OpenGadget3-development team, in particular Antonio Ragagnin.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef GREENTREE_HPP
#define GREENTREE_HPP

#include <stdio.h>
#include <stdlib.h>

#ifdef AR_TREE_STATS
#include <map>
#endif // AR_TREE_STATS

#ifndef KD_EXTEND_SEARCH
#define KD_EXTEND_SEARCH 1.00001
#endif // KD_EXTEND_SEARCH

#define PTWO(x) x *x

/**
 * Enum representing the different phases of an iteration.
 */
enum IterationPhase
{
    IterationPhase_IPhaseComplete, /* neighbour search + fills of export buffer */
    IterationPhase_IIPhase, /* second phase */
    IterationPhase_IPhaseExportBuffer, /* only export buffer (used when doing neighbour search on GPU) */
    IterationPhase_IPhaseComputation, /* only neighbour search (possibly used when mergin OpenACC code to this routines)
                                       */
};

/* is_active functions for hydro and conduction criteria. Maybe they should sttay in their respective modules (as
 * density_isactive). */

/**
 * Check if the particle with the given index is active.
 *
 * This is a dummy function that always returns true, indicating that every particle
 * is considered active.
 *
 * @param n The index of the particle to check.
 * @return Always returns true.
 */
static inline int dummy_isactive(int n)
{
    return 1; // every particle is a good particle
}

/**
 * Check if the particle with the given index is active for hydrodynamics.
 *
 * A particle is considered active for hydrodynamics if its type is 0 and, optionally,
 * if it has positive mass depending on whether BLACK_HOLES is defined.
 *
 * @param n The index of the particle to check.
 * @return True if the particle is active for hydrodynamics, false otherwise.
 */
static inline int hydro_isactive(int n)
{
#ifdef BLACK_HOLES
    return (P[n].Type == 0 && P[n].Mass > 0);
#else // BLACK_HOLES
    return (P[n].Type == 0);
#endif // BLACK_HOLES
}

/**
 * Check if the particle with the given index is active for conduction.
 *
 * A particle is considered active for conduction if its type is 0 and its mass is greater than 0.
 *
 * @param n The index of the particle to check.
 * @return True if the particle is active for conduction, false otherwise.
 */
static inline int conduction_isactive(int n)
{
    return P[n].Type == 0 && P[n].Mass > 0;
}

/**
 * Check if the particle with the given index is active for diffusion.
 * A particle is considered active if its type is 0 and its mass is greater than 0.
 *
 * @param n The index of the particle to check.
 * @return True if the particle is active for diffusion, false otherwise.
 */
static inline int cr_diffusion_isactive(int n)
{
    return P[n].Type == 0 && P[n].Mass > 0;
}

/**
 * Check if a particle meets the distance criteria for grouping.
 *
 * This function determines if a particle meets the distance criteria for grouping based on
 * the distance from the center squared and the maximum allowed distance squared.
 *
 * @param dist_from_center_2 The squared distance from the center.
 * @param max_allowed_dist_2 The squared maximum allowed distance.
 * @param P Pointer to the particle data.
 * @param next_particle Index of the next particle.
 * @param first_particle Index of the first particle.
 * @return True if the particle meets the distance criteria for grouping, false otherwise.
 */
static inline int group_criteria_distance(MyFloat dist_from_center_2, MyFloat max_allowed_dist_2, void *P,
                                          int next_particle, int first_particle)
{
    return dist_from_center_2 <= max_allowed_dist_2;
}

/**
 * Filter function to exclude non-gas particles in tree walks.
 *
 * This function filters out non-gas particles during tree walks based on their type.
 *
 * @param P Pointer to the particle data array.
 * @param p Index of the particle to check.
 * @param DataGet Pointer to additional data.
 * @param targets Array of target indices.
 * @param n_targets Number of targets in the array.
 * @param mode The current iteration phase.
 * @return True if the particle is gas, false otherwise.
 */
static inline int treewalk_filter_gas_only(particle_data *P, int p, void *DataGet, int *targets, int n_targets,
                                           enum IterationPhase mode)
{
    return P[p].Type > 0;
}

/**
 * Enumeration representing different modules for debugging purposes.
 *
 * For debugging purposes it is useful to know which module is executing one of the following functions.
 * This struct is  a parameter of most of the functions below.
 */
enum Module
{
    Module_Gravity,
    Module_HSML,
    Module_Gradients,
    Module_Density,
    Module_Hydro,
#if GADGET_HYDRO == HYDRO_MFM
    Module_MfmLimiters,
    Module_MfmFluxes,
#endif // ADGET_HYDRO == HYDRO_MFM
    Module_Conduction,
#ifdef mSIDM
    Module_mSIDM,
#endif // mSIDM
    Module_Hsml_pt1,
    Module_CRDiffusion
};

/**
 * Enumeration representing different neighbor search criteria.
 */
enum NeighbourSearchCriteria
{
    NeighbourSearchCriteria_Density, /* tree walk only takes particles within a given searching radius h */
    NeighbourSearchCriteria_Hydro, /* tree walk takes particles whose smothing length intercept a given searching radius
                                      h */
#ifdef mSIDM
    NeighbourSearchCriteria_mSIDM,
#endif // mSIDM
#ifdef PXP
#ifndef ADAPTGRAVSOFT
    NeighbourSearchCriteria_Hsml_pt1
#endif // ADAPTGRAVSOFT
#endif // PXP
};

/**
 * Enumeration representing methods for choosing the next particle in a loop.
 *
 * - NextParticleLoop_ActiveParticleList uses the active particle list.
 * - NextParticleLoop_ZeroToNgas counts from 0 to N_gas (as done in conduction.c).
 */
enum NextParticleLoop
{
    NextParticleLoop_ActiveParticleList,
    NextParticleLoop_ZeroToNgas,
};

/**
 * Enumeration representing types of neighbor search algorithms.
 *
 * Ideally, in the future, you can switch to the 'good old Gadget2 tree walk' for debugging purposes.
 */
enum TreeWalkType
{
    TreeWalkType_GreenTreeWalk,
    TreeWalkType_Gadget3TreeWalk
};

/**
 * Ultra unified neighbor finding routine.
 *
 * This function performs neighbor finding using the AR Green algorithm in an OpenMP parallelized manner.
 *
 * @tparam TreewakFilterCriteria The criteria for filtering particles during the tree walk.
 * @tparam data_in The type of additional data required for the neighbor finding.
 * @param searchcenter The center of the search region.
 * @param phsml The smoothing length parameter.
 * @param pradius The search radius parameter.
 * @param startnode Pointer to the starting node.
 * @param mode The current iteration phase.
 * @param exportflag Pointer to the export flag array.
 * @param exportnodecount Pointer to the export node count array.
 * @param exportindex Pointer to the export index array.
 * @param ngblist Pointer to the neighbor list.
 * @param targets Array of target indices.
 * @param n_targets Number of targets in the array.
 * @param should_drift Flag indicating whether particle drifting should be performed.
 * @param tree_walk_search_type The type of neighbor search criteria.
 * @param max_ngbs Maximum number of neighbors to find (0 for unlimited).
 * @param module The module executing the function.
 * @param treewalk_filter_criteria The criteria for filtering particles during the tree walk.
 * @param DataGet Additional data required for the neighbor finding.
 * @return The number of neighbors found.
 */
template <typename TreewakFilterCriteria, typename data_in>
static inline int ar_green_ngb_openmp(MyLongDouble searchcenter[3], MyFloat phsml, MyFloat pradius, int *startnode,
                                      enum IterationPhase mode, int *exportflag, int *exportnodecount, int *exportindex,
                                      int *ngblist, int *targets, int n_targets, int should_drift,
                                      enum NeighbourSearchCriteria tree_walk_search_type, int max_ngbs,
                                      enum Module module, TreewakFilterCriteria treewalk_filter_criteria,
                                      data_in DataGet) /* max_ngbs==0 => all neighbours */
{
    int p, task, nexp;
    struct NODE *current;
    // cache some global vars locally for improved compiler alias analysis
    int maxPart = All.MaxPart;
    int maxNodes = MaxNodes;
    int bunchSize = All.BunchSize;
    MyFloat hsml = phsml; // KD_EXTEND_SEARCH;
    MyFloat radius = pradius; // KD_EXTEND_SEARCH;
    MyFloat hsmlpradius = hsml + radius;
    integertime ti_Current = All.Ti_Current;
    MyLongDouble dist(0.0), dx, dy, dz;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC

    int numngb = 0;
    int no = *startnode;

    while (no >= 0)
    {
        if (max_ngbs > 0 && numngb >= max_ngbs)
        { /* GPU processes neighbours in chunks of (typically) 32 */
            *startnode = no;
            return numngb;
        }

        if (no < maxPart) /* single particle */
        {
            p = no;
            no = Nextnode[no];

            if (treewalk_filter_criteria(P, p, DataGet, targets, n_targets, mode))
            {
                continue;
            }

#ifdef BLACK_HOLES
            if (P[p].Mass == 0)
            {
                continue;
            }
#endif // BLACK_HOLES

            if (should_drift)
            {
                if (P[p].Ti_current != ti_Current)
                {
#pragma omp critical(_partnodedrift_)
                    drift_particle(p, ti_Current);
                }
            }

            if (mode == IterationPhase_IPhaseExportBuffer)
            {
                continue;
            }

            if (tree_walk_search_type == NeighbourSearchCriteria_Hydro)
            {
                dist = std::max(P[p].Hsml, hsml) + radius;
            }
            else if (tree_walk_search_type == NeighbourSearchCriteria_Density)
            {
                dist = hsmlpradius;
            }
#ifdef mSIDM
#ifndef ADAPTGRAVSOFT
            // this is only an approximate criterion
            else if (tree_walk_search_type == NeighbourSearchCriteria_mSIDM)
            {
                dist = P[p].Hsml + hsml + radius;
            }
#else // ADAPTGRAVSOFT
      // this is only an approximate criterion
            else if (tree_walk_search_type == NeighbourSearchCriteria_mSIDM)
            {
                dist = P[p].AGS_Hsml + hsml + radius;
            }
#endif // ADAPTGRAVSOFT
#endif // mSIDM
#if defined(mSIDM) || defined(mIDM)
#ifndef ADAPTGRAVSOFT
            else if (tree_walk_search_type == NeighbourSearchCriteria_Hsml_pt1)
            {
                dist = hsmlpradius;
            }
#endif // ADAPTGRAVSOFT
#endif // defined(mSIDM) || defined(mIDM)
            else
            {
                ENDRUN_MAIN("TreeWalkType got wrong");
            }

            dx = NGB_PERIODIC_LONG_X(P[p].Pos[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            dy = NGB_PERIODIC_LONG_Y(P[p].Pos[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            dz = NGB_PERIODIC_LONG_Z(P[p].Pos[2] - searchcenter[2]);
            if (dz > dist)
            {
                continue;
            }
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

#ifdef AR_GRAVCOST_IN_DENSITY
            P[p].GravCost[TakeLevel] += 0.2;
#endif // AR_GRAVCOST_IN_DENSITY

            ngblist[numngb++] = p; /* Note: unlike in previous versions of the code, the buffer
                          can hold up to all particles */
        }
        else
        {
            if (no >= maxPart + maxNodes) /* pseudo particle */
            {
                int i_target = 0;
                ENDRUN_IF(mode == IterationPhase_IIPhase,
                          "mode =%d task=%d no=%d maxpart=%d maxnodes=%d +=%d startnode=%d ", mode, ThisTask, no,
                          All.MaxPart, maxNodes, maxPart + maxNodes, *startnode);

                if (targets[i_target] >= 0) /* if no target is given, export will not occur */
                {

                    if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != targets[i_target])
                    {
                        exportflag[task] = targets[i_target];
                        exportnodecount[task] = NODELISTLENGTH;
                    }

                    if (exportnodecount[task] == NODELISTLENGTH)
                    {
                        int exitFlag = 0;

#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
                        {
                            if (Nexport + n_targets - 1 >= bunchSize)
                            {
                                // out of buffer space. Need to discard work for this particle and interrupt
                                BufferFullFlag = 1;
                                exitFlag = 1;
                            }
                            else
                            {
                                nexp = Nexport;
                                Nexport += n_targets;
                            }
                        }

                        if (exitFlag)
                        {
                            return -1;
                        }
                        /*
                           #pragma omp atomic capture
                           nexp = Nexport++;
                         */
                        /*                      if(nexp>=bunchSize)
                           {
                           BufferFullFlag=1;
                           Nexport = bunchSize;
                           return -1;
                           }
                         */
                        //   if (exitFlag)
                        // {
                        //   return -1;
                        // }
                        exportnodecount[task] = 0;
                        exportindex[task] = nexp; // per questo task, indice della coppia nexp->particella
                        for (int i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataIndexTable[nexp + i_target].Task = task;
                            DataIndexTable[nexp + i_target].Index = targets[i_target];
                            DataIndexTable[nexp + i_target].IndexGet = nexp + i_target;
                        }
                    }

                    for (int i_target = 0; i_target < n_targets; i_target++)
                    {
                        DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];
                    }
                    exportnodecount[task]++;
                    if (exportnodecount[task] < NODELISTLENGTH)
                    {
                        for (int i_target = 0; i_target < n_targets; i_target++)
                        {
                            DataNodeList[exportindex[task] + i_target].NodeList[exportnodecount[task]] = -1;
                        }
                    }
                }
                no = Nextnode[no - maxNodes];
                continue;
            }

            current = &Nodes[no];
            if (mode == IterationPhase_IIPhase)
            {
                if (current->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                {
                    *startnode = -1;
                    return numngb;
                }
            }

            if (should_drift)
            {
                if (current->Ti_current != ti_Current)
                {
#pragma omp critical
                    force_drift_node(no, ti_Current);
                }
            }

            if (!(current->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
            {
                if (current->u.d.mass) /* open cell */
                {
                    no = current->u.d.nextnode;
                    continue;
                }
            }

            /* on I phase hybrid in CPU, we stay only in the toplevel nodes to fill the export buffer */
            if (mode == IterationPhase_IPhaseExportBuffer && !(Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
            {
                no = Nodes[no].u.d.sibling;
                continue;
            }

            if (tree_walk_search_type == NeighbourSearchCriteria_Hydro)
            {
                dist = std::max(Extnodes[no].hmax, hsml) + 0.5 * current->len + radius;
            }
            if (tree_walk_search_type == NeighbourSearchCriteria_Density)
            {
                dist = hsmlpradius + 0.5 * current->len;
            }
#ifdef mSIDM
            // this is only an approximate criterion
            if (tree_walk_search_type == NeighbourSearchCriteria_mSIDM)
            {
                dist = current->maxsoft + hsml + 0.5 * current->len + radius;
            }
#endif // mSIDM
#if defined(mSIDM) || defined(mIDM)
#ifndef ADAPTGRAVSOFT
            if (tree_walk_search_type == NeighbourSearchCriteria_Hsml_pt1)
            {
                dist = hsmlpradius + 0.5 * current->len;
            }
#endif // ADAPTGRAVSOFT
#endif // defined(mSIDM) || defined(mIDM)
            no = current->u.d.sibling; /* in case the node can be discarded */

            dx = NGB_PERIODIC_LONG_X(current->center[0] - searchcenter[0]);
            if (dx > dist)
            {
                continue;
            }
            dy = NGB_PERIODIC_LONG_Y(current->center[1] - searchcenter[1]);
            if (dy > dist)
            {
                continue;
            }
            dz = NGB_PERIODIC_LONG_Z(current->center[2] - searchcenter[2]);
            if (dz > dist)
            {
                continue;
            }
            /* now test against the minimal sphere enclosing everything */
            dist += FACT1 * current->len;
            if (dx * dx + dy * dy + dz * dz > dist * dist)
            {
                continue;
            }

            no = current->u.d.nextnode; /* ok, we need to open the node */
        }
    }

    *startnode = -1;
    return numngb;
}

/**
 * Picks the next particle to group in the first phase.
 *
 * This function does the GreenTree grouping on the first phase.
 *
 * @tparam IsActiveFunction Type of the function used to determine if a particle is active.
 * @tparam GroupCriteria Type of the function used to determine grouping criteria.
 * @param is_active Function to determine if a particle is active.
 * @param next_particle_loop Criteria to know how to increment NextParticle.
 * @param group_ids List of group IDs where to store the grouped particles.
 * @param group_len Length of the group.
 * @param max_r2 Output variable storing the maximum distance square between particles.
 * @param max_h Output variable storing the maximum h.
 * @param group_particles Flag indicating whether to group particles.
 * @param module Current module for debugging purposes.
 * @param processed_particles Number of processed particles.
 * @param group_criteria Criteria for grouping particles.
 * @param ProcessedFlag_this ProcessedFlag containing bit to be used for this loop.
 * @return 1 if successful, -1 otherwise.
 */
template <typename IsActiveFunction, typename GroupCriteria>
static inline int ar_pick_next_particle_to_group_in_i_phase(IsActiveFunction is_active,
                                                            NextParticleLoop next_particle_loop, int *group_ids,
                                                            int &group_len, MyFloat &max_r2, MyFloat &max_h,
                                                            int group_particles, Module module,
                                                            int &processed_particles, GroupCriteria group_criteria,
                                                            int ProcessedFlag_this, int max_group_len)
{
    int first_group_particle = -1;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC
    MyLongDouble dx, dy, dz, dist2, cluster_dist_2 = -1.;

    group_len = 0;
    if (BufferFullFlag != 0 || NextParticle < 0)
    {
        return -1;
    }

    while (1)
    { /* we search for the first particle of the group, but it must be active! */
        if ((next_particle_loop == NextParticleLoop_ActiveParticleList && NextParticle < 0) ||
            (next_particle_loop == NextParticleLoop_ZeroToNgas && NextParticle == N_gas))
        {
            /*we reached the last particle without even starting to group them.. gadget iteration is done! */
            return -1;
        }
        if (!is_active(NextParticle) || (ProcessedFlag[NextParticle] & ProcessedFlag_this))
        { /* skip the particle */
            ProcessedFlag[NextParticle] = (ProcessedFlag[NextParticle] & ProcessedFlag_this) | ProcessedFlag_finished;
            if (next_particle_loop == NextParticleLoop_ActiveParticleList)
            {
                NextParticle = NextActiveParticle[NextParticle];
            }
            if (next_particle_loop == NextParticleLoop_ZeroToNgas)
            {
                NextParticle++;
            }
            processed_particles++;
            continue;
        }

        max_r2 = 0.;
        first_group_particle = NextParticle;

        max_h = P[first_group_particle].Hsml;
#if (defined(mSIDM) || defined(mIDM)) && defined(ADAPTGRAVSOFT)
        max_h += (P[first_group_particle].AGS_Hsml - P[first_group_particle].Hsml) *
                 (module == Modulde_mSIDM || module == Module_mIDM);
#endif // defined(mSIDM) && defined(ADAPTGRAVSOFT)
        group_ids[group_len] = first_group_particle;
        group_len = 1;

#if defined(mSIDM) && defined(ADAPTGRAVSOFT)
        cluster_dist_2 = P[first_group_particle].AGS_Hsml * AR_GREEN_TREE_HSML_FRACTION;
#else // defined(mSIDM) && defined(ADAPTGRAVSOFT)
        cluster_dist_2 = P[first_group_particle].Hsml * AR_GREEN_TREE_HSML_FRACTION;
#endif // defined(mSIDM) && defined(ADAPTGRAVSOFT)
        cluster_dist_2 *= cluster_dist_2;
        ProcessedFlag[first_group_particle] = 0;
        if (next_particle_loop == NextParticleLoop_ActiveParticleList)
        {
            NextParticle = NextActiveParticle[NextParticle];
        }
        if (next_particle_loop == NextParticleLoop_ZeroToNgas)
        {
            NextParticle++;
        }

        break;
    }
    //  return 1;
    if (!group_particles)
    {
        return 1;
    }

#ifdef AR_ONE_PARTICLE_PER_GROUP_I
    return 1;
#endif // AR_ONE_PARTICLE_PER_GROUP_I

    /* I prefear to do black holes alone - may not be necessary */
    if (P[first_group_particle].Type == 5)
    {
        return 1;
    }

    while (1)
    {
        if ((next_particle_loop == NextParticleLoop_ActiveParticleList && NextParticle < 0) ||
            (next_particle_loop == NextParticleLoop_ZeroToNgas && NextParticle == N_gas))
        {

            break;
        }
        if (!is_active(NextParticle))
        {
            ProcessedFlag[NextParticle] = 1;
            if (next_particle_loop == NextParticleLoop_ActiveParticleList)
            {
                NextParticle = NextActiveParticle[NextParticle];
            }
            if (next_particle_loop == NextParticleLoop_ZeroToNgas)
            {
                NextParticle++;
            }

            continue;
        }
        if (group_len >= max_group_len)
            break;

#ifdef AR_GREEN_TREE_MAX_GROUP_SIZE
        if (group_len >= AR_GREEN_TREE_MAX_GROUP_SIZE)
        {
            break;
        }
#endif // AR_GREEN_TREE_MAX_GROUP_SIZE

        /* I prefear to do black holes alone - may not be necessary */
        if (P[NextParticle].Type == 5)
        {
            break;
        }

        dx = NGB_PERIODIC_LONG_X(P[first_group_particle].Pos[0] - P[NextParticle].Pos[0]);
        dy = NGB_PERIODIC_LONG_Y(P[first_group_particle].Pos[1] - P[NextParticle].Pos[1]);
        dz = NGB_PERIODIC_LONG_Z(P[first_group_particle].Pos[2] - P[NextParticle].Pos[2]);
        dist2 = dx * dx + dy * dy + dz * dz;

        if (group_criteria(dist2, cluster_dist_2, P, NextParticle, first_group_particle))
        { /* found a new particle to group */
            if (dist2 > max_r2)
            {
                max_r2 = dist2; // update max group size
            }
#ifdef mSIDM
            if (module == Module_mSIDM)
            {
#ifdef ADAPTGRAVSOFT
                max_h = std::max(P[NextParticle].AGS_Hsml, max_h);
#else // ADAPTGRAVSOFT
                max_h = std::max(P[NextParticle].Hsml, max_h);
#endif // ADAPTGRAVSOFT
            }
#endif // mSIDM
#if defined(mSIDM) || defined(mIDM)
                else
                {
#endif // defined(mSIDM) || defined(mIDM)
                    max_h = std::max(P[NextParticle].Hsml, max_h);
#if defined(mSIDM) || defined(mIDM)
                }
#endif // defined(mSIDM) || defined(mIDM)
            group_ids[group_len] = NextParticle; // add particle to group
            group_len++;
            ProcessedFlag[NextParticle] = 0;
            if (next_particle_loop == NextParticleLoop_ActiveParticleList)
            {
                NextParticle = NextActiveParticle[NextParticle];
            }
            if (next_particle_loop == NextParticleLoop_ZeroToNgas)
            {
                NextParticle++;
            }
        }
        else
        {
            break; /* ...next! */
        }

    } /* while ends */
    return 1; /*ok! */
}

/**
 * Picks the next particle to group in the second phase.
 *
 * This function groups guest particles.
 *
 * @tparam data_in Type of the data to retrieve.
 * @tparam GroupCriteria Type of the function used to determine grouping criteria.
 * @param DataGet Function to retrieve data.
 * @param first_group_particle Output array with the list of grouped particles.
 * @param group_len Output size of the group.
 * @param max_r2 Output variable storing the maximum distance square.
 * @param max_h Output variable storing the maximum smoothing length.
 * @param module Module for debugging purposes.
 * @param group_criteria Criteria for grouping particles.
 * @return 1 if successful, -1 otherwise.
 */
template <typename data_in, typename GroupCriteria>
static inline int ar_pick_next_particle_to_group_in_ii_phase(data_in DataGet, int &first_group_particle, int &group_len,
                                                             MyFloat &max_r2, MyFloat &max_h, Module module,
                                                             GroupCriteria group_criteria, int max_group_len)
{
    first_group_particle = -1;
#ifdef PERIODIC
    MyLongDouble xtmp;
#endif // PERIODIC
    MyLongDouble dx, dy, dz, dist2, cluster_dist_2 = -1.;
    /*
       if(NextJ<Nimport){
       first_group_particle=NextJ;
       group_len=1;
       NextJ++;
       max_r2=0.;
       max_h = DataGet[first_group_particle].Hsml;
       return 1;
       }else{
       return -1;
       }
     */
    int n_nodes = 0;
    while (1)
    {
        int same_nodes;
        if (NextJ >= Nimport)
        { /* we reached the last particle: end import */
            if (first_group_particle == -1)
            {
                return -1;
            }
            break;
        }

        if (first_group_particle == -1)
        {
            first_group_particle = NextJ;
            group_len++;
            NextJ++;
            n_nodes = 0;

            while (DataNodeListGet[first_group_particle].NodeList[n_nodes] >= 0)
            { /* II phase group also particles with same NodeList */
                n_nodes++;
                if (n_nodes == NODELISTLENGTH)
                {
#ifdef mSIDM
                    if (module == Module_mSIDM)
                    {
                        std::cout << "mSIDM Warning: pseudo particle, NODELISTLENGTH is too small (" << NODELISTLENGTH
                                  << ")" << std::endl;
#ifndef mSIDM_IGNOREWARNING
                        ENDRUN("This can be resolved by increasing NODELISTLENGTH_PXP in Config.sh.");
#endif // mSIDM_IGNOREWARNING
                    }
#endif // mSIDM

                    break;
                }
            }

            max_h = DataGet[first_group_particle].Hsml;
            max_r2 = 0.;
            cluster_dist_2 = DataGet[first_group_particle].Hsml * AR_GREEN_TREE_HSML_FRACTION;
            cluster_dist_2 *= cluster_dist_2;
            // #if defined(LT_STELLAREVOLUTION)
            //	  if (DataGet[first_group_particle].Type == 5)
            //	    {
            //	      break;		/* black holes are don alone */
            //	    }
            // #endif // defined(LT_STELLAREVOLUTION)

            continue;
        }

        data_in D = &DataGet[first_group_particle];
        // #if defined(LT_STELLAREVOLUTION)
        //       //    if(D->Type==5)break; /* black hole will be processed in the next group - alone*/
        // #endif
        same_nodes = 1;
        int i_node = 0;

        while (DataNodeListGet[NextJ].NodeList[i_node] >= 0)
        { /* check if the current particle share the same node list of all particles in the group  */
            if (DataNodeListGet[first_group_particle].NodeList[i_node] != DataNodeListGet[NextJ].NodeList[i_node])
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
        {
            break;
        }

        data_in N = &DataGet[NextJ];
        if (group_len >= max_group_len)
            break;
#ifdef AR_ONE_PARTICLE_PER_GROUP_II
        break;
#endif // AR_ONE_PARTICLE_PER_GROUP_II
       //   if (same_nodes != 1)
       // {
       //   break;
       // }
        dx = NGB_PERIODIC_LONG_X(D->Pos[0] - N->Pos[0]);
        dy = NGB_PERIODIC_LONG_Y(D->Pos[1] - N->Pos[1]);
        dz = NGB_PERIODIC_LONG_Z(D->Pos[2] - N->Pos[2]);
        dist2 = dx * dx + dy * dy + dz * dz;

        if (group_criteria(dist2, cluster_dist_2, DataGet, NextJ, first_group_particle)) // && same_nodes == 1)
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
            NextJ++;
        }
        else
        {
            break;
        }
    }
    return 1; /*ok! */
}

/**
 * Perform an inner loop for the green tree algorithm.
 *
 * Takes the list of grouped particles and a possible list of nodes (in case of the secondary phase), and performs:
 * - For each node:
 *   - Perform both a single tree walk for the whole group.
 *   - For each particle in the group:
 *     - Evaluate().
 *
 * @tparam IsActiveFunction Type of the function used to determine if a particle is active.
 * @tparam EvaluateFunction Type of the function used to evaluate particles.
 * @tparam DataIn Type of the input data.
 * @tparam TreeWalkFilter Type of the function used for tree walk filtering.
 * @param P_In Input data.
 * @param is_active Function to determine if a particle is active.
 * @param evaluate Function to evaluate particles.
 * @param tree_walk_search_type Criteria for neighbor search.
 * @param targets List of target particles.
 * @param n_targets Number of target particles.
 * @param thread_id Thread ID.
 * @param max_r Maximum distance.
 * @param max_h Maximum smoothing length.
 * @param mode Iteration phase.
 * @param should_drift Flag indicating whether to drift particles.
 * @param nodes_list List of nodes.
 * @param module Module for debugging purposes.
 * @param data Additional data.
 * @param tree_walk_filter Tree walk filter function.
 * @param ProcessedFlag_this ProcessedFlag containing bit to be used for this loop.
 * @return 1 if successful, -1 otherwise.
 */
template <typename IsActiveFunction, typename EvaluateFunction, typename DataIn, typename TreeWalkFilter>
static inline int ar_green_tree_inner_loop(DataIn P_In, IsActiveFunction is_active, EvaluateFunction evaluate,
                                           NeighbourSearchCriteria tree_walk_search_type, int *targets, int n_targets,
                                           int thread_id, MyFloat max_r, MyFloat max_h, IterationPhase mode,
                                           int should_drift, int *nodes_list, Module module, void *data,
                                           TreeWalkFilter tree_walk_filter, int ProcessedFlag_this)
{
    int *exportflag, *exportnodecount, *exportindex, *ngblist;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    if (mode != IterationPhase_IIPhase)
    {
        exportflag = Exportflag + thread_id * NTask;
        exportnodecount = Exportnodecount + thread_id * NTask;
        exportindex = Exportindex + thread_id * NTask;
    }
    else
    {
        exportflag = exportnodecount = exportindex = NULL;
    }
    int is_loop_ok = 1;
    int i_node = 0; /*on the secondary phase this will increase with the varioujs NODELISTs, and will signal the
                       evaluate function that it has to to DensDataOut instead of assigning. */

    do
    { /* loop over the list of nodes, that is not empty in the II phase */
        int extended_numngb_inbox = -2;
        int evaluate_in_node = All.MaxPart;
        int target0 =
            targets[0]; /* the tree walk is performed around the first particle, ofc over a length of max_h + max_r */

        if (i_node >= NODELISTLENGTH)
        { /* we finished the nodes to evaluate */
            return 1;
        }

        if (nodes_list != NULL)
        { /*if we got a node list, we use it */
            evaluate_in_node = nodes_list[i_node];
        }

        if (evaluate_in_node < 0)
        {
            return 1; /* we finished the nodes to evaluate */
        }
        if (evaluate_in_node > All.MaxPart + MaxNodes)
        {
            for (int i = 0; i < NODELISTLENGTH; i++)
            {
                VERBOSE_ALL(2, "%d ", nodes_list[i]);
            }
        }
        ENDRUN_IF(evaluate_in_node > All.MaxPart + MaxNodes,
                  "Bad node no=%d > %d+%d = %d; i_node=%d<NODELISTLENGTH=%d, nodes=%p", evaluate_in_node, All.MaxPart,
                  MaxNodes, All.MaxPart + MaxNodes, i_node, NODELISTLENGTH, nodes_list);
        ENDRUN_IF(Nodes[evaluate_in_node].u.d.nextnode >= All.MaxPart + MaxNodes,
                  "Very bad nextnode = %d > %d; mode=%d", Nodes[evaluate_in_node].u.d.nextnode, All.MaxPart + MaxNodes,
                  mode);

        if (nodes_list != NULL)
        { /* enter the node if mode==1 */
            evaluate_in_node = Nodes[evaluate_in_node].u.d.nextnode;
        }

        int i_startnode = evaluate_in_node;
        MyLongDouble *pos = P_In[target0].Pos;
        extended_numngb_inbox = ar_green_ngb_openmp(pos, max_h, max_r, &i_startnode, mode, exportflag, exportnodecount,
                                                    exportindex, ngblist, targets, n_targets, should_drift,
                                                    tree_walk_search_type, 0, module, tree_walk_filter, P_In);
        ENDRUN_IF(mode == IterationPhase_IPhaseExportBuffer && extended_numngb_inbox > 0,
                  "mode == IterationPhase_IPhaseExportBuffer && extended_numngb_inbox = %d>0", extended_numngb_inbox);

        /* MyFloat x = max_h + max_r;
           extended_numngb_inbox = ar_green_ngb_treefind_variable_threads(pos, &x, &i_startnode,
           (mode==IterationPhase::IterationPhase_IPhaseComplete)?0:1, exportflag, exportnodecount, exportindex, ngblist,
           targets, n_targets, should_drift); */
        ENDRUN_IF(extended_numngb_inbox < 0 && mode == IterationPhase_IIPhase, "Error in ngb search; maxh=%f maxr=%f",
                  max_h, max_r);

        if (extended_numngb_inbox < 0)
        {
            return -1;
        }

        ENDRUN_IF(extended_numngb_inbox > ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)),
                  "mode=%d,  numngb = %d max=numpart/density_less_ngb_factor=%d  ", mode, extended_numngb_inbox,
                  ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)));

        int i_target = 0;

#ifdef GREEN_TREE_OUTER_LOOP_NUM_THREADS
        int num_threads_inner_loop = maxThreads / GREEN_TREE_OUTER_LOOP_NUM_THREADS;
#pragma omp parallel for num_threads(num_threads_inner_loop)
#error ""
#endif // GREEN_TREE_OUTER_LOOP_NUM_THREADS
        for (i_target = 0; i_target < n_targets; i_target++)
        { /* loop over the grouped particles */
            //    VERBOSE(-1,"1. evaluate itarget=%d",i_target);
            if (!is_loop_ok)
            {
                continue;
            }
            int target = targets[i_target];
            ENDRUN_IF(mode != IterationPhase_IIPhase && !is_active(target),
                      "Processing a non active particle. mode=%d, target=%d is_active=%d", mode, target,
                      is_active(target));
            //      VERBOSE(-1,"2. evaluate %d i_startnode=%d maxpart=%d",target, i_startnode, All.MaxPart);

            if (mode != IterationPhase_IPhaseExportBuffer && evaluate(target, mode, ngblist, extended_numngb_inbox,
                                                                      &i_startnode, evaluate_in_node, i_node, data) < 0)
            {
                is_loop_ok = 0;
                ENDRUN_IF(mode == IterationPhase_IIPhase, "tragic error in evaluate");
            }
            else
            {
                if (mode == IterationPhase_IPhaseComplete || mode == IterationPhase_IPhaseExportBuffer)
                {
                    ProcessedFlag[target] =
                        ProcessedFlag_finished | ProcessedFlag_this; /* particle successfully finished */
                }
            }
        }
        if (is_loop_ok != 1)
        {
            return -1; /* buffer is full */
        }
        if (nodes_list != NULL)
        {
            i_node++;
        }
    } while ((nodes_list != NULL && nodes_list[i_node] >= 0 && i_node < NODELISTLENGTH));
    return is_loop_ok ? 1 : -1;
}

/*
enum NextParticleLoop{
  NextParticleLoop_ActiveParticleList,
  NextParticleLoop_ZeroToNgas,
};
*/

/**
 * Perform evaluation in the primary phase of the green tree algorithm.
 *
 * @tparam IsActive Type of the function used to determine if a particle is active.
 * @tparam Evaluate Type of the function used to evaluate particles.
 * @tparam GroupCriteria Type of the function used for grouping criteria.
 * @tparam TreeWalkFilter Type of the function used for tree walk filtering.
 * @param is_active Function to determine if a particle is active.
 * @param evaluate Function to evaluate particles.
 * @param module Module for debugging purposes.
 * @param search_type Type of tree walk.
 * @param next_particle_loop Type of iteration loop for next particle.
 * @param tree_walk_type Criteria for tree walk.
 * @param group_criteria Criteria for grouping particles.
 * @param treewalk_filter Tree walk filter function.
 * @param data Additional data.
 * @param is_primary_on_gpu Flag indicating if the primary phase is on GPU.
 * @param should_drift Flag indicating whether to drift particles.
 * @param group_particles Flag indicating whether to group particles.
 * @return void pointer.
 */
template <typename IsActive, typename Evaluate, typename GroupCriteria, typename TreeWalkFilter>
static inline void *ar_green_evaluate_primary(IsActive is_active, Evaluate evaluate, Module module,
                                              TreeWalkType search_type, NextParticleLoop next_particle_loop,
                                              NeighbourSearchCriteria tree_walk_type, GroupCriteria group_criteria,
                                              TreeWalkFilter treewalk_filter, void *data, int is_primary_on_gpu,
                                              int should_drift, int group_particles)
{
    return ar_green_evaluate_primary(is_active, evaluate, module, search_type, next_particle_loop, tree_walk_type,
                                     group_criteria, treewalk_filter, data, is_primary_on_gpu, should_drift,
                                     group_particles,
                                     ProcessedFlag_null); /* default behavior by using no bitwise information */
}

/**
 * Perform evaluation in the primary phase of the green tree algorithm.
 *
 * @tparam IsActive Type of the function used to determine if a particle is active.
 * @tparam Evaluate Type of the function used to evaluate particles.
 * @tparam GroupCriteria Type of the function used for grouping criteria.
 * @tparam TreeWalkFilter Type of the function used for tree walk filtering.
 * @param is_active Function to determine if a particle is active.
 * @param evaluate Function to evaluate particles.
 * @param module Module for debugging purposes.
 * @param search_type Type of tree walk.
 * @param next_particle_loop Type of iteration loop for next particle.
 * @param tree_walk_type Criteria for tree walk.
 * @param group_criteria Criteria for grouping particles.
 * @param treewalk_filter Tree walk filter function.
 * @param data Additional data.
 * @param is_primary_on_gpu Flag indicating if the primary phase is on GPU.
 * @param should_drift Flag indicating whether to drift particles.
 * @param group_particles Flag indicating whether to group particles.
 * @param ProcessedFlag_this ProcessedFlag containing bit to be used for this loop.
 * @return void pointer.
 */
template <typename IsActive, typename Evaluate, typename GroupCriteria, typename TreeWalkFilter>
static inline void *ar_green_evaluate_primary(IsActive is_active, Evaluate evaluate, Module module,
                                              TreeWalkType search_type, NextParticleLoop next_particle_loop,
                                              NeighbourSearchCriteria tree_walk_type, GroupCriteria group_criteria,
                                              TreeWalkFilter treewalk_filter, void *data, int is_primary_on_gpu,
                                              int should_drift, int group_particles, int ProcessedFlag_this)
{
    // int thread_id = *(int *) data;
    int thread_id = *static_cast<int *>(data);
    IterationPhase mode = IterationPhase_IPhaseComplete;
    if (is_primary_on_gpu)
    {
        mode = IterationPhase_IPhaseExportBuffer;
    }

    int *recyclingbuffer, *exportflag;
    int max_group_len = ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    recyclingbuffer = RecyclingBuffer + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    if (search_type == TreeWalkType_Gadget3TreeWalk)
    {
        group_particles = 0;
    }
    int id0; // extreme case where NumPart=0 and the recycling buffers all collapse to the same buffer
    int *group_ids;
    if (max_group_len > 0)
        group_ids = recyclingbuffer;
    else
    {
        // ENDRUN("NumPart/DENSITY_LESS_NGB_FACTOR=%d/%d=%d<=0",NumPart,DENSITY_LESS_NGB_FACTOR,max_group_len);
        group_ids = &id0;
    }

    exportflag = Exportflag + thread_id * NTask;

    /* Note: exportflag is local to each thread */
    for (int j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }

    int exit_flag = 0;
    int processed_particles = 0;
    ENDRUN_IF(thread_id == -123, "GREPME in ar_green_evaluate_primary");

    while (1)
    {
        int group_len = 0;
        MyFloat max_r2, max_h, max_r1;

        if ((next_particle_loop == NextParticleLoop_ActiveParticleList && NextParticle < 0) ||
            (next_particle_loop == NextParticleLoop_ZeroToNgas && NextParticle == N_gas))
        {
            return NULL;
        }

        // note that processed_particles is passsed as reference, see signature of
        // ar_pick_next_particle_to_group_in_i_phase.
#pragma omp critical
        exit_flag = ar_pick_next_particle_to_group_in_i_phase(
            is_active, next_particle_loop, recyclingbuffer, group_len, max_r2, max_h, group_particles, module,
            processed_particles, group_criteria, ProcessedFlag_this, max_group_len);
        /*note: ar_pick_next_particle_to_group get paramas by reference */

        if (exit_flag == -1)
        {
            return NULL;
        }

        max_r1 = (MyFloat)std::sqrt((MyAtLeastDouble)max_r2);

        if (group_len > 0)
        {
            int was_green_outerloop_ok = ar_green_tree_inner_loop(
                P, is_active, evaluate, tree_walk_type, group_ids, group_len, thread_id, max_r1, max_h, mode,
                should_drift, NULL, module, data, treewalk_filter, ProcessedFlag_this);
            if (was_green_outerloop_ok != 1)
            {
                return NULL; /* ar_first_phase_inner_loop returns -1 on error, as tradition of Gadget */
            }
        }
    }
}

/**
 * This function evaluates the primary particles.
 *
 * @tparam IsActive The type of the function that determines if a particle is active.
 * @tparam Evaluate The type of the function used to evaluate particles.
 * @param is_active Function to determine if a particle is active.
 * @param evaluate Function used to evaluate particles.
 * @param module The module to work with.
 * @param search_type The type of tree walk search.
 * @param next_particle_loop The loop for the next particle.
 * @param tree_walk_type The type of tree walk.
 * @param data The data to work with.
 * @param is_primary_on_gpu Flag indicating if primary particles are on the GPU.
 * @param should_drift Flag indicating if drifting should be performed.
 * @param group_particles Flag indicating if particles should be grouped.
 * @param ProcessedFlag_this
 * @return A pointer to the evaluated primary particles.
 */
template <typename IsActive, typename Evaluate>
static inline void *ar_green_evaluate_primary(IsActive is_active, Evaluate evaluate, Module module,
                                              TreeWalkType search_type, NextParticleLoop next_particle_loop,
                                              NeighbourSearchCriteria tree_walk_type, void *data, int is_primary_on_gpu,
                                              int should_drift, int group_particles, int ProcessedFlag_this
                                              //,GroupCriteria group_criteria, TreeWalkFilter tree_walk_filter
)
{
    return ar_green_evaluate_primary(is_active, evaluate, module, search_type, next_particle_loop, tree_walk_type,
                                     group_criteria_distance, treewalk_filter_gas_only, data, is_primary_on_gpu,
                                     should_drift, group_particles, ProcessedFlag_this);
}

/**
 * This function evaluates the primary particles.
 *
 * @tparam IsActive The type of the function that determines if a particle is active.
 * @tparam Evaluate The type of the function used to evaluate particles.
 * @param is_active Function to determine if a particle is active.
 * @param evaluate Function used to evaluate particles.
 * @param module The module to work with.
 * @param search_type The type of tree walk search.
 * @param next_particle_loop The loop for the next particle.
 * @param tree_walk_type The type of tree walk.
 * @param data The data to work with.
 * @param is_primary_on_gpu Flag indicating if primary particles are on the GPU.
 * @param should_drift Flag indicating if drifting should be performed.
 * @param group_particles Flag indicating if particles should be grouped.
 * @return A pointer to the evaluated primary particles.
 */
template <typename IsActive, typename Evaluate>
static inline void *ar_green_evaluate_primary(IsActive is_active, Evaluate evaluate, Module module,
                                              TreeWalkType search_type, NextParticleLoop next_particle_loop,
                                              NeighbourSearchCriteria tree_walk_type, void *data, int is_primary_on_gpu,
                                              int should_drift, int group_particles
                                              //,GroupCriteria group_criteria, TreeWalkFilter tree_walk_filter
)
{
    return ar_green_evaluate_primary(is_active, evaluate, module, search_type, next_particle_loop, tree_walk_type,
                                     group_criteria_distance, treewalk_filter_gas_only, data, is_primary_on_gpu,
                                     should_drift, group_particles);
}

/**
 * This function evaluates the secondary particles.
 *
 * @tparam data_in The type of input data.
 * @tparam Evaluate The type of function used to evaluate particles.
 * @tparam GroupCriteria The type of criteria used for grouping particles.
 * @tparam TreeWalkFilter The type of filter used for tree walking.
 * @param DataGet Function to retrieve input data.
 * @param evaluate Function used to evaluate particles.
 * @param module The module to work with.
 * @param search_type The type of tree walk search.
 * @param ngb_criteria The criteria for neighboring search.
 * @param group_criteria The criteria for grouping particles.
 * @param treewalk_filter The filter for tree walking.
 * @param data The data to work with.
 * @param should_drift Flag indicating if drifting should be performed.
 * @return A pointer to the evaluated secondary particles.
 */
template <typename data_in, typename Evaluate, typename GroupCriteria, typename TreeWalkFilter>
static inline void *ar_green_evaluate_secondary(data_in DataGet, Evaluate evaluate, Module module,
                                                TreeWalkType search_type, NeighbourSearchCriteria ngb_criteria,
                                                GroupCriteria group_criteria, TreeWalkFilter treewalk_filter,
                                                void *data, int should_drift)
{
    // int thread_id = *(int *) data;
    int thread_id = *static_cast<int *>(data);
    int *recyclingbuffer = 0;

    // auto group_criteria = boh1;
    // auto treewalk_filter = boh2;

    // ngblist = Ngblist + thread_id * ((int) (NumPart / DENSITY_LESS_NGB_FACTOR));
    recyclingbuffer = RecyclingBuffer + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    int max_group_len = ((int)(NumPart / DENSITY_LESS_NGB_FACTOR)); // in case Nimport is larger than NumPart...
    //  printf("Numport=%d NextJ=%d\n",Nimport, NextJ);
    int id0; // extreme case for NumPart==0 => recyling buffer all collapse
    int *group_ids;
    if (max_group_len > 0)
    {
        group_ids = recyclingbuffer;
    }
    else
    {
        // ENDRUN("NumPart/DENSITY_LESS_NGB_FACTOR=%d/%d=%d<=0",NumPart,DENSITY_LESS_NGB_FACTOR,max_group_len);
        group_ids = &id0;
    }

    int exit_flag;

    while (1)
    {
        int group_len = 0;
        MyFloat max_h;
        MyFloat max_r2;
        MyFloat max_r1;

        int first_group_particle;
        /*note: ar_pick_next_particle_to_group get paramas by reference */

#pragma omp critical(_nexport_)
        exit_flag = ar_pick_next_particle_to_group_in_ii_phase(DataGet, first_group_particle, group_len, max_r2, max_h,
                                                               module, group_criteria, max_group_len);

        if (exit_flag != 1)
        {
            return NULL;
        }

        if (first_group_particle == -1 || first_group_particle >= Nimport)
        {
            return NULL; /* w finished */
        }
        max_r1 = (MyFloat)std::sqrt((MyAtLeastDouble)max_r2);

        for (int k = 0; k < group_len; k++)
        {
            group_ids[k] = first_group_particle;
            first_group_particle++;
        }

        int *nodes = DataNodeListGet[group_ids[0]].NodeList;
        int was_green_outerloop_ok = ar_green_tree_inner_loop(
            DataGet, dummy_isactive, evaluate, ngb_criteria, group_ids, group_len, thread_id, max_r1, max_h,
            IterationPhase_IIPhase, should_drift, nodes, module, data, treewalk_filter, ProcessedFlag_null);
        ENDRUN_IF(was_green_outerloop_ok != 1, "II phase failed miserably");
        if (exit_flag == -1)
        {
            return NULL;
        }
    }
}

/**
 * This function evaluates the secondary particles.
 *
 * @tparam data_in The type of input data.
 * @tparam Evaluate The type of function used to evaluate particles.
 * @param DataGet Function to retrieve input data.
 * @param evaluate Function used to evaluate particles.
 * @param module The module to work with.
 * @param search_type The type of tree walk search.
 * @param ngb_criteria The criteria for neighboring search.
 * @param data The data to work with.
 * @param should_drift Flag indicating if drifting should be performed.
 * @return A pointer to the evaluated secondary particles.
 */
template <typename data_in, typename Evaluate>
static inline void *ar_green_evaluate_secondary(data_in DataGet, Evaluate evaluate, Module module,
                                                TreeWalkType search_type, NeighbourSearchCriteria ngb_criteria,
                                                void *data, int should_drift)
{
    return ar_green_evaluate_secondary(DataGet, evaluate, module, search_type, ngb_criteria, group_criteria_distance,
                                       treewalk_filter_gas_only, data, should_drift);
}

/* we do statistics of the tree here below */
#ifdef AR_TREE_STATS
/**
 * Checks if a given tree index is a root index.
 *
 * @param n The tree index to check.
 * @return True if the index is a root index, otherwise false.
 */
inline static int ar_is_tree_index_root(int n)
{
    return n == All.MaxPart;
}

/**
 * Calculates the number of levels in the tree hierarchy for a given index.
 *
 * @param n The tree index to calculate levels for.
 * @return The number of levels in the tree hierarchy.
 */
inline static int ar_get_tree_index_levels(int n)
{
    int levels = 0;
    while (!ar_is_tree_index_root(n))
    {
        n = Nodes[n].u.d.father;
        levels++;
    }
    return levels;
}

/**
 * Prints statistics about the tree hierarchy.
 *
 * This function calculates and prints statistics about the tree hierarchy,
 * including the number of particles at each level.
 */
inline static void ar_tree_stats(void)
{
    std::map<int, int> mappa;
    int max_level = 0;

    for (int i = 0; i < NumPart; i++)
    {
        int level = ar_get_tree_index_levels(Father[i]);
        if (mappa.find(level) == mappa.end())
        {
            // mappa[level] = 0;
            mappa.try_emplace(level, 0);
        }

        mappa[level] += 1; // mappa[level] + 1;
        if (level > max_level)
        {
            max_level = level;
        }
    }

    for (int i = 0; i <= max_level; i++)
    {
        if (mappa.find(i) != mappa.end())
        {
            VERBOSE(0, "Level %d has %d particles", i, mappa[i]);
        }
    }
}
#endif // AR_TREE_STATS

#ifdef CS_DECOUPLING
/* Additional neighbour search routines related to the multiphase gas model */

template <typename group_particle_data> // will be either particle_data or DataGet depending on the I or II phase
static inline int cs_group_criteria(MyFloat dist_from_center_2, MyFloat max_allowed_dist_2, group_particle_data P,
                                    int next_particle, int first_particle)
{
    return 0; // the function is not called
}

// BEGINNING OF TREEWALK VARIABLE DECOUPLING
static inline int cs_treewalk_filter_variable_decoupling_criteria(int neighbour_index, int target_type,
                                                                  double target_pos[3], double target_vel[3],
                                                                  double target_hsml, double target_densityold,
                                                                  double target_entropy)
{

    if (P[neighbour_index].Type > 0)
    {
        return 1; // filter out all non-gas particles regardless
    }

    if (target_type > 0)
    {
        return 0; // non-gas-targets will end the filtering here so they will get all gas as neighbours
    }

    /* extra checks for decoupling. return 1 => ngb excluded, return 0 => ngb kept */
    MyAtLeastDouble fac_mu;

    if (All.ComovingIntegrationOn)
    {
        fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
    }
    else
    {
        fac_mu = 1.;
    }

    MyAtLeastDouble dx = target_pos[0] - P[neighbour_index].Pos[0];
    MyAtLeastDouble dy = target_pos[1] - P[neighbour_index].Pos[1];
    MyAtLeastDouble dz = target_pos[2] - P[neighbour_index].Pos[2];
#ifdef PERIODIC /*  now find the closest image in the given box size  */
    dx = NEAREST_X(dx);
    dy = NEAREST_Y(dy);
    dz = NEAREST_Z(dz);
#endif // PERIODIC
    MyAtLeastDouble r2 = dx * dx + dy * dy + dz * dz;

    MyAtLeastDouble dvx = target_vel[0] - SphP[neighbour_index].VelPred[0];
    MyAtLeastDouble dvy = target_vel[1] - SphP[neighbour_index].VelPred[1];
    MyAtLeastDouble dvz = target_vel[2] - SphP[neighbour_index].VelPred[2];
    /* NOTE: this is vi-vj but standard is vj-vi, so vdotr is positive if two particles are receding */
    MyAtLeastDouble vdotr = dx * dvx + dy * dvy + dz * dvz;

    MyAtLeastDouble soundspeed_i = sqrt(GAMMA * target_entropy * pow(target_densityold, GAMMA_MINUS1));
    MyAtLeastDouble soundspeed_j =
        sqrt(GAMMA * SphP[neighbour_index].Entropy * pow(SphP[neighbour_index].CS_DensityOld, GAMMA_MINUS1));
    MyAtLeastDouble c_ij = 0.5 * (soundspeed_i + soundspeed_j);
    MyAtLeastDouble mu_ij, h_ij; // CS CHECK CECILIA

    if (vdotr < 0)
    {
#ifndef CONVENTIONAL_VISCOSITY
        mu_ij = fac_mu * (-vdotr) / sqrt(r2); /* note: this is positive */
#else // CONVENTIONAL_VISCOSITY
        h_ij = 0.5 * (target_hsml + P[neighbour_index].Hsml);
        mu_ij = fac_mu * h_ij * (-vdotr) / (r2 + 0.0001 * h_ij * h_ij);
#endif // CONVENTIONAL_VISCOSITY
    }
    else /* particles receding, then no viscosity */
    {
        mu_ij = 0;
        h_ij = 0;
    }

    if ((target_entropy > All.DecouplingParam * SphP[neighbour_index].Entropy && mu_ij < c_ij) &&
        SphP[neighbour_index].CS_DensityOld > All.DensityTailThreshold)
    {
        return 1; /* exclude neighbour */
    }
    return 0; // accept all
}

template <typename group_particle_data>
static inline int cs_treewalk_filter_variable_decoupling_criteria_IPhase(particle_data *P, int p,
                                                                         group_particle_data P_In, int *targets,
                                                                         int n_targets, enum IterationPhase mode)
{
    /* target (central): info in P (indludes both P and SphP) if local, in P_In if exported
     * p --> ngb index */
    double target_pos[3], target_vel[3];
    int target_index = targets[0];
    int target_type = P[target_index].Type;
    target_pos[0] = P[target_index].Pos[0];
    target_pos[1] = P[target_index].Pos[1];
    target_pos[2] = P[target_index].Pos[2];
    target_vel[0] = SphP[target_index].VelPred[0]; // VelPred here is SphP.VelPred
    target_vel[1] = SphP[target_index].VelPred[1];
    target_vel[2] = SphP[target_index].VelPred[2];
    double target_hsml = P[target_index].Hsml;
    double target_densityold = SphP[target_index].CS_DensityOld;
    double target_entropy = SphP[target_index].Entropy;

    return cs_treewalk_filter_variable_decoupling_criteria(p, target_type, target_pos, target_vel, target_hsml,
                                                           target_densityold, target_entropy);
}

template <typename group_particle_data>
static inline int cs_treewalk_filter_variable_decoupling_criteria_IIPhase(particle_data *P, int p,
                                                                          group_particle_data P_In, int *targets,
                                                                          int n_targets, enum IterationPhase mode)
{
    double target_vel[3], target_pos[3];
    int target_index = targets[0];
    int target_type = P_In[target_index].Type;
    target_pos[0] = P_In[target_index].Pos[0];
    target_pos[1] = P_In[target_index].Pos[1];
    target_pos[2] = P_In[target_index].Pos[2];
    target_vel[0] = P_In[target_index].VelPred[0];
    target_vel[1] = P_In[target_index].VelPred[1];
    target_vel[2] = P_In[target_index].VelPred[2];
    double target_hsml = P_In[target_index].Hsml;
    double target_densityold = P_In[target_index].CS_DensityOld;
    double target_entropy = P_In[target_index].Entropy;

    return cs_treewalk_filter_variable_decoupling_criteria(p, target_type, target_pos, target_vel, target_hsml,
                                                           target_densityold, target_entropy);
}
// END OF TREEWALK VARIABLE DECOUPLING

// BEGINNING OF TREEWALK PAIRS DECOUPLING
static inline int cs_treewalk_filter_pairs_decoupling_criteria(int neighbour_index, int target_type,
                                                               double target_pos[3], double target_vel[3],
                                                               double target_hsml, double target_densityold,
                                                               double target_entropy)
{

    if (P[neighbour_index].Type > 0)
    {
        return 1; // filter out all non-gas particles regardless
    }

    if (target_type > 0)
    {
        return 0; // non-gas-targets will end the filtering here so they will get all gas as neighbours
    }

    /* extra checks for decoupling. return 1 => ngb excluded, return 0 => ngb kept */
    MyAtLeastDouble fac_mu;

    if (All.ComovingIntegrationOn)
    {
        fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
    }
    else
    {
        fac_mu = 1.;
    }

    MyAtLeastDouble dx = target_pos[0] - P[neighbour_index].Pos[0];
    MyAtLeastDouble dy = target_pos[1] - P[neighbour_index].Pos[1];
    MyAtLeastDouble dz = target_pos[2] - P[neighbour_index].Pos[2];
#ifdef PERIODIC /*  now find the closest image in the given box size  */
    dx = NEAREST_X(dx);
    dy = NEAREST_Y(dy);
    dz = NEAREST_Z(dz);
#endif // PERIODIC
    MyAtLeastDouble r2 = dx * dx + dy * dy + dz * dz;

    MyAtLeastDouble dvx = target_vel[0] - SphP[neighbour_index].VelPred[0];
    MyAtLeastDouble dvy = target_vel[1] - SphP[neighbour_index].VelPred[1];
    MyAtLeastDouble dvz = target_vel[2] - SphP[neighbour_index].VelPred[2];
    /* NOTE: this is vi-vj but standard is vj-vi, so vdotr is positive if two particles are receding */
    MyAtLeastDouble vdotr = dx * dvx + dy * dvy + dz * dvz;

    MyAtLeastDouble soundspeed_i = sqrt(GAMMA * target_entropy * pow(target_densityold, GAMMA_MINUS1));
    MyAtLeastDouble soundspeed_j =
        sqrt(GAMMA * SphP[neighbour_index].Entropy * pow(SphP[neighbour_index].Density, GAMMA_MINUS1));
    MyAtLeastDouble c_ij = 0.5 * (soundspeed_i + soundspeed_j);
    MyAtLeastDouble mu_ij, h_ij;

    if (vdotr < 0)
    {
#ifndef CONVENTIONAL_VISCOSITY
        mu_ij = fac_mu * (-vdotr) / sqrt(r2); /* note: this is positive */
#else // CONVENTIONAL_VISCOSITY
        h_ij = 0.5 * (target_hsml + P[neighbour_index].Hsml);
        mu_ij = fac_mu * h_ij * (-vdotr) / (r2 + 0.0001 * h_ij * h_ij);
#endif // CONVENTIONAL_VISCOSITY
    }
    else /* particles receding, then no viscosity */
    {
        mu_ij = 0;
        h_ij = 0;
    }

    if ((target_entropy > All.DecouplingParam * SphP[neighbour_index].Entropy ||
         SphP[neighbour_index].Entropy > All.DecouplingParam * target_entropy) &&
        mu_ij < c_ij) // || SphP[neighbour_index].CS_DensityOld > All.DensityTailThreshold))
    {
        return 1; /* exclude neighbour */
    }
    return 0; // accept all
}

// cs_treewalk_filter_pairs_decoupling_criteria_IPhase
template <typename group_particle_data>
static inline int cs_treewalk_filter_pairs_decoupling_criteria_IPhase(particle_data *P, int p, group_particle_data P_In,
                                                                      int *targets, int n_targets,
                                                                      enum IterationPhase mode)
{
    double target_pos[3], target_vel[3];
    int target_index = targets[0]; // if Type==5 then the group has one element only (see group_criteria_w_ptype)
    int target_type = P[target_index].Type;
    target_pos[0] = P[target_index].Pos[0];
    target_pos[1] = P[target_index].Pos[1];
    target_pos[2] = P[target_index].Pos[2];
    target_vel[0] = P[target_index].Vel[0];
    target_vel[1] = P[target_index].Vel[1];
    target_vel[2] = P[target_index].Vel[2];
    double target_hsml = P[target_index].Hsml;
    double target_densityold = SphP[target_index].Density;
    double target_entropy = SphP[target_index].Entropy;

    return cs_treewalk_filter_pairs_decoupling_criteria(p, target_type, target_pos, target_vel, target_hsml,
                                                        target_densityold, target_entropy);
}

template <typename group_particle_data>
static inline int cs_treewalk_filter_pairs_decoupling_criteria_IIPhase(particle_data *P, int p,
                                                                       group_particle_data P_In, int *targets,
                                                                       int n_targets, enum IterationPhase mode)
{
    double target_vel[3], target_pos[3];
    int target_index = targets[0];
    int target_type = P_In[target_index].Type;
    target_pos[0] = P_In[target_index].Pos[0];
    target_pos[1] = P_In[target_index].Pos[1];
    target_pos[2] = P_In[target_index].Pos[2];
    target_vel[0] = P_In[target_index].Vel[0];
    target_vel[1] = P_In[target_index].Vel[1];
    target_vel[2] = P_In[target_index].Vel[2];
    double target_hsml = P_In[target_index].Hsml;
    double target_densityold = P_In[target_index].Density;
    double target_entropy = P_In[target_index].Entropy;

    return cs_treewalk_filter_pairs_decoupling_criteria(p, target_type, target_pos, target_vel, target_hsml,
                                                        target_densityold, target_entropy);
}
// END OF TREEWALK PAIRS DECOUPLING

#endif // CS_DECOUPLING

#endif // GREENTREE_HPP
