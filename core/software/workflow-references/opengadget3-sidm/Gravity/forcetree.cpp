/**
 * @file
 *
 * This file is part of the developer version of OpenGadget3 and contains
 * the license conditions for its usage.
 *
 * @brief Gravity Tree and Ewald corrections.
 *
 * @author OpenGadget3-development team, led by Volker Springel and Klaus Dolag.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Received source code may be modified and used as convenient.
 *
 * 2. Redistributions of source code or in binary form is only possible with
 *    explicit agreement of the copyright holders.
 *
 * 3. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 4. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 5. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Changes by NVIDIA Corp. (see change logs) are licensed as follows:
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */

#ifndef ACC_INCLUDE_GRAVITY

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef SUBFIND
#include "../FofSubfind/subfind.hpp"
#endif // SUBFIND

#ifdef ACC
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // ACC

// using mymath::abs;

/*! auxialiary variable used to set-up non-recursive walk */
static int last;

/*! length of lock-up table for short-range force kernel in TreePM algorithm */
#define NTAB 1000
/*! variables for short-range lookup table */

#ifndef ACC // current PGI compiler doesn't like those static variables..
static
#endif // ACC
    float shortrange_table[NTAB],
    shortrange_table_potential[NTAB];
/*! toggles after first tree-memory allocation, has only influence on log-files */
static int first_flag = 0;
static int tree_allocated_flag = 0;

#ifdef PERIODIC
/*! Size of 3D lock-up table for Ewald correction force */

#define EN 64
/*! 3D lock-up table for Ewald correction to force and potential. Only one
 *  octant is stored, the rest constructed by using the symmetry
 */
#ifndef ACC // pgi openac does not support static on gpu
static
#endif // ACC
    MyFloat fcorrx[EN + 1][EN + 1][EN + 1];
#ifndef ACC // pgi openac does not support static on gpu
static
#endif // ACC
    MyFloat fcorry[EN + 1][EN + 1][EN + 1];
#ifndef ACC // pgi openac does not support static on gpu
static
#endif // ACC
    MyFloat fcorrz[EN + 1][EN + 1][EN + 1];
#ifndef ACC // pgi openac does not support static on gpu
static
#endif // ACC
    MyFloat potcorr[EN + 1][EN + 1][EN + 1];
#ifndef ACC // pgi openac does not support static on gpu
static
#endif // ACC
    MyAtLeastDouble fac_intp;
#endif // PERIODIC

/**
 * @brief Builds the gravitational oct-tree in parallel.
 *
 * This function is a driver routine for constructing the gravitational
 * oct-tree, which is done by calling a small number of other functions.
 *
 * @param npart Number of particles.
 * @param mp Pointer to the unbind_data structure.
 */
int force_treebuild(int npart, struct unbind_data *mp)
{
    TIMER_PUSH(1, "TREE_BUILD");
    int flag;
    VERBOSE(3, "before tree build\n");
    do
    {
#ifdef AR_XMAS_TREE
        Numnodestree = force_treebuild_parallel(npart, mp);
#else // AR_XMAS_TREE
        Numnodestree = force_treebuild_single(npart, mp);
#endif // AR_XMAS_TREE
        VERBOSE(4, "before mpi_allreduce\n");
        MPI_Allreduce(&Numnodestree, &flag, 1, MPI_INT, MPI_MIN, MYMPI_COMM_WORLD);
        VERBOSE(4, "after mpi_allreduce\n");
        if (flag == -1)
        {
            force_treefree();
            VERBOSE(2, "Increasing TreeAllocFactor=%g", All.TreeAllocFactor);
            All.TreeAllocFactor *= 1.15;
            VERBOSE(2, "new value=%g\n", All.TreeAllocFactor);
            force_treeallocate((int)(All.TreeAllocFactor * All.MaxPart) + NTopnodes, All.MaxPart);
        }
    } while (flag == -1);

    force_flag_localnodes();
    force_exchange_pseudodata();
    force_treeupdate_pseudos(All.MaxPart);

    TimeOfLastTreeConstruction = All.Time;

#ifdef ACC
    acc_all.no_tree_flag = 1;
#endif // ACC
    TIMER_POP(1, "TREE_BUILD");
    return Numnodestree;
}

/**
 * @brief Construct the gravitational oct-tree.
 *
 * The index convention for accessing tree nodes is the following: the
 * indices 0...NumPart-1 reference single particles, the indices
 * All.MaxPart.... All.MaxPart+nodes-1 reference tree nodes. `Nodes_base'
 * points to the first tree node, while `nodes' is shifted such that
 * nodes[All.MaxPart] gives the first tree node. Finally, node indices
 * with values 'All.MaxPart + MaxNodes' and larger indicate "pseudo
 * particles", i.e. multipole moments of top-level nodes that lie on
 * different CPUs. If such a node needs to be opened, the corresponding
 * particle must be exported to that CPU. The 'Extnodes' structure
 * parallels that of 'Nodes'. Its information is only needed for the SPH
 * part of the computation. (The data is split onto these two structures
 * as a tuning measure.  If it is merged into 'Nodes' a somewhat bigger
 * size of the nodes also for gravity would result, which would reduce
 * cache utilization slightly.
 *
 * @param npart Number of particles.
 * @param mp Pointer to the unbind_data structure.
 */
int force_treebuild_single(int npart, struct unbind_data *mp)
{
    int subnode = 0, shift, parent, numnodes, rep;
    int nfree, th, nn, no;
    struct NODE *nfreep;
    MyFloat lenhalf;
    peanokey key, morton, th_key, *morton_list;

    TIMER_PUSH(1, "TREE_BUILD_ADD_PART");

    /* create an empty root node  */
    nfree = All.MaxPart; /* index of first free node */
    nfreep = &Nodes[nfree]; /* select first node */

    nfreep->len = DomainLen;
    for (int j = 0; j < 3; j++)
    {
        nfreep->center[j] = DomainCenter[j];
    }
    for (int j = 0; j < 8; j++)
    {
        nfreep->u.suns[j] = -1;
    }

    numnodes = 1;
    nfreep++;
    nfree++;

    /* create a set of empty nodes corresponding to the top-level domain
     * grid. We need to generate these nodes first to make sure that we have a
     * complete top-level tree which allows the easy insertion of the
     * pseudo-particles at the right place
     */
    force_create_empty_nodes(All.MaxPart, 0, 1, 0, 0, 0, &numnodes, &nfree);

    /* if a high-resolution region in a global tree is used, we need to generate
     * an additional set empty nodes to make sure that we have a complete
     * top-level tree for the high-resolution inset
     */

    nfreep = &Nodes[nfree];
    parent = -1; /* note: will not be used below before it is changed */

    morton_list = (peanokey *)mymalloc("morton_list", NumPart * sizeof(peanokey));

#ifdef AR_TREEBUILD_PARALLEL
    int must_return = 0; // openmp cant return from region
    int do_omp_parallel = All.NumCurrentTiStep > 1;
    // it is better to limit the number of threads to prevent cache misses in shared regions of the three
#pragma omp parallel for num_threads(AR_TREEBUILD_PARALLEL) private(i, rep, key, shift, no, morton, th, subnode,       \
                                                                    parent, nn, lenhalf, th_key) if (do_omp_parallel)
#endif // AR_TREEBUILD_PARALLEL
    /* now we insert all particles */
    for (int k = 0; k < npart; k++)
    {
#ifdef AR_TREEBUILD_PARALLEL
        int thread_id = omp_get_thread_num();
        if (do_omp_parallel)
        {
            if (must_return)
            { // cant break inside openmp
                continue;
            }
        }
        int node_locked = -1; // rememebr which node we will lock
#endif // AR_TREEBUILD_PARALLEL
        int i;
        if (mp)
        {
            i = mp[k].index;
        }
        else
        {
            i = k;
        }

#ifdef NEUTRINOS
        if ((P[i].Type == type_neutrino) && (All.Time <= All.Time_tree_on_nu))
        {
            continue;
        }
#endif // NEUTRINOS

        rep = 0;
        key = peano_and_morton_key((int)((P[i].Pos[0] - DomainCorner[0]) * DomainFac),
                                   (int)((P[i].Pos[1] - DomainCorner[1]) * DomainFac),
                                   (int)((P[i].Pos[2] - DomainCorner[2]) * DomainFac), BITS_PER_DIMENSION, &morton);
        morton_list[i] = morton;

        /* reset particle's dp for next tree update */
        P[i].dp[0] = 0.0;
        P[i].dp[1] = 0.0;
        P[i].dp[2] = 0.0;

        shift = 3 * (BITS_PER_DIMENSION - 1);

        no = 0;
        /* search for the TopNode that contains the current  particle, given its peano key. */
        while (TopNodes[no].Daughter >= 0)
        {
            no = TopNodes[no].Daughter + (key - TopNodes[no].StartKey) / (TopNodes[no].Size / 8);
            shift -= 3;
            rep++;
        }

        no = TopNodes[no].Leaf;
        th = DomainNodeIndex[no];
        /* walk from the topnode down to the first leaf where we can append it. */
        while (1)
        {
            if (th >= All.MaxPart) /* we are dealing with an internal node */
            {
                if (shift >= 0)
                {
                    subnode = ((morton >> shift) & 7);
                }
                else
                {
                    subnode = 0;
                    if (P[i].Pos[0] > Nodes[th].center[0])
                    {
                        subnode += 1;
                    }
                    if (P[i].Pos[1] > Nodes[th].center[1])
                    {
                        subnode += 2;
                    }
                    if (P[i].Pos[2] > Nodes[th].center[2])
                    {
                        subnode += 4;
                    }
                }

#ifndef NOTREERND
                if (Nodes[th].len < 1.0e-3 * All.ForceSoftening[P[i].Type])
                {
                    /* seems like we're dealing with particles at identical (or extremely close)
                     * locations. Randomize subnode index to allow tree construction. Note: Multipole moments
                     * of tree are still correct, but this will only happen well below gravitational softening
                     * length-scale anyway.
                     */
                    subnode = (int)(8.0 * get_random_number((P[i].ID + rep) % (RNDTABLE + (rep & 3))));

                    if (subnode >= 8)
                    {
                        subnode = 7;
                    }
                }
#endif // NOTREERND
#ifdef AR_TREEBUILD_PARALLEL
                // if we locked a node, there are no locked nodes below our level and we proceed witout checking

                if (do_omp_parallel && node_locked != -1)
                {
                    nn = Nodes[th].u.suns[subnode];
                    shift -= 3;

                    if (nn >= 0) /* ok, something is in the daughter slot already, need to continue */
                    {
                        parent = th;
                        th = nn;
                        rep++;
                    }
                    else
                    {
                        /* here we have found an empty slot where we can attach
                         * the new particle as a leaf.
                         */
                        Nodes[th].u.suns[subnode] = i;
                        ENDRUN_IF(node_locked >= (NumberOfAllocatedNodeLocks - 1),
                                  " node_locked=%d > = NumberOfAllocatedNodeLocks =%d", node_locked,
                                  NumberOfAllocatedNodeLocks);
                        omp_unset_lock(&NodesLockList[node_locked]); // for sure we locked a node, so now we unlock
                        node_locked = -1;
                        break;
                    }
                }
                else
                {
                    node_locked = (th - All.MaxPart); // remember which node we will lock
                    if (do_omp_parallel)
                    {
                        ENDRUN_IF(node_locked >= (NumberOfAllocatedNodeLocks - 1),
                                  " node_locked=%d > = (NumberOfAllocatedNodeLocks-1) =%d", node_locked,
                                  (NumberOfAllocatedNodeLocks - 1));
                        omp_set_lock(&NodesLockList[node_locked]);
                    }
#endif // AR_TREEBUILD_PARALLEL
                    nn = Nodes[th].u.suns[subnode];
                    shift -= 3;
                    if (nn >= 0) /* ok, something is in the daughter slot already, need to continue */
                    {
                        parent = th;
                        th = nn;
                        rep++;
#ifdef AR_TREEBUILD_PARALLEL
                        if (th >= All.MaxPart) // next iteration we will touch an internal node, so we unlock
                        {
                            if (do_omp_parallel)
                            {
                                ENDRUN_IF(node_locked >= (NumberOfAllocatedNodeLocks - 1),
                                          " node_locked=%d > = (NumberOfAllocatedNodeLocks-1) =%d", node_locked,
                                          (NumberOfAllocatedNodeLocks - 1));
                                omp_unset_lock(&NodesLockList[node_locked]);
                            }
                            node_locked = -1;
                        }
#endif // AR_TREEBUILD_PARALLEL
                    }
                    else
                    {
                        /* here we have found an empty slot where we can attach
                         * the new particle as a leaf.
                         */
                        Nodes[th].u.suns[subnode] = i;
#ifdef AR_TREEBUILD_PARALLEL
                        if (do_omp_parallel)
                        {
                            ENDRUN_IF(node_locked >= (NumberOfAllocatedNodeLocks - 1),
                                      " node_locked=%d > = (NumberOfAllocatedNodeLocks-1) =%d", node_locked,
                                      (NumberOfAllocatedNodeLocks - 1));
                            omp_unset_lock(&NodesLockList[node_locked]);
                        }
                        node_locked = -1;
#endif // AR_TREEBUILD_PARALLEL
                        break; /* done for this particle */
                    }
#ifdef AR_TREEBUILD_PARALLEL
                } // end of if               if(node_locked!=-1)
#endif // AR_TREEBUILD_PARALLEL
            }
            else
            {
                struct NODE *_nfreep;
                // int _numnodes, _nfree;
                int _nfree;

#ifdef AR_TREEBUILD_PARALLEL
                int _numnodes;
#pragma omp atomic capture
                {
                    _numnodes = numnodes;
                    numnodes++;
                }
                _nfree = _numnodes + All.MaxPart;
                _nfreep = Nodes + _nfree;
                if ((numnodes) >= MaxNodes - 1)
                {
                    omp_unset_lock(&NodesLockList[node_locked]);
                    node_locked = -1; // unlock what we locked
                    must_return = 1; // we cant return from parallel region
                    continue;
                }
#else // AR_TREEBUILD_PARALLEL
      //_numnodes = numnodes;
                _nfree = nfree;
                _nfreep = nfreep;
#endif // AR_TREEBUILD_PARALLEL

                /* We try to insert into a leaf with a single particle.  Need
                 * to generate a new internal node at this point.
                 */
                Nodes[parent].u.suns[subnode] = _nfree;

                _nfreep->len = 0.5 * Nodes[parent].len;
                lenhalf = 0.25 * Nodes[parent].len;

                if (subnode & 1)
                {
                    _nfreep->center[0] = Nodes[parent].center[0] + lenhalf;
                }
                else
                {
                    _nfreep->center[0] = Nodes[parent].center[0] - lenhalf;
                }

                if (subnode & 2)
                {
                    _nfreep->center[1] = Nodes[parent].center[1] + lenhalf;
                }
                else
                {
                    _nfreep->center[1] = Nodes[parent].center[1] - lenhalf;
                }

                if (subnode & 4)
                {
                    _nfreep->center[2] = Nodes[parent].center[2] + lenhalf;
                }
                else
                {
                    _nfreep->center[2] = Nodes[parent].center[2] - lenhalf;
                }

                _nfreep->u.suns[0] = -1;
                _nfreep->u.suns[1] = -1;
                _nfreep->u.suns[2] = -1;
                _nfreep->u.suns[3] = -1;
                _nfreep->u.suns[4] = -1;
                _nfreep->u.suns[5] = -1;
                _nfreep->u.suns[6] = -1;
                _nfreep->u.suns[7] = -1;

                if (shift >= 0)
                {
                    th_key = morton_list[th];
                    subnode = ((th_key >> shift) & 7);
                }
                else
                {
                    subnode = 0;
                    if (P[th].Pos[0] > _nfreep->center[0])
                    {
                        subnode += 1;
                    }
                    if (P[th].Pos[1] > _nfreep->center[1])
                    {
                        subnode += 2;
                    }
                    if (P[th].Pos[2] > _nfreep->center[2])
                    {
                        subnode += 4;
                    }
                }

#ifndef NOTREERND
                if (_nfreep->len < 1.0e-3 * All.ForceSoftening[P[th].Type])
                {
                    /* seems like we're dealing with particles at identical (or extremely close)
                     * locations. Randomize subnode index to allow tree construction. Note: Multipole moments
                     * of tree are still correct, but this will only happen well below gravitational softening
                     * length-scale anyway.
                     */
                    subnode = (int)(8.0 * get_random_number((P[th].ID + rep) % (RNDTABLE + (rep & 3))));

                    if (subnode >= 8)
                    {
                        subnode = 7;
                    }
                }
#endif // NOTREERND
                _nfreep->u.suns[subnode] = th;

                th = _nfree; /* resume trying to insert the new particle at
                              * the newly created internal node
                              */

#ifndef AR_TREEBUILD_PARALLEL
                numnodes++;
                nfree++;
                nfreep++;
#endif // AR_TREEBUILD_PARALLEL
                if ((numnodes) >= MaxNodes)
                {
                    VERBOSE_ALL(5, "task %d: maximum number %d of tree-nodes reached for particle %d.\n", ThisTask,
                                MaxNodes, i);

                    if (All.TreeAllocFactor > All.MaxTreeAllocFactor)
                    {
                        dump_particles();
                        ENDRUN("task %d: looks like a serious problem (MaxTreeAllocFactor exceeded, max: %f, "
                               "current: %f) for particle %d, stopping with particle dump.\n",
                               ThisTask, All.MaxTreeAllocFactor, All.TreeAllocFactor, i);
                    }
                    else
                    {
#ifndef AR_TREEBUILD_PARALLEL // we dealt with this case before
                        myfree(morton_list);
                        TIMER_POP(1, "TREE_BUILD_ADD_PART");
                        return -1;
#endif // AR_TREEBUILD_PARALLEL
                    }
                }
            }
        }
    }

    myfree(morton_list);
#ifdef AR_TREEBUILD_PARALLEL
    if (must_return)
    { // cant return inside openmp parallel, we can return only now
        // lckily this return happens only in case of serious allocation problems
        TIMER_POP(1, "TREE_BUILD_ADD_PART");
        return -1;
    }
#endif // AR_TREEBUILD_PARALLEL

    /* insert the pseudo particles that represent the mass distribution of other domains */
    force_insert_pseudo_particles();

    TIMER_POPPUSH(1, "TREE_BUILD_ADD_PART", "TREE_BUILD_UPDATE_NODES");

    /* now compute the multipole moments recursively */
    last = -1;

#ifdef AR_NODE_RECURSIVE_PARALLEL // in this implementation last is private to the function
    NumUsedThreads = 0;
    last = force_update_node_recursive(All.MaxPart, -1, -1, 0, 0, last);
#else // AR_NODE_RECURSIVE_PARALLEL
    force_update_node_recursive(All.MaxPart, -1, -1, 0);
#endif // AR_NODE_RECURSIVE_PARALLEL

    if (last >= All.MaxPart)
    {
        if (last >= All.MaxPart + MaxNodes) /* a pseudo-particle */
        {
            Nextnode[last - MaxNodes] = -1;
        }
        else
        {
            Nodes[last].u.d.nextnode = -1;
        }
    }
    else
    {
        Nextnode[last] = -1;
    }

    TIMER_POP(1, "TREE_BUILD_UPDATE_NODES");

#ifdef GROUP_LEAVES
    TIMER_PUSH(1, "TREE_BUILD_GROUP_LEAVES");
    force_group_leaves(GROUP_LEAVES);
    TIMER_POP(1, "TREE_BUILD_GROUP_LEAVES");
#endif // GROUP_LEAVES

    return numnodes;
}

/**
 * @brief Create a set of empty tree nodes.
 *
 * This function recursively creates a set of empty tree nodes which
 * corresponds to the top-level tree for the domain grid. This is done to
 * ensure that this top-level tree is always "complete" so that we can easily
 * associate the pseudo-particles of other CPUs with tree-nodes at a given
 * level in the tree, even when the particle population is so sparse that
 * some of these nodes are actually empty.
 *
 * @param no Index of the current node.
 * @param topnode Index of the top-level node.
 * @param bits Number of bits used for the Peano-Hilbert key.
 * @param x X-coordinate of the current node.
 * @param y Y-coordinate of the current node.
 * @param z Z-coordinate of the current node.
 * @param nodecount Number of nodes.
 * @param nextfree Index of the next free node.
 */
void force_create_empty_nodes(int no, int topnode, int bits, int x, int y, int z, int *nodecount, int *nextfree)
{
    if (TopNodes[topnode].Daughter >= 0)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    int sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);
                    int count = i + 2 * j + 4 * k;

                    Nodes[no].u.suns[count] = *nextfree;
                    MyFloat lenhalf = 0.25 * Nodes[no].len;
                    Nodes[*nextfree].len = 0.5 * Nodes[no].len;
                    Nodes[*nextfree].center[0] = Nodes[no].center[0] + (2 * i - 1) * lenhalf;
                    Nodes[*nextfree].center[1] = Nodes[no].center[1] + (2 * j - 1) * lenhalf;
                    Nodes[*nextfree].center[2] = Nodes[no].center[2] + (2 * k - 1) * lenhalf;

                    for (int n = 0; n < 8; n++)
                    {
                        Nodes[*nextfree].u.suns[n] = -1;
                    }

                    if (TopNodes[TopNodes[topnode].Daughter + sub].Daughter == -1)
                    {
                        DomainNodeIndex[TopNodes[TopNodes[topnode].Daughter + sub].Leaf] = *nextfree;
                    }

                    *nextfree = *nextfree + 1;
                    *nodecount = *nodecount + 1;

                    if ((*nodecount) >= MaxNodes || (*nodecount) >= MaxTopNodes)
                    {
                        VERBOSE_ALL(0,
                                    "task %d: maximum number MaxNodes=%d of tree-nodes reached."
                                    "MaxTopNodes=%d NTopnodes=%d NTopleaves=%d nodecount=%d\n",
                                    ThisTask, MaxNodes, MaxTopNodes, NTopnodes, NTopleaves, *nodecount);
                        VERBOSE_ALL(0, "in create empty nodes\n");
                        dump_particles();
                        ENDRUN("Error in create empty nodes.");
                    }

                    force_create_empty_nodes(*nextfree - 1, TopNodes[topnode].Daughter + sub, bits + 1, 2 * x + i,
                                             2 * y + j, 2 * z + k, nodecount, nextfree);
                }
            }
        }
    }
}

/**
 * @brief Insert pseudo-particles into the tree.
 *
 * This function inserts pseudo-particles which will represent the mass
 * distribution of the other CPUs. Initially, the mass of the
 * pseudo-particles is set to zero, and their coordinate is set to the
 * center of the domain-cell they correspond to. These quantities will be
 * updated later on.
 */
void force_insert_pseudo_particles(void)
{
    for (int i = 0; i < NTopleaves; i++)
    {
        int index = DomainNodeIndex[i];
        if (DomainTask[i] != ThisTask)
        {
            Nodes[index].u.suns[0] = All.MaxPart + MaxNodes + i;
        }
    }
}

/**
 * @brief Determine the multipole moments for a given node and all its subnodes.
 *
 * This routine determines the multipole moments for a given internal node
 * and all its subnodes using a recursive computation.  The result is
 * stored in the Nodes[] structure in the sequence of this tree-walk.
 *
 * Note that the bitflags-variable for each node is used to store in the
 * lowest bits some special information: Bit 0 flags whether the node
 * belongs to the top-level tree corresponding to the domain
 * decomposition, while Bit 1 signals whether the top-level node is
 * dependent on local mass.
 *
 * bits 2-4 give the particle type with
 * the maximum softening among the particles in the node, and bit 5
 * flags whether the node contains any particles with lower softening
 * than that.
 *
 * @param no Index of the current node.
 * @param sib Index of the sibling node.
 * @param father Index of the father node.
 * @param LocSoft Local softening length.
 * @param layer Layer of the tree.
 * @param last Index of the last node.
 */
#ifdef AR_NODE_RECURSIVE_PARALLEL
// in this implementation `last` is private to the function and we return the new `last`
int force_update_node_recursive(int no, int sib, int father, float LocSoft, int layer, int last)
#else // AR_NODE_RECURSIVE_PARALLEL
void force_update_node_recursive(int no, int sib, int father, float LocSoft)
#endif // AR_NODE_RECURSIVE_PARALLEL
{
    int p, pp, nextsib, suns[8], count_particles, multiple_flag;
    MyFloat hmax, vmax, v;
    // integertime ti_step;
    integertime time0, time1;
    MyAtLeastDouble dt_drift_hmax;
    MyFloat s[3], vs[3], mass;
    struct particle_data *pa;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    int maxsofttype, diffsoftflag;
#else // ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat maxsoft;
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#if defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
    MyFloat maxsoft;
#endif // defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)

    if (no >= All.MaxPart && no < All.MaxPart + MaxNodes) /* internal node */
    {
        for (int j = 0; j < 8; j++)
        {
            suns[j] = Nodes[no].u.suns[j]; /* this "backup" is necessary because the nextnode entry will overwrite one
                                              element (union!) */
        }

        if (last >= 0)
        {
            if (last >= All.MaxPart)
            {
                if (last >= All.MaxPart + MaxNodes) /* a pseudo-particle */
                {
                    Nextnode[last - MaxNodes] = no;
                }
                else
                {
                    Nodes[last].u.d.nextnode = no;
                }
            }
            else
            {
                Nextnode[last] = no;
            }
        }

        last = no;

        mass = 0;
        s[0] = 0;
        s[1] = 0;
        s[2] = 0;
        vs[0] = 0;
        vs[1] = 0;
        vs[2] = 0;
        hmax = 0;
        vmax = 0;

        count_particles = 0;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
        maxsofttype = 7;
        diffsoftflag = 0;
#else // ADAPTIVE_GRAVSOFT_FORGAS
        maxsoft = 0;
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#if defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
        maxsoft = 0;
#endif //  defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)

#ifdef AR_NODE_RECURSIVE_PARALLEL
        int do_omp_parallel = layer < AR_NODE_RECURSIVE_PARALLEL; // change criteria as you whish
        int _num_used_threads = 0;
#pragma omp atomic capture
        {
            NumUsedThreads += 8;
            _num_used_threads = NumUsedThreads;
        }
        do_omp_parallel = do_omp_parallel && (_num_used_threads < maxThreads);

        int lasts[8];
        int _ps[8];
        VERBOSE_IF(do_omp_parallel, -1, "force_update_node_recursive() on level %d: do parallel", layer);

#pragma omp parallel for if (do_omp_parallel)
        for (int j = 0; j < 8; j++)
        {
            int p, jj, nextsib; // I prefer declare private here
            if ((p = suns[j]) >= 0)
            {
                /* check if we have a sibling on the same level */
                for (jj = j + 1; jj < 8; jj++)
                {
                    if ((pp = suns[jj]) >= 0)
                    {
                        break;
                    }
                }

                if (jj < 8) /* yes, we do */
                {
                    nextsib = pp;
                }
                else
                {
                    nextsib = sib;
                }
                if (do_omp_parallel)
                {
                    /*
                       in parallel we do not update NextNode and Nodes[].NextNode of the first `no` because
                       we do not know the value of `last`
                     */
                    lasts[j] = force_update_node_recursive(p, nextsib, no, LocSoft, layer + 1, -1);
                    _ps[j] = p;
                }
                else
                {
                    last = force_update_node_recursive(p, nextsib, no, LocSoft, layer + 1, last);
                }
            }
        }

        if (do_omp_parallel)
        { // we stitch the parallelised cubes together
            for (int j = 0; j < 8; j++)
            {
                int _last = j == 0 ? last : lasts[j - 1];
                int _no = suns[j];
                if (_no >= 0)
                {
                    if (_last >= 0)
                    {
                        if (_last >= All.MaxPart)
                        {
                            if (_last >= All.MaxPart + MaxNodes) /* a pseudo-particle */
                            {
                                Nextnode[_last - MaxNodes] = _no;
                            }
                            else
                            {
                                Nodes[_last].u.d.nextnode = _no;
                            }
                        }
                        else
                        {
                            Nextnode[_last] = _no;
                        }
                    }

                    _last = no;
                }
            }
            last = lasts[7];
        }
#endif // end of AR_NODE_RECURSIVE_PARALLEL

        for (int j = 0; j < 8; j++)
        {
            if ((p = suns[j]) >= 0)
            {
                int jj;
                /* check if we have a sibling on the same level */
                for (jj = j + 1; jj < 8; jj++)
                    if ((pp = suns[jj]) >= 0)
                    {
                        break;
                    }

                if (jj < 8) /* yes, we do */
                {
                    nextsib = pp;
                }
                else
                {
                    nextsib = sib;
                }
#ifdef AR_NODE_RECURSIVE_PARALLEL
                // if AR_NODE_RECURSIVE_PARALLEL, we did force_update_node_recursive in a previous separate loop
#else // AR_NODE_RECURSIVE_PARALLEL
                force_update_node_recursive(p, nextsib, no, LocSoft);
#endif // AR_NODE_RECURSIVE_PARALLEL
                if (p >= All.MaxPart) /* an internal node or pseudo particle */
                {
                    if (p >= All.MaxPart + MaxNodes) /* a pseudo particle */
                    {
                        /* nothing to be done here because the mass of the
                         * pseudo-particle is still zero. This will be changed
                         * later.
                         */
                    }
                    else
                    {
                        mass += (Nodes[p].u.d.mass);
                        s[0] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[0]);
                        s[1] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[1]);
                        s[2] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[2]);
                        vs[0] += (Nodes[p].u.d.mass * Extnodes[p].vs[0]);
                        vs[1] += (Nodes[p].u.d.mass * Extnodes[p].vs[1]);
                        vs[2] += (Nodes[p].u.d.mass * Extnodes[p].vs[2]);

                        if (Nodes[p].u.d.mass > 0)
                        {
                            if (Nodes[p].u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES))
                            {
                                count_particles += 2;
                            }
                            else
                            {
                                count_particles++;
                            }
                        }

                        if (Extnodes[p].hmax > hmax)
                        {
                            hmax = Extnodes[p].hmax;
                        }

                        if (Extnodes[p].vmax > vmax)
                        {
                            vmax = Extnodes[p].vmax;
                        }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                        diffsoftflag |= maskout_different_softening_flag(Nodes[p].u.d.bitflags);

                        if (maxsofttype == 7)
                        {
                            maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
                        }
                        else
                        {
                            int current_maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
                            if (current_maxsofttype != 7)
                            {
                                if (All.ForceSoftening[current_maxsofttype] > All.ForceSoftening[maxsofttype])
                                {
                                    maxsofttype = current_maxsofttype;
                                    diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                                }
                                else
                                {
                                    if (All.ForceSoftening[current_maxsofttype] < All.ForceSoftening[maxsofttype])
                                    {
                                        diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                                    }
                                }
                            }
                        }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                        if (Nodes[p].maxsoft > maxsoft)
                        {
                            maxsoft = Nodes[p].maxsoft;
                        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#if defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
                        /* update of the maximum gravitational softening in the node */
                        if (Nodes[p].maxsoft > maxsoft)
                        {
                            maxsoft = Nodes[p].maxsoft;
                        }
#endif // defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
                    }
                }
                else /* a particle */
                {
                    count_particles++;
                    pa = &P[p];
                    mass += (pa->Mass);

                    s[0] += (pa->Mass * pa->Pos[0]);
                    s[1] += (pa->Mass * pa->Pos[1]);
                    s[2] += (pa->Mass * pa->Pos[2]);

                    vs[0] += (pa->Mass * pa->Vel[0]);
                    vs[1] += (pa->Mass * pa->Vel[1]);
                    vs[2] += (pa->Mass * pa->Vel[2]);

                    if (pa->Type == 0)
                    {
                        hmax = std::max(hmax, P[p].Hsml);
                        time0 = P[p].Ti_current;
                        time1 = All.Ti_Current;
                        if (time0 != time1)
                        {
                            if (All.ComovingIntegrationOn)
                            {
                                dt_drift_hmax = get_drift_factor(time0, time1);
                            }
                            else
                            {
                                dt_drift_hmax = (time1 - time0) * All.Timebase_interval;
                            }
                            MyAtLeastDouble hsml_evol_fak = exp(0.333333333333 * SphP[p].DivVel * dt_drift_hmax);
                            if (hsml_evol_fak > (1 + All.CourantFac))
                            {
#ifdef DEBUG
                                VERBOSE_ALL(0,
                                            "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange "
                                            "hsml_max evol value in force_update: %e",
                                            ThisTask, (unsigned long long)P[p].ID, (float)P[p].Mass, SphP[p].DivVel,
                                            dt_drift_hmax, hsml_evol_fak);
#endif // DEBUG
                                hsml_evol_fak = 1 + All.CourantFac;
                            }
                            if (hsml_evol_fak < 1. / (1 + All.CourantFac))
                            {
#ifdef DEBUG
                                VERBOSE_ALL(0,
                                            "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange "
                                            "hsml_max evol value in force_update: %e",
                                            ThisTask, (unsigned long long)P[p].ID, (float)P[p].Mass, SphP[p].DivVel,
                                            dt_drift_hmax, hsml_evol_fak);
#endif // DEBUG
                                hsml_evol_fak = 1. / (1 + All.CourantFac);
                            }
                            MyFloat hmax_est = P[p].Hsml * hsml_evol_fak;
                            hmax = std::max(hmax, hmax_est);
                        }
                    }

                    for (int k = 0; k < 3; k++)
                    {
                        if ((v = abs(pa->Vel[k])) > vmax)
                        {
                            vmax = v;
                        }
                    }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                    if (maxsofttype == 7)
                    {
                        maxsofttype = pa->Type;
                    }
                    else
                    {
                        if (All.ForceSoftening[pa->Type] > All.ForceSoftening[maxsofttype])
                        {
                            maxsofttype = pa->Type;
                            diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                        }
                        else
                        {
                            if (All.ForceSoftening[pa->Type] < All.ForceSoftening[maxsofttype])
                            {
                                diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                            }
                        }
                    }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                    if (pa->Type == 0)
                    {
                        if (P[p].Hsml > maxsoft)
                        {
                            maxsoft = P[p].Hsml;
                        }
                    }
                    else
                    {
                        if (All.ForceSoftening[pa->Type] > maxsoft)
                            maxsoft = All.ForceSoftening[pa->Type];
                    }
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#ifdef ADAPTGRAVSOFT
                    /* update of the maximum gravitational softening  */
                    if (pa->AGS_Hsml > maxsoft)
                        maxsoft = pa->AGS_Hsml;
#elif defined(mSIDM) || defined(mIDM)
                    /* update of the maximum gravitational softening  */
                    if (pa->Hsml > maxsoft)
                        maxsoft = pa->Hsml;
#endif // defined(mSIDM) || defined(mIDM)
                }
            }
        }

        if (mass)
        {
            s[0] /= mass;
            s[1] /= mass;
            s[2] /= mass;
            vs[0] /= mass;
            vs[1] /= mass;
            vs[2] /= mass;
        }
        else
        {
            s[0] = Nodes[no].center[0];
            s[1] = Nodes[no].center[1];
            s[2] = Nodes[no].center[2];
            vs[0] = 0;
            vs[1] = 0;
            vs[2] = 0;
        }

        Nodes[no].Ti_current = All.Ti_Current;
        Nodes[no].u.d.mass = mass;
        Nodes[no].u.d.s[0] = s[0];
        Nodes[no].u.d.s[1] = s[1];
        Nodes[no].u.d.s[2] = s[2];
        Nodes[no].GravCost = 0;

        Extnodes[no].Ti_lastkicked = All.Ti_Current;
        Extnodes[no].Flag = GlobFlag;
        Extnodes[no].vs[0] = vs[0];
        Extnodes[no].vs[1] = vs[1];
        Extnodes[no].vs[2] = vs[2];
        Extnodes[no].hmax = hmax;
        Extnodes[no].vmax = vmax;
        Extnodes[no].dp[0] = 0;
        Extnodes[no].dp[1] = 0;
        Extnodes[no].dp[2] = 0;

        if (count_particles > 1) /* this flags that the node represents more than one particle */
        {
            multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);
        }
        else
        {
            multiple_flag = 0;
        }

        Nodes[no].u.d.bitflags = multiple_flag;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
        Nodes[no].u.d.bitflags |= diffsoftflag + (maxsofttype << BITFLAG_MAX_SOFTENING_TYPE);
#else // ADAPTIVE_GRAVSOFT_FORGAS
        Nodes[no].maxsoft = maxsoft;
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#if defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
        Nodes[no].maxsoft = maxsoft;
#endif // defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
        Nodes[no].u.d.sibling = sib;
        Nodes[no].u.d.father = father;
    }
    else /* single particle or pseudo particle */
    {
        if (last >= 0)
        {
            if (last >= All.MaxPart)
            {
                if (last >= All.MaxPart + MaxNodes) /* a pseudo-particle */
                {
                    Nextnode[last - MaxNodes] = no;
                }
                else
                {
                    Nodes[last].u.d.nextnode = no;
                }
            }
            else
            {
                Nextnode[last] = no;
            }
        }

        last = no;
        if (no < All.MaxPart) /* only set it for single particles */
        {
            ENDRUN_IF(no == father, "Task=%d Fixed point  no == father == %d  detected\n", ThisTask, no);
            Father[no] = father;
        }
    }
#ifdef AR_NODE_RECURSIVE_PARALLEL
    return last;
#endif // AR_NODE_RECURSIVE_PARALLEL
}

/**
 * @brief Communicate the values of the multipole moments of the top-level tree-nodes.
 *
 * This function communicates the values of the multipole moments of the
 * top-level tree-nodes of the domain grid. This data can then be used to
 * update the pseudo-particles on each CPU accordingly.
 */
void force_exchange_pseudodata(void)
{
    int no, recvTask;
    int *recvcounts, *recvoffset;
    struct DomainNODE
    {
        MyFloat s[3];
        MyFloat vs[3];
        MyFloat mass;
        MyFloat hmax;
        MyFloat vmax;

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
        MyFloat maxsoft;
#endif // defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defiend(mIDM)

        unsigned int bitflags;
#ifdef PAD_STRUCTURES
#ifndef DOUBLEPRECISION
        int pad[5];
#else // DOUBLEPRECISION
#if (DOUBLEPRECISION + 0) == 2
        /* mixed precision */
        int pad[5];
#else // (DOUBLEPRECISION+0) == 2
        int pad[3];
#endif // (DOUBLEPRECISION+0) == 2
#endif // DOUBLEPRECISION
#endif // PAD_STRUCTURES
    } *DomainMoment;

    DomainMoment = (struct DomainNODE *)mymalloc("DomainMoment", NTopleaves * sizeof(struct DomainNODE));

    for (int m = 0; m < Multipledomains; m++)
    {
        for (int i = DomainStartList[ThisTask * Multipledomains + m];
             i <= DomainEndList[ThisTask * Multipledomains + m]; i++)
        {
            no = DomainNodeIndex[i];

            /* read out the multipole moments from the local base cells */
            DomainMoment[i].s[0] = Nodes[no].u.d.s[0];
            DomainMoment[i].s[1] = Nodes[no].u.d.s[1];
            DomainMoment[i].s[2] = Nodes[no].u.d.s[2];
            DomainMoment[i].vs[0] = Extnodes[no].vs[0];
            DomainMoment[i].vs[1] = Extnodes[no].vs[1];
            DomainMoment[i].vs[2] = Extnodes[no].vs[2];
            DomainMoment[i].mass = Nodes[no].u.d.mass;
            DomainMoment[i].hmax = Extnodes[no].hmax;
            DomainMoment[i].vmax = Extnodes[no].vmax;
            DomainMoment[i].bitflags = Nodes[no].u.d.bitflags;

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
            DomainMoment[i].maxsoft = Nodes[no].maxsoft;
#endif // defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
        }
    }

    /* share the pseudo-particle data accross CPUs */
    recvcounts = (int *)mymalloc("recvcounts", sizeof(int) * NTask);
    recvoffset = (int *)mymalloc("recvoffset", sizeof(int) * NTask);

    for (int m = 0; m < Multipledomains; m++)
    {
        for (recvTask = 0; recvTask < NTask; recvTask++)
        {
            recvcounts[recvTask] =
                (DomainEndList[recvTask * Multipledomains + m] - DomainStartList[recvTask * Multipledomains + m] + 1) *
                sizeof(struct DomainNODE);
            recvoffset[recvTask] = DomainStartList[recvTask * Multipledomains + m] * sizeof(struct DomainNODE);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &DomainMoment[0], recvcounts, recvoffset, MPI_BYTE,
                       MYMPI_COMM_WORLD);
    }
    myfree(recvoffset);
    myfree(recvcounts);

    for (int ta = 0; ta < NTask; ta++)
    {
        if (ta != ThisTask)
        {
            for (int m = 0; m < Multipledomains; m++)
            {
                for (int i = DomainStartList[ta * Multipledomains + m]; i <= DomainEndList[ta * Multipledomains + m];
                     i++)
                {
                    no = DomainNodeIndex[i];

                    Nodes[no].u.d.s[0] = DomainMoment[i].s[0];
                    Nodes[no].u.d.s[1] = DomainMoment[i].s[1];
                    Nodes[no].u.d.s[2] = DomainMoment[i].s[2];
                    Extnodes[no].vs[0] = DomainMoment[i].vs[0];
                    Extnodes[no].vs[1] = DomainMoment[i].vs[1];
                    Extnodes[no].vs[2] = DomainMoment[i].vs[2];
                    Nodes[no].u.d.mass = DomainMoment[i].mass;
                    Extnodes[no].hmax = DomainMoment[i].hmax;
                    Extnodes[no].vmax = DomainMoment[i].vmax;

                    Nodes[no].u.d.bitflags =
                        (Nodes[no].u.d.bitflags & (~BITFLAG_MASK)) | (DomainMoment[i].bitflags & BITFLAG_MASK);

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
                    Nodes[no].maxsoft = DomainMoment[i].maxsoft;
#endif // defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
                }
            }
        }
    }
    myfree(DomainMoment);
}

/**
 * @brief Update the top-level tree.
 *
 * This function updates the top-level tree after the multipole moments of
 * the pseudo-particles have been updated.
 *
 * @param no Index of the current node.
 */
void force_treeupdate_pseudos(int no)
{
    int multiple_flag;
    MyFloat s[3], vs[3];

#if defined(ADAPTGRAVSOFT) || defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(mSIDM) || defined(mIDM)
    MyFloat maxsoft = 0;
#endif // defined(ADAPTGRAVSOFT) || defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(mSIDM) || defined(mIDM)

    MyFloat mass = 0;
    s[0] = 0;
    s[1] = 0;
    s[2] = 0;
    vs[0] = 0;
    vs[1] = 0;
    vs[2] = 0;
    MyFloat hmax = 0;
    MyFloat vmax = 0;

    int count_particles = 0;
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    int maxsofttype = 7;
    int diffsoftflag = 0;
#endif // ADAPTIVE_GRAVSOFT_FORGAS

    int p = Nodes[no].u.d.nextnode;

    for (int j = 0; j < 8;
         j++) /* since we are dealing with top-level nodes, we now that there are 8 consecutive daughter nodes */
    {
        if (p >= All.MaxPart && p < All.MaxPart + MaxNodes) /* internal node */
        {
            if (Nodes[p].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL))
            {
                force_treeupdate_pseudos(p);
            }

            mass += (Nodes[p].u.d.mass);
            s[0] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[0]);
            s[1] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[1]);
            s[2] += (Nodes[p].u.d.mass * Nodes[p].u.d.s[2]);

            vs[0] += (Nodes[p].u.d.mass * Extnodes[p].vs[0]);
            vs[1] += (Nodes[p].u.d.mass * Extnodes[p].vs[1]);
            vs[2] += (Nodes[p].u.d.mass * Extnodes[p].vs[2]);

            if (Extnodes[p].hmax > hmax)
            {
                hmax = Extnodes[p].hmax;
            }
            if (Extnodes[p].vmax > vmax)
            {
                vmax = Extnodes[p].vmax;
            }

            if (Nodes[p].u.d.mass > 0)
            {
                if (Nodes[p].u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES))
                {
                    count_particles += 2;
                }
                else
                {
                    count_particles++;
                }
            }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
            diffsoftflag |= maskout_different_softening_flag(Nodes[p].u.d.bitflags);

            if (maxsofttype == 7)
            {
                maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
            }
            else
            {
                int current_maxsofttype = extract_max_softening_type(Nodes[p].u.d.bitflags);
                if (current_maxsofttype != 7)
                {
                    if (All.ForceSoftening[current_maxsofttype] > All.ForceSoftening[maxsofttype])
                    {
                        maxsofttype = current_maxsofttype;
                        diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                    }
                    else
                    {
                        if (All.ForceSoftening[current_maxsofttype] < All.ForceSoftening[maxsofttype])
                            diffsoftflag = (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE);
                    }
                }
            }
#else // ADAPTIVE_GRAVSOFT_FORGAS
            if (Nodes[p].maxsoft > maxsoft)
            {
                maxsoft = Nodes[p].maxsoft;
            }
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#if defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
            if (Nodes[p].maxsoft > maxsoft)
            {
                maxsoft = Nodes[p].maxsoft;
            }
#endif // defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
        }
        else
        {
            ENDRUN_MAIN("Error in the top-level tree reconstruction.");
        }

        p = Nodes[p].u.d.sibling;
    }

    if (mass)
    {
        s[0] /= mass;
        s[1] /= mass;
        s[2] /= mass;
        vs[0] /= mass;
        vs[1] /= mass;
        vs[2] /= mass;
    }
    else
    {
        s[0] = Nodes[no].center[0];
        s[1] = Nodes[no].center[1];
        s[2] = Nodes[no].center[2];
        vs[0] = 0;
        vs[1] = 0;
        vs[2] = 0;
    }

    Nodes[no].u.d.s[0] = s[0];
    Nodes[no].u.d.s[1] = s[1];
    Nodes[no].u.d.s[2] = s[2];
    Extnodes[no].vs[0] = vs[0];
    Extnodes[no].vs[1] = vs[1];
    Extnodes[no].vs[2] = vs[2];
    Nodes[no].u.d.mass = mass;

    Extnodes[no].hmax = hmax;
    Extnodes[no].vmax = vmax;

    Extnodes[no].Flag = GlobFlag;

    if (count_particles > 1)
    {
        multiple_flag = (1 << BITFLAG_MULTIPLEPARTICLES);
    }
    else
    {
        multiple_flag = 0;
    }

    Nodes[no].u.d.bitflags &= (~BITFLAG_MASK); /* this clears the bits */
    Nodes[no].u.d.bitflags |= multiple_flag;

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    Nodes[no].u.d.bitflags |= diffsoftflag + (maxsofttype << BITFLAG_MAX_SOFTENING_TYPE);
#else // ADAPTIVE_GRAVSOFT_FORGAS
    Nodes[no].maxsoft = maxsoft;
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#if defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
    Nodes[no].maxsoft = maxsoft;
#endif // defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
}

/**
 * @brief Flag nodes that are dependent on local particle data.
 *
 * This function flags nodes in the top-level tree that are dependent on
 * local particle data.
 */
void force_flag_localnodes(void)
{

    /* mark all top-level nodes */
    for (int i = 0; i < NTopleaves; i++)
    {
        int no = DomainNodeIndex[i];

        while (no >= 0)
        {
            if (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL))
            {
                break;
            }

            Nodes[no].u.d.bitflags |= (1 << BITFLAG_TOPLEVEL);
            no = Nodes[no].u.d.father;
        }

        /* mark also internal top level nodes */
        no = DomainNodeIndex[i];
        no = Nodes[no].u.d.father;

        while (no >= 0)
        {
            if (Nodes[no].u.d.bitflags & (1 << BITFLAG_INTERNAL_TOPLEVEL))
            {
                break;
            }

            Nodes[no].u.d.bitflags |= (1 << BITFLAG_INTERNAL_TOPLEVEL);

            no = Nodes[no].u.d.father;
        }
    }

    /* mark top-level nodes that contain local particles */

    for (int m = 0; m < Multipledomains; m++)
    {
        for (int i = DomainStartList[ThisTask * Multipledomains + m];
             i <= DomainEndList[ThisTask * Multipledomains + m]; i++)
        {
            int no = DomainNodeIndex[i];

            ENDRUN_IF(DomainTask[i] != ThisTask, "Error in domain task assignment");

            while (no >= 0)
            {
                if (Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS))
                {
                    break;
                }

                Nodes[no].u.d.bitflags |= (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS);
                no = Nodes[no].u.d.father;
            }
        }
    }
}

/**
 * @brief Add a new star particle to the tree.
 *
 * When a new additional star particle is created, we can put it into the
 * tree at the position of the spawning gas particle. This is possible
 * because the Nextnode[] array essentially describes the full tree walk as a
 * link list. Multipole moments of tree nodes need not be changed.
 *
 * @param igas Index of the spawning gas particle.
 * @param istar Index of the new star particle.
 */
void force_add_star_to_tree(int igas, int istar)
{
    int no = Nextnode[igas];
    Nextnode[igas] = istar;
    Nextnode[istar] = no;
    Father[istar] = Father[igas];
}

#endif // ACC_INCLUDE_GRAVITY

/**
 * @brief Compute the gravitational force for a given local particle.
 *
 * This routine computes the gravitational force for a given local
 * particle, or for a particle in the communication buffer. Depending on
 * the value of ErrTolTheta, either the geometrical BH cell-opening
 * criterion (!=0), or the `relative' opening criterion is used (=0).
 *
 * @param active_target Index of the active target particle.
 * @param p_target Index of the target particle.
 * @param mode Mode of the computation.
 * @param P Pointer to the gravity_walkdata struct.
 * @param GravDataGet Pointer to the gravdata_in struct.
 * @param GravDataResult Pointer to the gravdata_out struct.
 * @param NodeGravCosts Pointer to the NodeGravCosts array.
 * @param PGravCosts Pointer to the PGravCosts array.
 *
 * @return Number of interactions.
 */
#ifdef ACC_INCLUDE_GRAVITY
#warning "compile is entering the ACC_INCLUDE_GRAVITY region. Following errors are related to the GPU version."

ACC_DECLARE_ROUTINE_BEFORE
int force_treeevaluate_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                             struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult,
                             double *NodeGravCosts, double *PGravCosts);
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int force_treeevaluate_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                             struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult,
                             double *NodeGravCosts, double *PGravCosts)
#else // ACC_INCLUDE_GRAVITY
/**
 * @brief Compute the gravitational force for a given local particle.
 *
 * This routine computes the gravitational force for a given local
 * particle, or for a particle in the communication buffer. Depending on
 * the value of ErrTolTheta, either the geometrical BH cell-opening
 * criterion (!=0), or the `relative' opening criterion is used (=0).
 *
 * @param target Index of the active target particle.
 * @param mode Mode of the computation.
 * @param exportflag Pointer to the exportflag array.
 * @param exportnodecount Pointer to the exportnodecount array.
 * @param exportindex Pointer to the exportindex array.
 * Ifdef ACC
 * @param should_drift Flag to indicate if the particle should be drifted.
 *
 * @return Number of interactions.
 */
#ifndef ACC_GRAVITY
int force_treeevaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex)
#else // ACC_GRAVITY
int force_treeevaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int should_drift)
#endif // ACC_GRAVITY
#endif // ACC_INCLUDE_GRAVITY
{
    struct NODE *nop = 0;

#ifdef ACC_INCLUDE_GRAVITY
    struct NODE *Nodes = Nodes_base - All.MaxPart;
#endif // ACC_INCLUDE_GRAVITY
    int no, nodesinlist, ninteractions, ptype, listindex = 0;
#ifndef ACC_INCLUDE_GRAVITY
    int nexp, task;
#endif // ACC_INCLUDE_GRAVITY
    MyAtLeastDouble r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv;
    MyAtLeastDouble pos_x, pos_y, pos_z;
    MyLongDouble acc_x, acc_y, acc_z, aold;
    // cache some global vars in local vars to help compiler with alias analysis
    int maxPart = All.MaxPart;
#ifndef ACC_INCLUDE_GRAVITY
    int bunchSize = All.BunchSize;
#endif // ACC_INCLUDE_GRAVITY
    int maxNodes = MaxNodes;
    double errTol2 = All.ErrTolTheta * All.ErrTolTheta; /* constant over the whole walk, avoid re-squaring per node */

#ifdef EVALPOTENTIAL
    MyAtLeastDouble wp;
    MyLongDouble pot;
    pot = 0.0;
#endif // EVALPOTENTIAL

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyAtLeastDouble soft = 0;
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
    MyAtLeastDouble h_p, h_p_inv, h_p3_inv, u_p, fac_p, h_max;
    MyAtLeastDouble zeta, omega, dWdr, dWdr_p, corr;
    MyAtLeastDouble mass_target;
    int particle;
#ifdef AGS_OUTPUTCORR
    MyAtLeastDouble correction = 0;
#endif // AGS_OUTPUTCORR
#endif // ADAPTGRAVSOFT

#ifdef ACC_INCLUDE_GRAVITY
    int target;
    if (mode == 1)
    {
        target = active_target;
    }
    if (mode == 2)
    {
        target = p_target;
    }
// else // ACC_INCLUDE_GRAVITY
//  ENDRUN_MAIN("not tested with openacc");
#endif // ACC_INCLUDE_GRAVITY

    acc_x = 0;
    acc_y = 0;
    acc_z = 0;
    ninteractions = 0;
    nodesinlist = 0;

    if (mode == 0 || mode == 3 || mode == 2)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];

        ptype = P[target].Type;
        aold = All.ErrTolForceAcc * P[target].OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = P[target].Hsml;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
        zeta = P[target].AGS_zeta;
        omega = P[target].AGS_omega;
        h = P[target].AGS_Hsml;
        mass_target = P[target].Mass;
#endif // ADAPTGRAVSOFT
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        ptype = GravDataGet[target].Type;
        aold = All.ErrTolForceAcc * GravDataGet[target].OldAcc;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = GravDataGet[target].Soft;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
        zeta = GravDataGet[target].AGS_zeta;
        omega = GravDataGet[target].AGS_omega;
        h = GravDataGet[target].AGS_Hsml;
        mass_target = GravDataGet[target].Mass;
#endif // ADAPTGRAVSOFT
    }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    double softening_self = All.ForceSoftening[ptype]; /* invariant for the whole walk of this target */
#endif // ADAPTIVE_GRAVSOFT_FORGAS

    if (mode == 0 || mode == 3 || mode == 2)
    {
        no = maxPart; /* root node */
    }
    else
    {
        nodesinlist++;
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {

            if (no < maxPart) /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */
#ifndef ACC_INCLUDE_GRAVITY
#ifdef ACC_GRAVITY
                if (should_drift)
#endif // ACC
                    if (P[no].Ti_current != All.Ti_Current)
                    {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                        drift_particle(no, All.Ti_Current);
                    }
#endif // ACC_INCLUDE_GRAVITY

                if (mode == 3)
                {
                    no = Nextnode[no];
                    continue;
                }

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                mass = P[no].Mass;
            }
            else
            {
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    if (mode == 0 || mode == 3)
                    {
#ifndef ACC_INCLUDE_GRAVITY
                        if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            exportflag[task] = target;
                            exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (exportnodecount[task] == NODELISTLENGTH)
                        {
#pragma omp atomic capture
                            nexp = Nexport++;
                            if (nexp >= bunchSize)
                            {
                                BufferFullFlag = 1;
                                Nexport = bunchSize;
                                return -1;
                            }

                            exportnodecount[task] = 0;
                            exportindex[task] = nexp;
                            DataIndexTable[nexp].Task = task;
                            DataIndexTable[nexp].Index = target;
                            DataIndexTable[nexp].IndexGet = nexp;
                        }

                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];

                        if (exportnodecount[task] < NODELISTLENGTH)
                            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
#endif // ACC_INCLUDE_GRAVITY
                    }
                    no = Nextnode[no - maxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                /*              assert(no-All.MaxPart<Numnodestree);
                   if(no-All.MaxPart>=Numnodestree) ENDRUN_MAIN("@@@@@@@@@ %d %d",no-All.MaxPart, Numnodestree ); */
                mass = nop->u.d.mass;

                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    if (mass)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                if (mode == 3 && !(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
                {
                    no = nop->u.d.sibling;
                    continue;
                }

#ifndef ACC_INCLUDE_GRAVITY
#ifdef ACC_GRAVITY
                if (should_drift)
#endif // ACC
                    if (nop->Ti_current != All.Ti_Current)
                    {
#pragma omp critical(_partnodedrift_)
                        force_drift_node(no, All.Ti_Current);
                    }
#endif // ACC_INCLUDE_GRAVITY
                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
            }

#if defined(PERIODIC)
            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);
#endif // defined(PERIODIC)
            r2 = dx * dx + dy * dy + dz * dz;

            if (no < maxPart)
            {
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }

                if (P[no].Type == 0)
                {
                    if (h < P[no].Hsml)
                        h = P[no].Hsml;
                }
                else
                {
                    if (h < All.ForceSoftening[P[no].Type])
                        h = All.ForceSoftening[P[no].Type];
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                h = softening_self;
                if (h < All.ForceSoftening[P[no].Type])
                {
                    h = All.ForceSoftening[P[no].Type];
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#ifdef ADAPTGRAVSOFT
                /* need to keep track of the softening (h_p) of the particle */
                h_p = P[no].AGS_Hsml;
                particle = no;
#endif // ADAPTGRAVSOFT
                if (TakeLevel >= 0)
                {

#ifdef ACC_INCLUDE_GRAVITY

#ifdef ACC_ATOMIC
#warning "OMP/ACC ATOMIC switched on"

#ifdef ACC_OPENACC
#pragma acc atomic
#elif defined(ACC_OPENMP)
#pragma omp atomic update
#endif // ACC_OPENMP

#endif // ACC_ATOMIC
                    PGravCosts[no] += 1;
#else // ACC_INCLUDE_GRAVITY
                    if (mode == 1 || mode == 0 || mode == 3)
                    {
#ifndef NO_THREAD_SAFE_COSTS
#pragma omp atomic
#endif // NO_THREAD_SAFE_COSTS
                        P[no].GravCost[TakeLevel] += 1.0;
                    }
#endif // ACC_INCLUDE_GRAVITY
                }
                no = Nextnode[no];
#ifdef ADAPTGRAVSOFT
                /* otherwise r=0 and corr=nan. This check must be put after the next node has been selected! */
                if (target == particle)
                {
                    continue;
                }
#endif // ADAPTGRAVSOFT
            }
            else /* we have an  internal node. Need to check opening criterion */
            {
                if (All.ErrTolTheta) /* check Barnes-Hut opening criterion */
                {
                    if (nop->len * nop->len > r2 * errTol2)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                else /* check relative opening criterion */
                {
                    if (mass * nop->len * nop->len > r2 * r2 * aold)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    /* check in addition whether we lie inside the cell */
                    MyAtLeastDouble node_dx = nop->center[0] - pos_x;
#if defined(PERIODIC)
                    node_dx = NEAREST_X(node_dx);
#endif // defined(PERIODIC)
                    if (abs(node_dx) < 0.60 * nop->len)
                    {
                        MyAtLeastDouble node_dy = nop->center[1] - pos_y;
#if defined(PERIODIC)
                        node_dy = NEAREST_Y(node_dy);
#endif // defined(PERIODIC
                        if (abs(node_dy) < 0.60 * nop->len)
                        {
                            MyAtLeastDouble node_dz = nop->center[2] - pos_z;
#if defined(PERIODIC)
                            node_dz = NEAREST_Z(node_dz);
#endif // defined(PERIODIC)
                            if (abs(node_dz) < 0.60 * nop->len)
                            {
                                no = nop->u.d.nextnode;
                                continue;
                            }
                        }
                    }
                }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = softening_self;
                if (h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
                {
                    h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
                    if (r2 < h * h)
                    {
                        if (maskout_different_softening_flag(nop->u.d.bitflags)) /* signals that there are particles of
                                                                                    different softening in the node */
                        {
                            no = nop->u.d.nextnode;
                            continue;
                        }
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }

                if (h < nop->maxsoft)
                {
                    h = nop->maxsoft;
                    if (r2 < h * h)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#ifdef ADAPTGRAVSOFT
                h_p = nop->maxsoft; /* softening of the node */
                h_max = h >= h_p ? h : h_p;

                if (r2 < h_max * h_max)
                {
                    no = nop->u.d.nextnode;
                    continue; /* discard cases where the particle-node separation is less than the bigger
                             between the target softening and the node softening
                           */
                }

                particle = no;
#endif // ADAPTGRAVSOFT
                if (TakeLevel >= 0)
                {
#ifdef ACC_INCLUDE_GRAVITY

#ifdef ACC_ATOMIC
#warning "OMP/ACC ATOMIC switched on"

#ifdef ACC_OPENACC
#pragma acc atomic
#elif defined(ACC_OPENMP)
#pragma omp atomic update
#endif // ACC_OPENMP

#endif // ACC_ATOMIC
                    NodeGravCosts[no] += 1;
#else // ACC_INCLUDE_GRAVITY
                    if (mode == 1 || mode == 0 || mode == 3)
                    {

#ifndef NO_THREAD_SAFE_COSTS
#pragma omp atomic update
#endif // NO_THREAD_SAFE_COSTS
                        nop->GravCost += 1.0;
                    }
#endif // ACC_INCLUDE_GRAVITY
                }
                no = nop->u.d.sibling; /* ok, node can be used */
            }

            if (mode == 3)
            {
                continue;
            }

            r = sqrt(r2);

#ifdef ADAPTGRAVSOFT
            h_max = h >= h_p ? h : h_p;
            if (r >= h_max)
            {
                /* no need to worry about softening lengths: interaction is newtonian.
                 * All particle - node interactions fall here */
                fac = mass / (r2 * r);
                corr = 0;
            }
#else // ADAPTGRAVSOFT
            if (r >= h)
            {
                fac = mass / (r2 * r);
#ifdef EVALPOTENTIAL
                pot += -mass / r;
#endif // EVALPOTENTIAL
            }
#endif // ADAPTGRAVSOFT
            else
            {
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
#ifndef ADAPTGRAVSOFT
                u = r * h_inv;
                if (u < 0.5)
                {
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                }
                else
                {
                    fac = mass * h3_inv *
                          (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u -
                           0.066666666667 / (u * u * u));
                }
#ifdef EVALPOTENTIAL
                if (u < 0.5)
                {
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                }
                else
                {
                    wp = -3.2 + 0.066666666667 / u +
                         u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                }
                pot += mass * h_inv * wp;
#endif // EVALPOTENTIAL
#else // ADAPTGRAVSOFT
                /* interaction is smoothed: only particle-particle, no nodes down here! */
                dWdr = dWdr_p = corr = 0;
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
                u = r * h_inv;

                if (u > 1.)
                {
                    fac = mass / (r2 * r);
                }

                else if (u < 0.5)
                {
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                    dWdr = 6. * KERNEL_COEFF_1 * h3_inv * h_inv * (-2. * u + 3. * u * u);
                }
                else
                {
                    fac = mass * h3_inv *
                          (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u -
                           0.066666666667 / (u * u * u));
                    dWdr = -6. * KERNEL_COEFF_1 * h3_inv * h_inv * ((1. - u) * (1. - u));
                }

                h_p_inv = 1.0 / h_p;
                h_p3_inv = h_p_inv * h_p_inv * h_p_inv;
                u_p = r * h_p_inv;

                if (u_p > 1.)
                {
                    fac_p = mass / (r2 * r);
                }

                else if (u_p < 0.5)
                {
                    fac_p = mass * h_p3_inv * (10.666666666667 + u_p * u_p * (32.0 * u_p - 38.4));
                    dWdr_p = 6. * KERNEL_COEFF_1 * h_p3_inv * h_p_inv * (-2. * u_p + 3. * u_p * u_p);
                }
                else
                {
                    fac_p = mass * h_p3_inv *
                            (21.333333333333 - 48.0 * u_p + 38.4 * u_p * u_p - 10.666666666667 * u_p * u_p * u_p -
                             0.066666666667 / (u_p * u_p * u_p));
                    dWdr_p = -6. * KERNEL_COEFF_1 * h_p3_inv * h_p_inv * (1. - u_p) * (1. - u_p);
                }

                /* force = (fac + fac_p)/2 */
                fac += fac_p;
                fac /= 2.;

#ifndef AGS_NOCORRECTION
                /* correction term */
                corr =
                    (zeta * omega * dWdr + mass / mass_target * P[particle].AGS_zeta * P[particle].AGS_omega * dWdr_p) /
                    r;

                corr /= 2.;

#ifdef AGS_OUTPUTCORR
                correction += corr;
#endif // AGS_OUTPUTCORR
                /***************************************************************************************************/
                /* The original correction term is:                                                                */
                /*   corr = mass * (termI / termII * dWdr +  P[particle].termI / P[particle].termII * dWdr_p) / r; */
                /*   corr /= 2.;                                                                                   */
                /* But remember that termI and termII would have a different definition.                           */
                /***************************************************************************************************/
#endif // AGS_NOCORRECTION
                if (particle >= maxPart)
                {
                    VERBOSE(1, "*PROBLEM in ADAPTGRAVSOFT*: we are having a smoothed particle-node interaction!\n");
                    VERBOSE(1, "Particle, Node: %i, %i \n", target, particle);
                }
#endif // ADAPTGRAVSOFT
            }

#ifdef EVALPOTENTIAL
#ifdef PERIODIC
            pot += mass * ewald_pot_corr(dx, dy, dz);
#endif // PERIODIC
#endif // EVALPOTENTIAL

            acc_x += dx * fac;
            acc_y += dy * fac;
            acc_z += dz * fac;

#ifdef ADAPTGRAVSOFT
            acc_x += dx * corr;
            acc_y += dy * corr;
            acc_z += dz * corr;
#endif // ADAPTGRAVSOFT

            if (mass > 0)
            {
                ninteractions++;
            }
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                {
                    nodesinlist++;
                    no = Nodes[no].u.d.nextnode; /* open it */
                }
            }
        }
    }

#ifndef ACC_INCLUDE_GRAVITY
    if (mode == 3)
    {
        return 0;
    }
#endif // ACC_INCLUDE_GRAVITY

#ifdef ACC_INCLUDE_GRAVITY
    target = active_target; // holds for both mode==2 and mode==1 when in the GPU
#endif // ACC_INCLUDE_GRAVITY

#ifndef ACC_INCLUDE_GRAVITY

    /* store result at the proper place */
    if (mode == 0)
    {
        P[target].GravAccel[0] = acc_x;
        P[target].GravAccel[1] = acc_y;
        P[target].GravAccel[2] = acc_z;
#ifdef FOLLOW_ID
        if (P[target].ID == FOLLOW_ID)
        {
            printf("Task %d (mode=0): acc=(%g,%g,%g)\n", ThisTask, P[target].GravAccel[0], P[target].GravAccel[1],
                   P[target].GravAccel[2]);
        }
#endif // FOLLOW_ID
#ifdef EVALPOTENTIAL
        P[target].Potential = pot;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
        P[target].AGS_corr = correction;
#endif //  defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
    }
    else
#endif // ACC_INCLUDE_GRAVITY
    {
        GravDataResult[target].Acc[0] = acc_x;
        GravDataResult[target].Acc[1] = acc_y;
        GravDataResult[target].Acc[2] = acc_z;
#ifdef FOLLOW_ID
        if (GravDataGet[target].ID == FOLLOW_ID)
        {
            printf("Task %d (mode=%1d): acc=(%g,%g,%g)\n", ThisTask, mode, GravDataResult[target].Acc[0],
                   GravDataResult[target].Acc[1], GravDataResult[target].Acc[2]);
        }
#endif // FOLLOW_ID
#ifdef EVALPOTENTIAL
        GravDataResult[target].Potential = pot;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
        GravDataResult[target].AGS_corr = correction;
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
#ifndef ACC_INCLUDE_GRAVITY
        *exportflag = nodesinlist;
#endif // ACC_INCLUDE_GRAVITY
       //      *exportflag = nodesinlist;
    }
    return ninteractions;
}
#ifdef ACC_INCLUDE_GRAVITY
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_GRAVITY

#ifdef PMGRID
/**
 * @brief Compute the short-range gravitational force for a given local particle.
 *
 * In the TreePM algorithm, the tree is walked only locally around the
 * target coordinate. Tree nodes that fall outside a box of half
 * side-length Rcut= RCUT*ASMTH*MeshSize can be discarded. The short-range
 * potential is modified by a complementary error function, multiplied
 * with the Newtonian form. The resulting short-range suppression compared
 * to the Newtonian force is tabulated, because looking up from this table
 * is faster than recomputing the corresponding factor, despite the
 * memory-access panelty (which reduces cache performance) incurred by the
 * table.
 *
 * @param active_target Index of the active target particle.
 * @param p_target Index of the target particle.
 * @param mode Mode of the computation.
 * @param P Pointer to the gravity_walkdata struct.
 * @param GravDataGet Pointer to the gravdata_in struct.
 * @param GravDataResult Pointer to the gravdata_out struct.
 * @param NodeGravCosts Pointer to the NodeGravCosts array.
 * @param PGravCosts Pointer to the PGravCosts array.
 *
 * @return Number of interactions.
 */
#ifdef ACC_INCLUDE_GRAVITY
#warning "compile is entering the ACC_INCLUDE_GRAVITY region. Following errors are related to the GPU version."

ACC_DECLARE_ROUTINE_BEFORE
#ifdef INTEL_UPDATE
int force_treeevaluate_shortrange_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                                        struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult, d\
ouble *NodeGravCosts,
                                        double *PGravCosts, int &Var_updt);
#else // INTEL_UPDATE
int force_treeevaluate_shortrange_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                                        struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult, d\
ouble *NodeGravCosts,
                                        double *PGravCosts);
#endif // INTEL_UPDATE
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
#if defined(ACC_OPENMP) && defined(INTEL_UPDATE)
int force_treeevaluate_shortrange_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                                        struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult, do\
uble *NodeGravCosts,
                                        double *PGravCosts, int &Var_updt)
#else // ACC_OPENMP && INTEL_UPDATE
int force_treeevaluate_shortrange_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                                        struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult, do\
uble *NodeGravCosts,
                                        double *PGravCosts)
#endif // ACC_OPENMP && INTEL_UPDATE
#else // ACC_INCLUDE_GRAVITY - standard gadget.
/**
 * @brief Compute the short-range gravitational force for a given local particle.
 *
 * In the TreePM algorithm, the tree is walked only locally around the
 * target coordinate. Tree nodes that fall outside a box of half
 * side-length Rcut= RCUT*ASMTH*MeshSize can be discarded. The short-range
 * potential is modified by a complementary error function, multiplied
 * with the Newtonian form. The resulting short-range suppression compared
 * to the Newtonian force is tabulated, because looking up from this table
 * is faster than recomputing the corresponding factor, despite the
 * memory-access panelty (which reduces cache performance) incurred by the
 * table.
 *
 * @param target Index of the active target particle.
 * @param mode Mode of the computation.
 * @param exportflag Pointer to the exportflag array.
 * @param exportnodecount Pointer to the exportnodecount array.
 * @param exportindex Pointer to the exportindex array.
 * Ifdef ACC
 * @param should_drift Flag to indicate if the particle should be drifted.
 *
 * @return Number of interactions.
 */
#ifndef ACC_GRAVITY
int force_treeevaluate_shortrange(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex)
#else // ACC
int force_treeevaluate_shortrange(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                  int should_drift)
#endif // ACC
#endif // ACC_INCLUDE_GRAVITY
{
    struct NODE *nop = 0;
#ifdef ACC_INCLUDE_GRAVITY
    struct NODE *Nodes = Nodes_base - All.MaxPart;
#endif // ACC_INCLUDE_GRAVITY

    int nodesinlist, ptype, ninteractions, nexp, tabindex, task, listindex = 0;
    MyAtLeastDouble r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv;
    MyAtLeastDouble pos_x, pos_y, pos_z, aold;
    MyAtLeastDouble eff_dist;
    MyAtLeastDouble rcut, asmth, asmthfac, rcut2, dist;
    MyLongDouble acc_x, acc_y, acc_z;
    // cache some global vars in local vars to help compiler with alias analysis
    int maxPart = All.MaxPart;
    int bunchSize = All.BunchSize;
    int maxNodes = MaxNodes;
    integertime ti_Current = All.Ti_Current;
    MyAtLeastDouble errTol2 = All.ErrTolTheta * All.ErrTolTheta;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat soft = 0;
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef EVALPOTENTIAL
    MyAtLeastDouble wp, facpot;
    MyLongDouble pot;
    pot = 0;
#endif // EVALPOTENTIAL

#ifdef ADAPTGRAVSOFT
    MyAtLeastDouble h_p, h_p_inv, h_p3_inv, u_p, fac_p, h_max;
    MyAtLeastDouble zeta, omega, dWdr, dWdr_p, corr;
    MyAtLeastDouble mass_target;
    int particle;
#ifdef AGS_OUTPUTCORR
    MyAtLeastDouble correction = 0;
#endif // AGS_OUTPUTCORR
#endif // ADAPTGRAVSOFT

    acc_x = 0;
    acc_y = 0;
    acc_z = 0;
    ninteractions = 0;
    nodesinlist = 0;

    rcut = All.Rcut[0];
    asmth = All.Asmth[0];
#ifdef ACC_INCLUDE_GRAVITY
    int target;
#endif // ACC_INCLUDE_GRAVITY

#ifndef ACC_INCLUDE_GRAVITY
    if (mode != 0 && mode != 1 && mode != 3)
    {
        ENDRUN("%d %d %d %d %d\n", target, mode, *exportflag, *exportnodecount, *exportindex);
    }
#endif // ACC_INCLUDE_GRAVITY

#ifdef ACC_INCLUDE_GRAVITY
    if (mode == 1)
    {
        target = active_target;
    }
    if (mode == 2)
    {
        target = p_target;
    }
#endif // ACC_INCLUDE_GRAVITY

    // #if !defined(ACC_INCLUDE_GRAVITY)
    if (mode == 0 || mode == 3 || mode == 2)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
        ptype = P[target].Type;
        aold = All.ErrTolForceAcc * P[target].OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = P[target].Hsml;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef PLACEHIGHRESREGION
        if (pmforce_is_particle_high_res(ptype, P[target].Pos))
        {
            rcut = All.Rcut[1];
            asmth = All.Asmth[1];
        }
#endif // PLACEHIGHRESREGION
#ifdef ADAPTGRAVSOFT
        zeta = P[target].AGS_zeta;
        omega = P[target].AGS_omega;
        h = P[target].AGS_Hsml;
        mass_target = P[target].Mass;
#endif // ADAPTGRAVSOFT
    }
    else if (mode == 1)
    // #endif
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        aold = All.ErrTolForceAcc * GravDataGet[target].OldAcc;

        ptype = GravDataGet[target].Type;

#ifdef PLACEHIGHRESREGION
        if (pmforce_is_particle_high_res(ptype, GravDataGet[target].Pos))
        {
            rcut = All.Rcut[1];
            asmth = All.Asmth[1];
        }
#endif // PLACEHIGHRESREGION

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = GravDataGet[target].Soft;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
        zeta = GravDataGet[target].AGS_zeta;
        omega = GravDataGet[target].AGS_omega;
        h = GravDataGet[target].AGS_Hsml;
        mass_target = GravDataGet[target].Mass;
#endif // ADAPTGRAVSOFT
    }

    rcut2 = rcut * rcut;
    asmthfac = 0.5 / asmth * (NTAB / 3.0);

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
    double softening_self = All.ForceSoftening[ptype]; /* invariant for the whole walk of this target */
#endif // ADAPTIVE_GRAVSOFT_FORGAS

    int no = -1;
    if (mode == 0 || mode == 3 || mode == 2)
    {
        no = maxPart; /* root node */
    }
    else if (mode == 1)
    {
        nodesinlist++;
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {
            if (no < maxPart)
            {
                // #ifdef ACC_INCLUDE_GRAVITY
                //            //              if(no>=NumPart) return -1;
                // #endif // ACC_INCLUDE_GRAVITY

#ifndef ACC_INCLUDE_GRAVITY
                if (no > NumPart)
                {
                    ENDRUN("ThisTask =%d, no = %d numpart=%d", ThisTask, no, NumPart);
                }

#ifdef ACC_GRAVITY
                if (should_drift)
#endif // ACC
                    if (P[no].Ti_current != ti_Current)
                    {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                        drift_particle(no, ti_Current);
                    }
#endif // ACC_GRAVITY

#ifndef ACC_INCLUDE_GRAVITY
                if (mode == 3)
                {
                    no = Nextnode[no];
                    continue;
                }
#endif // ACC_INCLUDE_GRAVITY

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
#ifdef PERIODIC
                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);
#endif // PERIODIC

                r2 = dx * dx + dy * dy + dz * dz;
                mass = P[no].Mass;

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }

                if (P[no].Type == 0)
                {
                    if (h < P[no].Hsml)
                    {
                        h = P[no].Hsml;
                    }
                }
                else
                {
                    if (h < All.ForceSoftening[P[no].Type])
                    {
                        h = All.ForceSoftening[P[no].Type];
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                h = softening_self;

                if (h < All.ForceSoftening[P[no].Type])
                    h = All.ForceSoftening[P[no].Type];
#endif // ADAPTIVE_GRAVSOFT_FORGAS
       //              PRINTF_NUKE("going to takelevel  no=%d, no-maxPart=%d",no,no-maxPart);
                if (TakeLevel >= 0)
                {
#ifdef ACC_INCLUDE_GRAVITY
#ifdef ACC_ATOMIC
#warning "OMP/ACC ATOMIC switched on"

#ifdef ACC_OPENACC
#pragma acc atomic
#elif defined(ACC_OPENMP)
#pragma omp atomic update
#endif // ACC_OPENMP

#endif // ACC_ATOMIC

                    PGravCosts[no] += 1;
#if defined(ACC_OPENMP) && defined(INTEL_UPDATE)
                    Var_updt |= 1 << 0; // For PGravCosts
#endif // ACC_OPENMP && INTEL_UPDATE
#else // ACC_INCLUDE_GRAVITY
                    if (mode == 1 || mode == 0 || mode == 3)
                    {
#ifndef NO_THREAD_SAFE_COSTS
#pragma omp atomic update
#endif // NO_THREAD_SAFE_COSTS
                        P[no].GravCost[TakeLevel] += 1.0;
                    }
#endif // ACC_INCLUDE_GRAVITY
                }
#ifdef ADAPTGRAVSOFT
                /* need to keep track of the softening (h_p) of the particle */
                h_p = P[no].AGS_Hsml;
                particle = no;
#endif // ADAPTGRAVSOFT
                no = Nextnode[no];
                //              PRINTF_NUKE("no=nn[no],  no=%d, no-maxPart=%d",no,no-maxPart);
#ifdef ADAPTGRAVSOFT
                /* otherwise r=0 and corr=nan. This check must be put after the next node has been selected! */
                if (target == particle)
                {
                    continue;
                }
#endif // ADAPTGRAVSOFT
            }
            else /* we have an  internal node */
            {
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    //          PRINTF_NUKE("nodo cattivo,  no=%d, no-maxPart=%d",no,no-maxPart);
#ifndef ACC_INCLUDE_GRAVITY
                    if (no > maxPart + maxNodes + NTopnodes)
                    {
                        ENDRUN("Error in tree. No=%d > maxPart + maxNodes + NTopnodes", no);
                    }
                    /* pseudo particle */
                    if (mode == 0 || mode == 3)
                    {
                        if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            exportflag[task] = target;
                            exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (exportnodecount[task] == NODELISTLENGTH)
                        {
#pragma omp atomic capture
                            nexp = Nexport++;
                            if (nexp >= bunchSize)
                            {
                                BufferFullFlag = 1;
                                Nexport = bunchSize;
                                return -1;
                            }
                            /*
                               int exitFlag = 0;
                               #pragma omp critical
                               {
                               nexp = Nexport++;

                               if(nexp >= bunchSize)
                               {
                               Nexport = bunchSize;
                               // out of buffer space. Need to discard work for this particle and interrupt
                               BufferFullFlag = 1;
                               exitFlag = 1;
                               }
                               }
                               if(exitFlag)
                               return -1;
                             */

                            exportnodecount[task] = 0;
                            exportindex[task] = nexp;
                            DataIndexTable[nexp].Task = task;
                            DataIndexTable[nexp].Index = target;
                            DataIndexTable[nexp].IndexGet = nexp;
                        }

                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];

                        if (exportnodecount[task] < NODELISTLENGTH)
                        {
                            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                        }
                    }
// #else // ACC_INCLUDE_GRAVITY
//              printf("no=%d maxNodes=%d\n",no,maxNodes);
#endif // ACC_INCLUDE_GRAVITY
                    no = Nextnode[no - maxNodes];
                    continue;
                }
                //              PRINTF_NUKE("nodo interno nno=%d, no-maxPart=%d",no,no-maxPart);e

#ifndef ACC_INCLUDE_GRAVITY
                assert(no - All.MaxPart < Numnodestree);
#endif // ACC_INCLUDE_GRAVITY
       //              if(no-All.MaxPart>Numnodestree) return -no;
#ifdef ACC_GRAVITY
                int _no = no; // save old value because others (as GPU NodeGravCosts) may need it before it get
                              // converted to nextnode;
#endif // ACC_GRAVITY
                nop = &Nodes[no];

                // GPU runs with mode==2 and needs to do everynode.
                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    no = nop->u.d.nextnode;
                    continue;
                }

                if (mode == 3 && !(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
                {
                    no = nop->u.d.sibling;
                    continue;
                }

#ifndef ACC_INCLUDE_GRAVITY
#ifdef ACC_GRAVITY
                if (should_drift)
#endif // ACC
                    if (nop->Ti_current != ti_Current)
                    {
#pragma omp critical(_partnodedrift_)
                        force_drift_node(no, ti_Current);
                    }
#endif // ACC_INCLUDE_GRAVITY
                mass = nop->u.d.mass;

                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;

                dx = NEAREST(dx);
                dy = NEAREST(dy);
                dz = NEAREST(dz);
                r2 = dx * dx + dy * dy + dz * dz;

                /* check whether we can stop walking along this branch */
                if (r2 > rcut2)
                {
                    eff_dist = rcut + 0.5 * nop->len;

                    dist = NEAREST(nop->center[0] - pos_x);
                    if (dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }

                    dist = NEAREST(nop->center[1] - pos_y);
                    if (dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }

                    dist = NEAREST(nop->center[2] - pos_z);
                    if (dist < -eff_dist || dist > eff_dist)
                    {
                        no = nop->u.d.sibling;
                        continue;
                    }
                }

                if (errTol2) /* check Barnes-Hut opening criterion */
                {
                    if (nop->len * nop->len > r2 * errTol2)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                else /* check relative opening criterion */
                {
                    if (mass * nop->len * nop->len > r2 * r2 * aold)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    /* check in addition whether we lie inside the cell */

                    if (abs(nop->center[0] - pos_x) < 0.60 * nop->len)
                    {
                        if (abs(nop->center[1] - pos_y) < 0.60 * nop->len)
                        {
                            if (abs(nop->center[2] - pos_z) < 0.60 * nop->len)
                            {
                                no = nop->u.d.nextnode;
                                continue;
                            }
                        }
                    }
                }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = softening_self;
                if (h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
                {
                    h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
                    if (r2 < h * h)
                    {
                        if (maskout_different_softening_flag(
                                nop->u.d.bitflags)) /* bit-5 signals that there are particles of different softening in
                                                       the node */
                        {
                            no = nop->u.d.nextnode;
                            continue;
                        }
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }

                if (h < nop->maxsoft)
                {
                    h = nop->maxsoft;
                    if (r2 < h * h)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
                h_p = nop->maxsoft; /* softening of the node */
                h_max = h >= h_p ? h : h_p;

                if (r2 < h_max * h_max)
                {
                    no = nop->u.d.nextnode;
                    continue; /* discard cases where the particle-node separation is less than the bigger
                             between the target softening and the node softening
                           */
                }
                particle = no;
#endif // ADAPTGRAVSOFT

                if (TakeLevel >= 0)
                {
#ifdef ACC_INCLUDE_GRAVITY

#ifdef ACC_ATOMIC
#warning "OMP/ACC ATOMIC switched on"

#ifdef ACC_OPENACC
#pragma acc atomic
#elif defined(ACC_OPENMP)
#pragma omp atomic update
#endif // ACC_OPENMP

#endif // ACC_ATOMIC

                    NodeGravCosts[_no] += 1;

#if defined(ACC_OPENMP) && defined(INTEL_UPDATE)
                    Var_updt |= 1 << 1; // for NodeGravCosts
#endif // ACC_OPENMP && INTEL_UPDATE

#else // ACC_INCLUDE_GRAVITY
                    if (mode == 1 || mode == 0 || mode == 3)
                    {
#ifndef NO_THREAD_SAFE_COSTS
#pragma omp atomic update
#endif // NO_THREAD_SAFE_COSTS
                        nop->GravCost += 1.0;
                    }
#endif // ACC_INCLUDE_GRAVITY
                }
                no = nop->u.d.sibling; /* ok, node can be used */
            }

#ifndef ACC_INCLUDE_GRAVITY
            if (mode == 3)
            {
                continue;
            }
#endif // ACC_INCLUDE_GRAVITY

            //          PRINTF_NUKE("forza e coraggio,  no=%d, no-maxPart=%d",no,no-maxPart);
            r = sqrt(r2);

#ifdef ADAPTGRAVSOFT
            h_max = h >= h_p ? h : h_p;
            if (r >= h_max)
            {
                /* no need to worry about softening lengths: interaction is newtonian.
                 * All particle-node interactions fall here */
                fac = mass / (r2 * r);
                corr = 0;
            }
#else // ADAPTGRAVSOFT
            if (r >= h)
            {
                fac = mass / (r2 * r);

#ifdef EVALPOTENTIAL
                facpot = -mass / r;
#endif // EVALPOTENTIAL
            }
#endif // ADAPTGRAVSOFT
            else
            {
#ifndef ADAPTGRAVSOFT
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
                u = r * h_inv;
                if (u < 0.5)
                {
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                }
                else
                {
                    fac = mass * h3_inv *
                          (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u -
                           0.066666666667 / (u * u * u));
                }
#ifdef EVALPOTENTIAL
                if (u < 0.5)
                {
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                }
                else
                {
                    wp = -3.2 + 0.066666666667 / u +
                         u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                }
                facpot = mass * h_inv * wp;
#endif // EVALPOTENTIAL

#else // ADAPTGRAVSOFT
                /* interaction is smoothed: only particle-particle, no nodes down here! */
                dWdr = dWdr_p = corr = 0;
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
                u = r * h_inv;

                if (u > 1.)
                {
                    fac = mass / (r2 * r);
                }

                else if (u < 0.5)
                {
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                    dWdr = 6. * KERNEL_COEFF_1 * h3_inv * h_inv * (-2. * u + 3. * u * u);
                }
                else
                {
                    fac = mass * h3_inv *
                          (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u -
                           0.066666666667 / (u * u * u));
                    dWdr = -6. * KERNEL_COEFF_1 * h3_inv * h_inv * ((1. - u) * (1. - u));
                }

                h_p_inv = 1.0 / h_p;
                h_p3_inv = h_p_inv * h_p_inv * h_p_inv;
                u_p = r * h_p_inv;

                if (u_p > 1.)
                {
                    fac_p = mass / (r2 * r);
                }

                else if (u_p < 0.5)
                {
                    fac_p = mass * h_p3_inv * (10.666666666667 + u_p * u_p * (32.0 * u_p - 38.4));
                    dWdr_p = 6. * KERNEL_COEFF_1 * h_p3_inv * h_p_inv * (-2. * u_p + 3. * u_p * u_p);
                }
                else
                {
                    fac_p = mass * h_p3_inv *
                            (21.333333333333 - 48.0 * u_p + 38.4 * u_p * u_p - 10.666666666667 * u_p * u_p * u_p -
                             0.066666666667 / (u_p * u_p * u_p));
                    dWdr_p = -6. * KERNEL_COEFF_1 * h_p3_inv * h_p_inv * (1. - u_p) * (1. - u_p);
                }

                /* force = (fac + fac_p)/2 */
                fac += fac_p;
                fac /= 2.;

#ifndef AGS_NOCORRECTION
                /* correction term */
                corr =
                    (zeta * omega * dWdr + mass / mass_target * P[particle].AGS_zeta * P[particle].AGS_omega * dWdr_p) /
                    r;
                corr /= 2.;
#ifdef AGS_OUTPUTCORR
                correction += corr;
#endif // AGS_OUTPUTCORR
                /***************************************************************************************************/
                /* The original correction term is:                                                                */
                /*   corr = mass * (termI / termII * dWdr +  P[particle].termI / P[particle].termII * dWdr_p) / r; */
                /*   corr /= 2.;                                                                                   */
                /* But remember that termI and termII would have a different definition.                           */
                /***************************************************************************************************/
#endif // AGS_NOCORRECTION
                if (particle >= All.MaxPart)
                {
                    VERBOSE_ALL(0, "*PROBLEM in ADAPTGRAVSOFT*: we are having a smoothed particle-node interaction!\n");
                    VERBOSE_ALL(0, "Particle, Node: %i, %i \n", target, particle);
                }
#endif // ADAPTGRAVSOFT
            }

            tabindex = (int)(asmthfac * r);

            if (tabindex < NTAB)
            {
                /*
                   #ifdef ACC_INCLUDE_GRAVITY
                   {
                   int i = tabindex;
                   MyAtLeastDouble u = 3.0 / NTAB * (i + 0.5);
                   fac = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
                   }
                   #else */
                fac *= shortrange_table[tabindex];
                /*
                   #endif
                 */
                acc_x += dx * fac;
                acc_y += dy * fac;
                acc_z += dz * fac;

#ifdef ADAPTGRAVSOFT
                corr *= shortrange_table[tabindex];
                acc_x += dx * corr;
                acc_y += dy * corr;
                acc_z += dz * corr;
#endif // ADAPTGRAVSOFT

#ifdef EVALPOTENTIAL
                pot += facpot * shortrange_table_potential[tabindex];
#endif // EVALPOTENTIAL
            }
            ninteractions++;
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                {
                    nodesinlist++;
                    no = Nodes[no].u.d.nextnode; /* open it */
                }
            }
        }
    } // while

    if (mode == 3)
    {
        return 0;
    }

#ifdef ACC_INCLUDE_GRAVITY
    target = active_target; // holds for both mode==2 and mode==1 when in the GPU
#endif // ACC_INCLUDE_GRAVITY
#ifndef ACC_INCLUDE_GRAVITY
    /* store result at the proper place */
    if (mode == 0) // if mode ==3 you are not updating anything anyway
    {
        P[target].GravAccel[0] = acc_x;
        P[target].GravAccel[1] = acc_y;
        P[target].GravAccel[2] = acc_z;
#ifdef EVALPOTENTIAL
        P[target].Potential = pot;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
        P[target].AGS_corr = correction;
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
    }
    else
#endif // ACC_INCLUDE_GRAVITY
        if (mode == 1 || mode == 2)
        {
            //      if(target>NActivePart) return -1;
            GravDataResult[target].Acc[0] = acc_x;
            GravDataResult[target].Acc[1] = acc_y;
            GravDataResult[target].Acc[2] = acc_z;
#ifdef EVALPOTENTIAL
            GravDataResult[target].Potential = pot;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
            GravDataResult[target].AGS_corr = correction;
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
#ifndef ACC_INCLUDE_GRAVITY
            *exportflag = nodesinlist;
#endif // ACC_INCLUDE_GRAVITY
#if defined(ACC_OPENMP) && defined(INTEL_UPDATE) && defined(ACC_INCLUDE_GRAVITY)
            Var_updt |= 1 << 2; // For LitePOut
#endif // ACC_OPENMP && INTEL_UPDATE && ACC_INCLUDE_GRAVITY
        }
    return ninteractions;
}
#ifdef ACC_INCLUDE_GRAVITY
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_GRAVITY
#endif // PMGRID

#ifdef PERIODIC
/**
 * @brief Compute the Ewald correction for a given local particle.
 *
 * This function computes the Ewald correction, and is needed if periodic
 * boundary conditions together with a pure tree algorithm are used. Note
 * that the ordinary tree walk does not carry out this correction directly
 * as it was done in Gadget-1.1. Instead, the tree is walked a second
 * time. This is actually faster because the "Ewald-Treewalk" can use a
 * different opening criterion than the normal tree walk. In particular,
 * the Ewald correction is negligible for particles that are very close,
 * but it is large for particles that are far away (this is quite
 * different for the normal direct force). So we can here use a different
 * opening criterion. Sufficient accuracy is usually obtained if the node
 * length has dropped to a certain fraction ~< 0.25 of the
 * BoxLength. However, we may only short-cut the interaction list of the
 * normal full Ewald tree walk if we are sure that the whole node and all
 * daughter nodes "lie on the same side" of the periodic boundary,
 * i.e. that the real tree walk would not find a daughter node or particle
 * that was mapped to a different nearest neighbour position when the tree
 * walk would be further refined.
 *
 * @param active_target The index of the particle for which the Ewald correction is computed.
 * @param p_target The index of the particle for which the Ewald correction is computed.
 * @param mode The mode of the Ewald correction.
 * @param P The particle data.
 * @param GravDataGet The gravity data.
 * @param GravDataResult The gravity data.
 * @param NodeGravCosts The cost of the gravity calculation for each node.
 * @param PGravCosts The cost of the gravity calculation for each particle.
 *
 * @return The number of interactions.
 */
#ifdef ACC_INCLUDE_GRAVITY
#warning "compile is entering the ACC_INCLUDE_GRAVITY region. Following errors are related to the GPU version."

ACC_DECLARE_ROUTINE_BEFORE
int force_treeevaluate_ewald_correction_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                                              struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult,
                                              double *NodeGravCosts, double *PGravCosts);
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int force_treeevaluate_ewald_correction_local(int active_target, int p_target, int mode, struct gravity_walkdata *P,
                                              struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult,
                                              double *NodeGravCosts, double *PGravCosts)
#else // standard gadget.
/**
 * @brief Compute the Ewald correction for a given local particle.
 *
 * This function computes the Ewald correction, and is needed if periodic
 * boundary conditions together with a pure tree algorithm are used. Note
 * that the ordinary tree walk does not carry out this correction directly
 * as it was done in Gadget-1.1. Instead, the tree is walked a second
 * time. This is actually faster because the "Ewald-Treewalk" can use a
 * different opening criterion than the normal tree walk. In particular,
 * the Ewald correction is negligible for particles that are very close,
 * but it is large for particles that are far away (this is quite
 * different for the normal direct force). So we can here use a different
 * opening criterion. Sufficient accuracy is usually obtained if the node
 * length has dropped to a certain fraction ~< 0.25 of the
 * BoxLength. However, we may only short-cut the interaction list of the
 * normal full Ewald tree walk if we are sure that the whole node and all
 * daughter nodes "lie on the same side" of the periodic boundary,
 * i.e. that the real tree walk would not find a daughter node or particle
 * that was mapped to a different nearest neighbour position when the tree
 * walk would be further refined.
 *
 * @param target The index of the particle for which the Ewald correction is computed.
 * @param mode The mode of the Ewald correction.
 * @param exportflag The export flag.
 * @param exportnodecount The export node count.
 * @param exportindex The export index.
 * Ifdef ACC
 * @param should_drift The drift flag.
 *
 * @return The number of interactions.
 */
#ifndef ACC_GRAVITY
int force_treeevaluate_ewald_correction(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex)
#else // ACC_GRAVITY
int force_treeevaluate_ewald_correction(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                        int should_drift)
#endif // ACC_GRAVITY
#endif // ACC_INCLUDE_GRAVITY
{
    struct NODE *nop = 0;

#ifdef ACC_INCLUDE_GRAVITY

    struct NODE *Nodes = Nodes_base - All.MaxPart;
#endif // ACC_INCLUDE_GRAVITY

    int no, listindex = 0;
    MyAtLeastDouble dx, dy, dz, mass, r2;
    int signx, signy, signz;
    int i, j, k, openflag;
#ifndef ACC_INCLUDE_GRAVITY
    int nexp, task;
#endif // ACC_INCLUDE_GRAVITY
    MyAtLeastDouble u, v, w;
    MyAtLeastDouble f1, f2, f3, f4, f5, f6, f7, f8;
    MyAtLeastDouble boxsize, boxhalf;
    MyAtLeastDouble pos_x, pos_y, pos_z, aold;
    // cache some global vars in local vars to help compiler with alias analysis
    int maxPart = All.MaxPart;
#ifndef ACC_INCLUDE_GRAVITY
    int bunchSize = All.BunchSize;
#endif // ACC_INCLUDE_GRAVITY
    int maxNodes = MaxNodes;
    double errTol2 = All.ErrTolTheta * All.ErrTolTheta; /* constant over the whole walk, avoid re-squaring per node */

    boxsize = All.BoxSize;
    boxhalf = 0.5 * All.BoxSize;

#ifdef ACC_INCLUDE_GRAVITY
    int target;
    if (mode == 1)
    {
        target = active_target;
    }
    if (mode == 2)
    {
        target = p_target;
    }
#endif // ACC_INCLUDE_GRAVITY

    MyLongDouble acc_x = 0;
    MyLongDouble acc_y = 0;
    MyLongDouble acc_z = 0;
    int cost = 0;

    if (mode == 0 || mode == 3 || mode == 2)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
        aold = All.ErrTolForceAcc * P[target].OldAcc;
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        aold = All.ErrTolForceAcc * GravDataGet[target].OldAcc;
    }

    if (mode == 0 || mode == 3 || mode == 2)
    {
        no = maxPart; /* root node */
    }
    else
    {
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {
            if (no < maxPart) /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */
#ifndef ACC_INCLUDE_GRAVITY

#ifdef ACC_GRAVITY
                if (should_drift)
#endif // ACC

                    if (P[no].Ti_current != All.Ti_Current)
                    {
#pragma omp critical(_partnodedrift_)
                        drift_particle(no, All.Ti_Current);
                    }

                if (mode == 3)
                {
                    no = Nextnode[no];
                    continue;
                }
#endif // ACC_INCLUDE_GRAVITY
                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                mass = P[no].Mass;
            }
            else /* we have an  internal node */
            {
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    if (mode == 0 || mode == 3)
                    {
#ifndef ACC_INCLUDE_GRAVITY
                        if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            exportflag[task] = target;
                            exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (exportnodecount[task] == NODELISTLENGTH)
                        {
#pragma omp atomic capture
                            nexp = Nexport++;
                            if (nexp >= bunchSize)
                            {
                                BufferFullFlag = 1;
                                Nexport = bunchSize;
                                return -1;
                            }
                            exportnodecount[task] = 0;
                            exportindex[task] = nexp;
                            DataIndexTable[nexp].Task = task;
                            DataIndexTable[nexp].Index = target;
                            DataIndexTable[nexp].IndexGet = nexp;
                        }

                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];

                        if (exportnodecount[task] < NODELISTLENGTH)
                            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
#endif // ACC_INCLUDE_GRAVITY
                    }

                    no = Nextnode[no - maxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    no = nop->u.d.nextnode;
                    continue;
                }
#ifndef ACC_INCLUDE_GRAVITY

                if (mode == 3 && !(nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)))
                {
                    no = nop->u.d.sibling;
                    continue;
                }
#endif // ACC_INCLUDE_GRAVITY

#ifndef ACC_INCLUDE_GRAVITY
#ifdef ACC_GRAVITY
                if (should_drift)
#endif // ACC

                    if (nop->Ti_current != All.Ti_Current)
                    {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                        force_drift_node(no, All.Ti_Current);
                    }
#endif // ACC_INCLUDE_GRAVITY
                mass = nop->u.d.mass;
                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
            }

            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);

            if (no < maxPart)
            {
                no = Nextnode[no];
            }
            else /* we have an  internal node. Need to check opening criterion */
            {
                openflag = 0;
                r2 = dx * dx + dy * dy + dz * dz;
                if (All.ErrTolTheta) /* check Barnes-Hut opening criterion */
                {
                    if (nop->len * nop->len > r2 * errTol2)
                    {
                        openflag = 1;
                    }
                }
                else /* check relative opening criterion */
                {
                    if (mass * nop->len * nop->len > r2 * r2 * aold)
                    {
                        openflag = 1;
                    }
                    else
                    {
                        if (abs(nop->center[0] - pos_x) < 0.60 * nop->len)
                        {
                            if (abs(nop->center[1] - pos_y) < 0.60 * nop->len)
                            {
                                if (abs(nop->center[2] - pos_z) < 0.60 * nop->len)
                                {
                                    openflag = 1;
                                }
                            }
                        }
                    }
                }

                if (openflag)
                {
                    /* now we check if we can avoid opening the cell */

                    u = nop->center[0] - pos_x;
                    if (u > boxhalf)
                    {
                        u -= boxsize;
                    }
                    if (u < -boxhalf)
                    {
                        u += boxsize;
                    }
                    if (abs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[1] - pos_y;
                    if (u > boxhalf)
                    {
                        u -= boxsize;
                    }
                    if (u < -boxhalf)
                    {
                        u += boxsize;
                    }
                    if (abs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[2] - pos_z;
                    if (u > boxhalf)
                    {
                        u -= boxsize;
                    }
                    if (u < -boxhalf)
                    {
                        u += boxsize;
                    }
                    if (abs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    /* if the cell is too large, we need to refine
                     * it further
                     */
                    if (nop->len > 0.20 * boxsize)
                    {
                        /* cell is too large */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }

                no = nop->u.d.sibling; /* ok, node can be used */
            }
#ifndef ACC_INCLUDE_GRAVITY
            if (mode == 3)
            {
                continue;
            }
#endif // ACC_INCLUDE_GRAVITY

            /* compute the Ewald correction force */

            if (dx < 0)
            {
                dx = -dx;
                signx = +1;
            }
            else
            {
                signx = -1;
            }
            if (dy < 0)
            {
                dy = -dy;
                signy = +1;
            }
            else
            {
                signy = -1;
            }
            if (dz < 0)
            {
                dz = -dz;
                signz = +1;
            }
            else
            {
                signz = -1;
            }
            u = dx * fac_intp;
            i = (int)u;
            if (i >= EN)
            {
                i = EN - 1;
            }
            u -= i;
            v = dy * fac_intp;
            j = (int)v;
            if (j >= EN)
            {
                j = EN - 1;
            }
            v -= j;
            w = dz * fac_intp;
            k = (int)w;
            if (k >= EN)
            {
                k = EN - 1;
            }
            w -= k;
            /* compute factors for trilinear interpolation */
            f1 = (1 - u) * (1 - v) * (1 - w);
            f2 = (1 - u) * (1 - v) * (w);
            f3 = (1 - u) * (v) * (1 - w);
            f4 = (1 - u) * (v) * (w);
            f5 = (u) * (1 - v) * (1 - w);
            f6 = (u) * (1 - v) * (w);
            f7 = (u) * (v) * (1 - w);
            f8 = (u) * (v) * (w);
            acc_x += mass * signx *
                     (fcorrx[i][j][k] * f1 + fcorrx[i][j][k + 1] * f2 + fcorrx[i][j + 1][k] * f3 +
                      fcorrx[i][j + 1][k + 1] * f4 + fcorrx[i + 1][j][k] * f5 + fcorrx[i + 1][j][k + 1] * f6 +
                      fcorrx[i + 1][j + 1][k] * f7 + fcorrx[i + 1][j + 1][k + 1] * f8);
            acc_y += mass * signy *
                     (fcorry[i][j][k] * f1 + fcorry[i][j][k + 1] * f2 + fcorry[i][j + 1][k] * f3 +
                      fcorry[i][j + 1][k + 1] * f4 + fcorry[i + 1][j][k] * f5 + fcorry[i + 1][j][k + 1] * f6 +
                      fcorry[i + 1][j + 1][k] * f7 + fcorry[i + 1][j + 1][k + 1] * f8);
            acc_z += mass * signz *
                     (fcorrz[i][j][k] * f1 + fcorrz[i][j][k + 1] * f2 + fcorrz[i][j + 1][k] * f3 +
                      fcorrz[i][j + 1][k + 1] * f4 + fcorrz[i + 1][j][k] * f5 + fcorrz[i + 1][j][k + 1] * f6 +
                      fcorrz[i + 1][j + 1][k] * f7 + fcorrz[i + 1][j + 1][k + 1] * f8);
            cost++;
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                    no = Nodes[no].u.d.nextnode; /* open it */
            }
        }
    }
#ifndef ACC_INCLUDE_GRAVITY
    if (mode == 3)
    {
        return 0;
    }
#endif // ACC_INCLUDE_GRAVITY
#ifdef ACC_INCLUDE_GRAVITY
    target = active_target;
#endif // ACC_INCLUDE_GRAVITY

    /* add the result at the proper place */
#if !defined(ACC_INCLUDE_GRAVITY)
    if (mode == 0)
    {
        P[target].GravAccel[0] += acc_x;
        P[target].GravAccel[1] += acc_y;
        P[target].GravAccel[2] += acc_z;
    }
    else
#endif
    {
        GravDataResult[target].Acc[0] = acc_x;
        GravDataResult[target].Acc[1] = acc_y;
        GravDataResult[target].Acc[2] = acc_z;
    }

    return cost;
}
#ifdef ACC_INCLUDE_GRAVITY
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_GRAVITY
#endif // PERIODIC

#ifndef ACC_INCLUDE_GRAVITY // again, worst thing ever, we want to include two times this file :/
#if !defined(PMGRID) && !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL))
/**
 * @brief Compute the gravitational potential from the tree at a given particle position.
 *
 * This routine computes the gravitational potential by walking the
 * tree. The same opening criteria is used as for the gravitational force
 * walk.
 *
 * @param target The index of the particle for which the potential is computed.
 * @param mode The mode of the potential computation.
 * @param next The next particle.
 * @param nsend_local The number of particles to be sent to other processors.
 *
 * @return 0 if successful, -1 if buffer is too small to process even a single particle.
 */
int force_treeevaluate_potential(int target, int mode, int *nexport, int *nsend_local)
{
    struct NODE *nop = 0;
    MyLongDouble pot = 0;
    int no, ptype, task, nexport_save, listindex = 0;
    MyAtLeastDouble r2, dx, dy, dz, mass, r, u, h, h_inv, wp;
    MyAtLeastDouble pos_x, pos_y, pos_z, aold;
    // cache some global vars in local vars to help compiler with alias analysis
    int maxPart = All.MaxPart;
    int bunchSize = All.BunchSize;
    int maxNodes = MaxNodes;
    double errTol2 = All.ErrTolTheta * All.ErrTolTheta; /* constant over the whole walk, avoid re-squaring per node */

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyAtLeastDouble soft = 0;
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#ifdef ADAPTGRAVSOFT
    MyAtLeastDouble h_p, h_p_inv, u_p, wp_p, h_max;
    int particle;
#endif // ADAPTGRAVSOFT

    nexport_save = *nexport;
    if (mode == 0)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
        ptype = P[target].Type;
        aold = All.ErrTolForceAcc * P[target].OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = P[target].Hsml;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
        h = P[target].AGS_Hsml;
#endif // ADAPTGRAVSOFT
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        ptype = GravDataGet[target].Type;
        aold = All.ErrTolForceAcc * GravDataGet[target].OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = GravDataGet[target].Soft;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
        h = GravDataGet[target].AGS_Hsml;
#endif // ADAPTGRAVSOFT
    }

    if (mode == 0)
    {
        no = maxPart; /* root node */
    }
    else
    {
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {
            if (no < maxPart) /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */

                if (P[no].Ti_current != All.Ti_Current)
                {
                    drift_particle(no, All.Ti_Current);
                }
                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                mass = P[no].Mass;
            }
            else
            {
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    if (mode == 0)
                    {
                        if (Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            Exportflag[task] = target;
                            Exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (Exportnodecount[task] == NODELISTLENGTH)
                        {
                            if (*nexport >= bunchSize)
                            {
                                *nexport = nexport_save;

                                ENDRUN_IF(nexport_save == 0,
                                          "Error due to too small buffer to even process a single particle.");

                                for (int task = 0; task < NTask; task++)
                                {
                                    nsend_local[task] = 0;
                                }
                                for (int no = 0; no < nexport_save; no++)
                                {
                                    nsend_local[DataIndexTable[no].Task]++;
                                }
                                return -1;
                            }
                            Exportnodecount[task] = 0;
                            Exportindex[task] = *nexport;
                            DataIndexTable[*nexport].Task = task;
                            DataIndexTable[*nexport].Index = target;
                            DataIndexTable[*nexport].IndexGet = *nexport;
                            *nexport = *nexport + 1;
                            nsend_local[task]++;
                        }

                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];
                        if (Exportnodecount[task] < NODELISTLENGTH)
                        {
                            DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                        }
                    }
                    no = Nextnode[no - MaxNodes];
                    continue;
                }

                nop = &Nodes[no];
                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    no = nop->u.d.nextnode;
                    continue;
                }

                if (nop->Ti_current != All.Ti_Current)
                {
                    force_drift_node(no, All.Ti_Current);
                }
                mass = nop->u.d.mass;
                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
            }

#ifdef PERIODIC
            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);
#endif // PERIODIC
            r2 = dx * dx + dy * dy + dz * dz;
            if (no < maxPart)
            {
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }
                if (P[no].Type == 0)
                {
                    if (h < P[no].Hsml)
                        h = P[no].Hsml;
                }
                else
                {
                    if (h < All.ForceSoftening[P[no].Type])
                        h = All.ForceSoftening[P[no].Type];
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if (h < All.ForceSoftening[P[no].Type])
                {
                    h = All.ForceSoftening[P[no].Type];
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
                /* need to keep track of the softening (h_p) of the particle */
                h_p = P[no].AGS_Hsml;
                particle = no;
#endif // ADAPTGRAVSOFT
                no = Nextnode[no];
            }
            else /* we have an internal node. Need to check opening criterion */
            {
                if (All.ErrTolTheta) /* check Barnes-Hut opening criterion */
                {
                    if (nop->len * nop->len > r2 * errTol2)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                else /* check relative opening criterion */
                {
                    if (mass * nop->len * nop->len > r2 * r2 * aold)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    if (abs(nop->center[0] - pos_x) < 0.60 * nop->len)
                    {
                        if (abs(nop->center[1] - pos_y) < 0.60 * nop->len)
                        {
                            if (abs(nop->center[2] - pos_z) < 0.60 * nop->len)
                            {
                                no = nop->u.d.nextnode;
                                continue;
                            }
                        }
                    }
                }
#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if (h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
                {
                    h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
                    if (r2 < h * h)
                    {
                        if (maskout_different_softening_flag(
                                nop->u.d.bitflags)) /* bit-5 signals that there are particles of different softening in
                                                       the node */
                        {
                            no = nop->u.d.nextnode;
                            continue;
                        }
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }
                if (h < nop->maxsoft)
                {
                    h = nop->maxsoft;
                    if (r2 < h * h)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT

                h_p = nop->maxsoft; /* softening of the node */
                h_max = h >= h_p ? h : h_p;

                if (r2 < h_max * h_max)
                {
                    no = nop->u.d.nextnode;
                    continue; /* discard cases where the particle-node separation is less than the bigger
                             between the target softening and the node softening
                           */
                }

                particle = no;
#endif // ADAPTGRAVSOFT
                no = nop->u.d.sibling; /* node can be used */
            }

            r = sqrt(r2);

#ifdef ADAPTGRAVSOFT
            h_max = h >= h_p ? h : h_p;
            if (r >= h_max)
            {
                pot += -mass / r;
            }
#else // ADAPTGRAVSOFT
            if (r >= h)
            {
                pot += -mass / r;
            }
#endif // ADAPTGRAVSOFT
            else
            {
                h_inv = 1.0 / h;
#ifdef ADAPTGRAVSOFT

                h_inv = 1.0 / h;
                u = r * h_inv;

                if (u > 1.)
                {
                    wp = -1. / u;
                }
                else if (u < 0.5)
                {
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                }
                else
                {
                    wp = -3.2 + 0.066666666667 / u +
                         u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                }

                h_p_inv = 1. / h_p;
                u_p = r * h_p_inv;

                if (u_p > 1.)
                {
                    wp_p = -1. / u_p;
                }
                else if (u_p < 0.5)
                {
                    wp_p = -2.8 + u_p * u_p * (5.333333333333 + u_p * u_p * (6.4 * u_p - 9.6));
                }
                else
                {
                    wp_p = -3.2 + 0.066666666667 / u_p +
                           u_p * u_p * (10.666666666667 + u_p * (-16.0 + u_p * (9.6 - 2.133333333333 * u_p)));
                }

                pot += (mass * h_inv * wp + mass * h_p_inv * wp_p) / 2.;
#else // ADAPTGRAVSOFT
                u = r * h_inv;
                if (u < 0.5)
                {
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                }
                else
                {
                    wp = -3.2 + 0.066666666667 / u +
                         u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                }
                pot += mass * h_inv * wp;
#endif // ADAPTGRAVSOFT
            }
#ifdef PERIODIC
            pot += mass * ewald_pot_corr(dx, dy, dz);
#endif // defined(PERIODIC)
        }
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                {
                    no = Nodes[no].u.d.nextnode; /* open it */
                }
            }
        }
    }

    /* store result at the proper place */

    if (mode == 0)
    {
        P[target].Potential = pot;
    }
    else
    {
        PotDataResult[target].Potential = pot;
    }
    return 0;
}

#endif // !defined(PMGRID) && !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL))

#ifdef SUBFIND
/**
 * @brief Evaluate gravitational potential at a given target particle position for subfind.
 *
 * This function computes the gravitational potential at the position of a target particle
 * for subfind algorithm by traversing the gravitational tree structure. It iterates through
 * nodes and particles, considering their masses and distances to the target particle.
 *
 * @param target Index of the target particle.
 * @param mode   Mode of operation: 0 for regular mode, 1 for buffer mode.
 * @param nexport Pointer to the number of exported particles.
 * @param nsend_local Pointer to an array storing the number of particles sent to each task.
 * @return 0 if successful, -1 if buffer is too small to process even a single particle.
 */
int subfind_force_treeevaluate_potential(int target, int mode, int *nexport, int *nsend_local)
{
    struct NODE *nop = 0;
    MyLongDouble pot;
    int no, ptype, task, nexport_save, listindex = 0;
    MyAtLeastDouble r2, dx, dy, dz, mass, r, u, h, h_inv, wp;
    MyAtLeastDouble pos_x, pos_y, pos_z;
#ifdef ADAPTGRAVSOFT
    MyAtLeastDouble h_p, h_p_inv, u_p, wp_p, h_max;
    int particle;
#endif // ADAPTGRAVSOFT

    nexport_save = *nexport;
    pot = 0;
    if (mode == 0)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
        ptype = P[target].Type;
#ifdef ADAPTGRAVSOFT
        h = P[target].AGS_Hsml;
#endif // ADAPTGRAVSOFT
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        ptype = GravDataGet[target].Type;
#ifdef ADAPTGRAVSOFT
        h = GravDataGet[target].AGS_Hsml;
#endif // ADAPTGRAVSOFT
    }

#ifndef ADAPTGRAVSOFT
    h = All.ForceSoftening[ptype];
    h_inv = 1.0 / h;
#endif // ADAPTGRAVSOFT

    if (mode == 0)
    {
        no = All.MaxPart; /* root node */
    }
    else
    {
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {
            if (no < All.MaxPart) /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */
                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                mass = P[no].Mass;
            }
            else
            {
                if (no >= All.MaxPart + MaxNodes) /* pseudo particle */
                {
                    if (mode == 0)
                    {
                        if (Exportflag[task = DomainTask[no - (All.MaxPart + MaxNodes)]] != target)
                        {
                            Exportflag[task] = target;
                            Exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (Exportnodecount[task] == NODELISTLENGTH)
                        {
                            if (*nexport >= All.BunchSize)
                            {
                                *nexport = nexport_save;

                                ENDRUN_IF(nexport_save == 0,
                                          "The buffer is too small to process even a single particle");

                                for (task = 0; task < NTask; task++)
                                {
                                    nsend_local[task] = 0;
                                }
                                for (no = 0; no < nexport_save; no++)
                                {
                                    nsend_local[DataIndexTable[no].Task]++;
                                }
                                return -1;
                            }
                            Exportnodecount[task] = 0;
                            Exportindex[task] = *nexport;
                            DataIndexTable[*nexport].Task = task;
                            DataIndexTable[*nexport].Index = target;
                            DataIndexTable[*nexport].IndexGet = *nexport;
                            *nexport = *nexport + 1;
                            nsend_local[task]++;
                        }

                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                            DomainNodeIndex[no - (All.MaxPart + MaxNodes)];
                        if (Exportnodecount[task] < NODELISTLENGTH)
                        {
                            DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                        }
                    }
                    no = Nextnode[no - MaxNodes];
                    continue;
                }

                nop = &Nodes[no];
                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                mass = nop->u.d.mass;
                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    if (mass)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }

                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
            }

#ifdef PERIODIC
            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);
#endif // PERIODIC
            r2 = dx * dx + dy * dy + dz * dz;
            if (no < All.MaxPart)
            {
#ifdef ADAPTGRAVSOFT
                /* need to keep track of the softening (h_p) of the particle */
                h_p = P[no].AGS_Hsml;
                particle = no;
#endif // ADAPTGRAVSOFT
                no = Nextnode[no];
            }
            else /* we have an internal node. Need to check opening criterion */
            {
                /* check Barnes-Hut opening criterion */

                if (nop->len * nop->len > r2 * All.ErrTolThetaSubfind * All.ErrTolThetaSubfind)
                {
                    /* open cell */
                    if (mass)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
#ifdef ADAPTGRAVSOFT
                h_p = nop->maxsoft; /* softening of the node */
                h_max = h >= h_p ? h : h_p;

                if (r2 < h_max * h_max)
                {
                    no = nop->u.d.nextnode;
                    continue; /* discard cases where the particle-node separation is less than the bigger
                             between the target softening and the node softening
                           */
                }

                particle = no;
#endif // ADAPTGRAVSOFT
                no = nop->u.d.sibling; /* node can be used */
            }

            r = sqrt(r2);
#ifdef ADAPTGRAVSOFT
            h_max = h >= h_p ? h : h_p;
            if (r >= h_max)
            {
                pot += -mass / r;
            }
#else // ADAPTGRAVSOFT
            if (r >= h)
            {
                pot += -mass / r;
            }
#endif // ADAPTGRAVSOFT
            else
            {
#ifdef ADAPTGRAVSOFT
                h_inv = 1.0 / h;
                u = r * h_inv;

                if (u > 1.)
                {
                    wp = -1. / u;
                }
                else if (u < 0.5)
                {
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                }
                else
                {
                    wp = -3.2 + 0.066666666667 / u +
                         u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                }

                h_p_inv = 1. / h_p;
                u_p = r * h_p_inv;

                if (u_p > 1.)
                {
                    wp_p = -1. / u_p;
                }
                else if (u_p < 0.5)
                {
                    wp_p = -2.8 + u_p * u_p * (5.333333333333 + u_p * u_p * (6.4 * u_p - 9.6));
                }
                else
                {
                    wp_p = -3.2 + 0.066666666667 / u_p +
                           u_p * u_p * (10.666666666667 + u_p * (-16.0 + u_p * (9.6 - 2.133333333333 * u_p)));
                }

                pot += (mass * h_inv * wp + mass * h_p_inv * wp_p) / 2.;
#else // ADAPTGRAVSOFT
                u = r * h_inv;
                if (u < 0.5)
                {
                    wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                }
                else
                {
                    wp = -3.2 + 0.066666666667 / u +
                         u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                }
                pot += mass * h_inv * wp;
#endif // ADAPTGRAVSOFT
            }
        }
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                {
                    no = Nodes[no].u.d.nextnode; /* open it */
                }
            }
        }
    }

    /* store result at the proper place */

    if (mode == 0)
    {
        P[target].u.DM_Potential = pot;
    }
    else
    {
        PotDataResult[target].Potential = pot;
    }
    return 0;
}
#endif // SUBFIND

#if defined(PMGRID) && !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL))
/**
 * @brief Evaluate the short-range gravitational potential for a target particle.
 *
 * This function calculates the short-range gravitational potential at the position of a target particle.
 * It traverses the gravitational tree structure to compute the potential contribution from nearby particles
 * and nodes within the short-range cutoff distance. This potential is the Newtonian potential, modified
 * by a complementary error function.
 *
 * @param target The index of the target particle for which the potential is being evaluated.
 * @param mode The mode of operation: 0 for regular mode, 1 for buffer mode.
 * @param nexport Pointer to the number of particles being exported.
 * @param nsend_local Pointer to the number of particles being sent to local tasks.
 * @return Returns 0 if the potential evaluation is successful, -1 if the buffer is too small to process even a single
 * particle.
 */
int force_treeevaluate_potential_shortrange(int target, int mode, int *nexport, int *nsend_local)
{
    struct NODE *nop = 0;
    MyLongDouble pot;
    int no, ptype, tabindex, task, nexport_save, listindex = 0;
    MyAtLeastDouble r2, dx, dy, dz, mass, r, u, h, h_inv, wp;
    MyAtLeastDouble pos_x, pos_y, pos_z, aold;
    MyAtLeastDouble eff_dist, fac, rcut, asmth, asmthfac;
    MyAtLeastDouble dxx, dyy, dzz;
    // cache some global vars in local vars to help compiler with alias analysis
    int maxPart = All.MaxPart;
    int bunchSize = All.BunchSize;
    int maxNodes = MaxNodes;
    double errTol2 = All.ErrTolTheta * All.ErrTolTheta; /* constant over the whole walk, avoid re-squaring per node */

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyAtLeastDouble soft = 0;
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
    MyAtLeastDouble h_p, h_p_inv, u_p, wp_p, h_max;
    int particle;
#endif // ADAPTGRAVSOFT

    nexport_save = *nexport;
    pot = 0;

    rcut = All.Rcut[0];
    asmth = All.Asmth[0];

    if (mode == 0)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
        ptype = P[target].Type;
        aold = All.ErrTolForceAcc * P[target].OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = P[target].Hsml;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef PLACEHIGHRESREGION
        if (pmforce_is_particle_high_res(ptype, P[target].Pos))
        {
            rcut = All.Rcut[1];
            asmth = All.Asmth[1];
        }
#endif // PLACEHIGHRESREGION
#ifdef ADAPTGRAVSOFT
        h = P[target].AGS_Hsml;
#endif // ADAPTGRAVSOFT
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        ptype = GravDataGet[target].Type;
        aold = All.ErrTolForceAcc * GravDataGet[target].OldAcc;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = GravDataGet[target].Soft;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef PLACEHIGHRESREGION
        if (pmforce_is_particle_high_res(ptype, GravDataGet[target].Pos))
        {
            rcut = All.Rcut[1];
            asmth = All.Asmth[1];
        }
#endif // PLACEHIGHRESREGION
#ifdef ADAPTGRAVSOFT
        h = GravDataGet[target].AGS_Hsml;
#endif // ADAPTGRAVSOFT
    }

    asmthfac = 0.5 / asmth * (NTAB / 3.0);
    if (mode == 0)
    {
        no = maxPart; /* root node */
    }
    else
    {
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {
            if (no < maxPart) /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign  */
#ifndef SUBFIND_RESHUFFLE_AND_POTENTIAL
                if (P[no].Ti_current != All.Ti_Current)
                {
                    drift_particle(no, All.Ti_Current);
                }
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                mass = P[no].Mass;
            }
            else
            {
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    if (mode == 0)
                    {
                        if (Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            Exportflag[task] = target;
                            Exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (Exportnodecount[task] == NODELISTLENGTH)
                        {
                            if (*nexport >= bunchSize)
                            {
                                *nexport = nexport_save;

                                ENDRUN_IF(nexport_save == 0,
                                          "Error due to too small buffer to even process a single particle.");

                                for (int task = 0; task < NTask; task++)
                                {
                                    nsend_local[task] = 0;
                                }
                                for (int no = 0; no < nexport_save; no++)
                                {
                                    nsend_local[DataIndexTable[no].Task]++;
                                }
                                return -1;
                            }
                            Exportnodecount[task] = 0;
                            Exportindex[task] = *nexport;
                            DataIndexTable[*nexport].Task = task;
                            DataIndexTable[*nexport].Index = target;
                            DataIndexTable[*nexport].IndexGet = *nexport;
                            *nexport = *nexport + 1;
                            nsend_local[task]++;
                        }

                        DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];
                        if (Exportnodecount[task] < NODELISTLENGTH)
                            DataNodeList[Exportindex[task]].NodeList[Exportnodecount[task]] = -1;
                    }
                    no = Nextnode[no - maxNodes];
                    continue;
                }

                nop = &Nodes[no];
                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    no = nop->u.d.nextnode;
                    continue;
                }
#ifndef SUBFIND_RESHUFFLE_AND_POTENTIAL
                if (nop->Ti_current != All.Ti_Current)
                {
                    force_drift_node(no, All.Ti_Current);
                }
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
                mass = nop->u.d.mass;
                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
            }

#if defined(PERIODIC)
            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);
#endif // defined(PERIODIC)
            r2 = dx * dx + dy * dy + dz * dz;
            if (no < maxPart)
            {
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }
                if (P[no].Type == 0)
                {
                    if (h < P[no].Hsml)
                    {
                        h = P[no].Hsml;
                    }
                }
                else
                {
                    if (h < All.ForceSoftening[P[no].Type])
                    {
                        h = All.ForceSoftening[P[no].Type];
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if (h < All.ForceSoftening[P[no].Type])
                {
                    h = All.ForceSoftening[P[no].Type];
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
                /* need to keep track of the softening (h_p) of the particle */
                h_p = P[no].AGS_Hsml;
                particle = no;
#endif // ADAPTGRAVSOFT
                no = Nextnode[no];
            }
            else /* we have an  internal node. Need to check opening criterion */
            {
                /* check whether we can stop walking along this branch */
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    if (mode == 0)
                    {
                        if (Exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            Exportflag[task] = target;
                            DataIndexTable[*nexport].Index = target;
                            DataIndexTable[*nexport].Task = task; /* Destination task */
                            *nexport = *nexport + 1;
                            nsend_local[task]++;
                        }
                    }
                    no = Nextnode[no - maxNodes];
                    continue;
                }

                eff_dist = rcut + 0.5 * nop->len;
                dxx = nop->center[0] - pos_x; /* observe the sign ! */
                dyy = nop->center[1] - pos_y; /* this vector is -y in my thesis notation */
                dzz = nop->center[2] - pos_z;
#ifdef PERIODIC
                dxx = NEAREST(dxx);
                dyy = NEAREST(dyy);
                dzz = NEAREST(dzz);
#endif // PERIODIC
                if (dxx < -eff_dist || dxx > eff_dist)
                {
                    no = nop->u.d.sibling;
                    continue;
                }

                if (dyy < -eff_dist || dyy > eff_dist)
                {
                    no = nop->u.d.sibling;
                    continue;
                }

                if (dzz < -eff_dist || dzz > eff_dist)
                {
                    no = nop->u.d.sibling;
                    continue;
                }

                if (All.ErrTolTheta) /* check Barnes-Hut opening criterion */
                {
                    if (nop->len * nop->len > r2 * errTol2)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                else /* check relative opening criterion */
                {
                    if (mass * nop->len * nop->len > r2 * r2 * aold)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    if (abs(dxx) < 0.60 * nop->len)
                    {
                        if (abs(dyy) < 0.60 * nop->len)
                        {
                            if (abs(dzz) < 0.60 * nop->len)
                            {
                                no = nop->u.d.nextnode;
                                continue;
                            }
                        }
                    }
                }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if (h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
                {
                    h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
                    if (r2 < h * h)
                    {
                        /* bit-5 signals that there are particles of
                         * different softening in the node
                         */
                        if (maskout_different_softening_flag(nop->u.d.bitflags))
                        {
                            no = nop->u.d.nextnode;
                            continue;
                        }
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }
                if (h < nop->maxsoft)
                {
                    h = nop->maxsoft;
                    if (r2 < h * h)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT

                h_p = nop->maxsoft; /* softening of the node */
                h_max = h >= h_p ? h : h_p;

                if (r2 < h_max * h_max)
                {
                    no = nop->u.d.nextnode;
                    continue; /* discard cases where the particle-node separation is less than the bigger
                             between the target softening and the node softening
                           */
                }

                particle = no;
#endif // ADAPTGRAVSOFT
                no = nop->u.d.sibling; /* node can be used */
            }

            r = sqrt(r2);
            tabindex = (int)(r * asmthfac);
            if (tabindex < NTAB)
            {
                fac = shortrange_table_potential[tabindex];
#ifdef ADAPTGRAVSOFT
                h_max = h >= h_p ? h : h_p;
                if (r >= h_max)
                {
                    pot += -fac * mass / r;
                }
#else // ADAPTGRAVSOFT
                if (r >= h)
                {
                    pot += -fac * mass / r;
                }
#endif // ADAPTGRAVSOFT
                else
                {
                    h_inv = 1.0 / h;
#ifdef ADAPTGRAVSOFT
                    // h_inv = 1.0 / h;
                    u = r * h_inv;

                    if (u > 1.)
                    {
                        wp = -1. / u;
                    }
                    else if (u < 0.5)
                    {
                        wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                    }
                    else
                    {
                        wp = -3.2 + 0.066666666667 / u +
                             u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                    }

                    h_p_inv = 1.0 / h_p;
                    u_p = r * h_p_inv;

                    if (u_p > 1.)
                    {
                        wp_p = -1. / u_p;
                    }
                    else if (u_p < 0.5)
                    {
                        wp_p = -2.8 + u_p * u_p * (5.333333333333 + u_p * u_p * (6.4 * u_p - 9.6));
                    }
                    else
                    {
                        wp_p = -3.2 + 0.066666666667 / u_p +
                               u_p * u_p * (10.666666666667 + u_p * (-16.0 + u_p * (9.6 - 2.133333333333 * u_p)));
                    }

                    pot += fac * (mass * h_inv * wp + mass * h_p_inv * wp_p) / 2.;

#else // ADAPTGRAVSOFT
                    u = r * h_inv;
                    if (u < 0.5)
                    {
                        wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6));
                    }
                    else
                    {
                        wp = -3.2 + 0.066666666667 / u +
                             u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)));
                    }
                    pot += fac * mass * h_inv * wp;
#endif // ADAPTGRAVSOFT
                }
            }
        }
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                    no = Nodes[no].u.d.nextnode; /* open it */
            }
        }
    }

    /* store result at the proper place */
    if (mode == 0)
    {
        P[target].Potential = pot;
    }
    else
    {
        PotDataResult[target].Potential = pot;
    }
    return 0;
}
#endif // defined(PMGRID) && !defined(EVALPOTENTIAL) && (defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL))

/**
 * @brief Allocate memory for the tree structure.
 *
 * This function allocates the memory used for storage of the tree and of
 * auxiliary arrays needed for tree-walk and link-lists.  Usually,
 * maxnodes approximately equal to 0.7*maxpart is sufficient to store the
 * tree for up to maxpart particles.
 *
 * @param maxnodes Maximum number of nodes in the tree.
 * @param maxpart Maximum number of particles.
 */
void force_treeallocate(int maxnodes, int maxpart)
{
    size_t bytes;
    MyAtLeastDouble allbytes = 0, allbytes_topleaves = 0;

    tree_allocated_flag = 1;
    DomainNodeIndex = (int *)mymalloc("DomainNodeIndex", bytes = NTopleaves * sizeof(int));
    allbytes_topleaves += bytes;
    MaxNodes = maxnodes;
    if (!(Nodes_base = (struct NODE *)mymalloc("Nodes_base", bytes = (MaxNodes + 1) * sizeof(struct NODE))))
    {
        ENDRUN("Failed to allocate memory for %d tree-nodes (%g MB).", MaxNodes, bytes / (1024.0 * 1024.0));
    }
    allbytes += bytes;
    if (!(Extnodes_base = (struct extNODE *)mymalloc("Extnodes_base", bytes = (MaxNodes + 1) * sizeof(struct extNODE))))
    {
        ENDRUN("Failed to allocate memory for %d tree-extnodes (%g MB).", MaxNodes, bytes / (1024.0 * 1024.0));
    }
    allbytes += bytes;
    Nodes = Nodes_base - All.MaxPart;
    Extnodes = Extnodes_base - All.MaxPart;
    if (!(Nextnode = (int *)mymalloc("Nextnode", bytes = (maxpart + NTopnodes) * sizeof(int))))
    {
        VERBOSE_ALL(0, "Failed to allocate %d spaces for 'Nextnode' array (%g MB)", maxpart + NTopnodes,
                    bytes / (1024.0 * 1024.0));
        exit(0);
    }

    allbytes += bytes;
#ifdef GROUP_LEAVES
    NextGroupedNode = (int *)mymalloc("NextGroupedNode", bytes = (maxpart) * sizeof(int));
#endif // GROUP_LEAVES
    allbytes += bytes;
    if (!(Father = (int *)mymalloc("Father", bytes = (maxpart) * sizeof(int))))
    {
        VERBOSE_ALL(0, "Failed to allocate %d spaces for 'Father' array (%g MB)", maxpart, bytes / (1024.0 * 1024.0));
        exit(0);
    }
    allbytes += bytes;
    if (first_flag == 0)
    {
        first_flag = 1;
        VERBOSE(2, "\nAllocated %g MByte for BH-tree, and %g Mbyte for top-leaves.  (presently allocted %g MB)\n",
                allbytes / (1024.0 * 1024.0), allbytes_topleaves / (1024.0 * 1024.0),
                AllocatedBytes / (1024.0 * 1024.0));
        for (int i = 0; i < NTAB; i++)
        {
            MyAtLeastDouble u = 3.0 / NTAB * (i + 0.5);
            shortrange_table[i] = erfc(u) + 2.0 * u / sqrt(M_PI) * exp(-u * u);
            shortrange_table_potential[i] = erfc(u);
        }
    }
#ifdef ACC
    acc_all.len_nextnode = (maxpart + NTopnodes);
    acc_all.len_nodes_base = (MaxNodes + 1);
    acc_all.no_tree_flag = 1;
#endif // ACC
}

/**
 * @brief Free the memory allocated for the tree.
 *
 * This function frees the memory allocated for the tree, i.e. it frees
 * the space allocated by the function force_treeallocate().
 */
void force_treefree(void)
{
    if (tree_allocated_flag)
    {
        myfree(Father);

#ifdef GROUP_LEAVES
        myfree(NextGroupedNode);
#endif // GROUP_LEAVES
        myfree(Nextnode);
        myfree(Extnodes_base);
        myfree(Nodes_base);
        myfree(DomainNodeIndex);
        tree_allocated_flag = 0;
    }
}

/**
 * @brief Evaluate the gravitational force for a target particle.
 *
 * This function does the force computation with direct summation for the
 * specified particle in the communication buffer. This can be useful for
 * debugging purposes, in particular for explicit checks of the force
 * accuracy.
 *
 * @param target The index of the target particle for which the force is being evaluated.
 * @param mode The mode of operation: 0 for regular mode, 1 for buffer mode.
 */
#ifdef FORCETEST
int force_treeevaluate_direct(int target, int mode)
{
    int ptype;
    MyAtLeastDouble pos_x, pos_y, pos_z;

#ifdef PERIODIC
    MyAtLeastDouble fcorr[3];
    MyAtLeastDouble boxsize, boxhalf;
    boxsize = All.BoxSize;
    boxhalf = 0.5 * All.BoxSize;
#endif // PERIODIC
    MyLongDouble acc_x = 0;
    MyLongDouble acc_y = 0;
    MyLongDouble acc_z = 0;
    if (mode == 0)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
        ptype = P[target].Type;
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        ptype = GravDataGet[target].Type;
    }

    for (int i = 0; i < NumPart; i++)
    {
        MyAtLeastDouble fac;
        MyAtLeastDouble epsilon = std::max(All.ForceSoftening[P[i].Type], All.ForceSoftening[ptype]);
        MyAtLeastDouble h = epsilon;
        MyAtLeastDouble h_inv = 1 / h;
        MyAtLeastDouble dx = P[i].Pos[0] - pos_x;
        MyAtLeastDouble dy = P[i].Pos[1] - pos_y;
        MyAtLeastDouble dz = P[i].Pos[2] - pos_z;
#ifdef PERIODIC
        while (dx > boxhalf)
        {
            dx -= boxsize;
        }
        while (dy > boxhalf)
        {
            dy -= boxsize;
        }
        while (dz > boxhalf)
        {
            dz -= boxsize;
        }
        while (dx < -boxhalf)
        {
            dx += boxsize;
        }
        while (dy < -boxhalf)
        {
            dy += boxsize;
        }
        while (dz < -boxhalf)
        {
            dz += boxsize;
        }
#endif // PERIODIC
        MyAtLeastDouble r2 = dx * dx + dy * dy + dz * dz;
        MyAtLeastDouble r = sqrt(r2);
        MyAtLeastDouble u = r * h_inv;
        if (u >= 1)
        {
            MyAtLeastDouble r_inv = 1 / r;
            fac = P[i].Mass * r_inv * r_inv * r_inv;
        }
        else
        {
            if (u < 0.5)
            {
                fac = P[i].Mass * h_inv * h_inv * h_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
            }
            else
            {
                fac = P[i].Mass * h_inv * h_inv * h_inv *
                      (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u -
                       0.066666666667 / (u * u * u));
            }
        }

        acc_x += dx * fac;
        acc_y += dy * fac;
        acc_z += dz * fac;
#ifdef PERIODIC
        if (u > 1.0e-5)
        {
            ewald_corr(dx, dy, dz, fcorr);
            acc_x += P[i].Mass * fcorr[0];
            acc_y += P[i].Mass * fcorr[1];
            acc_z += P[i].Mass * fcorr[2];
        }
#endif // PERIODIC
    }

    if (mode == 0)
    {
        P[target].GravAccelDirect[0] = acc_x;
        P[target].GravAccelDirect[1] = acc_y;
        P[target].GravAccelDirect[2] = acc_z;
    }
    else
    {
        GravDataResult[target].Acc[0] = acc_x;
        GravDataResult[target].Acc[1] = acc_y;
        GravDataResult[target].Acc[2] = acc_z;
    }

    return NumPart;
}
#endif // FORCETEST

#ifdef PHIDOT
/**
 * @brief Evaluate the gravitational potential time derivative at a target particle the tree.
 *
 * This function calculates the time derivative of the gravitational potential at a target particle
 * by traversing the gravitational tree structure. It considers interactions between particles
 * and nodes of the tree.
 *
 * @param target The index of the target particle.
 * @param mode   The mode of operation.
 *               - 0: Evaluation mode for target particles.
 *               - 1: Evaluation mode for particles in the given node list.
 *               - 2: Export mode for particles not in the given node list.
 *               - 3: Export mode for particles not in the given node list and node list is exhausted.
 * @param exportflag      Export flag indicating whether the target particle should be exported.
 * @param exportnodecount Export node count for exporting particles.
 * @param exportindex     Export index for exporting particles.
 * @return The number of interactions performed during the evaluation.
 */
int phidot_treeevaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex)
{
    struct NODE *nop = 0;
    int no, nexp, ptype, task, listindex = 0;
    MyAtLeastDouble r2, dx, dy, dz, vx, vy, vz, mass, r, fac, u, h, h_inv, h3_inv;
    MyAtLeastDouble pos_x, pos_y, pos_z;
    // cache some global vars in local vars to help compiler with alias analysis
    int maxPart = All.MaxPart;
    int bunchSize = All.BunchSize;
    int maxNodes = MaxNodes;
    double errTol2 = All.ErrTolTheta * All.ErrTolTheta; /* constant over the whole walk, avoid re-squaring per node */

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyAtLeastDouble soft = 0;
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#ifdef ADAPTGRAVSOFT
    MyAtLeastDouble h_p, h_p_inv, h_p3_inv, u_p, fac_p, h_max;
    MyAtLeastDouble zeta, omega, dWdr, dWdr_p, corr;
    MyAtLeastDouble mass_target;
    int particle;
#endif // ADAPTGRAVSOFT

    MyDouble phidot = 0;
    int ninteractions = 0;
    int nodesinlist = 0;

    if (mode == 0)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];

        ptype = P[target].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = P[target].Hsml;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
        zeta = P[target].AGS_zeta;
        omega = P[target].AGS_omega;
        h = P[target].AGS_Hsml;
        mass_target = P[target].Mass;
#endif // ADAPTGRAVSOFT
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
        ptype = GravDataGet[target].Type;
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
        if (ptype == 0)
        {
            soft = GravDataGet[target].Soft;
        }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
        zeta = GravDataGet[target].AGS_zeta;
        omega = GravDataGet[target].AGS_omega;
        h = GravDataGet[target].AGS_Hsml;
        mass_target = GravDataGet[target].Mass;
#endif // ADAPTGRAVSOFT
    }

    if (mode == 0 || mode == 3 || mode == 2)
    {
        no = maxPart; /* root node */
    }
    else
    {
        nodesinlist++;
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {
            if (no < maxPart) /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */

                if (P[no].Ti_current != All.Ti_Current)
                {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                    drift_particle(no, All.Ti_Current);
                }

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;

                vx = P[no].Vel[0];
                vy = P[no].Vel[1];
                vz = P[no].Vel[2];

                mass = P[no].Mass;
            }
            else
            {
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    if (mode == 0)
                    {
                        if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            exportflag[task] = target;
                            exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (exportnodecount[task] == NODELISTLENGTH)
                        {
#pragma omp atomic capture
                            nexp = Nexport++;
                            if (nexp >= bunchSize)
                            {
                                BufferFullFlag = 1;
                                Nexport = bunchSize;
                                return -1;
                            }
                            exportnodecount[task] = 0;
                            exportindex[task] = nexp;
                            DataIndexTable[nexp].Task = task;
                            DataIndexTable[nexp].Index = target;
                            DataIndexTable[nexp].IndexGet = nexp;
                        }

                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];

                        if (exportnodecount[task] < NODELISTLENGTH)
                        {
                            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                        }
                    }
                    no = Nextnode[no - maxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                mass = nop->u.d.mass;

                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    if (mass)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }

                if (nop->Ti_current != All.Ti_Current)
                {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                    force_drift_node(no, All.Ti_Current);
                }

                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;

                vx = Extnodes[no].vs[0];
                vy = Extnodes[no].vs[1];
                vz = Extnodes[no].vs[2];
            }

#if defined(PERIODIC)
            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);
#endif // defined(PERIODIC)
            r2 = dx * dx + dy * dy + dz * dz;

            if (no < maxPart)
            {
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }

                if (P[no].Type == 0)
                {
                    if (h < P[no].Hsml)
                    {
                        h = P[no].Hsml;
                    }
                }
                else
                {
                    if (h < All.ForceSoftening[P[no].Type])
                    {
                        h = All.ForceSoftening[P[no].Type];
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if (h < All.ForceSoftening[P[no].Type])
                {
                    h = All.ForceSoftening[P[no].Type];
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTGRAVSOFT
                /* need to keep track of the softening (h_p) of the particle */
                h_p = P[no].AGS_Hsml;
                particle = no;
#endif // ADAPTGRAVSOFT

                no = Nextnode[no];

#ifdef ADAPTGRAVSOFT
                /* otherwise r=0 and corr=nan. This check must be put after the next node has been selected! */
                if (target == particle)
                {
                    continue;
                }
#endif // ADAPTGRAVSOFT
            }
            else /* we have an  internal node. Need to check opening criterion */
            {

                if (All.ErrTolTheta) /* check Barnes-Hut opening criterion */
                {
                    if (nop->len * nop->len > r2 * errTol2)
                    {
                        /* open cell */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
                else /* check relative opening criterion */
                {
                    /* check in addition whether we lie inside the cell */
                    if (abs(nop->center[0] - pos_x) < 0.60 * nop->len)
                    {
                        if (abs(nop->center[1] - pos_y) < 0.60 * nop->len)
                        {
                            if (abs(nop->center[2] - pos_z) < 0.60 * nop->len)
                            {
                                no = nop->u.d.nextnode;
                                continue;
                            }
                        }
                    }
                }

#ifndef ADAPTIVE_GRAVSOFT_FORGAS
                h = All.ForceSoftening[ptype];
                if (h < All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)])
                {
                    h = All.ForceSoftening[extract_max_softening_type(nop->u.d.bitflags)];
                    if (r2 < h * h)
                    {
                        if (maskout_different_softening_flag(nop->u.d.bitflags)) /* signals that there are particles of
                                                                                    different softening in the node */
                        {
                            no = nop->u.d.nextnode;
                            continue;
                        }
                    }
                }
#else // ADAPTIVE_GRAVSOFT_FORGAS
                if (ptype == 0)
                {
                    h = soft;
                }
                else
                {
                    h = All.ForceSoftening[ptype];
                }

                if (h < nop->maxsoft)
                {
                    h = nop->maxsoft;
                    if (r2 < h * h)
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }
#endif // ADAPTIVE_GRAVSOFT_FORGAS

#ifdef ADAPTGRAVSOFT
                h_p = nop->maxsoft; /* softening of the node */
                h_max = h >= h_p ? h : h_p;

                if (r2 < h_max * h_max)
                {
                    no = nop->u.d.nextnode;
                    continue; /* discard cases where the particle-node separation is less than the bigger
                             between the target softening and the node softening
                           */
                }

                particle = no;
#endif // ADAPTGRAVSOFT

                no = nop->u.d.sibling; /* ok, node can be used */
            }

            r = sqrt(r2);

#ifdef ADAPTGRAVSOFT
            h_max = h >= h_p ? h : h_p;
            if (r >= h_max)
            {
                /* no need to worry about softening lengths: interaction is newtonian.
                 * All particle - node interactions fall here */
                fac = mass / (r2 * r);
                corr = 0;
            }
#else // ADAPTGRAVSOFT
            if (r >= h)
            {
                fac = mass / (r2 * r);
            }
#endif // ADAPTGRAVSOFT
            else
            {
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
#ifndef ADAPTGRAVSOFT
                u = r * h_inv;
                if (u < 0.5)
                {
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                }
                else
                {
                    fac = mass * h3_inv *
                          (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u -
                           0.066666666667 / (u * u * u));
                }

#else // ADAPTGRAVSOFT
                /* interaction is smoothed: only particle-particle, no nodes down here! */
                dWdr = dWdr_p = corr = 0;
                h_inv = 1.0 / h;
                h3_inv = h_inv * h_inv * h_inv;
                u = r * h_inv;

                if (u > 1.)
                {
                    fac = -mass / (r2 * r);
                }

                else if (u < 0.5)
                {
                    fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
                    dWdr = 6. * KERNEL_COEFF_1 * h3_inv * h_inv * (-2. * u + 3. * u * u);
                }
                else
                {
                    fac = mass * h3_inv *
                          (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u -
                           0.066666666667 / (u * u * u));
                    dWdr = -6. * KERNEL_COEFF_1 * h3_inv * h_inv * ((1. - u) * (1. - u));
                }

                h_p_inv = 1.0 / h_p;
                h_p3_inv = h_p_inv * h_p_inv * h_p_inv;
                u_p = r * h_p_inv;

                if (u_p > 1.)
                {
                    fac_p = mass / (r2 * r);
                }

                else if (u_p < 0.5)
                {
                    fac_p = mass * h_p3_inv * (10.666666666667 + u_p * u_p * (32.0 * u_p - 38.4));
                    dWdr_p = 6. * KERNEL_COEFF_1 * h_p3_inv * h_p_inv * (-2. * u_p + 3. * u_p * u_p);
                }
                else
                {
                    fac_p = mass * h_p3_inv *
                            (21.333333333333 - 48.0 * u_p + 38.4 * u_p * u_p - 10.666666666667 * u_p * u_p * u_p -
                             0.066666666667 / (u_p * u_p * u_p));
                    dWdr_p = -6. * KERNEL_COEFF_1 * h_p3_inv * h_p_inv * (1. - u_p) * (1. - u_p);
                }

                /* force = (fac + fac_p)/2 */
                fac += fac_p;
                fac /= 2.;

#ifndef AGS_NOCORRECTION
                /* correction term */
                corr = -(zeta * omega * dWdr +
                         mass / mass_target * P[particle].AGS_zeta * P[particle].AGS_omega * dWdr_p) /
                       r;

                corr /= 2.;

                /***************************************************************************************************/
                /* The original correction term is:                                                                */
                /*   corr = mass * (termI / termII * dWdr +  P[particle].termI / P[particle].termII * dWdr_p) / r; */
                /*   corr /= 2.;                                                                                   */
                /* But remember that termI and termII would have a different definition.                           */
                /***************************************************************************************************/
#endif // AGS_NOCORRECTION
                if (particle >= maxPart)
                {
                    VERBOSE_ALL(1, "*PROBLEM in ADAPTGRAVSOFT*: we are having a smoothed particle-node interaction!");
                    VERBOSE_ALL(1, "Particle, Node: %i, %i", target, particle);
                }
#endif // ADAPTGRAVSOFT
            }

            phidot += fac * (dx * vx + dy * vy + dz * vz);

#ifdef ADAPTGRAVSOFT
            phidot += corr * (dx * vx + dy * vy + dz * vz);
#endif // ADAPTGRAVSOFT

            if (mass > 0)
                ninteractions++;
        }
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                {
                    nodesinlist++;
                    no = Nodes[no].u.d.nextnode; /* open it */
                }
            }
        }
    }

    /* store result at the proper place */

    if (mode == 0)
    {
        SphP[target].PhiDot = phidot;
    }
    else
    {
        PhiDotDataResult[target].PhiDot = phidot;
        *exportflag = nodesinlist;
    }

#ifdef PERIODIC
    phidot_treeevaluate_ewald_correction(target, mode, exportflag, exportnodecount, exportindex);
#endif // PERIODIC

    return ninteractions;
}

#ifdef PERIODIC
/**
 * @brief Evaluate the Ewald correction for the time derivative of the gravitational potential at a target particle.
 *
 * This function computes the Ewald correction for the time derivative of the gravitational potential
 * at a target particle. It calculates the correction using trilinear interpolation based on particle
 * positions and stores the result in the appropriate place.
 *
 * @param target          The index of the target particle.
 * @param mode            The mode of operation:
 *                        - 0: Evaluation mode for target particles.
 *                        - 1: Evaluation mode for particles in the given node list.
 * @param exportflag      Export flag indicating whether the target particle should be exported.
 * @param exportnodecount Export node count for exporting particles.
 * @param exportindex     Export index for exporting particles.
 * @return The cost of computing the Ewald correction.
 */
int phidot_treeevaluate_ewald_correction(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex)
{
    struct NODE *nop = 0;
    int no, listindex = 0;
    MyAtLeastDouble dx, dy, dz, vx, vy, vz, mass, r2;
    int signx, signy, signz, nexp;
    int i, j, k, openflag, task;
    MyAtLeastDouble u, v, w;
    MyAtLeastDouble f1, f2, f3, f4, f5, f6, f7, f8;
    MyAtLeastDouble pos_x, pos_y, pos_z;
    // cache some global vars in local vars to help compiler with alias analysis
    int maxPart = All.MaxPart;
    int bunchSize = All.BunchSize;
    int maxNodes = MaxNodes;
    double errTol2 = All.ErrTolTheta * All.ErrTolTheta; /* constant over the whole walk, avoid re-squaring per node */

    MyAtLeastDouble boxsize = All.BoxSize;
    MyAtLeastDouble boxhalf = 0.5 * All.BoxSize;

    MyFloat phidot = 0;
    int cost = 0;

    if (mode == 0)
    {
        pos_x = P[target].Pos[0];
        pos_y = P[target].Pos[1];
        pos_z = P[target].Pos[2];
    }
    else
    {
        pos_x = GravDataGet[target].Pos[0];
        pos_y = GravDataGet[target].Pos[1];
        pos_z = GravDataGet[target].Pos[2];
    }

    if (mode == 0)
    {
        no = maxPart; /* root node */
    }
    else
    {
        no = GravDataGet[target].NodeList[0];
        no = Nodes[no].u.d.nextnode; /* open it */
    }

    while (no >= 0)
    {
        while (no >= 0)
        {
            if (no < maxPart) /* single particle */
            {
                /* the index of the node is the index of the particle */
                /* observe the sign */
                if (P[no].Ti_current != All.Ti_Current)
                {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                    drift_particle(no, All.Ti_Current);
                }

                dx = P[no].Pos[0] - pos_x;
                dy = P[no].Pos[1] - pos_y;
                dz = P[no].Pos[2] - pos_z;
                vx = P[no].Vel[0];
                vy = P[no].Vel[1];
                vz = P[no].Vel[2];
                mass = P[no].Mass;
            }
            else /* we have an  internal node */
            {
                if (no >= maxPart + maxNodes) /* pseudo particle */
                {
                    if (mode == 0)
                    {
                        if (exportflag[task = DomainTask[no - (maxPart + maxNodes)]] != target)
                        {
                            exportflag[task] = target;
                            exportnodecount[task] = NODELISTLENGTH;
                        }

                        if (exportnodecount[task] == NODELISTLENGTH)
                        {
#pragma omp atomic capture
                            nexp = Nexport++;
                            if (nexp >= bunchSize)
                            {
                                BufferFullFlag = 1;
                                Nexport = bunchSize;
                                return -1;
                            }
                            exportnodecount[task] = 0;
                            exportindex[task] = nexp;
                            DataIndexTable[nexp].Task = task;
                            DataIndexTable[nexp].Index = target;
                            DataIndexTable[nexp].IndexGet = nexp;
                        }

                        DataNodeList[exportindex[task]].NodeList[exportnodecount[task]++] =
                            DomainNodeIndex[no - (maxPart + maxNodes)];

                        if (exportnodecount[task] < NODELISTLENGTH)
                        {
                            DataNodeList[exportindex[task]].NodeList[exportnodecount[task]] = -1;
                        }
                    }
                    no = Nextnode[no - maxNodes];
                    continue;
                }

                nop = &Nodes[no];

                if (mode == 1)
                {
                    if (nop->u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node again, which means
                                                                        that we are done with the branch */
                    {
                        no = -1;
                        continue;
                    }
                }

                if (!(nop->u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)))
                {
                    /* open cell */
                    no = nop->u.d.nextnode;
                    continue;
                }

                if (nop->Ti_current != All.Ti_Current)
                {
#ifdef _OPENMP
#pragma omp critical(_partnodedrift_)
#endif // _OPENMP
                    force_drift_node(no, All.Ti_Current);
                }

                mass = nop->u.d.mass;
                dx = nop->u.d.s[0] - pos_x;
                dy = nop->u.d.s[1] - pos_y;
                dz = nop->u.d.s[2] - pos_z;
                vx = Extnodes[no].vs[0];
                vy = Extnodes[no].vs[1];
                vz = Extnodes[no].vs[2];
            }

            dx = NEAREST(dx);
            dy = NEAREST(dy);
            dz = NEAREST(dz);

            if (no < maxPart)
            {
                no = Nextnode[no];
            }
            else /* we have an  internal node. Need to check opening criterion */
            {
                openflag = 0;
                r2 = dx * dx + dy * dy + dz * dz;
                if (All.ErrTolTheta) /* check Barnes-Hut opening criterion */
                {
                    if (nop->len * nop->len > r2 * errTol2)
                    {
                        openflag = 1;
                    }
                }
                else /* check relative opening criterion */
                {
                    if (abs(nop->center[0] - pos_x) < 0.60 * nop->len)
                    {
                        if (abs(nop->center[1] - pos_y) < 0.60 * nop->len)
                        {
                            if (abs(nop->center[2] - pos_z) < 0.60 * nop->len)
                            {
                                openflag = 1;
                            }
                        }
                    }
                }

                if (openflag)
                {
                    /* now we check if we can avoid opening the cell */

                    u = nop->center[0] - pos_x;
                    if (u > boxhalf)
                    {
                        u -= boxsize;
                    }
                    if (u < -boxhalf)
                    {
                        u += boxsize;
                    }
                    if (abs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[1] - pos_y;
                    if (u > boxhalf)
                    {
                        u -= boxsize;
                    }
                    if (u < -boxhalf)
                    {
                        u += boxsize;
                    }
                    if (abs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    u = nop->center[2] - pos_z;
                    if (u > boxhalf)
                    {
                        u -= boxsize;
                    }
                    if (u < -boxhalf)
                    {
                        u += boxsize;
                    }
                    if (abs(u) > 0.5 * (boxsize - nop->len))
                    {
                        no = nop->u.d.nextnode;
                        continue;
                    }

                    /* if the cell is too large, we need to refine
                     * it further
                     */
                    if (nop->len > 0.20 * boxsize)
                    {
                        /* cell is too large */
                        no = nop->u.d.nextnode;
                        continue;
                    }
                }

                no = nop->u.d.sibling; /* ok, node can be used */
            }

            /* compute the Ewald correction force */

            if (dx < 0)
            {
                dx = -dx;
                signx = +1;
            }
            else
            {
                signx = -1;
            }
            if (dy < 0)
            {
                dy = -dy;
                signy = +1;
            }
            else
            {
                signy = -1;
            }
            if (dz < 0)
            {
                dz = -dz;
                signz = +1;
            }
            else
            {
                signz = -1;
            }
            u = dx * fac_intp;
            i = (int)u;
            if (i >= EN)
            {
                i = EN - 1;
            }
            u -= i;
            v = dy * fac_intp;
            j = (int)v;
            if (j >= EN)
            {
                j = EN - 1;
            }
            v -= j;
            w = dz * fac_intp;
            k = (int)w;
            if (k >= EN)
            {
                k = EN - 1;
            }
            w -= k;
            /* compute factors for trilinear interpolation */
            f1 = (1 - u) * (1 - v) * (1 - w);
            f2 = (1 - u) * (1 - v) * (w);
            f3 = (1 - u) * (v) * (1 - w);
            f4 = (1 - u) * (v) * (w);
            f5 = (u) * (1 - v) * (1 - w);
            f6 = (u) * (1 - v) * (w);
            f7 = (u) * (v) * (1 - w);
            f8 = (u) * (v) * (w);
            phidot += vx * mass * signx *
                      (fcorrx[i][j][k] * f1 + fcorrx[i][j][k + 1] * f2 + fcorrx[i][j + 1][k] * f3 +
                       fcorrx[i][j + 1][k + 1] * f4 + fcorrx[i + 1][j][k] * f5 + fcorrx[i + 1][j][k + 1] * f6 +
                       fcorrx[i + 1][j + 1][k] * f7 + fcorrx[i + 1][j + 1][k + 1] * f8);
            phidot += vy * mass * signy *
                      (fcorry[i][j][k] * f1 + fcorry[i][j][k + 1] * f2 + fcorry[i][j + 1][k] * f3 +
                       fcorry[i][j + 1][k + 1] * f4 + fcorry[i + 1][j][k] * f5 + fcorry[i + 1][j][k + 1] * f6 +
                       fcorry[i + 1][j + 1][k] * f7 + fcorry[i + 1][j + 1][k + 1] * f8);
            phidot += vz * mass * signz *
                      (fcorrz[i][j][k] * f1 + fcorrz[i][j][k + 1] * f2 + fcorrz[i][j + 1][k] * f3 +
                       fcorrz[i][j + 1][k + 1] * f4 + fcorrz[i + 1][j][k] * f5 + fcorrz[i + 1][j][k + 1] * f6 +
                       fcorrz[i + 1][j + 1][k] * f7 + fcorrz[i + 1][j + 1][k + 1] * f8);
            cost++;
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                no = GravDataGet[target].NodeList[listindex];
                if (no >= 0)
                    no = Nodes[no].u.d.nextnode; /* open it */
            }
        }
    }

    /* add the result at the proper place */

    if (mode == 0)
    {
        SphP[target].PhiDot += phidot;
    }
    else
    {
        PhiDotDataResult[target].PhiDot += phidot;
    }

    return cost;
}
#endif // PERIODIC

// double phidot_ewald_corr(double dx, double dy, double dz, double vx, double vy, double vz)
// {
//   int signx, signy, signz;
//   int i, j, k;
//   double u, v, w;
//   double f1, f2, f3, f4, f5, f6, f7, f8;
//   double fper[3], phidot;

//   if (dx < 0)
//     {
//       dx = -dx;
//       signx = +1;
//     }
//   else
//     {
//       signx = -1;
//     }
//   if (dy < 0)
//     {
//       dy = -dy;
//       signy = +1;
//     }
//   else
//     {
//       signy = -1;
//     }
//   if (dz < 0)
//     {
//       dz = -dz;
//       signz = +1;
//     }
//   else
//     {
//       signz = -1;
//     }
//   u = dx * fac_intp;
//   i = (int) u;
//   if (i >= EN)
//     {
//       i = EN - 1;
//     }
//   u -= i;
//   v = dy * fac_intp;
//   j = (int) v;
//   if (j >= EN)
//     {
//       j = EN - 1;
//     }
//   v -= j;
//   w = dz * fac_intp;
//   k = (int) w;
//   if (k >= EN)
//     {
//       k = EN - 1;
//     }
//   w -= k;
//   f1 = (1 - u) * (1 - v) * (1 - w);
//   f2 = (1 - u) * (1 - v) * (w);
//   f3 = (1 - u) * (v) * (1 - w);
//   f4 = (1 - u) * (v) * (w);
//   f5 = (u) * (1 - v) * (1 - w);
//   f6 = (u) * (1 - v) * (w);
//   f7 = (u) * (v) * (1 - w);
//   f8 = (u) * (v) * (w);
//   fper[0] = signx * (fcorrx[i][j][k] * f1 +
// 		     fcorrx[i][j][k + 1] * f2 +
// 		     fcorrx[i][j + 1][k] * f3 +
// 		     fcorrx[i][j + 1][k + 1] * f4 +
// 		     fcorrx[i + 1][j][k] * f5 +
// 		     fcorrx[i + 1][j][k + 1] * f6 +
// 		     fcorrx[i + 1][j + 1][k] * f7 + fcorrx[i + 1][j + 1][k + 1] * f8);
//   fper[1] =
//     signy * (fcorry[i][j][k] * f1 + fcorry[i][j][k + 1] * f2 +
// 	     fcorry[i][j + 1][k] * f3 + fcorry[i][j + 1][k + 1] * f4 +
// 	     fcorry[i + 1][j][k] * f5 + fcorry[i + 1][j][k + 1] * f6 +
// 	     fcorry[i + 1][j + 1][k] * f7 + fcorry[i + 1][j + 1][k + 1] * f8);
//   fper[2] =
//     signz * (fcorrz[i][j][k] * f1 + fcorrz[i][j][k + 1] * f2 +
// 	     fcorrz[i][j + 1][k] * f3 + fcorrz[i][j + 1][k + 1] * f4 +
// 	     fcorrz[i + 1][j][k] * f5 + fcorrz[i + 1][j][k + 1] * f6 +
// 	     fcorrz[i + 1][j + 1][k] * f7 + fcorrz[i + 1][j + 1][k + 1] * f8);

//   phidot = fper[0] * vx + fper[1] * vy + fper[2] * vz;

//   return phidot;
// }
#endif // PHIDOT

/**
 * @brief Write particle data to file
 *
 * This function dumps some of the basic particle data to a file. In case
 * the tree construction fails, it is called just before the run
 * terminates with an error message. Examination of the generated file may
 * then give clues to what caused the problem.
 */
void dump_particles(void)
{
    FILE *fd;
    char buffer[200];

    sprintf(buffer, "particles%d.dat", ThisTask);
    fd = fopen(buffer, "w");
    my_fwrite(&NumPart, 1, sizeof(int), fd);
    for (int i = 0; i < NumPart; i++)
    {
        my_fwrite(&P[i].Pos[0], 3, sizeof(MyFloat), fd);
    }
    for (int i = 0; i < NumPart; i++)
    {
        my_fwrite(&P[i].Vel[0], 3, sizeof(MyFloat), fd);
    }
    for (int i = 0; i < NumPart; i++)
    {
        my_fwrite(&P[i].ID, 1, sizeof(int), fd);
    }
    fclose(fd);
}

#ifdef PERIODIC
/**
 * @brief Initialize the Ewald correction tables
 *
 * This function initializes tables with the correction force and the
 * correction potential due to the periodic images of a point mass located
 * at the origin. These corrections are obtained by Ewald summation. (See
 * Hernquist, Bouchet, Suto, ApJS, 1991, 75, 231) The correction fields
 * are used to obtain the full periodic force if periodic boundaries
 * combined with the pure tree algorithm are used. For the TreePM
 * algorithm, the Ewald correction is not used.
 *
 * The correction fields are stored on disk once they are computed. If a
 * corresponding file is found, they are loaded from disk to speed up the
 * initialization.  The Ewald summation is done in parallel, i.e. the
 * processors share the work to compute the tables if needed.
 */
void ewald_init(void)
{
#ifndef NOGRAVITY
    char buf[200];

    VERBOSE(2, "initialize Ewald correction...\n");

#ifdef DOUBLEPRECISION
    sprintf(buf, "ewald_spc_table_%d_dbl%d.dat", EN, DOUBLEPRECISION);
#else // DOUBLEPRECISION
    sprintf(buf, "ewald_spc_table_%d.dat", EN);
#endif // DOUBLEPRECISION

    int flag = 0;
    if (ThisTask == 0)
    {
        FILE *fd;
        if ((fd = fopen(buf, "r")))
        {

            VERBOSE(2, "\nreading Ewald tables from file `%s'", buf);

            int size = (EN + 1) * (EN + 1) * (EN + 1);

            my_fread(&fcorrx[0][0][0], sizeof(MyFloat), size, fd);
            my_fread(&fcorry[0][0][0], sizeof(MyFloat), size, fd);
            my_fread(&fcorrz[0][0][0], sizeof(MyFloat), size, fd);
            my_fread(&potcorr[0][0][0], sizeof(MyFloat), size, fd);

            fclose(fd);

            flag = 1;
        }
    }

    MPI_Bcast(&flag, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    if (flag == 1)
    {
        // task 0 managed to read the data, so communicate it now.
        // As the whole file was read by task 0, we can communicate everything at once.
        int size = (EN + 1) * (EN + 1) * (EN + 1);
        MPI_Bcast(&fcorrx[0][0][0], size * sizeof(MyFloat), MPI_BYTE, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(&fcorry[0][0][0], size * sizeof(MyFloat), MPI_BYTE, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(&fcorrz[0][0][0], size * sizeof(MyFloat), MPI_BYTE, 0, MYMPI_COMM_WORLD);
        MPI_Bcast(&potcorr[0][0][0], size * sizeof(MyFloat), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    }
    else
    {
        VERBOSE(2, "\nNo Ewald tables in file `%s' found.\nRecomputing them...", buf);

        /* ok, let's recompute things. Actually, we do that in parallel. */

        int size = (EN + 1) * (EN + 1) * (EN + 1) / NTask;
        int beg = ThisTask * size;
        int len = size;
        // if (ThisTask == (NTask - 1))
        // {
        //     int len = (EN + 1) * (EN + 1) * (EN + 1) - beg;
        // }

        for (int i = 0, count = 0; i <= EN; i++)
        {
            for (int j = 0; j <= EN; j++)
            {
                for (int k = 0; k <= EN; k++)
                {
                    int n = (i * (EN + 1) + j) * (EN + 1) + k;
                    if (n >= beg && n < (beg + len))
                    {
                        if (ThisTask == 0)
                        {
                            if ((count % (len / 20)) == 0)
                            {
                                VERBOSE_ALL(2, "%4.1f percent done\n", count / (len / 100.0));
                            }
                        }

                        MyAtLeastDouble x[3], force[3];
                        x[0] = 0.5 * ((MyAtLeastDouble)i) / EN;
                        x[1] = 0.5 * ((MyAtLeastDouble)j) / EN;
                        x[2] = 0.5 * ((MyAtLeastDouble)k) / EN;
                        ewald_force(i, j, k, x, force);
                        fcorrx[i][j][k] = force[0];
                        fcorry[i][j][k] = force[1];
                        fcorrz[i][j][k] = force[2];
                        if (i + j + k == 0)
                        {
                            potcorr[i][j][k] = 2.8372975;
                        }
                        else
                        {
                            potcorr[i][j][k] = ewald_psi(x);
                        }
                        count++;
                    }
                }
            }
        }

        for (int task = 0; task < NTask; task++)
        {
            int beg = task * size;
            int len = size;
            if (task == (NTask - 1))
            {
                len = (EN + 1) * (EN + 1) * (EN + 1) - beg;
            }
            MPI_Bcast(&fcorrx[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MYMPI_COMM_WORLD);
            MPI_Bcast(&fcorry[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MYMPI_COMM_WORLD);
            MPI_Bcast(&fcorrz[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MYMPI_COMM_WORLD);
            MPI_Bcast(&potcorr[0][0][beg], len * sizeof(MyFloat), MPI_BYTE, task, MYMPI_COMM_WORLD);
        }

        if (ThisTask == 0)
        {
            VERBOSE(2, "\nwriting Ewald tables to file `%s'", buf);

            FILE *fd;
            if ((fd = fopen(buf, "w")))
            {
                my_fwrite(&fcorrx[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&fcorry[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&fcorrz[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                my_fwrite(&potcorr[0][0][0], sizeof(MyFloat), (EN + 1) * (EN + 1) * (EN + 1), fd);
                fclose(fd);
            }
        }
    }

    fac_intp = 2 * EN / All.BoxSize;
    for (int i = 0; i <= EN; i++)
    {
        for (int j = 0; j <= EN; j++)
        {
            for (int k = 0; k <= EN; k++)
            {
                potcorr[i][j][k] /= All.BoxSize;
                fcorrx[i][j][k] /= All.BoxSize * All.BoxSize;
                fcorry[i][j][k] /= All.BoxSize * All.BoxSize;
                fcorrz[i][j][k] /= All.BoxSize * All.BoxSize;
            }
        }
    }

    VERBOSE(2, "initialization of periodic boundaries finished.");

#endif // NOGRAVITY
}

#ifdef FORCETEST
/**
 * @brief Compute the Ewald correction force
 *
 * This function looks up the correction force due to the infinite number
 * of periodic particle/node images. We here use trilinear interpolation
 * to get it from the precomputed tables, which contain one octant
 * around the target particle at the origin. The other octants are
 * obtained from it by exploiting the symmetry properties.
 *
 * @param dx The x-coordinate of the distance vector.
 * @param dy The y-coordinate of the distance vector.
 * @param dz The z-coordinate of the distance vector.
 * @param fper The correction force.
 */
void ewald_corr(MyAtLeastDouble dx, MyAtLeastDouble dy, MyAtLeastDouble dz, MyAtLeastDouble *fper)
{
    int signx, signy, signz;
    if (dx < 0)
    {
        dx = -dx;
        signx = +1;
    }
    else
    {
        signx = -1;
    }

    if (dy < 0)
    {
        dy = -dy;
        signy = +1;
    }
    else
    {
        signy = -1;
    }

    if (dz < 0)
    {
        dz = -dz;
        signz = +1;
    }
    else
    {
        signz = -1;
    }
    double u = dx * fac_intp;
    int i = (int)u;
    if (i >= EN)
    {
        i = EN - 1;
    }
    u -= i;
    double v = dy * fac_intp;
    int j = (int)v;
    if (j >= EN)
    {
        j = EN - 1;
    }
    v -= j;
    double w = dz * fac_intp;
    int k = (int)w;
    if (k >= EN)
    {
        k = EN - 1;
    }
    w -= k;
    double f1 = (1 - u) * (1 - v) * (1 - w);
    double f2 = (1 - u) * (1 - v) * (w);
    double f3 = (1 - u) * (v) * (1 - w);
    double f4 = (1 - u) * (v) * (w);
    double f5 = (u) * (1 - v) * (1 - w);
    double f6 = (u) * (1 - v) * (w);
    double f7 = (u) * (v) * (1 - w);
    double f8 = (u) * (v) * (w);
    fper[0] = signx * (fcorrx[i][j][k] * f1 + fcorrx[i][j][k + 1] * f2 + fcorrx[i][j + 1][k] * f3 +
                       fcorrx[i][j + 1][k + 1] * f4 + fcorrx[i + 1][j][k] * f5 + fcorrx[i + 1][j][k + 1] * f6 +
                       fcorrx[i + 1][j + 1][k] * f7 + fcorrx[i + 1][j + 1][k + 1] * f8);
    fper[1] = signy * (fcorry[i][j][k] * f1 + fcorry[i][j][k + 1] * f2 + fcorry[i][j + 1][k] * f3 +
                       fcorry[i][j + 1][k + 1] * f4 + fcorry[i + 1][j][k] * f5 + fcorry[i + 1][j][k + 1] * f6 +
                       fcorry[i + 1][j + 1][k] * f7 + fcorry[i + 1][j + 1][k + 1] * f8);
    fper[2] = signz * (fcorrz[i][j][k] * f1 + fcorrz[i][j][k + 1] * f2 + fcorrz[i][j + 1][k] * f3 +
                       fcorrz[i][j + 1][k + 1] * f4 + fcorrz[i + 1][j][k] * f5 + fcorrz[i + 1][j][k + 1] * f6 +
                       fcorrz[i + 1][j + 1][k] * f7 + fcorrz[i + 1][j + 1][k + 1] * f8);
}
#endif // FORCETEST

#ifdef PERIODIC
/**
 * @brief Loop up the correction for the Ewald correction potential.
 *
 * This function looks up the correction potential due to the infinite
 * number of periodic particle/node images. We here use tri-linear
 * interpolation to get it from the precomputed table, which contains
 * one octant around the target particle at the origin. The other
 * octants are obtained from it by exploiting symmetry properties.
 *
 * @param dx The x-coordinate of the distance vector.
 * @param dy The y-coordinate of the distance vector.
 * @param dz The z-coordinate of the distance vector.
 *
 * @return The correction potential.
 */
double ewald_pot_corr(MyAtLeastDouble dx, MyAtLeastDouble dy, MyAtLeastDouble dz)
{
    if (dx < 0)
    {
        dx = -dx;
    }
    if (dy < 0)
    {
        dy = -dy;
    }
    if (dz < 0)
    {
        dz = -dz;
    }
    MyAtLeastDouble u = dx * fac_intp;
    int i = (int)u;
    if (i >= EN)
    {
        i = EN - 1;
    }
    u -= i;
    MyAtLeastDouble v = dy * fac_intp;
    int j = (int)v;
    if (j >= EN)
    {
        j = EN - 1;
    }
    v -= j;
    MyAtLeastDouble w = dz * fac_intp;
    int k = (int)w;
    if (k >= EN)
    {
        k = EN - 1;
    }
    w -= k;
    MyAtLeastDouble f1 = (1 - u) * (1 - v) * (1 - w);
    MyAtLeastDouble f2 = (1 - u) * (1 - v) * (w);
    MyAtLeastDouble f3 = (1 - u) * (v) * (1 - w);
    MyAtLeastDouble f4 = (1 - u) * (v) * (w);
    MyAtLeastDouble f5 = (u) * (1 - v) * (1 - w);
    MyAtLeastDouble f6 = (u) * (1 - v) * (w);
    MyAtLeastDouble f7 = (u) * (v) * (1 - w);
    MyAtLeastDouble f8 = (u) * (v) * (w);
    return potcorr[i][j][k] * f1 + potcorr[i][j][k + 1] * f2 + potcorr[i][j + 1][k] * f3 +
           potcorr[i][j + 1][k + 1] * f4 + potcorr[i + 1][j][k] * f5 + potcorr[i + 1][j][k + 1] * f6 +
           potcorr[i + 1][j + 1][k] * f7 + potcorr[i + 1][j + 1][k + 1] * f8;
}
#endif // PERIODIC

/**
 * @brief Compute the Ewald correction potential
 *
 * This function computes the potential correction term by means of Ewald
 * summation.
 *
 * @param x The position vector.
 *
 * @return The correction potential.
 */
MyAtLeastDouble ewald_psi(MyAtLeastDouble x[3])
{
    MyAtLeastDouble sum1, sum2;
    int n[3], h[3];

    MyAtLeastDouble alpha = 2.0;
    for (n[0] = -4, sum1 = 0; n[0] <= 4; n[0]++)
    {
        for (n[1] = -4; n[1] <= 4; n[1]++)
        {
            for (n[2] = -4; n[2] <= 4; n[2]++)
            {
                MyAtLeastDouble dx[3];
                for (int i = 0; i < 3; i++)
                {
                    dx[i] = x[i] - n[i];
                }
                MyAtLeastDouble r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                sum1 += erfc(alpha * r) / r;
            }
        }
    }

    for (h[0] = -4, sum2 = 0; h[0] <= 4; h[0]++)
    {
        for (h[1] = -4; h[1] <= 4; h[1]++)
        {
            for (h[2] = -4; h[2] <= 4; h[2]++)
            {
                MyAtLeastDouble hdotx = x[0] * h[0] + x[1] * h[1] + x[2] * h[2];
                int h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
                if (h2 > 0)
                {
                    sum2 += 1 / (M_PI * h2) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) * cos(2 * M_PI * hdotx);
                }
            }
        }
    }

    MyAtLeastDouble r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    MyAtLeastDouble psi = M_PI / (alpha * alpha) - sum1 - sum2 + 1 / r;
    return psi;
}

/**
 * @brief Compute the Ewald correction force
 *
 * This function computes the force correction term (difference between full
 * force of infinite lattice and nearest image) by Ewald summation.
 * It accounts for both the direct and reciprocal space contributions to the force.
 *
 * @param iii 	 The x-coordinate of the distance vector.
 * @param jjj    The y-coordinate of the distance vector
 * @param kkk    The z-coordinate of the distance vector
 * @param x      The coordinates of the particle.
 * @param force  The array to store the computed force components.
 */
void ewald_force(int iii, int jjj, int kkk, MyAtLeastDouble x[3], MyAtLeastDouble force[3])
{
    int h[3], n[3];

    MyAtLeastDouble alpha = 2.0;
    for (int i = 0; i < 3; i++)
    {
        force[i] = 0;
    }
    if (iii == 0 && jjj == 0 && kkk == 0)
    {
        return;
    }
    MyAtLeastDouble r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    for (int i = 0; i < 3; i++)
    {
        force[i] += x[i] / (r2 * sqrt(r2));
    }
    for (n[0] = -4; n[0] <= 4; n[0]++)
    {
        for (n[1] = -4; n[1] <= 4; n[1]++)
        {
            for (n[2] = -4; n[2] <= 4; n[2]++)
            {
                MyAtLeastDouble dx[3];
                for (int i = 0; i < 3; i++)
                {
                    dx[i] = x[i] - n[i];
                }
                MyAtLeastDouble r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
                MyAtLeastDouble val = erfc(alpha * r) + 2 * alpha * r / sqrt(M_PI) * exp(-alpha * alpha * r * r);
                for (int i = 0; i < 3; i++)
                {
                    force[i] -= dx[i] / (r * r * r) * val;
                }
            }
        }
    }

    for (h[0] = -4; h[0] <= 4; h[0]++)
    {
        for (h[1] = -4; h[1] <= 4; h[1]++)
        {
            for (h[2] = -4; h[2] <= 4; h[2]++)
            {
                MyAtLeastDouble hdotx = x[0] * h[0] + x[1] * h[1] + x[2] * h[2];
                int h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
                if (h2 > 0)
                {
                    MyAtLeastDouble val =
                        2.0 / ((MyAtLeastDouble)h2) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) * sin(2 * M_PI * hdotx);
                    for (int i = 0; i < 3; i++)
                    {
                        force[i] -= h[i] * val;
                    }
                }
            }
        }
    }
}
#endif // PERIODIC

#ifdef ADAPTGRAVSOFT
/**
 * @brief Update the gravitational softening of tree nodes
 *
 * This function updates the gravitational softening of tree nodes.
 * You need to call it after ags_density() and before the computation of the gravitational force.
 */
void ags_force_update_hmax(void)
{
    TIMER_PUSH(1, "AGSTREEHMAXUPD");
    int no, totDomainNumChanged;
    int *domainList_all;
    int *counts, *offset_list, *offset_hmax;
    MyFloat *domainHmax_loc, *domainHmax_all;

    GlobFlag++;

    DomainNumChanged = 0;
    DomainList = (int *)mymalloc("DomainList", NTopleaves * sizeof(int));
#ifndef AGS_UPDATEALLPARTICLES
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
#else // AGS_UPDATEALLPARTICLES
    for (int i = 0; i < NumPart; i++)
#endif // AGS_UPDATEALLPARTICLES
    {
        no = Father[i];

        while (no >= 0)
        {
            force_drift_node(no, All.Ti_Current);

            if (P[i].AGS_Hsml > Nodes[no].maxsoft)
            {

                Nodes[no].maxsoft = P[i].AGS_Hsml;

                if (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node */
                {
                    if (Nodes[no].Flag != GlobFlag)
                    {
                        Nodes[no].Flag = GlobFlag;
                        DomainList[DomainNumChanged++] = no;
                    }
                    break;
                }
            }
            else
            {
                break;
            }
            no = Nodes[no].u.d.father;
        }
    }
    /* share the hmax-data of the pseudo-particles accross CPUs */

    counts = (int *)mymalloc("counts", sizeof(int) * NTask);
    offset_list = (int *)mymalloc("offset_list", sizeof(int) * NTask);
    offset_hmax = (int *)mymalloc("offset_hmax", sizeof(int) * NTask);

    domainHmax_loc = (MyFloat *)mymalloc("domainHmax_loc", DomainNumChanged * 2 * sizeof(MyFloat));

    for (int i = 0; i < DomainNumChanged; i++)
    {
        domainHmax_loc[2 * i] = Nodes[DomainList[i]].maxsoft;
    }

    MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MYMPI_COMM_WORLD);
    totDomainNumChanged = 0, offset_list[0] = 0, offset_hmax[0] = 0;
    for (int ta = 0; ta < NTask; ta++)
    {
        totDomainNumChanged += counts[ta];
        if (ta > 0)
        {
            offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
            offset_hmax[ta] = offset_hmax[ta - 1] + counts[ta - 1] * 2 * sizeof(MyFloat);
        }
    }

    VERBOSE(2, "*ADAPTGRAVSOFT* Hmax exchange: %d topleaves out of %d", totDomainNumChanged, NTopleaves);

    domainHmax_all = (MyFloat *)mymalloc("domainHmax_all", totDomainNumChanged * 2 * sizeof(MyFloat));
    domainList_all = (int *)mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

    MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT, domainList_all, counts, offset_list, MPI_INT,
                   MYMPI_COMM_WORLD);

    for (int ta = 0; ta < NTask; ta++)
    {
        counts[ta] *= 2 * sizeof(MyFloat);
    }

    MPI_Allgatherv(domainHmax_loc, 2 * DomainNumChanged * sizeof(MyFloat), MPI_BYTE, domainHmax_all, counts,
                   offset_hmax, MPI_BYTE, MYMPI_COMM_WORLD);

    for (int i = 0; i < totDomainNumChanged; i++)
    {
        no = domainList_all[i];

        if (Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS)) /* to avoid that the hmax is updated twice */
        {
            no = Nodes[no].u.d.father;
        }

        while (no >= 0)
        {
            force_drift_node(no, All.Ti_Current);

            if (domainHmax_all[2 * i] > Nodes[no].maxsoft)
            {

                Nodes[no].maxsoft = domainHmax_all[2 * i];
            }
            else
            {
                break;
            }
            no = Nodes[no].u.d.father;
        }
    }

    myfree(domainList_all);
    myfree(domainHmax_all);
    myfree(domainHmax_loc);
    myfree(offset_hmax);
    myfree(offset_list);
    myfree(counts);
    myfree(DomainList);

    TIMER_POP(1, "AGSTREEHMAXUPD");
}
#endif // ADAPTGRAVSOFT

#ifdef AR_XMAS_TREE
/**
 * @brief Create empty tree nodes and build the array of leaf IDs.
 *
 * This function recursively creates a set of empty tree nodes which
 * corresponds to the top-level tree for the domain grid. This is done to
 * ensure that this top-level tree is always "complete" so that we can easily
 * associate the pseudo-particles of other CPUs with tree-nodes at a given
 * level in the tree, even when the particle population is so sparse that
 * some of these nodes are actually empty.
 *
 * @param no         The index of the current tree node.
 * @param topnode    The index of the top-level node.
 * @param bits       The bit resolution for the Peano-Hilbert key calculation.
 * @param x          The x-coordinate of the current node.
 * @param y          The y-coordinate of the current node.
 * @param z          The z-coordinate of the current node.
 * @param nodecount  Pointer to the count of nodes created.
 * @param nextfree   Pointer to the index of the next free node.
 * @param nleaves    Pointer to the count of leaf nodes.
 * @param leaf_ids   Array to store the IDs of leaf nodes.
 */
void force_create_empty_nodes_and_build_leaves_array(int no, int topnode, int bits, int x, int y, int z, int *nodecount,
                                                     int *nextfree, int *nleaves, int *leaf_ids)
{
    if (TopNodes[topnode].Daughter >= 0)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    int sub = 7 & peano_hilbert_key((x << 1) + i, (y << 1) + j, (z << 1) + k, bits);
                    int count = i + 2 * j + 4 * k;

                    Nodes[no].u.suns[count] = *nextfree;

                    MyFloat lenhalf = 0.25 * Nodes[no].len;
                    Nodes[*nextfree].len = 0.5 * Nodes[no].len;
                    Nodes[*nextfree].center[0] = Nodes[no].center[0] + (2 * i - 1) * lenhalf;
                    Nodes[*nextfree].center[1] = Nodes[no].center[1] + (2 * j - 1) * lenhalf;
                    Nodes[*nextfree].center[2] = Nodes[no].center[2] + (2 * k - 1) * lenhalf;

                    for (int n = 0; n < 8; n++)
                    {
                        Nodes[*nextfree].u.suns[n] = -1;
                    }

                    if (TopNodes[TopNodes[topnode].Daughter + sub].Daughter == -1)
                    {
                        DomainNodeIndex[TopNodes[TopNodes[topnode].Daughter + sub].Leaf] = *nextfree;
                    }

                    *nextfree = *nextfree + 1;
                    *nodecount = *nodecount + 1;

                    if ((*nodecount) >= MaxNodes || (*nodecount) >= MaxTopNodes)
                    {
                        VERBOSE_ALL(0,
                                    "task %d: maximum number MaxNodes=%d of tree-nodes reached."
                                    "MaxTopNodes=%d NTopnodes=%d NTopleaves=%d nodecount=%d",
                                    ThisTask, MaxNodes, MaxTopNodes, NTopnodes, NTopleaves, *nodecount);
                        VERBOSE_ALL(0, "in create empty nodes");
                        dump_particles();
                        ENDRUN("Error in tree structure.");
                    }

                    force_create_empty_nodes_and_build_leaves_array(*nextfree - 1, TopNodes[topnode].Daughter + sub,
                                                                    bits + 1, 2 * x + i, 2 * y + j, 2 * z + k,
                                                                    nodecount, nextfree, nleaves, leaf_ids);
                }
            }
        }
    }
    else
    {
        leaf_ids[topnode] = *nleaves;
        (*nleaves)++;
    }
}

/**
 * @brief Builds a parallel tree structure to represent the distribution of particles.
 *
 * This function constructs a hierarchical tree structure to efficiently represent the spatial
 * distribution of particles in a parallel environment. It utilizes OpenMP for parallelism
 * to distribute the workload across multiple threads.
 *
 * @param npart  The total number of particles.
 * @param mp     An array of unbind_data structs containing particle information. If NULL, particles are assumed to be
 * indexed from 0 to npart-1.
 * @return  The number of nodes created in the tree structure. Returns -1 if an error occurred during tree construction.
 */
int force_treebuild_parallel(int npart, struct unbind_data *mp)
{
    struct NODE *root_nfreep;

    TIMER_PUSH(1, "TREE_BUILD_ADD_PART");
    /* create an empty root node  */
    int root_nfree = All.MaxPart; /* index of first free node */
    root_nfreep = &Nodes[root_nfree]; /* select first node */

    root_nfreep->len = DomainLen;
    for (int j = 0; j < 3; j++)
    {
        root_nfreep->center[j] = DomainCenter[j];
    }
    for (int j = 0; j < 8; j++)
    {
        root_nfreep->u.suns[j] = -1;
    }

    int num_top_nodes = 1;
    int num_leaves = 1;
    root_nfreep++;
    root_nfree++;

    int error = 0;

    peanokey *morton_list = (peanokey *)mymalloc("morton_list", NumPart * sizeof(peanokey));
    int *leaf_ids = (int *)mymalloc("leaf_ids", NTopnodes * sizeof(int));
    int *particle_leaf_ids = (int *)mymalloc("particle_leaf", NumPart * sizeof(int));
    int *particles_nos = (int *)mymalloc("particle_leaf", NumPart * sizeof(int));
    int *particles_in_leaf = (int *)mymalloc("particle_leaf", NumPart * sizeof(int));
    int *particles_slots = (int *)mymalloc("particle_leaf", NumPart * sizeof(int));
    int *particles_reps = (int *)mymalloc("particle_leaf", NumPart * sizeof(int));
    int *particles_shifts = (int *)mymalloc("particle_leaf", NumPart * sizeof(int));

#pragma omp parallel for
    for (int i = 0; i < NTopnodes; i++)
    {
        leaf_ids[i] = -1;
    }

    force_create_empty_nodes_and_build_leaves_array(All.MaxPart, 0, 1, 0, 0, 0, &num_top_nodes, &root_nfree,
                                                    &num_leaves, leaf_ids);
    int *inserted_particles_in_leaf = (int *)mymalloc("particles_in_leaf", num_leaves * sizeof(int));
    int *beginning_index_particle_array = (int *)mymalloc("particles_in_leaf", num_leaves * sizeof(int));
#pragma omp parallel for
    for (int i = 0; i < num_leaves; i++)
    {
        particles_in_leaf[i] = 0;
    }

#pragma omp parallel for
    for (int i = 0; i < num_leaves; i++)
    {
        inserted_particles_in_leaf[i] = 0;
    }

    int *particles_in_leaf_private = (int *)mymalloc("particles_in_leaf", num_leaves * maxThreads * sizeof(int));

    int FreeSlot = 0;
    int SlotSize = AR_XMAS_TREE_FACTOR;

    VERBOSE(1, "SlotSiyz=%d", SlotSize);
#pragma omp parallel for
    for (int i = 0; i < num_leaves * maxThreads; i++)
    {
        particles_in_leaf_private[i] = 0;
    }

// reduction(+:particles_in_leaf[:num_leaves]) pgi compiler does not support redcution of array, but we are smart and we
// do it by our self!
#pragma omp parallel for
    for (int k = 0; k < npart; k++)
    {
        int i;
        if (mp)
        {
            i = mp[k].index;
        }
        else
        {
            i = k;
        }
        peanokey morton;
#ifdef NEUTRINOS
        if ((P[i].Type == type_neutrino) && (All.Time <= All.Time_tree_on_nu))
        {
            continue;
        }
#endif // NEUTRINOS
        peanokey key = peano_and_morton_key(
            (int)((P[i].Pos[0] - DomainCorner[0]) * DomainFac), (int)((P[i].Pos[1] - DomainCorner[1]) * DomainFac),
            (int)((P[i].Pos[2] - DomainCorner[2]) * DomainFac), BITS_PER_DIMENSION, &morton);
        morton_list[i] = morton;

        /* reset particle's dp for next tree update */
        P[i].dp[0] = 0.0;
        P[i].dp[1] = 0.0;
        P[i].dp[2] = 0.0;

        int shift = 3 * (BITS_PER_DIMENSION - 1);
        int rep = 0;
        int no = 0;
        while (TopNodes[no].Daughter >= 0)
        {
            no = TopNodes[no].Daughter + (key - TopNodes[no].StartKey) / (TopNodes[no].Size / 8);
            shift -= 3;
            rep++;
        }
        particles_nos[i] = no;
        particles_reps[i] = rep;
        particles_shifts[i] = shift;
        int oldno = no;
        no = TopNodes[no].Leaf; // only for debugging purposes
        if (leaf_ids[oldno] < 0 || leaf_ids[oldno] >= num_leaves)
        {
            ENDRUN_MAIN("i=%d, oldno=%d, leaf_ids[%d]=%d, leaf_ids[oldno]=%d", i, oldno, no, leaf_ids[no],
                        leaf_ids[oldno]);
        }
        int leaf_id = leaf_ids[oldno];
        particles_in_leaf_private[num_leaves * ThisThread + leaf_id]++;
        particle_leaf_ids[i] = leaf_id;
    }

    TIMER_POPPUSH(1, "TREE_BUILD_ADD_PART", "TREE_BUILD_JOINING_TREES");
    TIMER_PUSH(1, "part_in_leaf");

// we reduce the sum on the array because pgi compiler does not support reduction(+:arrayy[:size])
#pragma omp parallel for
#pragma ivdep
    for (int i = 0; i < num_leaves; i++)
    {
        for (int j = 0; j < maxThreads; j++)
        {
            particles_in_leaf[i] += particles_in_leaf_private[i + j * num_leaves];
        }
    }

    TIMER_POPPUSH(1, "part_in_leaf", "compute_index");

    int starting_point = 0;
    beginning_index_particle_array[0] = 0;
    // i am no parallel :(
    for (int i = 1; i < num_leaves; i++)
    {
        beginning_index_particle_array[i] = particles_in_leaf[i - 1] + starting_point;
        starting_point = beginning_index_particle_array[i];
    }

    TIMER_POPPUSH(1, "compute_index", "insert_part_in_leaf");

#pragma omp parallel for
#pragma novector
    // ivdep Nitya - this will have a conflict in the value of slot_id
    for (int k = 0; k < npart; k++)
    {
        int i;
        if (mp)
        {
            i = mp[k].index;
        }
        else
        {
            i = k;
        }
#ifdef NEUTRINOS
        if ((P[i].Type == type_neutrino) && (All.Time <= All.Time_tree_on_nu))
        {
            continue;
        }
#endif // NEUTRINOS
        int leaf_id = particle_leaf_ids[i];
        int i_p_in_leaf;
#pragma omp atomic capture
        i_p_in_leaf = inserted_particles_in_leaf[leaf_id]++;
        int slot_id = beginning_index_particle_array[leaf_id] + i_p_in_leaf;

        particles_slots[slot_id] = i;
    }

    TIMER_POPPUSH(1, "insert_part_in_leaf", "join_leafs");

    int numnodes = 0;
    int inserted_particles = 0;
    FreeSlot = 0;
#pragma omp parallel for reduction(max : numnodes) reduction(+ : inserted_particles)
    for (int leaf_id = 0; leaf_id < num_leaves; leaf_id++)
    {
        if (error)
        {
            continue;
        }
        int myslot = -1;
        int slot_size = SlotSize;
        int pointer_current = -1;
        int pointer_from = -1;
        int pointer_to = -1;
        int loop_from = beginning_index_particle_array[leaf_id];
        int loop_to = loop_from + particles_in_leaf[leaf_id];
        struct NODE *current_node = NULL;
        MyFloat lenhalf;

        for (int j = loop_from; j < loop_to; j++)
        {
            if (error)
            {
                break;
            }
            int i = particles_slots[j];

            int parent;
            peanokey th_key;

            int no = particles_nos[i];
            no = TopNodes[no].Leaf;
            int th = DomainNodeIndex[no];
            int rep = particles_reps[i];
            peanokey morton = morton_list[i];
            int subnode;
            inserted_particles++;
            int shift = particles_shifts[i];

            while (1)
            {
                if (th >= All.MaxPart)
                { /* we are dealing with an internal node */
                    if (shift >= 0)
                    {
                        subnode = ((morton >> shift) & 7);
                    }
                    else
                    {
                        subnode = 0;
                        if (P[i].Pos[0] > Nodes[th].center[0])
                        {
                            subnode += 1;
                        }
                        if (P[i].Pos[1] > Nodes[th].center[1])
                        {
                            subnode += 2;
                        }
                        if (P[i].Pos[2] > Nodes[th].center[2])
                        {
                            subnode += 4;
                        }
                    }
#ifndef NOTREERND
                    if (Nodes[th].len < 1.0e-3 * All.ForceSoftening[P[i].Type])
                    {
                        subnode = (int)(8.0 * get_random_number((P[i].ID + rep) % (RNDTABLE + (rep & 3))));
                        if (subnode >= 8)
                            subnode = 7;
                    }
#endif // NOTREERND
                    int nn = Nodes[th].u.suns[subnode];
                    shift -= 3;
                    if (nn >= 0)
                    { /* ok, something is in the daughter slot already, need to continue */
                        parent = th;
                        th = nn;
                        rep++;
                    }
                    else
                    {
                        /* here we have found an empty slot where we can attach
                         * the new particle as a leaf.
                         */
                        Nodes[th].u.suns[subnode] = i;
                        break; /* done for this particle */
                    }
                }
                else
                {
                    if ((pointer_current - pointer_from) >= slot_size || myslot == -1)
                    {
#pragma omp atomic capture
                        myslot = FreeSlot++;

                        if (myslot * slot_size >= MaxNodes)
                        {
                            error = 1;
                            break;
                        }

                        pointer_from = root_nfree + myslot * slot_size;
                        pointer_to = pointer_from + slot_size;
                        pointer_current = pointer_from;
                        current_node = &Nodes[pointer_current];

                        for (int l = pointer_current; l < pointer_to; l++)
                        {
                            Nodes[l].allocated = 0;
                        }
                    }
                    /* We try to insert into a leaf with a single particle.  Need
                     * to generate a new internal node at this point.
                     */
                    Nodes[parent].u.suns[subnode] = pointer_current;

                    current_node->allocated = 1;
                    current_node->len = 0.5 * Nodes[parent].len;
                    lenhalf = 0.25 * Nodes[parent].len;

                    if (subnode & 1)
                    {
                        current_node->center[0] = Nodes[parent].center[0] + lenhalf;
                    }
                    else
                    {
                        current_node->center[0] = Nodes[parent].center[0] - lenhalf;
                    }

                    if (subnode & 2)
                    {
                        current_node->center[1] = Nodes[parent].center[1] + lenhalf;
                    }
                    else
                    {
                        current_node->center[1] = Nodes[parent].center[1] - lenhalf;
                    }

                    if (subnode & 4)
                    {
                        current_node->center[2] = Nodes[parent].center[2] + lenhalf;
                    }
                    else
                    {
                        current_node->center[2] = Nodes[parent].center[2] - lenhalf;
                    }

                    current_node->u.suns[0] = -1;
                    current_node->u.suns[1] = -1;
                    current_node->u.suns[2] = -1;
                    current_node->u.suns[3] = -1;
                    current_node->u.suns[4] = -1;
                    current_node->u.suns[5] = -1;
                    current_node->u.suns[6] = -1;
                    current_node->u.suns[7] = -1;

                    if (shift >= 0)
                    {
                        th_key = morton_list[th];
                        subnode = ((th_key >> shift) & 7);
                    }
                    else
                    {
                        subnode = 0;
                        if (P[th].Pos[0] > current_node->center[0])
                        {
                            subnode += 1;
                        }
                        if (P[th].Pos[1] > current_node->center[1])
                        {
                            subnode += 2;
                        }
                        if (P[th].Pos[2] > current_node->center[2])
                        {
                            subnode += 4;
                        }
                    }

#ifndef NOTREERND

                    if (current_node->len < 1.0e-3 * All.ForceSoftening[P[th].Type])
                    {
                        /* seems like we're dealing with particles at identical (or extremely close)
                         * locations. Randomize subnode index to allow tree construction. Note: Multipole moments
                         * of tree are still correct, but this will only happen well below gravitational softening
                         * length-scale anyway.
                         */
                        subnode = (int)(8.0 * get_random_number((P[th].ID + rep) % (RNDTABLE + (rep & 3))));

                        if (subnode >= 8)
                        {
                            subnode = 7;
                        }
                    }
#endif // NOTREERND
                    current_node->u.suns[subnode] = th;

                    th = pointer_current; /* resume trying to insert the new particle at
                                           * the newly created internal node
                                           */

                    pointer_current++;
                    current_node++;
                    //  VERBOSE(-1, "numnodes %d -> %d ?", numnodes, pointer_current-All.MaxPart);
                    if (pointer_current - All.MaxPart > numnodes)
                    {
                        numnodes = pointer_current - All.MaxPart;
                    }
                }
            }
        }
    }

    TIMER_POP(1, "join_leafs");

    myfree(particles_in_leaf_private);
    myfree(beginning_index_particle_array);
    myfree(inserted_particles_in_leaf);
    myfree(particles_shifts);
    myfree(particles_reps);
    myfree(particles_slots);
    myfree(particles_in_leaf);
    myfree(particles_nos);
    myfree(particle_leaf_ids);
    myfree(leaf_ids);
    myfree(morton_list);

    TIMER_POPPUSH(1, "TREE_BUILD_JOINING_TREES", "TREE_BUILD_INSERT_PSEUDO");

    if (error == 1)
    {
        TIMER_POP(1, "TREE_BUILD_INSERT_PSEUDO");
        return -1;
    }
    /* insert the pseudo particles that represent the mass distribution of other domains */
    force_insert_pseudo_particles();

    TIMER_POPPUSH(1, "TREE_BUILD_INSERT_PSEUDO", "TREE_BUILD_UPDATE_NODES");

    /* now compute the multforce_update_node_reipole moments recursively */
    last = -1;
    force_update_node_recursive(All.MaxPart, -1, -1, 0);

    if (last >= All.MaxPart)
    {
        if (last >= All.MaxPart + MaxNodes) /* a pseudo-particle */
        {
            Nextnode[last - MaxNodes] = -1;
        }
        else
        {
            Nodes[last].u.d.nextnode = -1;
        }
    }
    else
    {
        Nextnode[last] = -1;
    }

    TIMER_POP(1, "TREE_BUILD_UPDATE_NODES");

#ifdef GROUP_LEAVES
    TIMER_PUSH(1, "TREE_BUILD_GROUP_LEAVES");
    force_group_leaves(GROUP_LEAVES);
    TIMER_POP(1, "TREE_BUILD_GROUP_LEAVES");
#endif // GROUP_LEAVES

    return numnodes;
}
#endif // AR_XMAS_TREE

// we include our self!
#ifdef ACC_GRAVITY
#define ACC_INCLUDE_GRAVITY 1
#include "forcetree.cpp"
#undef ACC_INCLUDE_GRAVITY
#endif // ACC_GRAVITY

#endif // ACC_INCLUDE_GRAVITY
