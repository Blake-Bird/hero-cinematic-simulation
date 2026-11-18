/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for gravity tree.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef FORCETREE_HPP
#define FORCETREE_HPP

#ifndef INLINE_FUNC
#ifdef INLINE
#define INLINE_FUNC inline
#else // INLINE
#define INLINE_FUNC
#endif // INLINE
#endif // INLINE_FUNC

#define BITFLAG_TOPLEVEL 0
#define BITFLAG_DEPENDS_ON_LOCAL_MASS 1
#define BITFLAG_MAX_SOFTENING_TYPE 2 /* bits 2-4 */
#define BITFLAG_MIXED_SOFTENINGS_IN_NODE 5
#define BITFLAG_INTERNAL_TOPLEVEL 6
#define BITFLAG_MULTIPLEPARTICLES 7
#define BITFLAG_NODEHASBEENKICKED 8
#define BITFLAG_INSIDE_LINKINGLENGTH 9

#define BITFLAG_MASK                                                                                                   \
    ((1 << BITFLAG_MULTIPLEPARTICLES) + (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE) + (7 << BITFLAG_MAX_SOFTENING_TYPE))
#define maskout_different_softening_flag(x) (x & (1 << BITFLAG_MIXED_SOFTENINGS_IN_NODE))
#define extract_max_softening_type(x) ((x >> BITFLAG_MAX_SOFTENING_TYPE) & 7)

void force_update_tree(void);
void force_flag_localnodes(void);

#ifdef ACC_GRAVITY
void *gravity_primary_loop(void *p, int is_primary_on_gpu, int should_drift);
#else // ACC_GRAVITY
void *gravity_primary_loop(void *p);
#endif // ACC_GRAVITY
void *gravity_secondary_loop(void *p);

#ifdef ACC_GRAVITY
int force_treeevaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int should_drift);
int force_treeevaluate_ewald_correction(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                        int should_drift);
int force_treeevaluate_shortrange(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                  int should_drift);
#else // ACC_GRAVITY
int force_treeevaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
int force_treeevaluate_ewald_correction(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
int force_treeevaluate_shortrange(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
#endif // ACC_GRAVITY

int force_treeevaluate_potential(int target, int type, int *nexport, int *nsend_local);
int force_treeevaluate_potential_shortrange(int target, int mode, int *nexport, int *nsend_local);

void force_drift_node(int no, int time1);

void force_tree_discardpartials(void);
void force_treeupdate_pseudos(int);
void force_update_pseudoparticles(void);

void force_kick_node(int i, MyFloat *dv);

void force_dynamic_update(void);
void force_dynamic_update_node(int no, int mode, MyFloat *minbound, MyFloat *maxbound);

// int force_treeevaluate_shortrange_local( int target, int mode, struct gravdata_lite *P, struct gravdata_in
// *GravDataGet, struct gravdata_out *GravDataResult, double* NodeGravCosts, double *PGravCosts);

void force_update_hmax(void);
#ifdef AXION_DM
void ax_force_update_hmax(void);
#endif // AXION_DM
void force_update_hmax_of_node(int no, int mode);

void force_finish_kick_nodes(void);

void force_create_empty_nodes(int no, int topnode, int bits, int x, int y, int z, int *nodecount, int *nextfree);

void force_exchange_pseudodata(void);

void force_insert_pseudo_particles(void);

void force_add_star_to_tree(int igas, int istar);

void force_costevaluate(void);
int force_getcost_single(void);
int force_getcost_quadru(void);
void force_resetcost(void);
void force_setupnonrecursive(int no);
void force_treeallocate(int maxnodes, int maxpart);
int force_treebuild(int npart, struct unbind_data *mp);
int force_treebuild_single(int npart, struct unbind_data *mp);
int force_treebuild_parallel(int npart, struct unbind_data *mp);
int force_treeevaluate_direct(int target, int mode);

void force_exchange_topleafdata_loc(void);
void force_update_node_recursive_loc(int no);
void force_treeupdate_toplevel_loc(
    int no /*!< node to be updated */, int topnode /*!< index of the node no in the #TopNodes array */,
    int bits
    /*!< 2^bits is the number of nodes per dimension at the level of the daughter nodes of node no */
    ,
    int x
    /*!< position of the node no in the x direction, falls in the range [0,2^(bits-1) - 1] */
    ,
    int y
    /*!< position of the node no in the y direction, falls in the range [0,2^(bits-1) - 1] */
    ,
    int z
    /*!< position of the node no in the z direction, falls in the range [0,2^(bits-1) - 1] */
);

void force_treefree(void);
void force_update_node(int no, int flag);

#ifdef AR_NODE_RECURSIVE_PARALLEL
int force_update_node_recursive(int no, int sib, int father, float LocSoft, int layer, int last);
#else // AR_NODE_RECURSIVE_PARALLEL
void force_update_node_recursive(int no, int sib, int father, float LocSoft);
#endif // AR_NODE_RECURSIVE_PARALLEL

void force_update_size_of_parent_node(int no);

void dump_particles(void);

MyFloat INLINE_FUNC ngb_periodic(MyFloat x);
MyFloat INLINE_FUNC ngb_periodic_longbox(MyFloat x);
MyFloat ngb_select_closest(int k, int n, MyFloat *arr, int *ind);
void ngb_treeallocate(int npart);
void ngb_treebuild(void);

void ngb_treefree(void);
void ngb_treesearch(int);
void ngb_treesearch_pairs(int);
void ngb_update_nodes(void);
void ngb_treesearch_notsee(int no);

int ngb_treefind_fof_primary(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                             int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES);
int ngb_clear_buf(MyLongDouble searchcenter[3], MyFloat hguess, int numngb);
void ngb_treefind_flagexport(MyLongDouble searchcenter[3], MyFloat hguess);

int ngb_treefind_blackhole(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                           int *nexport, int *nsend_local);
int ngb_treefind_variable(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                          int *nexport, int *nsend_local);

#ifdef GM_DMVELS
int ngb_treefind_variable_threads_dmvelfb(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                          int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                          int *ngblist);
#endif // GM_DMVELS

/*
template<typename FunctionPointer>
int ar_generic_treewalk(FunctionPointer particle, FunctionPointer pseudoparticle, MyLongDouble searchcenter[3], MyFloat
phsml, MyFloat radius, int *startnode, int mode){
{
  int no, nexp, p, task;
  struct NODE *current;
  // cache some global vars locally for improved compiler alias analysis
  int maxPart = All.MaxPart;
  int maxNodes = MaxNodes;
  int bunchSize = All.BunchSize;

  MyLongDouble dx, dy, dz, dist;
  MyLongDouble xtmp;
}
*/

#define NODELEVEL(no) (int)(log(Nodes[All.MaxPart].len / Nodes[no].len) / log(2))
#ifdef GROUP_LEAVES
/**
 * @brief Groups leaf nodes of the tree based on the specified level.
 *
 * This function groups the leaf nodes of the tree based on the specified level.
 * Grouping the leaf nodes can be useful for various algorithms or optimizations.
 *
 * @param level The level at which to group the leaf nodes. Higher levels result in coarser grouping.
 */
inline static void force_group_leaves(int level)
{
    int no = All.MaxPart;
    int previous_no = -1;
    int should_group_particles = 0;
    int maxNodes = MaxNodes;
    VERBOSE(0, "Begin collapse Nextnode after level %d", level);
    int max_level = -1;
    int maxPart = All.MaxPart;
    int grouped = 0;
    int groups = 0;
    while (no >= 0)
    {
        //    VERBOSE("no=%d",no);
        if (no < maxPart)
        {

            NextGroupedNode[no] = Nextnode[no]; /* if next node is a node and then  a particle again,then  this value
                                                   will be overwritten */

            if (previous_no != -1)
            {
                NextGroupedNode[previous_no] = no;
                grouped++;
            }

            if (should_group_particles == 1)
            {
                previous_no = no;
            }

            no = Nextnode[no];
            continue;
        }
        else
        {
            if (no >= maxPart + maxNodes)
            { /* pseudo particle */
                if (previous_no != -1)
                { /*we are laving a group of particles */
                    int myno = DomainNodeIndex[no - (maxPart + maxNodes)];
                    NextGroupedNode[previous_no] = myno;
                    /*VERBOSE(0,"pseudo particle on secondary phase, no=%d, level <-%d, level=%d
                       %X",no,myno,NODELEVEL(myno),Nodes[myno].u.d.bitflags); exit(0); */
                }
                should_group_particles = 0;
                previous_no = -1;

                no = Nextnode[no - maxNodes];
                continue;
            }

            int current_level = NODELEVEL(no);
            if (current_level > max_level)
            {
                max_level = current_level;
            }
            //      VERBOSE(0, "no=%d current_level=%d %X",no, level, Nodes[no].u.d.bitflags);
            if (current_level >= level && should_group_particles == 0)
            {
                groups++;
                //      VERBOSE(0, "group!");
                should_group_particles = 1;
                if (previous_no != -1)
                { /*we are laving a group of particles */
                    exit(1);
                }
                previous_no = -1;
            }

            /* we interrupt the grouping if we reach a top level, a "weird level" (bc I am not sure what is
             * BITFLAG_MULTIPLEPARTICLES ) or if we cross `level` */
            if ((Nodes[no].u.d.bitflags & (1 << BITFLAG_MULTIPLEPARTICLES)) ||
                (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) || (current_level < level))
            {
                if (previous_no != -1)
                { /*we are laving a group of particles */
                    //      VERBOSE(0, "ungroup & link %d -> %d!",previous_no,no);
                    NextGroupedNode[previous_no] = no;
                }
                else
                {
                    //      VERBOSE(0, "ungroup ");
                }
                should_group_particles = 0;
                previous_no = -1;
            }
            no = Nodes[no].u.d.nextnode; /*we open all nodes */
            continue;
        }
    }

    VERBOSE(0, "Done. MAx level = %d, groups=%d, grouped particles=%d", max_level, groups, grouped);
    return;
}
#endif // GROUP_LEAVES
#endif // FORCETREE_HPP
