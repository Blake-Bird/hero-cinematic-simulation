/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header to the substructure finding algorithm (Subfind).
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef SUBFIND_HPP
#define SUBFIND_HPP
#include "../IO/io.hpp"

typedef struct
{
    MyFloat r;
    MyFloat mass;
#ifdef LT_ADD_GAL_TO_SUB
    int index;
#ifdef DUSTATT
    MyFloat r2;
    MyFloat z;
#endif // DUSTATT
#endif // LT_ADD_GAL_TO_SUB
} sort_r2list;

#ifdef LT_ADD_GAL_TO_SUB
extern int bc_use_index[12];
extern char bc_name[12][2];
extern double CB07_used_lumdist, H0Filters, OMFilters, OLFiletrs;
#endif // LT_ADD_GAL_TO_SUB

void subfind_col_find_candidates(int num, int totgrouplen);

void read_hsml_files_voronoi(float *Values, int count, iofields blocknr, long long nskip);

void subfind_distlinklist_get_two_heads(long long ngb_index1, long long ngb_index2, long long *head,
                                        long long *head_attach);

int subfind_reshuffle_compare_ID_list_ID(const void *a, const void *b);
void read_hsml_table(void);
void get_hsml_file(long long nskip, int count, int *filenr, int *n_to_read, int *n_to_skip);
void read_subfind_ids(void);
int subfind_reshuffle_compare_ID_list_ID(const void *a, const void *b);

int subfind_distlinklist_get_tail_set_tail_increaselen(long long index, long long *tail, long long newtail);

void subfind_get_tasks(void);
void subfind_exchange(void);

void subfind(int num);
int subfind_overdensity_evaluate_dispersion(int target, int mode, int *exportflag, int *exportnodecount,
                                            int *exportindex, int *ngblist);
void *subfind_overdensity_evaluate_dispersion_primary(void *p);
void *subfind_overdensity_evaluate_dispersion_secondary(void *p);
int subfind_contamination_treefind(MyLongDouble *searchcenter, MyFloat hsml, int target, int *startnode, int mode,
                                   int *nexport, int *nsend_local, double *Mass);
int subfind_ovderdens_treefind_dispersion_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target,
                                                  int *startnode, int mode, int *exportflag, int *exportnodecount,
                                                  int *exportindex, int *ngblist);
int subfind_contamination_evaluate(int target, int mode, int *nexport, int *nsend_local);
void subfind_contamination(void);
int subfind_force_treeevaluate_potential(int target, int mode, int *nexport, int *nsend_local);
#ifdef DENSITY_SPLIT_BY_TYPE
void subfind_density(int j);
#else // DENSITY_SPLIT_BY_TYPE
void subfind_density(void);
#endif // DENSITY_SPLIT_BY_TYPE
void subfind_overdensity(void);
int subfind_overdensity_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
void *subfind_overdensity_evaluate_primary(int thread_id, char *todo, MyFloat *left, MyFloat *right);
void *subfind_overdensity_evaluate_secondary(void *p);
double subfind_ovderdens_treefind_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                          int mode, int *exportflag, int *exportnodecount, int *exportindex);
void subfind_save_densities(int num);
void subfind_save_local_densities(int num);
#ifdef DENSITY_SPLIT_BY_TYPE
void subfind_setup_smoothinglengths(int j);
int subfind_density_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                             int *ngblist, MyFloat *dist2list, int tp);
int subfind_ngb_treefind_linkpairs_threads(MyLongDouble *searchcenter, double hsml, int target, int *startnode,
                                           int mode, double *hmax, int *exportflag, int *exportnodecount,
                                           int *exportindex, int *ngblist, MyFloat *dist2list);
#else // DENSITY_SPLIT_BY_TYPE
void subfind_setup_smoothinglengths(void);
int subfind_density_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                             int *ngblist, MyFloat *dist2list);
#endif // DENSITY_SPLIT_BY_TYPE
void subfind_save_local_catalogue(int num);
void fill_subfind_io_buffer(void *IOBuffer, int bnr, int *offset, int nwrite);
void subfind_save_catalogues_collectively(int num);
void subfind_save_final(int num);
int subfind_linkngb_evaluate(int target, int mode, int *nexport, int *nsend_local);
int subfind_ngb_treefind_linkngb(MyLongDouble *searchcenter, double hsml, int target, int *startnode, int mode,
                                 double *hmax, int *nexport, int *nsend_local);
int subfind_ngb_treefind_linkngb_threads(MyLongDouble *searchcenter, double hsml, int target, int *startnode, int mode,
                                         double *hmax, int *exportflag, int *exportnodecount, int *exportindex,
                                         int *ngblist, MyFloat *dist2list);

int subfind_ngb_treefind_nearesttwo(MyLongDouble *searchcenter, double hsml, int target, int *startnode, int mode,
                                    double *hmax, int *nexport, int *nsend_local);

void subfind_distribute_particles(int mode);
void subfind_distribute_groups(int mode);

void subfind_unbind_independent_ones(int count);
void subfind_potential_compute(int num, struct unbind_data *d, int phase, double weakly_bound_limit);
void subfind_process_group_collectively(int num);
int subfind_col_unbind(struct unbind_data *d, int num, int *num_non_gas);
void subfind_determine_sub_halo_properties(struct unbind_data *d, int num, int collective,
                                           struct subgroup_properties *local_group, int subindex);
void subfind_find_linkngb(void);
int subfind_loctree_treebuild(int npart, struct unbind_data *mp);
void subfind_loctree_update_node_recursive(int no, int sib, int father);
double subfind_loctree_treeevaluate_potential(int target);
void subfind_loctree_copyExtent(void);
int subfind_locngb_treefind(MyLongDouble *xyz, int desngb, double hguess, int part_index);
void subfind_loctree_findExtent(int npart, struct unbind_data *mp);
int subfind_locngb_treefind_variable(MyLongDouble *searchcenter, double hguess, int part_index, int *numngb);
size_t subfind_loctree_treeallocate(int maxnodes, int maxpart);
void subfind_loctree_treefree(void);
void subfind_find_nearesttwo(void);
int subfind_nearesttwo_evaluate(int target, int mode, int *nexport, int *nsend_local);
int subfind_process_group_serial(int num, int gr, int offset);
int subfind_unbind(struct unbind_data *ud, int len, int *len_non_gas);
int subfind_compare_P_origindex(const void *a, const void *b);
int subfind_compare_P_GrNr_DM_Density(const void *a, const void *b);
int subfind_compare_P_GrNrGrNr(const void *a, const void *b);
int subfind_locngb_compare_key(const void *a, const void *b);
int subfind_compare_serial_candidates_subnr(const void *a, const void *b);
int subfind_compare_serial_candidates_rank(const void *a, const void *b);
int subfind_compare_candidates_boundlength(const void *a, const void *b);
int subfind_compare_candidates_nsubs(const void *a, const void *b);
int subfind_compare_serial_candidates_boundlength(const void *a, const void *b);
int subfind_compare_P_submark(const void *a, const void *b);
int subfind_compare_dist_rotcurve(const void *a, const void *b);
int subfind_compare_binding_energy(const void *a, const void *b);
int subfind_compare_densities(const void *a, const void *b);
int subfind_compare_candidates_rank(const void *a, const void *b);
int subfind_ngb_compare_dist(const void *a, const void *b);
int subfind_compare_hsml_data(const void *a, const void *b);
int subfind_compare_ID_list(const void *a, const void *b);
int subfind_compare_SubGroup_GrNr_SubNr(const void *a, const void *b);
int subfind_compare_candidates_subnr(const void *a, const void *b);
void subfind_poll_for_requests(void);
long long subfind_distlinklist_setrank_and_get_next(long long index, long long *rank);
long long subfind_distlinklist_get_rank(long long index);
void subfind_distlinklist_set_next(long long index, long long next);
void subfind_distlinklist_add_particle(long long index);
void subfind_distlinklist_add_bound_particles(long long index, int nsub);
void subfind_distlinklist_mark_particle(long long index, int target, int submark);
long long subfind_distlinklist_get_next(long long index);
long long subfind_distlinklist_get_head(long long index);
void subfind_distlinklist_set_headandnext(long long index, long long head, long long next);
void subfind_distlinklist_set_tailandlen(long long index, long long tail, int len);
void subfind_distlinklist_get_tailandlen(long long index, long long *tail, int *len);
void subfind_distlinklist_set_all(long long index, long long head, long long tail, int len, long long next);
long long subfind_distlinklist_set_head_get_next(long long index, long long head);

int get_sub_entrytype_of_block(siofields blocknr);
int get_values_per_sub(siofields blocknr);
void get_IO_Label_sub(siofields blocknr, char *label);
int get_datatype_in_sub(siofields blocknr);
int block_in_sub(siofields blocknr);

#ifdef LT_ADD_GAL_TO_SUB
void load_CB07_table(int IMFt, int num);
#ifndef OLD_FILTERS
void load_CB07_table_new(int IMFt);
#endif // OLD_FILTERS
void add_luminosity(int PART_index, int *FLAG_AGE, double *lum, double *lum_obs);
float ran1(long *idum);
#endif // LT_ADD_GAL_TO_SUB

#ifdef SO_BAR_INFO
int subfind_overdensity_evaluate_barinfo(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                         int *ngblist);
void *subfind_overdensity_evaluate_barinfo_primary(void *p);
void *subfind_overdensity_evaluate_barinfo_secondary(void *p);
#endif // SO_BAR_INFO

#ifdef KD_ALTERNATIVE_GROUP_SORT
int GroupToTask(int i, int totngroups);
void sort_groups();
void sort_sub_groups();
void sort_ids();
#endif // KD_ALTERNATIVE_GROUP_SORT

extern int Ncollective;
extern int MaxNsubgroups;
extern int Nsubgroups;
extern int TotNsubgroups;

#ifdef SUBFIND_COUNT_BIG_HALOS
extern int Nbiggroups;
extern int TotNbiggroups;
#endif // SUBFIND_COUNT_BIG_HALOS

extern int *grouptasks, ntaskspergroup;

extern int SubThisTask, SubNTask, color, ncolors;

extern MPI_Comm SUB_COMM;
extern MPI_Comm SWAP_COMM;

extern float gasdev(long *);

#ifdef LT_METAL_COOLING_WAL
extern double Redshift, DZ;
#endif // LT_METAL_COOLING_WAL

extern struct subgroup_properties
{
    int Len;
    int GrNr;
    int SubNr;
    int SubParent;
    unsigned int Offset;
    MyIDType SubMostBoundID;
    double Mass;
    double SubVelDisp;
    double SubVmax;
    double SubVmaxRad;
    double SubHalfMass;
    double SubStelAngMom;
    double SubBval;
    double SubStelHalfMass;
    double Pos[3];
    double CM[3];
    double Vel[3];
    double Spin[3];
    double MassTab[6];
#ifdef LT_ADD_GAL_TO_SUB
    double SubLum[LT_ADD_GAL_TO_SUB];
#ifdef DUSTATT
    double SubLumAtt[LT_ADD_GAL_TO_SUB];
    double DustProfile[DUSTATT];
#endif // DUSTATT
#ifdef OBSERVER_FRAME
    double SubLumObs[LT_ADD_GAL_TO_SUB];
#endif // OBSERVER_FRAME
#endif // LT_ADD_GAL_TO_SUB
    double SubMeanAge;
    double SubMeanZStar;
    double SubSFR;
    double SubHIMass;
    double SubMeanZGas;
    double SubMeanZColdGas;
    double SubSynchrotron;
} *SubGroup;

extern struct nearest_r2_data
{
    MyFloat dist[2];
} *R2Loc;

extern struct nearest_ngb_data
{
    long long index[2];
    int count;
} *NgbLoc;

extern struct r2data
{
    MyFloat r2;
    int index;
} *R2list;

extern MyFloat *Dist2list;

#define FAC_HSML pow(All.DesNumNgb / 20.0, 1.0 / 3.0)

#endif // SUBFIND_HPP
