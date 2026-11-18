/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This header file contains the declaration of various functions.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef PROTO_HPP
#define PROTO_HPP

#ifndef ALLVARS_H
#include "../CodeBase/allvars.hpp"
#endif // ALLVARS_H
#include "../CodeBase/precision.hpp"
#include "../Gravity/forcetree.hpp"
#include "../CodeBase/domain.hpp"
#ifdef LT_STELLAREVOLUTION
#include "../CoolingSfr/Sfr_LT/lt.hpp"
#endif // LT_STELLAREVOLUTION
#ifdef COOLING
#include "../CoolingSfr/cooling.hpp"
#endif // COOLING
#ifdef GT_DUST
#include "../Hydro/dust.hpp"
#endif // GT_DUST

#include <mpi.h>
#include <map>

#ifdef DARKENERGY
double DarkEnergy_a(double);
double DarkEnergy_t(double);
#ifdef TIMEDEPDE
void fwa_init(void);
double INLINE_FUNC fwa(double);
double INLINE_FUNC get_wa(double);
#ifdef TIMEDEPGRAV
double INLINE_FUNC dHfak(double a);
double INLINE_FUNC dGfak(double a);
#endif // TIMEDEPGRAV
#endif // TIMEDEPDE
#endif // DARKENERGY

void write_header_to_balance_file(FILE *file);
// CodeBase
void set_non_standard_physics_for_current_time(void);
void calculate_non_standard_physics(void);
void compute_statistics(void);
void execute_resubmit_command(void);
void output_extra_log_messages(void);
int check_stop_condition(void);
void assign_unique_ids(void);
int compare_IDs(const void *a, const void *b);
void test_id_uniqueness(void);
void check_particles_info(const char *func, const char *file, int linenr);
void check_node_info(const char *func, const char *file, int linenr);
void put_symbol(double t0, double t1, char c);
void write_cpu_log(void);
int peano_compare_key(const void *a, const void *b);
void mysort_peano(void *b, size_t n, size_t s, int (*cmp)(const void *, const void *));
void find_next_sync_point_and_drift(void);
void close_outputfiles(void);
void compute_global_quantities_of_system(void);
void energy_statistics(void);
void output_log_messages(void);
integertime find_next_outputtime(integertime time);
void open_outputfiles(void);
int read_outputlist(char *fname);
void read_parameter_file(char *fname, char *tags[], void **addr, int *id, int nt, int *default_indicator,
                         std::multimap<std::string, ParamInfo> &parameter_pointers_map);
void print_current_parameters(std::multimap<std::string, ParamInfo> parameter_pointers_map);
void reorder_gas(void);
void reorder_particles(void);
void readjust_timebase(double TimeMax_old, double TimeMax_new);
void omp_mysort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void serial_sort_omp(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void omp_qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

#ifndef TWOPOINT_ON_OUTPUT
void twopoint(void);
// void twopoint_save(void);
#else // TWOPOINT_ON_OUTPUT
void twopoint(int);
// void twopoint_save(int);
#endif // TWOPOINT_ON_OUTPUT

#ifdef _OPENMP
void kd_reorder_particles2(void);
void kd_reorder_gas2(void);
#endif // _OPENMP

#ifdef ALTERNATIVE_PSORT
void init_sort_ID(MyIDType *data, int ndata);
#endif // ALTERNATIVE_PSORT

// System
void write_ps_files(void);
void output_compile_time_options(void);
void report_pinning(void);
void pin_to_core_set(void);
void get_core_set(void);
void calc_memory_checksum(void *base, size_t bytes);
void create_snapshot_if_desired(void);
void mpi_report_comittable_memory(void);
void report_task_list(void);
long long report_comittable_memory(long long *MemTotal, long long *Committed_AS, long long *SwapTotal,
                                   long long *SwapFree);
int MPI_Check_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbufreal,
                       int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm,
                       MPI_Status *status);
int MPI_Sizelimited_Sendrecv(void *sendbuf, size_t sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                             void *recvbuf, size_t recvcount, MPI_Datatype recvtype, int source, int recvtag,
                             MPI_Comm comm, MPI_Status *status);
int mpi_calculate_offsets(int *send_count, int *send_offset, int *recv_count, int *recv_offset, int send_identical);
#if defined(X86FIX) && defined(SOFTDOUBLEDOUBLE)
void x86_fix(void);
#endif // // defined(X86FIX) && defined(SOFTDOUBLEDOUBLE)
#ifndef DISABLE_MEMORY_MANAGER
void *mymalloc_fullinfo(const char *varname, size_t n, const char *func, const char *file, int linenr);
void *mymalloc_movable_fullinfo(void *ptr, const char *varname, size_t n, const char *func, const char *file, int line);
void *myrealloc_fullinfo(void *p, size_t n, const char *func, const char *file, int line);
void *myrealloc_movable_fullinfo(void *p, size_t n, const char *func, const char *file, int line);
void myfree_fullinfo(void *p, const char *func, const char *file, int line);
void myfree_movable_fullinfo(void *p, const char *func, const char *file, int line);
void report_detailed_memory_usage_of_largest_task(size_t *OldHighMarkBytes, const char *label, const char *func,
                                                  const char *file, int line);
#endif // DISABLE_MEMORY_MANAGER
void mymalloc_init(void);

void set_random_numbers(void);
void catch_abort(int sig);
void enable_core_dumps_and_fpu_exceptions(void);
void write_pid_file(void);

#ifdef PAUSE_RUN_TO_ATTACH_DEBUGGER
void pause_run_to_attach_debugger();
#endif // PAUSE_RUN_TO_ATTACH_DEBUGGER

// Integrator
void do_first_halfstep_kick(void);
void do_second_halfstep_kick(void);
void find_timesteps(void);
void drift_sph_extra_physics(int i, integertime tstart, integertime tend, double dt_entr);
void make_list_of_active_particles(void);
void set_predicted_sph_quantities_for_extra_physics(int i);
void do_sph_kick_for_extra_physics(int i, integertime tstart, integertime tend, double dt_entr);
#ifdef KD_RESTRICT_MAXVEL
void restrict_maxvel(MyFloat *Vel, int i);
#endif // KD_RESTRICT_MAXVEL
void check_particle_for_temperature_minimum(int i);
MyFloat limited_divvel(int i);
void do_the_kick(int i, integertime tstart, integertime tend, integertime tcurrent);
void move_particles(integertime time1);
void find_dt_displacement_constraint(double hfac);
// static double drift_integ(double a, void *param);
// static double gravkick_integ(double a, void *param);
// static double hydrokick_integ(double a, void *param);
void init_drift_table(void);
void do_box_wrapping(void);

#ifdef MAKEGLASS
void do_glass_making_step(void);
#endif // MAKEGLASS

void process_wake_ups_new(void);

#ifdef AXION_DM
double get_ax_hydrokick_factor(integertime time0, integertime time1);
static double ax_hydrokick_integ(double a, void *param);
#endif // AXION_DM

double get_magkick_factor(integertime time0, integertime time1);

// Gravity
void compute_densities(void);
void compute_grav_accelerations(void);
void get_disk_forces(double RR, double zz, double *f_R, double *f_z);
double get_disk_mass(double time);
void growing_disk_init(void);
void kspace_neutrinos_init(void);
void long_range_init_regionsize(void);
void pm_init_nonperiodic_allocate(void);
void pm_init_nonperiodic_free(void);
void pm_init_periodic_allocate(void);
void pm_init_periodic_free(void);
void mysort_pmperiodic(void *b, size_t n, size_t s, int (*cmp)(const void *, const void *));
void mysort_pmnonperiodic(void *b, size_t n, size_t s, int (*cmp)(const void *, const void *));
void set_injection_accel(void);
void on_beginning_of_timestep();
void on_end_of_timestep();
peanokey peano_hilbert_key(int x, int y, int z, int bits);
peanokey peano_and_morton_key(int x, int y, int z, int bits, peanokey *morton);
peanokey morton_key(int x, int y, int z, int bits);
void compute_potential(void);
void gravity_forcetest(void);
MyAtLeastDouble enclosed_mass(MyAtLeastDouble R);
void ensure_neighbours(void);
void ewald_corr(MyAtLeastDouble dx, MyAtLeastDouble dy, MyAtLeastDouble dz, MyAtLeastDouble *fper);
void ewald_force(int ii, int jj, int kk, MyAtLeastDouble x[3], MyAtLeastDouble force[3]);
void ewald_init(void);
MyAtLeastDouble ewald_psi(MyAtLeastDouble x[3]);

#ifdef PERIODIC
ACC_DECLARE_ROUTINE_BEFORE
double ewald_pot_corr(MyAtLeastDouble dx, MyAtLeastDouble dy, MyAtLeastDouble dz);
ACC_DECLARE_ROUTINE_AFTER
#endif // PERIODIC

void gravity_tree(void);
void set_softenings(void);
void setup_smoothinglengths(void);
void long_range_init(void);
#ifdef ADAPTGRAVSOFT
void ags_setup_smoothinglengths(void);
int ags_ngb_treefind_variable_threads(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                                      int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);
void ags_density(void);
int ags_density_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);
void *ags_density_evaluate_primary(void *p);
void *ags_density_evaluate_secondary(void *p);
int ags_density_isactive(int n);
void ags_force_update_hmax(void);
#endif // ADAPTGRAVSOFT



#ifdef EXTERNALHUBBLE
double hubble_function_external(double a);
#endif // EXTERNALHUBBLE

#ifdef DISKPOT
void gravity_tree_subfunc_diskpot(void);
#endif // DISKPOT

#ifdef ACC
void on_beginning_of_timestep();
void on_end_of_timestep();
#endif // ACC

#ifdef AXION_DM
void ax_density(_Bool cycle);
#endif // AXION_DM

// ImportFromL3 & Gravity
double PowerSpec_Efstathiou(double k);
void powerspec_save(void);
void foldonitself(int *typelist);
void dump_potential(void);
void read_and_dump_potential(void);
ACC_DECLARE_ROUTINE_BEFORE
int pmforce_is_particle_high_res(int type, MyLongDouble *pos);
ACC_DECLARE_ROUTINE_AFTER
int pmforce(int mode_periodic); //, int mode=0, int *typelist=NULL);
void pm_setup_nonperiodic_kernel(void);

#ifdef _OPENMP
void omp_qsort(void *a, size_t n, size_t es, int (*cmp)(const void *, const void *));
void omp_mysort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void serial_sort_omp(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
int floorLog2(unsigned int n);
#endif // _OPENMP

// CoolingSFR
int snIaheating_evaluate(int target, int mode, int *nexport, int *nSend_local);
void set_units_sfr(void);
void cl_CoolingOnly(void);
void cooling_and_starformation(void);
void set_units(void);

#ifdef _OPENMP
void build_active_particle_list(void);
#endif // _OPENMP


#ifdef UM_METAL_COOLING /* not necessarily UM_CHEMISTRY */
#ifdef UM_e_MET_IMPACTS
double um_metal_line_cooling(double, double, int, double);
#else // UM_e_MET_IMPACTS
double um_metal_line_cooling(double, double, int);
#endif // UM_e_MET_IMPACTS
#endif // UM_METAL_COOLING

#ifdef UM_CHECK
void Um_cooling_check(void);
#endif // UM_CHECK


// Hydro
void conduction_matrix_multiply(double *in, double *out);
double conduction_vector_multiply(double *a, double *b);

// OpenACC
// void dump_memory_table();
void free_memory(void);

// Gravity & IO
void second_order_ics(void);

// Gravity  & Hydro
void smoothed_values(void);
void hydro_force(void);

// Blackholes
void report_internal_model_for_blackhole();

#ifdef GM_STARDENSITY
void density_stars(void);
int density_evaluate_stars(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);
void *density_evaluate_primary_stars(void *p);
void *density_evaluate_secondary_stars(void *p);
int density_isactive_stars(int n);
int ngb_treefind_variable_threads_stars(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode,
                                        int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                        int *ngblist);
#ifdef GM_REPOSITION_ON_STARDENSITY_MAX
void blackhole_stardens_reposition(int);
#endif // GM_REPOSITION_ON_STARDENSITY_MAX
#endif // GM_STARDENSITY

// SUBFIND
int fof_find_dmparticles_evaluate(int target, int mode, int *nexport, int *nsend_local);
void fof_compute_group_properties(int gr, int start, int len);
void fof_fof(int num);
void fof_find_groups(void);
void fof_compile_catalogue(void);
void fof_save_groups(int num);
void fof_save_local_catalogue(int num);
double fof_periodic(double x);
double fof_periodic_wrap(double x);
void fof_find_nearest_dmparticle(void);
int fof_find_nearest_dmparticle_evaluate(int target, int mode, int *nexport, int *nsend_local);
void fof_make_black_holes(void);
void read_fof(int num);
int fof_compare_ID_list_ID(const void *a, const void *b);
void parallel_sort_special_P_GrNr_ID(void);

#ifdef PB_INITIALIZE_BH_PART
void Initialize_Black_Hole(void);
#endif // PB_INITIALIZE_BH_PART

#ifdef AR_GREEN_TREE_FOF
int fof_find_nearest_dmparticle_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox,
                                         int *i_startnode, int node, int i_node);
#else // AR_GREEN_TREE_FOF
int powerspec_turb_find_nearest_evaluate(int target, int mode, int *nexport, int *nsend_local);
#endif // AR_GREEN_TREE_FOF

#ifdef SUBFIND
#ifdef AR_GREEN_TREE_FOF
int ngb_treefind_fof_nearest_openmp(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                                    int *exportflag, int *exportnodecount, int *exportindex, int *ngblist,
                                    int MyFOF_PRIMARY_LINK_TYPES);
int ar_green_ngb_treefind_fof_variable_threads(MyLongDouble searchcenter[3], MyFloat *phsml, int *startnode, int mode,
                                               int *exportflag, int *exportnodecount, int *exportindex, int *ngblist,
                                               int *targets, int n_targets, int should_drift, int);
#endif // AR_GREEN_TREE_FOF
#ifdef DENSITY_SPLIT_BY_TYPE
void *subfind_density_evaluate_primary(int thread_id, char *todo, int j_in, int j_target);
#else // DENSITY_SPLIT_BY_TYPE
void *subfind_density_evaluate_primary(int thread_id, char *todo);
#endif // DENSITY_SPLIT_BY_TYPE
#ifdef DENSITY_SPLIT_BY_TYPE
void *subfind_density_evaluate_secondary(int thread_id, int j_in);
#else // DENSITY_SPLIT_BY_TYPE
void *subfind_density_evaluate_secondary(int thread_id);
#endif // DENSITY_SPLIT_BY_TYPE
#endif // SUBFIND

// Greentree
void mysort_domain(void *b, size_t n, size_t s);
void check_wind_creation(void);
void treat_outflowing_particles(void);
void allocate_memory(void);
#ifdef ACC_OPENMP
void allocateAPL();
#endif // ACC_OPENMP

void begrun(void);
void peano_hilbert_order(void);
void serial_sort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void domain_diffuse();

// mSIDM
#if (defined(mSIDM) || defined(mIDM)) && !defined(ADAPTGRAVSOFT)
namespace scatter
{
int pt1_hsml_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                      int i_node, void *additional_data);
void pt1_force_update_hmax(void);
void pt1_find_hsml();
} // namespace scatter
#endif // (defined(mSIDM) || defined(mIDM)) && !defined(ADAPTGRAVSOFT)

#if defined(mSIDM) || defined(mIDM)
// static double scattering_integ(double a, void *param);
double get_scattering_kick_factor(integertime time0, integertime time1);
#endif // mSIDM || mIDM

// Chemistry
#ifdef CHEMISTRY
int compute_abundances(int mode, int ithis, double a_start, double a_end);
#endif // CHEMISTRY

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
int InitChem(void);
int init_rad(double);
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

// Chemistry & Cooling
#ifdef UM_CHEMISTRY
/* main call of chemistry treatment */
int compute_abundances(int mode, int ithis, double a_start, double a_end, double *um_energy);
/* main prototypes of chemistry treatment */
double INLINE_FUNC Um_Compute_MeanMolecularWeight(int);
double Um_AbundanceRatios(double, double, double *, double *, double *, double, double, int);

#if (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
double Um_DoCooling(double, double, double, double *, int, int); // std SFR with um_chemistry
double Um_GetCoolingTime(double, double, double *, int); // std SFR + um_chem
#else // (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
#ifndef LT_METAL_COOLING_WAL
double Um_DoCooling(double, double, double, double *, double, int, int);
double Um_GetCoolingTime(double, double, double *, double, int);
#else // LT_METAL_COOLING_WAL
double Um_DoCooling(double, double, double, double *, double *, double, double, int, int);
double Um_GetCoolingTime(double, double, double *, double *, double, double, int);
#endif // LT_METAL_COOLING_WAL
double Um_AbundanceRatios(double, double, double *, double *, double *, double, double, int);
#endif // (defined(SFR) || defined(COOLING)) && !(defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))

/* additional prototypes of chemistry treatment */
double INLINE_FUNC Um_Compute_basenumden(double, int);
double INLINE_FUNC Um_Compute_n_den(double, int);
double INLINE_FUNC Um_Compute_sum_nuclei(int);
double INLINE_FUNC Um_Compute_mean_nden(int);

#if defined(UM_METAL_COOLING) && (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) &&                       \
    defined(UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)
void Um_met_corrections(int);
#else // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
      // (UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)
void Std_corrections(int);
#endif // defined (UM_METAL_COOLING) && (defined (LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL)) && defined
       // (UM_MET_IN_NONEQ_COOLING) && !defined(UM_DISABLE_MET_NONEQ_CORRECTIONS)

#if defined(UM_METAL_COOLING) && defined(LT_METAL_COOLING) && defined(UM_MET_IN_NONEQ_COOLING)
double INLINE_FUNC Um_Compute_invden(int i);
#endif // defined (UM_METAL_COOLING) && defined (LT_METAL_COOLING) && defined (UM_MET_IN_NONEQ_COOLING)

/* these fcts are not passed anywhere outside chemistry (see chemistry_noneq.cpp) */
// int energy_solver(int, int, double, double double, double, double *, double *, double *);
// int energy_solver(int, int, double, double, double, double, double *, double *)
// int species_solver(int, int, double, double, double, double, double *);
// int radiative_rates(int, double, double);
#endif // UM_CHEMISTRY

// various
void init(void);
void set_cosmo_factors_for_current_time(void);
double get_pressure(int i);
double get_gamma(int i);
double get_gamma_minus1(int i);
void powerspec(int flag, int *typeflag);
void calculate_power_spectra(int num, long long *ntot_type_all);
void conduction(void);
int conduction_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                        int i_node, void *data);
void *conduction_evaluate_primary(void *p);
void *conduction_evaluate_secondary(void *p);
void parallel_sort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void drift_particle(int i, integertime time1);
int get_timestep_bin(integertime ti_step);
void bubble(void);
double INLINE_FUNC hubble_function(double a);
int ngb_treefind_fof_nearest(MyLongDouble searchcenter[3], MyFloat hsml, int target, int *startnode, int mode,
                             int *nexport, int *nsend_local, int MyFOF_PRIMARY_LINK_TYPES);
void blackhole(void);
double get_random_number(MyIDType id);
int data_index_compare(const void *a, const void *b);
void quicksort(void *a, size_t n, size_t es, int (*cmp)(const void *, const void *));
void mysort_dataindex(void *b, size_t n, size_t s, int (*cmp)(const void *, const void *));
void *density_evaluate_secondary(void *p);
int hsml_isactive(int n);
int gradients_isactive(int n);
int density_isactive(int n);
size_t sizemax(size_t a, size_t b);
void reconstruct_timebins(void);
void density(void);
void find_hsml(void);
void compute_unified_gradients(void);
integertime get_timestep(int p, double *a, int flag);
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE *stream);

// create macro for my_fread in order to pass information about where function is called
size_t my_fread_impl(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *caller_func,
                     const char *caller_file, int caller_line);
#define my_fread(ptr, size, nmemb, stream) my_fread_impl(ptr, size, nmemb, stream, __func__, __FILE__, __LINE__)

void predict(double time);
void read_ic(char *fname);
void read_ic_cluster(char *fname);
void rearrange_particle_sequence(void);
void restart(int mod);
void restart_timer(int modus);
void run(void);
void savepositions(int num, int flag);
double second(void);
void total_execution_times(double time_diff, const std::string &func_name, bool final_output = false);
int total_counts(const std::string &counter_name, bool final_output = false);
void sumup_large_ints(int n, int *src, long long *res);
void sumup_longs(int n, long long *src, long long *res);
#ifdef ENLARGE_DYNAMIC_RANGE_IN_TIME
void minimum_large_ints(int n, long long *src, long long *res);
#endif // ENLARGE_DYNAMIC_RANGE_IN_TIME
void statistics(void);
void veldisp(void);
double get_hydrokick_factor(integertime time0, integertime time1);
double get_gravkick_factor(integertime time0, integertime time1);
double get_drift_factor(integertime time0, integertime time1);
double report_time(void);
double timediff(double t0, double t1);
void long_range_force(void);
void pm_init(void);
void pm_init_regionsize(void);

#if GADGET_HYDRO == HYDRO_SPH || GADGET_HYDRO == HYDRO_PESPH
void compute_hydro_accelerations(void);
int hydro_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                   int i_node, void *data);
#ifdef ACC_HYDRA
void *hydro_evaluate_primary(void *p, int OnGPU);
#else // ACC_HYDRA
void *hydro_evaluate_primary(void *p);
#endif // ACC_HYDRA
void *hydro_evaluate_secondary(void *p);

#ifdef AXION_DM
void ax_hydro_force(void);
int ax_hydro_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist);
void *ax_hydro_evaluate_primary(void *p);
void *ax_hydro_evaluate_secondary(void *p);
#endif // AXION_DM

#elif GADGET_HYDRO == HYDRO_MFM
void compute_fluxes(void);
void compute_limiters(void);

void mfm_fluxes(void);
void mfm_limiters(void);
void mfm_fluxes_to_code(int i, integertime tstart_step, integertime tend_step);

int flux_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node, int i_node,
                  void *additional_data);

int slope_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                   int i_node, void *additional_data);


void mfm_synchronize_entropy(int i);
#ifdef MAGNETIC
void mfm_synchronize_bfield(int i);
#endif // MAGNETIC
void mfm_update_mass(int i, MyFloat old_mass, MyFloat new_mass);
void mfm_add_mass(int i, MyFloat mass_change);
#ifdef DIVBCLEANING_DEDNER
MyFloat get_t_damp_phi_inv(int i);
MyFloat get_fastest_wave_speed();
#endif // DIVBCLEANING_DEDNER
MyFloat get_mfm_cell_size(int i);
#endif // GADGET_HYDRO

int hsml_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node, int i_node,
                  void *additional_data);

int gradients_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                       int i_node, void *additional_data);

int density_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                     int i_node, void *additional_data);

#ifdef ACC_DENSITY
void *density_evaluate_primary(void *p, int OnGPU);
#else // ACC_DENSITY
void *density_evaluate_primary(void *p);
#endif // ACC_DENSITY

#if !defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)
double get_starformation_rate(int i);
#else // !defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)
double get_starformation_rate(int i, float *Temperature, float *xclouds);
#endif // !defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)

#ifndef LT_STELLAREVOLUTION
void init_clouds(void);
void integrate_sfr(void);
#else // LT_STELLAREVOLUTION
void init_clouds(int, double, double, double, double, double *, double *);
void integrate_sfr(double, double, double, double, double, double);
#endif // LT_STELLAREVOLUTION

#ifndef ACC
/* on some DEC Alphas, the correct prototype for pow() is missing,
   even when math.h is included ! */
double pow(double, double) noexcept(true);
#endif // ACC

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
double dmax(double, double);
double dmin(double, double);
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#ifdef STATICBRANDT
inline double OmegaR(int i, int mode);
#endif // STATICBRANDT

#ifdef LT_STELLAREVOLUTION
// int get_Yset(int);
void ReadYields(int, int, int);
void reading_thresholds_for_thermal_instability(void);
double GetMetalLambda(double, double);
void init_SN(void);
void TestStellarEvolution();
void TestStarsEvolution(int, int);
void get_metals_checksum(int, long double *);
unsigned int evolve_SN(void);
double INLINE_FUNC get_NextChemTime(double, int, int *);
// double INLINE_FUNC get_cost_SE(int);
int INLINE_FUNC get_chemstep(int, int, double *, double);
// int INLINE_FUNC get_current_chem_bin(double, int);
int get_chemstep_bin(double, double, int *, int);
// int compare_steps(const void *, const void *);
int append_chemicallyactive_particles(unsigned int *);
void drop_chemicallyactive_particles(void);
#ifdef GL_CR_DUST
void read_dust();
#endif
void read_metals(void);
void init_clouds_cm(int, double *, double *, double, double, double, int, double *);

int load_SFs_IMFs(void);

ACC_DECLARE_ROUTINE_BEFORE
int get_SF_index(int, int *, int *);
ACC_DECLARE_ROUTINE_AFTER

int write_eff_model(int, int);
int read_eff_model(int, int);

void initialize_star_lifetimes(void);
double INLINE_FUNC get_age(double);

int calculate_effective_yields(double, double, int, int);
double calculate_FactorSN(double, double, void *);
double get_imf_params(double);

void write_metallicity_stat(void);
double INLINE_FUNC get_metalmass(float *);
double INLINE_FUNC get_metallicity(int, int);

ACC_DECLARE_ROUTINE_BEFORE
double INLINE_FUNC get_PhysDensThresh(int);
ACC_DECLARE_ROUTINE_AFTER

#ifdef SUBFIND
double INLINE_FUNC get_metallicity_subfind(int);
double get_starformation_rate_subfind(int i, float *Temperature, float *xclouds);
int get_SF_index_subfind(int, int *, int *);
#endif // SUBFIND
double INLINE_FUNC get_metallicity_solarunits(MyFloat);

// double INLINE_FUNC get_entropy(int);

ACC_DECLARE_ROUTINE_BEFORE
int INLINE_FUNC getindex(double *, int, int, double *, int *);
ACC_DECLARE_ROUTINE_AFTER

int INLINE_FUNC perturb(double *, double *);
int is_chemically_active(int);
#endif // LT_STELLAREVOLUTION

#ifdef JD_DPP
void compute_Dpp(int ipart);
#endif // JD_DPP


#ifdef PHIDOT
void phidot(void);
int phidot_treeevaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
#ifdef PERIODIC
int phidot_treeevaluate_ewald_correction(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex);
#endif // PERIODIC
void *phidot_primary_loop(void *p);
void *phidot_secondary_loop(void *p);
// double phidot_ewald_corr(double dx, double dy, double dz, double vx, double vy, double vz);
#endif // PHIDOT

#ifdef GL_CR_DUST
void evolve_dust(void);
void compute_ism_properties_effective_model(double, double, double, double, double *, double *, double *, double *,
                                            double *);
#endif // GL_CR_DUST

#ifdef SPHERICAL_BOUNDARY // enforce a boundary at a given radius, that either reflects or stops the particles
void apply_spherical_boundary(const int i);
#endif // SPHERICAL_BOUNDARY

#endif // PROTO_HPP
