/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file  simulation.h
 *
 *  \brief declares the main simulation class holding its data and principle modules
 */

#ifndef SIMULATION_H
#define SIMULATION_H

#include "gadgetconfig.h"

#include <mpi.h>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/constants.h"
#include "../data/dtypes.h"
#include "../data/lcparticles.h"
#include "../data/macros.h"
#include "../data/mmparticles.h"
#include "../data/mymalloc.h"
#include "../data/simparticles.h"
#ifdef BLACK_HOLES
#include "../black_holes/research_sidm_drag_driver.h"
#include "../black_holes/research_sidm_drag_context.h"
#include "../black_holes/research_sidm_wake_measurement.h"
#endif
#include "../domain/domain.h"
#include "../fmm/fmm.h"
#include "../fof/fof.h"
#include "../gravity/ewald.h"
#include "../gravity/grav_forcetest.h"
#include "../gravtree/gravtree.h"
#include "../gravtree/gwalk.h"
#include "../io/parameters.h"
#include "../io/restart.h"
#include "../io/test_io_bandwidth.h"
#include "../lightcone/lightcone.h"
#include "../logs/logs.h"
#include "../mergertree/mergertree.h"
#include "../mpi_utils/mpi_utils.h"
#include "../mpi_utils/setcomm.h"
#include "../ngbtree/ngbtree.h"
#include "../ngenic/ngenic.h"
#include "../pm/pm.h"
#include "../sph/sph.h"
#include "../system/pinning.h"

class sim : public pinning, public test_io_bandwidth
{
 public:
  sim(MPI_Comm comm) : setcomm(comm), test_io_bandwidth(comm) {}

  /* here come the main classes the code operates on */

  simparticles Sp{Communicator}; /* stores all the simulation particles of the simulation */

  domain<simparticles> Domain{Communicator, &Sp}; /* get an instance of a domain decomposition, operating on Sp */

  /* Note: GravTree/NgbTree  inherit their communicator and NTask/ThisTask from their associated domain object  */

#ifdef FMM
  fmm GravTree; /* get an instance of a gravitational tree */
#else
  gwalk GravTree; /* get an instance of a gravitational tree */
#endif

  sph NgbTree; /* get an instance of a neighbour search tree */

#ifdef PMGRID
  pm PM{Communicator};
#endif

#ifdef MERGERTREE
  mergertree MergerTree{Communicator, &Sp};
#endif

#ifdef COOLING
  coolsfr CoolSfr{Communicator};
#endif

#ifdef NGENIC
  ngenic Ngenic{Communicator, &Sp};
#endif

#ifdef LIGHTCONE
#ifdef LIGHTCONE_PARTICLES
  lcparticles Lp{Communicator}; /* stores all the buffered light-cone particles of the simulation */
#endif
#ifdef LIGHTCONE_MASSMAPS
  mmparticles Mp{Communicator}; /* stores buffered particles to construct projected mass maps on the lightcone */
#endif

#if defined(LIGHTCONE_PARTICLES) && defined(LIGHTCONE_MASSMAPS) /* both particles and massmaps */
  lightcone LightCone{Communicator, &Sp, &Lp, &Mp};
#else
#if defined(LIGHTCONE_PARTICLES)
  lightcone LightCone{Communicator, &Sp, &Lp};
#else
  lightcone LightCone{Communicator, &Sp, &Mp};
#endif
#endif
#endif  // end of LIGHTCONE

#ifdef FORCETEST
  gravtest GravTest{&Sp, &GravTree, &Domain};
#endif

  void rearrange_lightcone(int argc, char **argv);
  void rearrange_snapshot(int argc, char **argv);

  template <typename partset>
  void rearrange_generic(partset &Tp, int conenr, int firstnr, int lastnr);

  template <typename partset>
  void rearrange_fill_treetable(partset &Tp);

  template <typename partset>
  void rearrange_read(partset &Tp, int num, int conenr);

  template <typename partset>
  void rearrange_write(partset &Tp, int num, int conenr);

 private:
#ifdef PERIODIC
  void check_omega(void);
#endif

  void setup_smoothinglengths(void);
  void recreate_unique_ids(void);
  void test_id_uniqueness(void);
  int check_for_interruption_of_run(void);
  void set_non_standard_physics_for_current_time(void);
  void calculate_non_standard_physics_end_of_step(void);
  integertime find_next_outputtime(integertime ti_curr);
  double measure_cpu_performance(MPI_Comm Communicator);
  double measure_hyper_cube_speed(const char *tag, MPI_Comm Communicator);
  void measure_iprobe_performance(const char *tag);

#ifdef SECOND_ORDER_LPT_ICS
  double F1_Omega(double a);
  double F2_Omega(double a);
  int executed = 0;
  void second_order_ic_correction(void);
#endif

 public:
  void hello(void);
  void endrun(void);
  void begrun1(const char *parameterFile);
  void begrun2(void);
  void init(int RestartSnapNum);
  void run(void);
#ifdef RESEARCH_SIDM_REFERENCE
  // Native synchronized elastic source; distributed path has a DM octree
  // population independent of NgbTree gas particles and supports hydro gravity.
  void sidm_reference_source_step(void);
  void sidm_distributed_source_step(void);
#endif
#ifdef STELLAR_EVOLUTION
  void stellar_initialize_composition(void);
  void stellar_source_step(void);
#endif
#ifdef BLACK_HOLES
  void black_hole_source_step(void);
  void black_hole_dynamics_step(void);
  void black_hole_sidm_wake_schedule_step(void);
  research_sidm_wake_measurement::measurement black_hole_sidm_wake_measurement(std::uint64_t carrier_id,
      const std::vector<double> &radial_edges_proper_kpc,std::uint64_t maximum_particles,
      std::uint64_t maximum_work,std::uint64_t maximum_workspace_bytes);
  void black_hole_sidm_publish_wake_measurement(std::uint64_t carrier_id,const std::vector<double> &radial_edges_proper_kpc,
      std::uint64_t maximum_particles,std::uint64_t maximum_work,std::uint64_t maximum_workspace_bytes,
      std::uint64_t measurement_id,std::uint64_t maximum_publication_bytes,const std::string &actual_physics_signature);
  void black_hole_sidm_drag_initialize(const research_sidm_drag_response::state_response &response,
      const research_sidm_drag_driver::context &context);
  void black_hole_sidm_drag_restore(const research_sidm_drag_response::state_response &response,
      const research_sidm_drag_driver::context &context);
  void black_hole_sidm_drag_transition_response(const research_sidm_drag_response::state_response &old_response,
      const research_sidm_drag_response::state_response &new_response,const research_sidm_drag_driver::context &context);
  void black_hole_sidm_drag_transition_assets(const research_sidm_drag_context::launch_settings &old_settings,
      const research_sidm_drag_context::launch_settings &new_settings,const std::string &actual_native_physics_signature,
      double proper_boundary_seconds);
  void black_hole_sidm_drag_initialize_assets(const research_sidm_drag_context::launch_settings &settings,
      const std::string &actual_native_physics_signature,double proper_origin_seconds,bool restore);
  std::vector<research_sidm_drag_driver::diagnostics> black_hole_sidm_drag_advance_assets(
      const research_sidm_drag_context::launch_settings &settings,const std::string &actual_native_physics_signature,
      double native_begin,double proper_begin_seconds,double proper_end_seconds);
  std::vector<research_sidm_drag_driver::diagnostics> black_hole_sidm_drag_advance(
      const research_sidm_drag_response::state_response &response,const research_sidm_drag_driver::context &context,
      std::size_t maximum_subcycles,std::uint64_t maximum_schedule_workspace_bytes);
  research_sidm_drag_driver::diagnostics black_hole_sidm_drag_source_step(
      const research_sidm_drag_response::state_response &response,const research_sidm_drag_driver::context &context);
  std::vector<research_sidm_drag_schedule::interval> black_hole_sidm_drag_plan(double native_begin,double proper_origin_seconds,
      double maximum_step_seconds,std::size_t maximum_subcycles,std::uint64_t maximum_workspace_bytes,
      double absolute_clock_tolerance_seconds,double relative_clock_tolerance);
#endif
  void set_units(void);
  void create_snapshot_if_desired(void);
  void healthtest(void);
  void mpi_report_comittable_memory(void);
  long long report_comittable_memory(long long *MemTotal, long long *Committed_AS, long long *SwapTotal, long long *SwapFree);
  long long report_free_size_in_tmpfs(void);
  void do_gravity_step_second_half(void);
  void find_timesteps_and_do_gravity_step_first_half(void);
  void do_hydro_step_first_half(void);
  void do_hydro_step_second_half(void);
  void find_global_timesteps(void);
  void find_hydro_timesteps(void);
  void gravity(int timebin);
  void gravity_long_range_force(void);
  void gravity_comoving_factors(int timebin);
  void gravity_pm(int timebin);
  void gravity_set_oldacc(int timebin);

  void hydro_force(int step_indicator);
  void compute_grav_accelerations(int timebin);
#ifdef FORCETEST_TESTFORCELAW
  void gravity_forcetest_testforcelaw(void);
#endif

#ifdef EXTERNALGRAVITY
  void gravity_external(void);
#endif
};

#endif
