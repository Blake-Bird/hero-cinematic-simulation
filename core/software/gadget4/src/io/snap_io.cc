/*******************************************************************************
 * \copyright   This file is part of the GADGET4 N-body/SPH code developed
 * \copyright   by Volker Springel. Copyright (C) 2014-2020 by Volker Springel
 * \copyright   (vspringel@mpa-garching.mpg.de) and all contributing authors.
 *******************************************************************************/

/*! \file snap_io.cc
 *
 *  \brief routines for I/O of snapshot files
 */

#include "gadgetconfig.h"

#include <errno.h>
#include <hdf5.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <type_traits>
#include <vector>

#include "../cooling_sfr/cooling.h"
#include "../data/allvars.h"
#include "../data/dtypes.h"
#include "../data/intposconvert.h"
#include "../data/mymalloc.h"
#include "../fof/fof.h"
#include "../gitversion/version.h"
#include "../gravtree/gravtree.h"
#include "../io/hdf5_util.h"
#include "../io/io.h"
#include "../io/snap_io.h"
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
#include "research_canonical_snapshot.h"
#endif
#include "../lightcone/lightcone.h"
#include "../logs/logs.h"
#include "../logs/timer.h"
#include "../main/main.h"
#include "../main/simulation.h"
#include "../mpi_utils/mpi_utils.h"
#include "../sort/peano.h"
#include "../src/pm/pm.h"
#include "../system/system.h"

/*!
 * \brief Function for field registering.
 *
 * For init_field arguments read the documentation of init_field.
 * Don't forget to add the new IO_FLAG to io_private.h
 */
void snap_io::init_basic(simparticles *Sp_ptr, int format)
{
#ifdef STARFORMATION
#ifdef GADGET2_HEADER
#error "Native SSP birth journal counters require the standard GADGET-4 HDF5 header"
#endif
  if(format != 3) Terminate("Native SSP birth journals require HDF5 snapshots and initial conditions (format 3)");
#endif
  Sp = Sp_ptr;

  this->N_IO_Fields  = 0;
  this->N_DataGroups = NTYPES + 1;  // the last data group is a tree table used only for storing/reading tree-reordered particle data
  this->header_size  = sizeof(header);
  this->header_buf   = &header;
  this->type_of_file = FILE_IS_SNAPSHOT;
  snprintf(this->info, MAXLEN_PATH, "SNAPSHOT: writing snapshot");

#ifdef PERIODIC
  init_field("PIVR","PeriodicImageVersion",MEM_UINT64,FILE_UINT64,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchPeriodicImages.version,NULL,ALL_TYPES,1,0,0,0,0,0,1.);
  init_field("PIST","PeriodicImageStatus",MEM_UINT64,FILE_UINT64,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchPeriodicImages.status,NULL,ALL_TYPES,1,0,0,0,0,0,1.);
  init_field("PIAT","PeriodicImageAnchorNativeTime",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchPeriodicImages.anchor_native_time,NULL,ALL_TYPES,1,0,0,0,0,0,1.);
  init_field("PIOF","PeriodicImageOffsets",MEM_INT64,FILE_INT64,READ_IF_PRESENT,3,A_P,
      &Sp->P[0].ResearchPeriodicImages.offsets[0],NULL,ALL_TYPES,1,0,0,0,0,0,1.);
#endif

#ifdef OUTPUT_COORDINATES_AS_INTEGERS
  init_field("IPOS", "IntCoordinates", MEM_MY_INTPOS_TYPE, FILE_MY_INTPOS_TYPE, READ_IF_PRESENT, 3, A_P, NULL, io_func_intpos,
             ALL_TYPES, 1, 1., -1., 1., 0., 0., All.UnitLength_in_cm * Sp->FacIntToCoord, true);
#else
  init_field("POS ", "Coordinates", MEM_MY_DOUBLE, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 3, A_P, NULL, io_func_pos, ALL_TYPES, 1, 1., -1.,
             1., 0., 0., All.UnitLength_in_cm, true);
#endif

#ifdef OUTPUT_VELOCITIES_IN_HALF_PRECISION
  init_field("VEL ", "Velocities", MEM_MY_FLOAT, FILE_HALF, READ_IF_PRESENT, 3, A_NONE, NULL, io_func_vel, ALL_TYPES, 1, 0.5, 0., 0.,
             0., 1., All.UnitVelocity_in_cm_per_s);
#else
  init_field("VEL ", "Velocities", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 3, A_NONE, NULL, io_func_vel, ALL_TYPES, 1, 0.5,
             0., 0., 0., 1., All.UnitVelocity_in_cm_per_s);
#endif

#ifdef RESEARCH_RESTART_STATE_CAPSULE
  init_field("RICF","ResearchIntegratorFloat64",MEM_DOUBLE,FILE_DOUBLE,SKIP_ON_READ,
      research_restart_integrator::floating_columns,A_NONE,NULL,io_func_integrator_floating,ALL_TYPES,0,0,0,0,0,0,0);
  init_field("RICI","ResearchIntegratorInt32Lanes",MEM_INT,FILE_INT,SKIP_ON_READ,
      research_restart_integrator::storage_lanes,A_NONE,NULL,io_func_integrator_integer,ALL_TYPES,0,0,0,0,0,0,0);
  init_field("RSCF","ResearchSourceStateFloat64",MEM_DOUBLE,FILE_DOUBLE,SKIP_ON_READ,
      research_restart::floating_columns,A_NONE,NULL,io_func_source_floating,ALL_TYPES,0,0,0,0,0,0,0);
  init_field("RSCI","ResearchSourceStateUInt64Lanes",MEM_INT,FILE_INT,SKIP_ON_READ,
      2*research_restart::integer_columns,A_NONE,NULL,io_func_source_integer,ALL_TYPES,0,0,0,0,0,0,0);
#define RESEARCH_STATE_FIELD(tag,name,memory,components,array,pointer,species) \
  static_assert(std::is_same<typename std::remove_reference<decltype(*(pointer))>::type, \
      typename std::conditional<memory==MEM_DOUBLE,double, \
        typename std::conditional<memory==MEM_UINT64,uint64_t,MyFloat>::type>::type>::value,"Restart diagnostic member type changed"); \
  init_field(tag,name,memory,memory==MEM_UINT64?FILE_UINT64:FILE_DOUBLE,SKIP_ON_READ,components,array,pointer,NULL,species,0,0,0,0,0,0,0);
#ifdef IMPROVED_VELOCITY_GRADIENTS
  static_assert(NUMDIMS==3,"Restart velocity-gradient field layout requires three dimensions");
#endif
#include "research_restart_hydro_fields.inc"
#undef RESEARCH_STATE_FIELD
#endif
  init_field("ID  ", "ParticleIDs", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P, NULL, io_func_id, ALL_TYPES, 0, 0, 0, 0,
             0, 0, 0, true);
#ifdef RESEARCH_SIDM_REFERENCE
  init_field("SICN","SIDMInteractionCount",MEM_UINT64,FILE_UINT64,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchSourceState.sidm_collision_count,NULL,2,0,0,0,0,0,0,1.);
  init_field("SEXP","SIDMExposure",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchSourceState.sidm_optical_depth,NULL,2,0,0,0,0,0,0,1.);
  init_field("SSUP","SIDMScatteringSupportLength",MEM_DOUBLE,FILE_DOUBLE,SKIP_ON_READ,1,A_NONE,NULL,
      io_func_sidm_support,2,1,0,0,1,0,0,3.0856775814913673e21);
  init_field("SRNG","SIDMOperatorSubstep",MEM_UINT64,FILE_UINT64,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchSourceState.sidm_random_counter,NULL,2,0,0,0,0,0,0,1.);
  init_field("SKCF","SIDMKernelCoarseSupportFraction",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchSourceState.sidm_kernel_coarse_support_fraction,NULL,2,0,0,0,0,0,0,1.);
  init_field("SOMT","SIDMOmittedScatteringUpperBound",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchSourceState.sidm_omitted_scattering_upper_bound,NULL,2,0,0,0,0,0,0,1.);
  init_field("SCGV","SIDMCoarseGuardVersion",MEM_INT64,FILE_INT64,READ_IF_PRESENT,1,A_P,
      &Sp->P[0].ResearchSourceState.sidm_coarse_guard_version,NULL,2,0,0,0,0,0,0,1.);
#endif

#ifndef LEAN
  init_field("MASS", "Masses", MEM_MY_DOUBLE, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_P, NULL, io_func_mass, MASS_BLOCK, 1, 0., -1.,
             0., 1., 0., All.UnitMass_in_g);
#endif

#ifdef INDIVIDUAL_GRAVITY_SOFTENING
  init_field("GSFT", "GravitySoftening", MEM_DOUBLE, FILE_DOUBLE, SKIP_ON_READ, 1, A_NONE, NULL,
             io_func_gravity_softening, ALL_TYPES, 1, 1., -1., 0., 1., 0., All.UnitLength_in_cm);
  init_field("GSCL", "GravitySofteningClass", MEM_INT, FILE_INT, SKIP_ON_READ, 1, A_NONE, NULL,
             io_func_gravity_softening_class, ALL_TYPES, 0, 0., 0., 0., 0., 0., 0.);
#endif

#ifdef BLACK_HOLES
  init_field("BARV","BHAccretionRoundoffVersion",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_accretion_roundoff_version,NULL,32,1,0,0,0,0,0,1.);
  init_field("BAPM","BHAccretionPendingMass",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_accretion_pending_mass,NULL,32,1,0,-1,0,1,0,All.UnitMass_in_g);
  init_field("BAPC","BHAccretionPendingCorrection",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_accretion_pending_correction,NULL,32,1,0,-1,0,1,0,All.UnitMass_in_g);
  init_field("BAGC","BHGravityMassCorrection",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_gravity_mass_correction,NULL,32,1,0,-1,0,1,0,All.UnitMass_in_g);
  init_field("BAPG","BHPhysicalMassCorrection",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_physical_mass_correction,NULL,32,1,0,-1,0,1,0,All.UnitMass_in_g);
  init_field("BHDE","BHDensityGRAM_CM3",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_density_g_cm3,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHCS","BHSoundSpeedCM_S",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sound_speed_cm_s,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHVR","BHRelativeSpeedCM_S",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_relative_speed_cm_s,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHJA","BHSpecificAngularMomentumCM2_S",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_specific_angular_momentum_cm2_s,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHRS","BHRotationalSuppression",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_rotational_suppression,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHED","BHEddingtonRatio",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_eddington_ratio,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHAE","BHAdvectedInternalEnergy",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_advected_internal_energy,NULL,32,1,0,-1,0,1,2,All.UnitEnergy_in_cgs);
  init_field("BHKE","BHKineticFeedbackEnergy",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_kinetic_feedback_energy,NULL,32,1,0,-1,0,1,2,All.UnitEnergy_in_cgs);
  init_field("BHMA", "BH_Mass", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P, &Sp->P[0].BHMass, NULL,
             32, 1, 0., -1., 0., 1., 0., All.UnitMass_in_g);
  init_field("BHMD", "BH_InflowRate", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P, &Sp->P[0].BHInflowRate, NULL,
             32, 1, 0., 0., -1., 1., 1., All.UnitMass_in_g / All.UnitTime_in_s);
  init_field("BHRM", "BH_RadiatedMass", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P, &Sp->P[0].BHCumulativeRadiatedMass, NULL,
             32, 1, 0., -1., 0., 1., 0., All.UnitMass_in_g);
  init_field("BHFE", "BH_FeedbackEnergy", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P, &Sp->P[0].BHCumulativeFeedbackEnergy, NULL,
             32, 1, 0., -1., 0., 1., 2., All.UnitEnergy_in_cgs);
  init_field("BSRV","BHSeedReservoirVersion",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_seed_reservoir_version,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BSAM","BHSeedAssembledMassGRAM",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_seed_assembled_mass_g,NULL,32,1,0,0,0,1,0,1.0);
  init_field("BSIE","BHSeedInternalEnergyERG",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_seed_internal_energy_erg,NULL,32,1,0,0,0,1,2,1.0);
  init_field("BSME","BHSeedMixingEnergyERG",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_seed_mixing_energy_erg,NULL,32,1,0,0,0,1,2,1.0);
  init_field("BSEM","BHSeedElementMassGRAM",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,11,A_P,&Sp->P[0].ResearchSourceState.bh_seed_element_mass_g,NULL,32,1,0,0,0,1,0,1.0);
  init_field("BSOA","BHSeedOrbitalAngularMomentumCGS",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_seed_orbital_angular_momentum_cgs,NULL,32,1,0,0,1,1,1,1.0);
  init_field("BHSV","BHStateVersion",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_state_version,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHSP","BHSpin",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_spin,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHCO","BHCompanionID",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_companion_id,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHPA","BHParentIDs",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,2,A_P,&Sp->P[0].ResearchSourceState.bh_parent_ids,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHGE","BHGeneration",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_generation,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHEC","BHEventCounter",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_event_counter,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHBC","BHBinaryCounter",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_counter,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHBA","BHBinarySemimajorAxisCM",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_a_cm,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHBE","BHBinaryEccentricity",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_eccentricity,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHBL","BHBinaryAxis",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_binary_axis,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHBR","BHBinaryRadial",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_binary_radial,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHBS","BHBinarySecondarySpin",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_binary_spin2,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHBP","BHBinaryPhase",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_phase,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHEG","BHBinaryGWOrbitalEnergyERG",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_gw_energy_erg,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHES","BHBinaryStellarScatteringEnergyERG",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_stellar_energy_erg,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHDW","BHFrictionHeatERG",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_friction_work_erg,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BHST","BHLastSourceNativeTime",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_last_source_time,NULL,32,1,0,0,0,0,0,All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);
  init_field("BHDT","BHLastDynamicsNativeTime",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_last_dynamics_time,NULL,32,1,0,0,0,0,0,All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);
  init_field("BSDV","BHSIDMDragVersion",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_version,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BSDC","BHSIDMDragCounter",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_counter,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BSCR","BHSIDMDragModelCRC32",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_model_crc32,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BSMS","BHSIDMDragModelSHA256",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,4,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_model_sha256,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BSSS","BHSIDMDragSignatureSHA256",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,4,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_signature_sha256,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BSNT","BHSIDMDragLastNativeTime",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_last_native_time,NULL,32,1,0,0,0,0,0,All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);
  init_field("BSPT","BHSIDMDragLastProperSeconds",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_last_proper_seconds,NULL,32,1,0,0,0,0,0,1.0);
  init_field("BSHE","BHSIDMDragHeatERG",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_heat_erg,NULL,32,1,0,0,1,0,2,1.0);
  init_field("BSRE","BHSIDMDragReferenceErrorCGS",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_reference_error_cm_s2,NULL,32,1,0,0,0,-1,2,1.0);
  init_field("BSER","BHSIDMDragHeldoutErrorCGS",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_heldout_error_cm_s2,NULL,32,1,0,0,0,-1,2,1.0);
  init_field("BSAC","BHSIDMDragLastAccelerationCGS",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_sidm_drag_last_acceleration_cm_s2,NULL,32,1,0,0,0,-1,2,1.0);
  init_field("BHBT","BHBinaryLastNativeTime",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_last_time,NULL,32,1,0,0,0,0,0,All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);
#endif

#ifdef BLACK_HOLES
  init_field("BHAR","BHBinaryAngularReservoirCGS",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_binary_angular_reservoir,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHGA","BHBinaryGWAngularMomentumCGS",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,3,A_P,&Sp->P[0].ResearchSourceState.bh_binary_gw_angular_momentum,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHBM","BHBinaryPreviousPhysicalMassGRAM",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,2,A_P,&Sp->P[0].ResearchSourceState.bh_binary_previous_physical_mass_g,NULL,32,1,0,0,0,0,0,1.);
  init_field("BHCE","BHBinaryConstraintReservoirERG",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.bh_binary_constraint_energy_erg,NULL,32,1,0,0,0,0,0,1.);
#endif

#ifdef SECOND_ORDER_LPT_ICS
  init_field("LPTM", "SecondOrderICMasses", MEM_FLOAT, FILE_NONE, READ_IF_PRESENT, 1, A_P, &Sp->P[0].OldAcc, NULL, ALL_TYPES, 0, 0, 0,
             0, 0, 0, 0);
#endif

  init_field("U   ", "InternalEnergy", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_NONE, NULL, io_func_u, GAS_ONLY, 1, 0.,
             0., 0., 0., 2., All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

  init_field("RHO ", "Density", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].Density, NULL, GAS_ONLY, 1,
             -3., 2., -3., 1., 0., All.UnitDensity_in_cgs);

  init_field("HSML", "SmoothingLength", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].Hsml, NULL, GAS_ONLY,
             1, 1., -1., 1., 0., 0., All.UnitLength_in_cm);

#ifdef OUTPUT_ACCELERATION
#ifdef OUTPUT_ACCELERATIONS_IN_HALF_PRECISION
  All.accel_normalize_fac = 10.0 * All.Hubble * (100.0 * 1.0e5 / All.UnitVelocity_in_cm_per_s);

  init_field("ACCE", "Acceleration", MEM_MY_FLOAT, FILE_HALF, SKIP_ON_READ, 3, A_NONE, 0, io_func_accel, ALL_TYPES, 1, -2.0, 1, -1, 0,
             2, All.accel_normalize_fac * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s / All.UnitLength_in_cm);
#else
  All.accel_normalize_fac = 1.0;

  init_field("ACCE", "Acceleration", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, SKIP_ON_READ, 3, A_NONE, 0, io_func_accel, ALL_TYPES, 1, -2.0, 1,
             -1, 0, 2, All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s / All.UnitLength_in_cm);
#endif

  /* Stored hydro acceleration becomes physical with cf_afac2, exactly as
   * get_timestep_hydro uses it: a^(2-3*GAMMA) in comoving integration. */
  init_field("HACC", "HydroAcceleration", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 3, A_SPHP, &Sp->SphP[0].HydroAccel, 0,
             GAS_ONLY, 1, 2.0 - 3.0 * GAMMA, 1., -1., 0., 2.,
             All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s / All.UnitLength_in_cm);
  init_field("HAET", "HydroForceEvaluationNativeTime", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_SPHP,
             &Sp->SphP[0].ResearchHydroForceNativeTime, 0, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.0);
  init_field("HAEV", "HydroForceEpochVersion", MEM_UINT64, FILE_UINT64, READ_IF_PRESENT, 1, A_SPHP,
             &Sp->SphP[0].ResearchHydroForceEpochVersion, 0, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.0);
#endif

#ifdef STARFORMATION

#if defined(IDS_32BIT) || defined(IDS_48BIT)
#error "Research SSP birth/parent/counter snapshots require IDS_64BIT"
#endif
  init_field("SIBM", "SSPInitialMass", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_initial_mass, NULL, STARS_ONLY,
             1, 0., -1., 0., 1., 0., All.UnitMass_in_g);
  init_field("SILM", "SSPInitialLockedMass", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_initial_locked_mass, NULL, STARS_ONLY,
             1, 0., -1., 0., 1., 0., All.UnitMass_in_g);
  init_field("SPRM", "SSPPromptReturnedMass", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_prompt_returned_mass, NULL, STARS_ONLY,
             1, 0., -1., 0., 1., 0., All.UnitMass_in_g);
  init_field("SCUM", "SSPCurrentMass", MEM_MY_DOUBLE, FILE_DOUBLE, SKIP_ON_READ, 1, A_P,
             NULL, io_func_mass, STARS_ONLY, 1, 0., -1., 0., 1., 0., All.UnitMass_in_g);
  init_field("SBTA", "SSPBirthScaleFactor", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_birth_scale_factor, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SBTN", "SSPBirthNativeTime", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_birth_time, NULL, STARS_ONLY, 1, 0, 0,
             All.ComovingIntegrationOn ? 0. : 1., 0., All.ComovingIntegrationOn ? 0. : -1.,
             All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);
  init_field("SBZM", "SSPBirthMetallicity", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_birth_metallicity, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SPID", "SSPParentGasID", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_parent_gas_id, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SSRV", "SSPBirthStateVersion", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_birth_state_version, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SBMD", "SSPBirthMode", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_birth_mode, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SRNG", "StellarRandomCounter", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_random_counter, NULL, GAS_AND_STARS, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SFR ", "StarFormationRate", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, All.RestartFlag == RST_FOF ? READ_IF_PRESENT : SKIP_ON_READ,
             1, A_NONE, 0, io_func_sfr, GAS_ONLY, 1, 0., 0., -1., 1., 1., SOLAR_MASS / SEC_PER_YEAR);
#ifdef STELLAR_EVOLUTION
  init_field("SPLV","SSPLiveMass",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.stellar_live_mass,NULL,STARS_ONLY,1,0,-1,0,1,0,All.UnitMass_in_g);
  init_field("SPRE","SSPRemnantMass",MEM_DOUBLE,FILE_DOUBLE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.stellar_remnant_mass,NULL,STARS_ONLY,1,0,-1,0,1,0,All.UnitMass_in_g);
  init_field("SPPV","SSPPopulationVersion",MEM_MY_ID_TYPE,FILE_MY_ID_TYPE,READ_IF_PRESENT,1,A_P,&Sp->P[0].ResearchSourceState.stellar_population_version,NULL,STARS_ONLY,1,0,0,0,0,0,1.);
  init_field("CHEM", "ElementMassFractions", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 11, A_P,
             &Sp->P[0].ResearchSourceState.element_mass_fraction, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("CHVR", "ChemicalStateVersion", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.chemical_state_version, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SBEF", "SSPBirthElementFractions", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 11, A_P,
             &Sp->P[0].ResearchSourceState.stellar_birth_element_fraction, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SBCV", "SSPBirthCompositionVersion", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_birth_composition_version, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SRMS", "SSPCumulativeReturnedMass", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_returned_mass, NULL, STARS_ONLY, 1, 0, -1, 0, 1, 0, All.UnitMass_in_g);
  init_field("SREL", "SSPCumulativeReturnedElements", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 11, A_P,
             &Sp->P[0].ResearchSourceState.stellar_returned_elements, NULL, STARS_ONLY, 1, 0, -1, 0, 1, 0, All.UnitMass_in_g);
  init_field("SREN", "SSPCumulativeInjectedEnergy", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_injected_energy, NULL, STARS_ONLY, 1, 0, -1, 0, 1, 2, All.UnitEnergy_in_cgs);
  init_field("SRCL", "SSPLastSourceNativeTime", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_last_source_time, NULL, STARS_ONLY, 1, 0, 0,
             All.ComovingIntegrationOn ? 0. : 1., 0, 0, All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);
  init_field("SREV", "SSPSourceEventCounter", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_event_counter, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("SRMD", "SSPSourceModelCRC32", MEM_MY_ID_TYPE, FILE_MY_ID_TYPE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.stellar_model_crc32, NULL, STARS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
#ifdef RESEARCH_GRACKLE
  init_field("GCHE", "GrackleAtomicSpeciesFractions", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 6, A_P,
             &Sp->P[0].ResearchSourceState.grackle_species_mass_fraction, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCMO", "GrackleMolecularSpeciesFractions", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 6, A_P,
             &Sp->P[0].ResearchSourceState.grackle_molecular_mass_fraction, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCGP", "GrackleGammaPressure", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.grackle_gamma_pressure, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCGS", "GrackleGammaSound", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.grackle_gamma_sound, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCTK", "GrackleTemperature", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.grackle_temperature_kelvin, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCMU", "GrackleMeanMolecularWeight", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.grackle_mean_molecular_weight, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCIC", "GrackleInverseCoolingTime", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.grackle_inverse_cooling_time_per_s, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCVR", "GrackleStateVersion", MEM_UINT64, FILE_UINT64, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.grackle_state_version, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCDT", "GrackleDataCRC32", MEM_UINT64, FILE_UINT64, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.grackle_data_crc32, NULL, GAS_ONLY, 1, 0, 0, 0, 0, 0, 1.);
  init_field("GCLT", "ChemistryLastUpdateNativeTime", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.chemical_last_update_time, NULL, GAS_ONLY, 1, 0, 0,
             All.ComovingIntegrationOn ? 0. : 1., 0, All.ComovingIntegrationOn ? 0. : -1., All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);
  init_field("GCER", "ChemistryCumulativeRadiatedEnergy", MEM_DOUBLE, FILE_DOUBLE, READ_IF_PRESENT, 1, A_P,
             &Sp->P[0].ResearchSourceState.chemical_radiated_energy, NULL, GAS_ONLY, 1, 0, -1, 0, 1, 2, All.UnitEnergy_in_cgs);
#endif
#endif

  init_field("AGE ", "StellarFormationTime", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_P, &Sp->P[0].StellarAge, NULL,
             AGE_BLOCK, /* scale factor in comoving runs; native L/V time in isolated runs */
             1, 0, 0, All.ComovingIntegrationOn ? 0. : 1., 0., All.ComovingIntegrationOn ? 0. : -1.,
             All.ComovingIntegrationOn ? 1. : All.UnitTime_in_s);

  init_field("Z   ", "Metallicity", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_NONE, 0, io_func_metallicity,
             Z_BLOCK, /* gas and star metal mass fraction */
             1, 0, 0, 0, 0, 0, 1.);
#endif

#if defined(PRESSURE_ENTROPY_SPH) && defined(OUTPUT_PRESSURE_SPH_DENSITY)
  init_field("PRHO", "PressureSphDensity", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].PressureSphDensity,
             NULL, GAS_ONLY, /* Pressure density */
             1, -3., 2., -3., 1., 0., All.UnitDensity_in_cgs);
#endif

#ifdef OUTPUT_PRESSURE
  init_field("PRES", "Pressure", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, SKIP_ON_READ, 1, A_NONE, 0, io_func_pressure,
             GAS_ONLY, /* particle pressure */
             1, -3 * GAMMA, 2., -3., 1., 2., All.UnitDensity_in_cgs * All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
#endif

#if defined(TIMEDEP_ART_VISC) && defined(OUTPUT_VISCOSITY_PARAMETER)
  init_field("ALP ", "ArtificialViscosityParameter", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].Alpha,
             NULL, GAS_ONLY, 1, -3., 2., -3., 1., 0., 1);
#endif

#ifdef OUTPUT_ENTROPY
  init_field("ENTR", "Entropy", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, SKIP_ON_READ, 1, A_SPHP, &Sp->SphP[0].Entropy, 0,
             GAS_ONLY, /* particle entropy */
             0, 0, 0, 0, 0, 0, 0);
#endif

#ifdef COOLING
  init_field("NE  ", "ElectronAbundance", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].Ne, 0,
             GAS_ONLY, /* electron abundance */
             0, 0, 0, 0, 0, 0, 0);
#endif

#ifdef OUTPUT_POTENTIAL
  init_field("POT ", "Potential", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_P, &Sp->P[0].Potential, 0,
             ALL_TYPES, /* potential */
             1, -1., 0., 0., 0., 2., All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);
#endif

#ifdef OUTPUT_CHANGEOFENTROPY
  init_field("ENDT", "RateOfChangeOfEntropy", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].DtEntropy, 0,
             GAS_ONLY, /* particle entropy change */
             0, 0, 0, 0, 0, 0, 0);
#endif

#ifdef OUTPUT_TIMESTEP
  init_field("TSTP", "TimeStep", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, SKIP_ON_READ, 1, A_NONE, 0, io_func_timestep,
             ALL_TYPES, /* time step */
             0, 0, 0, 0, 0, 0, 0);

  init_field("TSTH", "TimeStepHydro", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, SKIP_ON_READ, 1, A_NONE, 0, io_func_timestephydro,
             GAS_ONLY, /* hydro time step */
             0, 0, 0, 0, 0, 0, 0);
#endif

#ifdef OUTPUT_DIVVEL
  init_field("DIVV", "VelocityDivergence", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].DivVel, 0,
             GAS_ONLY, /* hydro velocity divergence */
             1, -2., 1., -1., 0., 1., All.UnitVelocity_in_cm_per_s / All.UnitLength_in_cm);
#endif

#ifdef OUTPUT_CURLVEL
  init_field("ROTV", "VelocityCurl", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].CurlVel, 0,
             GAS_ONLY, /* absolute value of rot v */
             1, -2., 1., -1., 0., 1., All.UnitVelocity_in_cm_per_s / All.UnitLength_in_cm);
#endif

#ifdef OUTPUT_VELOCITY_GRADIENT
#ifndef IMPROVED_VELOCITY_GRADIENTS
#error "The option OUTPUT_VELOCITY_GRADIENT requires IMPROVED_VELOCITY_GRADIENTS"
#endif
  init_field("GRAV", "VelocityGradient", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 9, A_SPHP, &Sp->SphP[0].dvel[0][0], 0,
             GAS_ONLY, 0, 0, 0, 0, 0, 0, 0);
#endif

#ifdef OUTPUT_COOLHEAT
  init_field("COHE", "CoolingHeatingEnergy", MEM_MY_FLOAT, FILE_MY_IO_FLOAT, READ_IF_PRESENT, 1, A_SPHP, &Sp->SphP[0].CoolHeat, 0,
             GAS_ONLY, 0, 0, 0, 0, 0, 0, 0);
#endif

#if defined(SUBFIND) && defined(SUBFIND_STORE_LOCAL_DENSITY)

  init_field("SFDE", "SubfindDensity", MEM_MY_FLOAT, All.RestartFlag != RST_CREATEICS ? FILE_MY_IO_FLOAT : FILE_NONE, SKIP_ON_READ, 1,
             A_PS, &Sp->PS[0].SubfindDensity, 0, ALL_TYPES, /* subfind density */
             1, -3., 2., -3., 1., 0., All.UnitDensity_in_cgs);

  init_field("SFHS", "SubfindHsml", MEM_MY_FLOAT, All.RestartFlag != RST_CREATEICS ? FILE_MY_IO_FLOAT : FILE_NONE, SKIP_ON_READ, 1,
             A_PS, &Sp->PS[0].SubfindHsml, 0, ALL_TYPES, /* subfind hsml */
             1, 1., -1., 1., 0., 0., All.UnitLength_in_cm);

  init_field("SFVD", "SubfindVelDisp", MEM_MY_FLOAT, All.RestartFlag != RST_CREATEICS ? FILE_MY_IO_FLOAT : FILE_NONE, SKIP_ON_READ, 1,
             A_PS, &Sp->PS[0].SubfindVelDisp, 0, ALL_TYPES, /* subfind velocity dispersion */
             1, 0., 0., 0., 0., 1., All.UnitVelocity_in_cm_per_s);
#endif

#if defined(GADGET2_HEADER) && defined(REARRANGE_OPTION) && defined(MERGERTREE)
  Terminate("GADGET2_HEADER does not work together with REARRANGE_OPTION\n");
#endif
}

#if defined(REARRANGE_OPTION) && defined(MERGERTREE)
void snap_io::init_extra(simparticles *Sp_ptr, mergertree *MergerTree_ptr)
{
  Sp         = Sp_ptr;
  MergerTree = MergerTree_ptr;

  init_field("MTRI", "TreeID", MEM_INT64, FILE_INT64, READ_IF_PRESENT, 1, A_P, &Sp->P[0].TreeID, NULL, ALL_TYPES, 0, 0, 0, 0, 0, 0, 0);

  init_field("MTRL", "ParticleCount", MEM_INT, FILE_INT, READ_IF_PRESENT, 1, A_TT, &MergerTree->TreeTable[0].HaloCount, NULL,
             TREETABLE, 0, 0, 0, 0, 0, 0, 0);
  init_field("MTRS", "ParticleFirst", MEM_INT64, FILE_INT64, READ_IF_PRESENT, 1, A_TT, &MergerTree->TreeTable[0].FirstHalo, NULL,
             TREETABLE, 0, 0, 0, 0, 0, 0, 0);
  init_field("MTRI", "TreeID", MEM_INT64, FILE_INT64, READ_IF_PRESENT, 1, A_TT, &MergerTree->TreeTable[0].TreeID, NULL, TREETABLE, 0,
             0, 0, 0, 0, 0, 0);
}
#endif

void snap_io::read_snapshot(int num, mysnaptype loc_snap_type)
{
  snap_type = loc_snap_type;

  char buf[MAXLEN_PATH_EXTRA];

  if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT)
    {
      if(All.NumFilesPerSnapshot > 1)
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s/snapdir_%03d/%s-prevmostboundonly_%03d", All.OutputDir, num, All.SnapshotFileBase, num);
      else
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s%s-prevmostboundonly_%03d", All.OutputDir, All.SnapshotFileBase, num);
    }
  else
    {
      if(All.NumFilesPerSnapshot > 1)
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s/snapdir_%03d/%s_%03d", All.OutputDir, num, All.SnapshotFileBase, num);
      else
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s%s_%03d", All.OutputDir, All.SnapshotFileBase, num);
    }

  read_ic(buf);
}

/*! \brief This function reads initial conditions that are in on of the default file formats
 * of Gadget.
 *
 * Snapshot files can be used as input files.  However, when a
 * snapshot file is used as input, not all the information in the header is
 * used: THE STARTING TIME NEEDS TO BE SET IN THE PARAMETERFILE.
 * Alternatively, the code can be started with All.RestartFlag 2, then snapshots
 * from the code can be used as initial conditions-files without having to
 * change the parameter file.  For gas particles, only the internal energy is
 * read, the density and mean molecular weight will be recomputed by the code.
 * When InitGasTemp>0 is given, the gas temperature will be initialized to this
 * value assuming a mean molecular weight either corresponding to complete
 * neutrality, or full ionization.
 *
 * \param fname file name of the ICs
 * \param readTypes readTypes is a bitfield that
 * determines what particle types to read, only if the bit
 * corresponding to a particle type is set, the corresponding data is
 * loaded, otherwise its particle number is set to zero. (This is
 * only implemented for HDF5 files.)
 */
void snap_io::read_ic(const char *fname)
{
  if(All.ICFormat < 1 || All.ICFormat > 4)
    Terminate("ICFormat = %d not supported.\n", All.ICFormat);

  TIMER_START(CPU_SNAPSHOT);

  double t0 = Logs.second();

  Sp->TotNumPart = 0;

  int num_files = find_files(fname, fname);

  reset_io_byte_count();

  /* we repeat reading the headers of the files two times. In the first iteration, only the
   * particle numbers ending up on each processor are assembled, followed by memory allocation.
   * In the second iteration, the data is actually read in.
   */
  for(int rep = 0; rep < 2; rep++)
    {
      Sp->NumPart = 0;
      Sp->NumGas  = 0;

      read_files_driver(fname, rep, num_files);

      /* now do the memory allocation */
      if(rep == 0)
        {
          int max_load, max_sphload;
          MPI_Allreduce(&Sp->NumPart, &max_load, 1, MPI_INT, MPI_MAX, Communicator);
          MPI_Allreduce(&Sp->NumGas, &max_sphload, 1, MPI_INT, MPI_MAX, Communicator);

#ifdef TILING
          max_load *= TILING * TILING * TILING;
          max_sphload *= TILING * TILING * TILING;
#endif

          Sp->MaxPart    = max_load / (1.0 - 2 * ALLOC_TOLERANCE);
          Sp->MaxPartSph = max_sphload / (1.0 - 2 * ALLOC_TOLERANCE);

          Sp->allocate_memory(); /* allocate particle storage */

#ifndef OUTPUT_COORDINATES_AS_INTEGERS
          Ptmp = (ptmp_data *)Mem.mymalloc_movable(&Ptmp, "Ptmp", Sp->NumPart * sizeof(ptmp_data));
#endif
        }
    }

  MPI_Barrier(Communicator);

  long long byte_count = get_io_byte_count(), byte_count_all;
  sumup_longs(1, &byte_count, &byte_count_all, Communicator);

  double t1 = Logs.second();

  mpi_printf("READIC: reading done. Took %g sec, total size %g MB, corresponds to effective I/O rate of %g MB/sec\n",
             Logs.timediff(t0, t1), byte_count_all / (1024.0 * 1024.0), byte_count_all / (1024.0 * 1024.0) / Logs.timediff(t0, t1));

  mpi_printf("\nREADIC: Total number of particles :  %lld\n\n", Sp->TotNumPart);

  snap_init_domain_mapping();

#ifndef OUTPUT_COORDINATES_AS_INTEGERS
  Mem.myfree(Ptmp);
#endif

#ifdef TILING

  MyIntPosType halfboxlen = ((MyIntPosType)1) << ((BITS_FOR_POSITIONS - 1));

  MyIntPosType len = (halfboxlen / TILING) + (halfboxlen / TILING);

  Sp->TotNumGas *= TILING * TILING * TILING;
  Sp->TotNumPart *= TILING * TILING * TILING;

  int add_numgas = Sp->NumGas * (TILING * TILING * TILING - 1);
  /* create a gap behind the existing gas particles where we will insert the new gas particles */
  memmove(static_cast<void *>(Sp->P + Sp->NumGas + add_numgas), static_cast<void *>(Sp->P + Sp->NumGas),
          (Sp->NumPart - Sp->NumGas) * sizeof(simparticles::pdata));

  int off = 0;

  for(int i = 0; i < TILING; i++)
    for(int j = 0; j < TILING; j++)
      for(int k = 0; k < TILING; k++)
        if(i != 0 || j != 0 || k != 0)
          for(int n = 0; n < Sp->NumGas; n++)
            {
              Sp->P[Sp->NumGas + off] = Sp->P[n];

              Sp->P[Sp->NumGas + off].IntPos[0] += i * len;
              Sp->P[Sp->NumGas + off].IntPos[1] += j * len;
              Sp->P[Sp->NumGas + off].IntPos[2] += k * len;

              off++;
            }

  if(off != add_numgas)
    Terminate("this should not happen");

  Sp->NumGas += add_numgas;
  Sp->NumPart += add_numgas;

  int add_numpart = (Sp->NumPart - Sp->NumGas) * (TILING * TILING * TILING - 1);

  off = 0;

  for(int i = 0; i < TILING; i++)
    for(int j = 0; j < TILING; j++)
      for(int k = 0; k < TILING; k++)
        if(i != 0 || j != 0 || k != 0)
          for(int n = Sp->NumGas; n < Sp->NumPart; n++)
            {
              Sp->P[Sp->NumPart + off] = Sp->P[n];

              Sp->P[Sp->NumPart + off].IntPos[0] += i * len;
              Sp->P[Sp->NumPart + off].IntPos[1] += j * len;
              Sp->P[Sp->NumPart + off].IntPos[2] += k * len;

              off++;
            }

  if(off != add_numpart)
    Terminate("this should not happen");

  Sp->NumPart += add_numpart;
#endif

#ifdef GADGET2_HEADER
#ifndef INITIAL_CONDITIONS_CONTAIN_ENTROPY
  if(header.flag_entropy_instead_u)
    Terminate("Initial condition file contains entropy, but INITIAL_CONDITIONS_CONTAIN_ENTROPY is not set\n");
#else
  if(!header.flag_entropy_instead_u)
    Terminate("Initial condition file contains uthermal, but INITIAL_CONDITIONS_CONTAIN_ENTROPY is set\n");
#endif
#endif

  TIMER_STOP(CPU_SNAPSHOT);
}

/*! This routine initializes the global domain mapping between integer coordinates and real space coordinates.
 *  If periodic is on, the extent is the box size. Otherwise we look at the maximum extent of the particles.
 */
void snap_io::snap_init_domain_mapping(void)
{
#ifdef PERIODIC

  Sp->RegionLen     = All.BoxSize;
  Sp->FacCoordToInt = pow(2.0, BITS_FOR_POSITIONS) / Sp->RegionLen;
  Sp->FacIntToCoord = Sp->RegionLen / pow(2.0, BITS_FOR_POSITIONS);

#if defined(NGENIC) && !defined(CREATE_GRID)
  if(All.RestartFlag == RST_BEGIN || All.RestartFlag == RST_CREATEICS)
    {
      // Make sure that the velocities are zero when a glass file is fed to IC creation
      mpi_printf("READIC: Setting velocities in glass file to zero.\n");
      for(int i = 0; i < Sp->NumPart; i++)
        for(int k = 0; k < 3; k++)
          Sp->P[i].Vel[k] = 0;
    }
#endif

#else

  double posmin[3], posmax[3];
  for(int k = 0; k < 3; k++)
    {
#ifdef EXTERNALGRAVITY
      posmin[k] = 0;  // make sure that origin is included in this case
      posmax[k] = 0;
#else
      posmin[k] = MAX_REAL_NUMBER;
      posmax[k] = -MAX_REAL_NUMBER;
#endif
    }

  for(int i = 0; i < Sp->NumPart; i++)
    for(int k = 0; k < 3; k++)
      {
        if(Ptmp[i].Pos[k] < posmin[k])
          posmin[k] = Ptmp[i].Pos[k];

        if(Ptmp[i].Pos[k] > posmax[k])
          posmax[k] = Ptmp[i].Pos[k];
      }

  double xyz[6] = {posmin[0], posmin[1], posmin[2], -posmax[0], -posmax[1], -posmax[2]};
  double xyz_glob[6];

  MPI_Allreduce(xyz, xyz_glob, 6, MPI_DOUBLE, MPI_MIN, Communicator);

  mpi_printf("READIC: Region covered with particles: (%g %g %g) -> (%g %g %g)\n", xyz_glob[0], xyz_glob[1], xyz_glob[2], -xyz_glob[3],
             -xyz_glob[4], -xyz_glob[5]);

  Sp->RegionLen = 0;
  for(int j = 0; j < 3; j++)
    if(-xyz_glob[j + 3] - xyz_glob[j] > Sp->RegionLen)
      Sp->RegionLen = -xyz_glob[j + 3] - xyz_glob[j];

  Sp->RegionLen *= 4.0;

  mpi_printf("READIC: Initial root node size: %g\n", Sp->RegionLen);

  for(int j = 0; j < 3; j++)
    {
      Sp->RegionCenter[j] = 0.5 * (xyz_glob[j] - xyz_glob[j + 3]);
      Sp->RegionCorner[j] = Sp->RegionCenter[j] - 0.5 * Sp->RegionLen;
    }

  Sp->FacCoordToInt = pow(2.0, BITS_FOR_POSITIONS) / Sp->RegionLen;
  Sp->FacIntToCoord = Sp->RegionLen / pow(2.0, BITS_FOR_POSITIONS);

#endif

#ifndef OUTPUT_COORDINATES_AS_INTEGERS
  for(int i = 0; i < Sp->NumPart; i++)
    Sp->pos_to_intpos(Ptmp[i].Pos, Sp->P[i].IntPos); /* converts floating point representation to integers */
#endif
}

int snap_io::get_type_of_element(int index)
{
  if(snap_type == NORMAL_SNAPSHOT)
    {
      return Sp->P[index].getType();
    }
  else if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT)
    {
      if(Sp->P[index].ID.is_previously_most_bound())
        return Sp->P[index].getType();
      else
        return -1;  // marks that particle has type that is not written out
    }
  else if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT_REORDERED)
    {
      return Sp->P[index].getType();
    }
  else
    Terminate("can't be");
}

void snap_io::set_type_of_element(int index, int type)
{
  if(type < NTYPES)
    Sp->P[index].setType(type);
}

void *snap_io::get_base_address_of_structure(enum arrays array, int index)
{
  switch(array)
    {
      case A_SPHP:
        return (void *)(Sp->SphP + index);
      case A_P:
        return (void *)(Sp->P + index);
      case A_PS:
        return (void *)(Sp->PS + index);
#ifdef LIGHTCONE_PARTICLES
      case A_LC:
        Terminate("a");  // return (void *) (Lp->P + index);
#endif
#ifdef LIGHTCONE_MASSMAPS
      case A_MM:
        Terminate("b");  // return (void *) (Lp->P + index);
#endif
#if defined(REARRANGE_OPTION) && defined(MERGERTREE)
      case A_TT:
        return (void *)(MergerTree->TreeTable + index);
#endif
      default:
        Terminate("we don't expect to get here");
    }

  return NULL;
}

/*! \brief Save snapshot to disk
 *
 * This function writes a snapshot of the particle distribution to one or
 * several files. If NumFilesPerSnapshot>1, the snapshot is distributed
 * into several files, which are written simultaneously. Each file contains
 * data from a group of processors of size roughly NTask/NumFilesPerSnapshot.
 * \param num the snapshot number
 */
void snap_io::write_snapshot(int num, mysnaptype loc_snap_type)
{
#ifdef DO_NOT_PRODUCE_BIG_OUTPUT
  if(snap_type != MOST_BOUND_PARTICLE_SNAPHOT)
    {
      mpi_printf("\nSNAPSHOT: We skip writing snapshot file #%d @ time %g\n", num, All.Time);
      return;
    }
#endif

  snap_type = loc_snap_type;

  TIMER_START(CPU_SNAPSHOT);

  mpi_printf("\nSNAPSHOT: writing snapshot file #%d @ time %g ... \n", num, All.Time);

  double t0 = Logs.second();
  reset_io_byte_count();

  All.set_cosmo_factors_for_current_time();

  int n_type[NTYPES];

  /* determine global and local particle numbers */
  for(int n = 0; n < NTYPES; n++)
    n_type[n] = 0;

  for(int n = 0; n < Sp->NumPart; n++)
    {
      int type = get_type_of_element(n);
      if(type >= 0)
        n_type[type]++;
    }

  sumup_large_ints(NTYPES, n_type, ntot_type_all, Communicator);

  if(All.NumFilesPerSnapshot > 1)
    {
      if(ThisTask == 0)
        {
          char buf[MAXLEN_PATH_EXTRA];
          snprintf(buf, MAXLEN_PATH_EXTRA, "%s/snapdir_%03d", All.OutputDir, num);
          mkdir(buf, 02755);
        }
      MPI_Barrier(Communicator);
    }

  char buf[MAXLEN_PATH_EXTRA];
  if(All.NumFilesPerSnapshot > 1)
    snprintf(buf, MAXLEN_PATH_EXTRA, "%s/snapdir_%03d/%s_%03d", All.OutputDir, num, All.SnapshotFileBase, num);
  else
    snprintf(buf, MAXLEN_PATH_EXTRA, "%s%s_%03d", All.OutputDir, All.SnapshotFileBase, num);

  if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT)
    {
      if(All.NumFilesPerSnapshot > 1)
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s/snapdir_%03d/%s-prevmostboundonly_%03d", All.OutputDir, num, All.SnapshotFileBase, num);
      else
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s%s-prevmostboundonly_%03d", All.OutputDir, All.SnapshotFileBase, num);
    }
  else if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT_REORDERED)
    {
      if(All.NumFilesPerSnapshot > 1)
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s/snapdir_%03d/%s-prevmostboundonly-treeorder_%03d", All.OutputDir, num,
                 All.SnapshotFileBase, num);
      else
        snprintf(buf, MAXLEN_PATH_EXTRA, "%s%s-prevmostboundonly-treeorder_%03d", All.OutputDir, All.SnapshotFileBase, num);
    }

  /* now write the files */
  write_multiple_files(buf, All.NumFilesPerSnapshot);

  long long byte_count = get_io_byte_count(), byte_count_all;
  sumup_longs(1, &byte_count, &byte_count_all, Communicator);

  double t1 = Logs.second();

  mpi_printf("SNAPSHOT: done with writing snapshot.  Took %g sec, total size %g MB, corresponds to effective I/O rate of %g MB/sec\n",
             Logs.timediff(t0, t1), byte_count_all / (1024.0 * 1024.0), byte_count_all / (1024.0 * 1024.0) / Logs.timediff(t0, t1));

  All.Ti_lastoutput = All.Ti_Current;

  TIMER_STOP(CPU_SNAPSHOT);
}

void snap_io::fill_file_header(int writeTask, int lastTask, long long *n_type, long long *ntot_type)
{
  /* determine global and local particle numbers */
  for(int n = 0; n < NTYPES + 1; n++)
    n_type[n] = 0;

  for(int n = 0; n < Sp->NumPart; n++)
    {
      int type = get_type_of_element(n);
      if(type >= 0)
        n_type[type]++;
    }

#if defined(REARRANGE_OPTION) && defined(MERGERTREE)
  if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT_REORDERED)
    n_type[NTYPES] = MergerTree->Ntrees;
#endif

  /* determine particle numbers of each type in file */
  if(ThisTask == writeTask)
    {
      for(int n = 0; n < NTYPES + 1; n++)
        ntot_type[n] = n_type[n];

      for(int task = writeTask + 1; task <= lastTask; task++)
        {
          long long nn[NTYPES + 1];
          MPI_Recv(&nn[0], NTYPES + 1, MPI_LONG_LONG, task, TAG_LOCALN, Communicator, MPI_STATUS_IGNORE);
          for(int n = 0; n < NTYPES + 1; n++)
            ntot_type[n] += nn[n];
        }

      for(int task = writeTask + 1; task <= lastTask; task++)
        MPI_Send(&ntot_type[0], NTYPES + 1, MPI_LONG_LONG, task, TAG_N, Communicator);
    }
  else
    {
      MPI_Send(&n_type[0], NTYPES + 1, MPI_LONG_LONG, writeTask, TAG_LOCALN, Communicator);
      MPI_Recv(&ntot_type[0], NTYPES + 1, MPI_LONG_LONG, writeTask, TAG_N, Communicator, MPI_STATUS_IGNORE);
    }

  /* fill file header */
#ifdef BLACK_HOLES
  header.bh_event_count=All.BHEventCount;header.bh_event_commit_count=All.BHEventCommitCount;header.bh_event_journal_schema=1;
#endif
#ifdef STARFORMATION
  header.stellar_birth_event_count = All.StellarBirthEventCount;
  header.stellar_birth_commit_count = All.StellarBirthCommitCount;
  header.stellar_birth_journal_schema = 1;
#ifdef STELLAR_EVOLUTION
  header.stellar_return_event_count=All.StellarReturnEventCount;
  header.stellar_return_commit_count=All.StellarReturnCommitCount;
  header.stellar_return_journal_schema=1;
#endif
#endif

  for(int n = 0; n < NTYPES; n++)
    {
      header.npart[n]      = ntot_type[n];
      header.npartTotal[n] = ntot_type_all[n];
    }

#if defined(MERGERTREE) && !defined(GADGET2_HEADER)
  if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT_REORDERED)
    {
      header.Ntrees = ntot_type[NTYPES];
#if defined(REARRANGE_OPTION) && defined(MERGERTREE)
      header.TotNtrees = MergerTree->TotNtrees;
#endif
    }
  else
    {
      header.Ntrees    = 0;
      header.TotNtrees = 0;
    }
#endif

  for(int n = 0; n < NTYPES; n++)
    header.mass[n] = All.MassTable[n];

  header.time = All.Time;

  if(All.ComovingIntegrationOn)
    header.redshift = 1.0 / All.Time - 1;
  else
    header.redshift = 0;

  header.num_files = All.NumFilesPerSnapshot;
  header.BoxSize   = All.BoxSize;

#ifdef GADGET2_HEADER
  for(int n = 0; n < NTYPES; n++)
    header.npartTotalLowWord[n] = ntot_type_all[n];

  header.flag_sfr      = 0;
  header.flag_feedback = 0;
  header.flag_cooling  = 0;

#ifdef COOLING
  header.flag_cooling = 1;
#endif

#ifdef STARFORMATION
  header.flag_sfr      = 1;
  header.flag_feedback = 1;
#endif
  header.Omega0      = All.Omega0;
  header.OmegaLambda = All.OmegaLambda;

#if !(defined(REARRANGE_OPTION) && defined(MERGERTREE))
  header.HubbleParam = All.HubbleParam;
  header.Hubble      = All.Hubble;
#endif

#ifdef OUTPUT_IN_DOUBLEPRECISION
  header.flag_doubleprecision = 1;
#else
  header.flag_doubleprecision = 0;
#endif
#endif
}

void snap_io::read_file_header(const char *fname, int filenr, int readTask, int lastTask, long long *n_type, long long *ntot_type,
                               int *nstart)
{
  n_type[NTYPES]    = 0;
  ntot_type[NTYPES] = 0;
#ifdef STARFORMATION
  if(stellar_birth_read_schema == UINT64_MAX) stellar_birth_read_schema = header.stellar_birth_journal_schema;
  else if(stellar_birth_read_schema != header.stellar_birth_journal_schema)
    Terminate("Snapshot shards mix SSP journal schemas");
  if(header.stellar_birth_journal_schema == 0 && All.RestartFlag == RST_STARTFROMSNAP)
    Terminate("Snapshot restart lacks authoritative SSP birth journal counters");
  if(Sp->TotNumPart == 0)
    {
      All.StellarBirthEventCount = header.stellar_birth_event_count;
      All.StellarBirthCommitCount = header.stellar_birth_commit_count;
    }
  else if(All.StellarBirthEventCount != header.stellar_birth_event_count ||
          All.StellarBirthCommitCount != header.stellar_birth_commit_count)
    Terminate("Snapshot shards disagree about SSP birth journal counters");
#ifdef STELLAR_EVOLUTION
  if(stellar_return_read_schema==UINT64_MAX)stellar_return_read_schema=header.stellar_return_journal_schema;
  else if(stellar_return_read_schema!=header.stellar_return_journal_schema)Terminate("Snapshot shards mix stellar return schemas");
  if(!header.stellar_return_journal_schema && All.RestartFlag==RST_STARTFROMSNAP)
    Terminate("Snapshot restart lacks authoritative stellar return counters");
  if(Sp->TotNumPart==0) {
    All.StellarReturnEventCount=header.stellar_return_event_count;
    All.StellarReturnCommitCount=header.stellar_return_commit_count;
  } else if(All.StellarReturnEventCount!=header.stellar_return_event_count || All.StellarReturnCommitCount!=header.stellar_return_commit_count)
    Terminate("Snapshot shards disagree about stellar return counters");
#endif
#endif

#ifdef BLACK_HOLES
  if(bh_event_read_schema==UINT64_MAX)bh_event_read_schema=header.bh_event_journal_schema;
  else if(bh_event_read_schema!=header.bh_event_journal_schema)Terminate("Snapshot shards mix BH journal schemas");
  if(!header.bh_event_journal_schema && All.RestartFlag==RST_STARTFROMSNAP)Terminate("Snapshot restart lacks authoritative BH event counters");
  if(Sp->TotNumPart==0){All.BHEventCount=header.bh_event_count;All.BHEventCommitCount=header.bh_event_commit_count;}
  else if(All.BHEventCount!=header.bh_event_count || All.BHEventCommitCount!=header.bh_event_commit_count)Terminate("Snapshot shards disagree about BH event counters");
#endif

#ifdef GADGET2_HEADER
  for(int i = 0; i < NTYPES_HEADER; i++)
    if(header.npartTotalLowWord[i] > 0)
      header.npartTotal[i] = header.npartTotalLowWord[i];  // + (((long long)header.npartTotalHighWord[i]) << 32);
#endif

  if(Sp->TotNumPart == 0)
    {
      if(header.num_files <= 1)
        for(int i = 0; i < NTYPES; i++)
          header.npartTotal[i] = header.npart[i];

      Sp->TotNumGas  = header.npartTotal[0];
      Sp->TotNumPart = 0;

      for(int i = 0; i < NTYPES; i++)
        Sp->TotNumPart += header.npartTotal[i];

#ifdef GADGET2_HEADER
#if defined(SECOND_ORDER_LPT_ICS)
      if(header.flag_ic_info == FLAG_SECOND_ORDER_ICS)
        {
          mpi_printf("READIC:  Second order ICs detected. Will complete them first before starting run.\n");
          All.LptScalingfactor = header.lpt_scalingfactor;
        }
      else
        Terminate("READIC:  No second order ICs detected even though you activated SECOND_ORDER_LPT_ICS.\n");
#else
      if(header.flag_ic_info == FLAG_SECOND_ORDER_ICS)
        Terminate("Detected second order ICs but SECOND_ORDER_LPT_ICS is not enabled.\n");
#endif
#endif

#ifdef GENERATE_GAS_IN_ICS
      if(All.RestartFlag == RST_BEGIN)
        {
#ifdef SPLIT_PARTICLE_TYPE
          for(int i = 0; i < NTYPES; i++)
            if((1 << i) & (SPLIT_PARTICLE_TYPE))
              {
                Sp->TotNumGas += header.npartTotal[i];
                Sp->TotNumPart += header.npartTotal[i];
              }
#else
          Sp->TotNumGas += header.npartTotal[1];
          Sp->TotNumPart += header.npartTotal[1];
#endif
        }
#endif

      for(int i = 0; i < NTYPES; i++)
        All.MassTable[i] = header.mass[i];

      if(All.RestartFlag == RST_BEGIN || All.RestartFlag == RST_RESUME || All.RestartFlag == RST_CREATEICS)
        All.Time = All.TimeBegin;
      else
        All.Time = All.TimeBegin = header.time;

      All.set_cosmo_factors_for_current_time();
    }

  if(ThisTask == readTask)
    {
      if(filenr == 0 && nstart == NULL)
        {
          mpi_printf(
              "\nREADIC: filenr=%d, '%s' contains:\n"
              "READIC: Type 0 (gas):   %8lld  (tot=%15lld) masstab= %g\n",
              filenr, fname, (long long)header.npart[0], (long long)header.npartTotal[0], All.MassTable[0]);

          for(int type = 1; type < NTYPES; type++)
            {
              mpi_printf("READIC: Type %d:         %8lld  (tot=%15lld) masstab= %g\n", type, (long long)header.npart[type],
                         (long long)header.npartTotal[type], All.MassTable[type]);
            }
          mpi_printf("\n");
        }
    }

  /* to collect the gas particles all at the beginning (in case several
     snapshot files are read on the current CPU) we move the collisionless
     particles such that a gap of the right size is created */

  long long nall = 0;
  for(int type = 0; type < NTYPES; type++)
    {
      ntot_type[type] = header.npart[type];

      long long n_in_file = header.npart[type];
      int ntask           = lastTask - readTask + 1;
      int n_for_this_task = n_in_file / ntask;
      if((ThisTask - readTask) < (n_in_file % ntask))
        n_for_this_task++;

      n_type[type] = n_for_this_task;

      nall += n_for_this_task;
    }

#if defined(MERGERTREE) && !defined(GADGET2_HEADER)
  if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT_REORDERED)
    ntot_type[NTYPES] = header.Ntrees;
#endif

  if(nstart)
    {
      memmove(static_cast<void *>(&Sp->P[Sp->NumGas + nall]), static_cast<void *>(&Sp->P[Sp->NumGas]),
              (Sp->NumPart - Sp->NumGas) * sizeof(particle_data));
#ifndef OUTPUT_COORDINATES_AS_INTEGERS
      memmove(&Ptmp[Sp->NumGas + nall], &Ptmp[Sp->NumGas], (Sp->NumPart - Sp->NumGas) * sizeof(ptmp_data));
#endif
      *nstart = Sp->NumGas;
    }
}

/*! \brief Write the fields contained in the header group of the HDF5 snapshot file
 *
 *  This function stores the fields of the structure io_header as attributes belonging
 *  to the header group of the HDF5 file.
 *
 *  \param handle contains a reference to the header group
 */
void snap_io::write_header_fields(hid_t handle)
{
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
  const uint64_t canonical_schema=1;
  write_scalar_attribute(handle,"Research_sidm_canonical_control_schema",&canonical_schema,H5T_NATIVE_UINT64);
  write_scalar_attribute(handle,"Research_sidm_canonical_initial_scale_factor",&All.SIDMCanonicalInitialScaleFactor,H5T_NATIVE_DOUBLE);
  write_string_attribute(handle,"Research_sidm_canonical_clock_SHA256",All.SIDMCanonicalClockSHA256);
  write_string_attribute(handle,"Research_sidm_physical_model_hash",All.SIDMAngularModelHash);
  write_string_attribute(handle,"Research_sidm_original_physical_state_SHA256",All.SIDMCanonicalPhysicalStateSHA256);
  write_string_attribute(handle,"Research_sidm_position_frame","canonical_x=r/a");
  write_string_attribute(handle,"Research_sidm_velocity_frame","canonical_u=a*v_pec");
  write_string_attribute(handle,"Research_sidm_clock_frame","tau_integral_dt_over_a_squared");
#endif
#include "research_restart_global_write.inc"
#ifdef BLACK_HOLES
  write_scalar_attribute(handle,"BlackHoleEventCount",&header.bh_event_count,H5T_NATIVE_UINT64);
  write_scalar_attribute(handle,"BlackHoleEventCommitCount",&header.bh_event_commit_count,H5T_NATIVE_UINT64);
  write_scalar_attribute(handle,"BlackHoleEventJournalSchema",&header.bh_event_journal_schema,H5T_NATIVE_UINT64);
#endif
#ifdef GADGET2_HEADER
  write_vector_attribute(handle, "NumPart_ThisFile", header.npart, H5T_NATIVE_UINT, NTYPES);
#else
  write_vector_attribute(handle, "NumPart_ThisFile", header.npart, H5T_NATIVE_UINT64, NTYPES);
#endif
  write_vector_attribute(handle, "NumPart_Total", header.npartTotal, H5T_NATIVE_UINT64, NTYPES);
  write_vector_attribute(handle, "MassTable", header.mass, H5T_NATIVE_DOUBLE, NTYPES);
  write_scalar_attribute(handle, "Time", &header.time, H5T_NATIVE_DOUBLE);
#ifdef RESEARCH_RESTART_STATE_CAPSULE
  write_string_attribute(handle,"ResearchSourceCapsuleLayoutSHA256",research_restart::layout_sha256);
  write_string_attribute(handle,"ResearchIntegratorLayoutSHA256",research_restart_integrator::layout_sha256);
  write_string_attribute(handle,"ResearchIntegratorActiveFloatColumns",research_restart_integrator::active_float_columns);
  const int coordinate_bits=BITS_FOR_POSITIONS;
  write_scalar_attribute(handle,"ResearchIntegratorCoordinateBits",&coordinate_bits,H5T_NATIVE_INT);
  write_string_attribute(handle,"ResearchHydroParticleLayoutSHA256","5df4032125ecc04e62cd3f9225001f9dcddda9d2cdf58b985793ff8003ace016");
  static const char active_state_fields[]=
#define RESEARCH_STATE_FIELD(tag,name,memory,components,array,pointer,species) name ";"
#include "research_restart_hydro_fields.inc"
#undef RESEARCH_STATE_FIELD
      "";
  write_string_attribute(handle,"ResearchHydroParticleActiveFields",active_state_fields);
#endif
#ifdef STARFORMATION
  write_scalar_attribute(handle, "StellarBirthEventCount", &header.stellar_birth_event_count, H5T_NATIVE_UINT64);
  write_scalar_attribute(handle, "StellarBirthCommitCount", &header.stellar_birth_commit_count, H5T_NATIVE_UINT64);
  write_scalar_attribute(handle, "StellarBirthJournalSchema", &header.stellar_birth_journal_schema, H5T_NATIVE_UINT64);
#ifdef STELLAR_EVOLUTION
  write_scalar_attribute(handle,"StellarReturnEventCount",&header.stellar_return_event_count,H5T_NATIVE_UINT64);
  write_scalar_attribute(handle,"StellarReturnCommitCount",&header.stellar_return_commit_count,H5T_NATIVE_UINT64);
  write_scalar_attribute(handle,"StellarReturnJournalSchema",&header.stellar_return_journal_schema,H5T_NATIVE_UINT64);
#endif
#endif
  write_scalar_attribute(handle, "Redshift", &header.redshift, H5T_NATIVE_DOUBLE);
  write_scalar_attribute(handle, "BoxSize", &header.BoxSize, H5T_NATIVE_DOUBLE);
  write_scalar_attribute(handle, "Omega0", &All.Omega0, H5T_NATIVE_DOUBLE);
  write_scalar_attribute(handle, "OmegaBaryon", &All.OmegaBaryon, H5T_NATIVE_DOUBLE);
  write_scalar_attribute(handle, "OmegaLambda", &All.OmegaLambda, H5T_NATIVE_DOUBLE);
  write_scalar_attribute(handle, "OmegaRadiation", &All.OmegaRadiation, H5T_NATIVE_DOUBLE);
  write_scalar_attribute(handle, "HubbleParam", &All.HubbleParam, H5T_NATIVE_DOUBLE);
  write_scalar_attribute(handle, "NumFilesPerSnapshot", &header.num_files, H5T_NATIVE_INT);
  write_string_attribute(handle, "Git_commit", GIT_COMMIT);
  write_string_attribute(handle, "Git_date", GIT_DATE);

#if defined(REARRANGE_OPTION) && defined(MERGERTREE)
  if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT_REORDERED)
    {
      write_scalar_attribute(handle, "Ntrees_ThisFile", &header.Ntrees, H5T_NATIVE_UINT64);
      write_scalar_attribute(handle, "Ntrees_Total", &header.TotNtrees, H5T_NATIVE_UINT64);
    }
#endif
}

/*! \brief This function reads the snapshot header in case of hdf5 files (i.e. format 3)
 *
 * \param fname file name of the snapshot as given in the parameter file
 */
void snap_io::read_header_fields(const char *fname)
{
  for(int i = 0; i < NTYPES; i++)
    {
      header.npart[i]      = 0;
      header.npartTotal[i] = 0;
      header.mass[i]       = 0;
    }

  int ntypes = NTYPES;

  hid_t hdf5_file = my_H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t handle    = my_H5Gopen(hdf5_file, "/Header");
  const htri_t canonical_marker=H5Aexists(handle,"Research_sidm_canonical_control_schema");
  if(canonical_marker<0)Terminate("Cannot inspect SIDM snapshot coordinate-frame marker");
#ifdef RESEARCH_SIDM_CANONICAL_CONTROL
  try {research_sidm::require_canonical_snapshot(handle,All.SIDMCanonicalClockSHA256,All.SIDMAngularModelHash,
      All.SIDMCanonicalPhysicalStateSHA256,All.SIDMCanonicalInitialScaleFactor);}
  catch(const std::exception &error){Terminate("Canonical SIDM snapshot binding failed: %s",error.what());}
#else
  if(canonical_marker)Terminate("Canonical SIDM IC/snapshot requires its mapped native control build");
#endif
#ifdef STARFORMATION
  header.stellar_birth_event_count = header.stellar_birth_commit_count = header.stellar_birth_journal_schema = 0;
  const char *journal_names[] = {"StellarBirthEventCount", "StellarBirthCommitCount", "StellarBirthJournalSchema"};
  int journal_present = 0;
  for(const char *name : journal_names)
    {
      const htri_t exists = H5Aexists(handle, name);
      if(exists < 0) Terminate("Cannot inspect SSP journal header attribute");
      if(exists)
        {
          ++journal_present;
          hid_t attr = my_H5Aopen_name(handle, name), type = H5Aget_type(attr), space = H5Aget_space(attr);
          if(H5Tget_class(type) != H5T_INTEGER || H5Tget_sign(type) != H5T_SGN_NONE || H5Tget_size(type) != 8 ||
             H5Sget_simple_extent_npoints(space) != 1)
            Terminate("SSP journal header needs exact scalar uint64 attributes");
          H5Tclose(type); H5Sclose(space); my_H5Aclose(attr, name);
        }
    }
  if(journal_present && journal_present != 3) Terminate("Partial SSP birth journal header");
  if(journal_present)
    {
      read_scalar_attribute(handle, journal_names[0], &header.stellar_birth_event_count, H5T_NATIVE_UINT64);
      read_scalar_attribute(handle, journal_names[1], &header.stellar_birth_commit_count, H5T_NATIVE_UINT64);
      read_scalar_attribute(handle, journal_names[2], &header.stellar_birth_journal_schema, H5T_NATIVE_UINT64);
      if(header.stellar_birth_journal_schema != 1 || header.stellar_birth_commit_count > header.stellar_birth_event_count)
        Terminate("Unsupported or inconsistent SSP journal header counters");
    }
#endif

#ifdef STELLAR_EVOLUTION
  header.stellar_return_event_count=header.stellar_return_commit_count=header.stellar_return_journal_schema=0;
  const char *return_names[]={"StellarReturnEventCount","StellarReturnCommitCount","StellarReturnJournalSchema"};
  uint64_t *return_values[]={&header.stellar_return_event_count,&header.stellar_return_commit_count,&header.stellar_return_journal_schema};
  int return_present=0;
  for(int i=0;i<3;i++) {
    htri_t exists=H5Aexists(handle,return_names[i]);if(exists<0)Terminate("Cannot inspect stellar return header");
    if(exists) {
      return_present++;
      hid_t attr=my_H5Aopen_name(handle,return_names[i]),type=H5Aget_type(attr),space=H5Aget_space(attr);
      if(H5Tget_class(type)!=H5T_INTEGER || H5Tget_sign(type)!=H5T_SGN_NONE || H5Tget_size(type)!=8 || H5Sget_simple_extent_type(space)!=H5S_SCALAR)
        Terminate("Stellar return header needs exact scalar uint64 attributes");
      H5Tclose(type);H5Sclose(space);my_H5Aclose(attr,return_names[i]);
      read_scalar_attribute(handle,return_names[i],return_values[i],H5T_NATIVE_UINT64);
    }
  }
  if(return_present && return_present!=3)Terminate("Partial stellar return header");
  if(return_present && (header.stellar_return_journal_schema!=1 || header.stellar_return_commit_count>header.stellar_return_event_count))
    Terminate("Invalid stellar return header schema/counters");
#endif

#ifdef BLACK_HOLES
  header.bh_event_count=header.bh_event_commit_count=header.bh_event_journal_schema=0;
  const char *bh_names[]={"BlackHoleEventCount","BlackHoleEventCommitCount","BlackHoleEventJournalSchema"};
  uint64_t *bh_values[]={&header.bh_event_count,&header.bh_event_commit_count,&header.bh_event_journal_schema};int bh_present=0;
  for(int i=0;i<3;i++){htri_t exists=H5Aexists(handle,bh_names[i]);if(exists<0)Terminate("Cannot inspect BH event header");if(exists){bh_present++;
    hid_t attr=my_H5Aopen_name(handle,bh_names[i]),type=H5Aget_type(attr),space=H5Aget_space(attr);
    if(H5Tget_class(type)!=H5T_INTEGER || H5Tget_sign(type)!=H5T_SGN_NONE || H5Tget_size(type)!=8 || H5Sget_simple_extent_type(space)!=H5S_SCALAR)Terminate("BH header requires scalar uint64 counters");
    H5Tclose(type);H5Sclose(space);my_H5Aclose(attr,bh_names[i]);read_scalar_attribute(handle,bh_names[i],bh_values[i],H5T_NATIVE_UINT64);}}
  if(bh_present && (bh_present!=3 || header.bh_event_journal_schema!=1 || header.bh_event_commit_count>header.bh_event_count))Terminate("Invalid/partial BH journal header");
#endif

  /* check if the file in question actually has this number of types */
  hid_t hdf5_attribute = my_H5Aopen_name(handle, "NumPart_ThisFile");
  hid_t space          = H5Aget_space(hdf5_attribute);
  hsize_t dims, len;
  H5Sget_simple_extent_dims(space, &dims, &len);
  H5Sclose(space);
  if(len != (size_t)ntypes)
    Terminate("Length of NumPart_ThisFile attribute (%d) does not match NTYPES(ICS) (%d)", (int)len, ntypes);
  my_H5Aclose(hdf5_attribute, "NumPart_ThisFile");

  /* now read the header fields */

#ifdef GADGET2_HEADER
  read_vector_attribute(handle, "NumPart_ThisFile", header.npart, H5T_NATIVE_UINT, ntypes);
#else
  read_vector_attribute(handle, "NumPart_ThisFile", header.npart, H5T_NATIVE_UINT64, ntypes);
#endif

  read_vector_attribute(handle, "NumPart_Total", header.npartTotal, H5T_NATIVE_UINT64, ntypes);

#if defined(MERGERTREE) && !defined(GADGET2_HEADER)
  if(snap_type == MOST_BOUND_PARTICLE_SNAPHOT_REORDERED)
    {
      read_scalar_attribute(handle, "Ntrees_ThisFile", &header.Ntrees, H5T_NATIVE_UINT64);
      read_scalar_attribute(handle, "Ntrees_Total", &header.TotNtrees, H5T_NATIVE_UINT64);
    }
#endif

  read_vector_attribute(handle, "MassTable", header.mass, H5T_NATIVE_DOUBLE, ntypes);
  read_scalar_attribute(handle, "Time", &header.time, H5T_NATIVE_DOUBLE);
  read_scalar_attribute(handle, "Redshift", &header.redshift, H5T_NATIVE_DOUBLE);
  read_scalar_attribute(handle, "BoxSize", &header.BoxSize, H5T_NATIVE_DOUBLE);
  read_scalar_attribute(handle, "NumFilesPerSnapshot", &header.num_files, H5T_NATIVE_INT);

#if defined(GADGET2_HEADER) && defined(SECOND_ORDER_LPT_ICS)
  read_scalar_attribute(handle, "LptScalingfactor", &header.lpt_scalingfactor, H5T_NATIVE_FLOAT);
#endif

  // Snapshot continuation must transport every selected physical source
  // field. READ_IF_PRESENT is appropriate for fresh ICs, never for restart.
  if(All.RestartFlag==RST_STARTFROMSNAP) {
    auto require_state=[&](int species,const char *name){
      if(header.npart[species]==0)return;
      char path[160];snprintf(path,sizeof(path),"/PartType%d/%s",species,name);
      if(H5Lexists(hdf5_file,path,H5P_DEFAULT)<=0)Terminate("Snapshot restart lacks required source field %s",path);
      hid_t dataset=H5Dopen2(hdf5_file,path,H5P_DEFAULT),space=H5Dget_space(dataset);
      hsize_t dims[2]={};int rank=H5Sget_simple_extent_dims(space,dims,NULL);
      if(rank<1 || rank>2 || dims[0]!=(hsize_t)header.npart[species])Terminate("Snapshot restart source extent disagrees with particle census: %s",path);
      int width=registered_dataset_width(name);
      if(width<1 || (width==1 ? rank!=1 : rank!=2 || dims[1]!=(hsize_t)width))Terminate("Snapshot restart source width differs from native registration: %s",path);
      hid_t dtype=H5Dget_type(dataset);
      size_t name_length=strlen(name);
      bool identity=(name_length>=2&&strcmp(name+name_length-2,"ID")==0)||(name_length>=3&&strcmp(name+name_length-3,"IDs")==0);
      bool signed_integer=strcmp(name,"PeriodicImageOffsets")==0;
      bool integer=strstr(name,"Version") || strstr(name,"Counter") || strstr(name,"CRC32") || strstr(name,"SHA256") || identity || strcmp(name,"BHGeneration")==0 || strcmp(name,"SSPBirthMode")==0 || strcmp(name,"PeriodicImageStatus")==0;
      if(signed_integer ? H5Tget_class(dtype)!=H5T_INTEGER || H5Tget_sign(dtype)!=H5T_SGN_2 || H5Tget_size(dtype)!=8 : integer ? H5Tget_class(dtype)!=H5T_INTEGER || H5Tget_sign(dtype)!=H5T_SGN_NONE || H5Tget_size(dtype)!=8 : H5Tget_class(dtype)!=H5T_FLOAT || H5Tget_size(dtype)!=8)Terminate("Snapshot restart source precision/type is invalid: %s",path);
      H5Tclose(dtype);H5Sclose(space);H5Dclose(dataset);
    };
#ifdef PERIODIC
    const char *image_fields[]={"PeriodicImageVersion","PeriodicImageStatus","PeriodicImageAnchorNativeTime","PeriodicImageOffsets"};
    for(int species=0;species<NTYPES;species++)for(const char *name:image_fields)require_state(species,name);
#endif
#ifdef BLACK_HOLES
    const char *bh_fields[]={"BH_Mass","BH_RadiatedMass","BH_FeedbackEnergy","BHStateVersion","BHSpin","BHCompanionID","BHParentIDs","BHGeneration","BHEventCounter","BHBinaryCounter","BHBinarySemimajorAxisCM","BHBinaryEccentricity","BHBinaryAxis","BHBinaryRadial","BHBinarySecondarySpin","BHBinaryPhase","BHBinaryGWOrbitalEnergyERG","BHBinaryStellarScatteringEnergyERG","BHFrictionHeatERG","BHLastSourceNativeTime","BHLastDynamicsNativeTime","BHBinaryLastNativeTime","BHBinaryPreviousPhysicalMassGRAM","BHBinaryConstraintReservoirERG","BHBinaryAngularReservoirCGS","BHBinaryGWAngularMomentumCGS","BHAdvectedInternalEnergy","BHKineticFeedbackEnergy"};
    for(const char *name:bh_fields)require_state(5,name);
    const char *drag_fields[]={"BHSIDMDragVersion","BHSIDMDragCounter","BHSIDMDragModelCRC32","BHSIDMDragModelSHA256","BHSIDMDragSignatureSHA256","BHSIDMDragLastNativeTime","BHSIDMDragLastProperSeconds","BHSIDMDragHeatERG","BHSIDMDragReferenceErrorCGS","BHSIDMDragHeldoutErrorCGS","BHSIDMDragLastAccelerationCGS"};
    for(const char *name:drag_fields)require_state(5,name);
#endif
#ifdef STARFORMATION
    const char *birth_fields[]={"SSPInitialMass","SSPInitialLockedMass","SSPPromptReturnedMass","SSPBirthScaleFactor","SSPBirthNativeTime","SSPBirthMetallicity","SSPParentGasID","SSPBirthStateVersion","SSPBirthMode","StellarRandomCounter"};
    for(const char *name:birth_fields)require_state(4,name);
#ifdef STELLAR_EVOLUTION
    const char *stellar_fields[]={"SSPLiveMass","SSPRemnantMass","SSPPopulationVersion","ElementMassFractions","ChemicalStateVersion","SSPBirthElementFractions","SSPBirthCompositionVersion","SSPCumulativeReturnedMass","SSPCumulativeReturnedElements","SSPCumulativeInjectedEnergy","SSPLastSourceNativeTime","SSPSourceEventCounter","SSPSourceModelCRC32"};
    for(const char *name:stellar_fields)require_state(4,name);
    require_state(0,"ElementMassFractions");require_state(0,"ChemicalStateVersion");
#ifdef RESEARCH_GRACKLE
    const char *chemistry_fields[]={"GrackleAtomicSpeciesFractions","GrackleStateVersion","GrackleDataCRC32","ChemistryLastUpdateNativeTime","ChemistryCumulativeRadiatedEnergy"};
    for(const char *name:chemistry_fields)require_state(0,name);
#endif
#endif
#endif
  }
  my_H5Gclose(handle, "/Header");
  my_H5Fclose(hdf5_file, fname);
}

int snap_io::get_filenr_from_header(void) { return header.num_files; }

void snap_io::set_filenr_in_header(int numfiles) { header.num_files = numfiles; }

void snap_io::read_increase_numbers(int type, int n_for_this_task)
{
  if(type < NTYPES)
    Sp->NumPart += n_for_this_task;

  if(type == 0)
    Sp->NumGas += n_for_this_task;
}

void snap_io::get_datagroup_name(int type, char *buf)
{
  if(type < NTYPES)
    snprintf(buf, MAXLEN_PATH, "/PartType%d", type);
  else if(type == NTYPES)
    snprintf(buf, MAXLEN_PATH, "/TreeTable");
  else
    Terminate("wrong group");
}
