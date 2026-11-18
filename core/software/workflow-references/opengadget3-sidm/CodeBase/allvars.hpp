/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief In this file the global variables are declared.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef ALLVARS_HPP
#define ALLVARS_HPP

/*!< code version string */
#define GADGETVERSION "0.42"

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#include "../CodeBase/switches.hpp"
#include "../CodeBase/switches_check.hpp"
#include "../CodeBase/precision.hpp"
#include "../CodeBase/constants.hpp"
#include "../CodeBase/verbose.hpp"
#include "../System/rng.hpp"

#ifdef FPE_TRAPPING
#define _GNU_SOURCE
#include <fenv.h>
#endif // FPE_TRAPPING

#ifdef MPISENDRECV_CHECKSUM
#define MPI_Sendrecv MPI_Check_Sendrecv
#endif // MPISENDRECV_CHECKSUM

#ifdef MPISENDRECV_SIZELIMIT
#define MPI_Sendrecv MPI_Sizelimited_Sendrecv
#endif // MPISENDRECV_SIZELIMIT

#include "../System/tags.hpp"
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
#include "../Chemistry/chemistry.hpp"
#endif // defined(CHEMISTRY) || defined (UM_CHEMISTRY)
#include "../System/assert.hpp"


#if GADGET_HYDRO == HYDRO_MFM
#include "../Mfm/Eos.hpp"
#include "../Mfm/FluidVector.hpp"
#include "../Mfm/FluxSolver.hpp"
#include "../Mfm/Matrix.hpp"
#include "../Mfm/Vector.hpp"
using namespace OpenGadget3;
extern Eos<mfm_numdims> *eos;
extern FluxSolver<mfm_numdims> *fluxSolver;
#if !defined(DONT_BIN_MFM_FLUXES) && defined(_OPENMP)
// locks for the MFM neighbor flux
extern omp_lock_t *MfmNgbLockList;
#endif // !defined(DONT_BIN_MFM_FLUXES) && defined(_OPENMP)

// has to be defined before the SphP struct
enum fluid_vector_index
{
    ivx = 0,
    ivy,
    ivz,
    irho = mfm_numdims,
    ipress, // mfm_numdims + 1
#ifdef MAGNETIC
    ibx, // mfm_numdims + 2
    iby,
    ibz,
#ifdef DIVBCLEANING_DEDNER
    iphi, // mfm_numdims + 5
#endif // DIVBCLEANING_DEDNER
#endif // MAGNETIC
    numvar
};
#endif // GADGET_HYDRO == HYDRO_MFM

#ifdef PXP_OPENMP
extern omp_lock_t *pxpPartLockList;
#endif // PXP_OPENMP

enum particle_type
{
    type_gas = 0,
    type_dm = 1,
#ifdef NEUTRINOS
    type_neutrino = 2,
#endif // NEUTRINOS
    type_star = 4,
    type_bh = 5
};

#ifdef BLACK_HOLES
#ifdef FG_REMOVE_BND
#define IS_SPH_ALIVE(Pi) (((Pi).Type == type_gas) && ((Pi).Mass > 0) && ((Pi).ID != 0))
#else // FG_REMOVE_BND
#define IS_SPH_ALIVE(Pi) (((Pi).Type == type_gas) && ((Pi).Mass > 0))
#endif // FG_REMOVE_BND
#define IS_BLACKHOLE(Pi) ((Pi).Type == type_bh)
#else // BLACK_HOLES
#ifdef FG_REMOVE_BND
#define IS_SPH_ALIVE(Pi) (((Pi).Type == type_gas) && ((Pi).ID != 0))
#else // FG_REMOVE_BND
#define IS_SPH_ALIVE(Pi) ((Pi).Type == type_gas)
#endif // FG_REMOVE_BND
#define IS_BLACKHOLE(Pi) 0
#endif // BLACK_HOLES

#if defined(LT_STELLAREVOLUTION)
#ifdef LT_STELLAREVOLUTION
#define IS_STAR_ACTIVE(Pp, n) (((Pp)[(n)].Type == type_star) && (is_chemically_active((n))))
#else // LT_STELLAREVOLUTION
#define IS_STAR_ACTIVE(Pp, n) ((Pp)[(n)].Type == type_star)
#endif // LT_STELLAREVOLUTION
#else // not LT_STELLAREVOLUTION
#define IS_STAR_ACTIVE(Pp, n) 0
#endif // defined(LT_STELLAREVOLUTION)

// see https://gcc.gnu.org/onlinedocs/cpp/Swallowing-the-Semicolon.html
#ifdef _OPENMP
#define THREAD_ID_OR_ZERO() omp_get_thread_num()
#else // _OPENMP
#define THREAD_ID_OR_ZERO() 0
#endif // _OPENMP

#define OMP_SORT_THRESH 1000

#ifdef _OPENMP
#define MYSORT_DATAINDEX omp_mysort
#else // _OPENMP
#define MYSORT_DATAINDEX mysort_dataindex
#endif // _OPENMP

#ifdef _OPENMP
#define STD_SORT omp_qsort
#else // _OPENMP
#define STD_SORT qsort
#endif // _OPENMP

// compiler specific data alignment hints
// XLC compiler
#ifdef __xlC__
#define ALIGN(n) __attribute__((__aligned__(n)))
// GNU compiler
#elif defined(__GNUC__)
#define ALIGN(n) __attribute__((__aligned__(n)))
// Intel Compiler
#elif defined(__INTEL_COMPILER)
// GNU Intel Compiler
#define ALIGN(n) __declspec(align(n))
// Unknown Compiler
#else // __xlC__
#define ALIGN(n)
#endif // __xlC__

#ifdef PERIODIC
#ifdef POWER6
#define NEAREST_X(x) ((x)-boxSize_X * __frin((x)*inverse_boxSize_X))
#define NEAREST_Y(y) ((y)-boxSize_Y * __frin((y)*inverse_boxSize_Y))
#define NEAREST_Z(z) ((z)-boxSize_Z * __frin((z)*inverse_boxSize_Z))
#define NEAREST(x) ((x)-boxSize * __frin((x)*inverse_boxSize))
#else // POWER6
#define NEAREST_X(x) (((x) > boxHalf_X) ? ((x)-boxSize_X) : (((x) < -boxHalf_X) ? ((x) + boxSize_X) : (x)))
#define NEAREST_Y(y) (((y) > boxHalf_Y) ? ((y)-boxSize_Y) : (((y) < -boxHalf_Y) ? ((y) + boxSize_Y) : (y)))
#define NEAREST_Z(z) (((z) > boxHalf_Z) ? ((z)-boxSize_Z) : (((z) < -boxHalf_Z) ? ((z) + boxSize_Z) : (z)))
#define NEAREST(x) (((x) > boxHalf) ? ((x)-boxSize) : (((x) < -boxHalf) ? ((x) + boxSize) : (x)))
#define __fsel(crit, age, alt) (((crit) >= 0.0) ? (age) : (alt))
#endif // POWER6
#else // PERIODIC
#define NEAREST_X(x) (x)
#define NEAREST_Y(x) (x)
#define NEAREST_Z(x) (x)
#define NEAREST(x) (x)
#endif // PERIODIC

#define ASSIGN_ADD(x, y, mode) (mode == 0 ? (x = y) : (x += y))
#define ASSIGN_MIN(x, y, mode) (mode == 0 ? (x = y) : (x = std::min(x, y)))
#define ASSIGN_MAX(x, y, mode) (mode == 0 ? (x = y) : (x = std::max(x, y)))

/* this is the number of past executions of a timebin that the reported average CPU-times average over */
#define NUMBER_OF_MEASUREMENTS_TO_RECORD 6

#if !defined(mSIDM) && !defined(mIDM)
#define NODELISTLENGTH 8
#else // !mSIDM && !mIDM
#define NODELISTLENGTH NODELISTLENGTH_default // 8
#endif // !mSIDM && !mIDM

#define BITS_PER_DIMENSION 21 /* for Peano-Hilbert order. Note: Maximum is 10 to fit in 32-bit integer ! */
#define PEANOCELLS (((peanokey)1) << (3 * BITS_PER_DIMENSION))

#define BITS_PER_DIMENSION_SAVE_KEYS 10
#define PEANOCELLS_SAVE_KEYS (((peanokey)1) << (3 * BITS_PER_DIMENSION_SAVE_KEYS))

#ifdef mSIDM
#include "../mSIDM/msidm.hpp"
#endif // mSIDM


#ifdef ANALYTIC_POTENTIALS
#include "../Gravity/analytic_potentials.hpp"
#endif // ANALYTIC_POTENTIALS

#ifdef LT_STELLAREVOLUTION
#include "../CoolingSfr/Sfr_LT/lt.hpp"
#endif // LT_STELLAREVOLUTION

using namespace constants;

#ifdef LT_ADD_GAL_TO_SUB
extern float *tempiAS, *CB07, *Filters_Effective_Wavelenght;
#ifdef OBSERVER_FRAME
extern float *CB07obs;
#ifdef INTERP_OBSERVER_FRAME
extern float CB07obs_redshifts[INTERP_OBSERVER_FRAME];
extern int flag_readCB07obs_redshifts;
#endif // INTERP_OBSERVER_FRAME
#endif // OBSERVER_FRAME
#endif // LT_ADD_GAL_TO_SUB


#define check_particles() check_particles_info(__FUNCTION__, __FILE__, __LINE__)
#define check_node() check_node_info(__FUNCTION__, __FILE__, __LINE__)

#ifndef DISABLE_MEMORY_MANAGER
#define mymalloc(x, y) mymalloc_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)
#define mymalloc_movable(x, y, z) mymalloc_movable_fullinfo(x, y, z, __FUNCTION__, __FILE__, __LINE__)

#define myrealloc(x, y) myrealloc_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)
#define myrealloc_movable(x, y) myrealloc_movable_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)

#ifdef SUPER_STRICT_MEMORY_MANAGER
// a block which is not last one is bever freed
#define myfree(x) myfree_fullinfo(x, __FUNCTION__, __FILE__, __LINE__)
#else // SUPER_STRICT_MEMORY_MANAGER
// a block which is not last one is still freed when only movable blocks are behind
#define myfree(x) myfree_movable_fullinfo(x, __FUNCTION__, __FILE__, __LINE__)
#endif // SUPER_STRICT_MEMORY_MANAGER
// a block which is not last one is still freed when only movable blocks are behind
#define myfree_movable(x) myfree_movable_fullinfo(x, __FUNCTION__, __FILE__, __LINE__)

#define report_memory_usage(x, y) report_detailed_memory_usage_of_largest_task(x, y, __FUNCTION__, __FILE__, __LINE__)
#else // DISABLE_MEMORY_MANAGER

#if (DISABLE_MEMORY_MANAGER == 2)
#define mymalloc(x, y) calloc(y, 1)
#define mymalloc_movable(x, y, z) calloc(z, 1)
#else // DISABLE_MEMORY_MANAGER==1
#define mymalloc(x, y) malloc(y)
#define mymalloc_movable(x, y, z) malloc(z)
#endif // DISABLE_MEMORY_MANAGER==X

#define myrealloc(x, y) realloc(x, y)
#define myrealloc_movable(x, y) realloc(x, y)

#define myfree(x) free(x)
#define myfree_movable(x) free(x)

#define report_memory_usage(x, y) VERBOSE(2, "Memory manager disabled.")
#endif // DISABLE_MEMORY_MANAGER

#ifndef HYDROGEN_ONLY
#define HYDROGEN_MASSFRAC 0.76 /*!< mass fraction of hydrogen, relevant only for radiative cooling */
#else // HYDROGEN_ONLY
#define HYDROGEN_MASSFRAC 1 /*!< mass fraction of hydrogen, relevant only for radiative cooling */
#endif // HYDROGEN_ONLY

#define METAL_YIELD 0.02 /*!< effective metal yield for star formation */

#define MAX_REAL_NUMBER 1e37
#define MIN_REAL_NUMBER 1e-37

#define RNDTABLE 8192

#ifdef NAVIERSTOKES
#define LOG_LAMBDA 37.8 /* logarithmic Coulomb factor */
#endif // NAVIERSTOKES

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY) || defined(INCLUDE_RADIATION) || defined(KSPACE_NEUTRINOS_2)
#define T_CMB0 2.7255 /* present-day CMB temperature, from Fixsen 2009 */
#endif //  defined(CHEMISTRY) || defined(UM_CHEMISTRY) || defined(INCLUDE_RADIATION) || defined(KSPACE_NEUTRINOS_2)

#if defined(KSPACE_NEUTRINOS_2) || defined(INCLUDE_RADIATION)
/* Note there is a slight correction from 4/11
 * due to the neutrinos being slightly coupled at e+- annihilation.
 * See Mangano et al 2005 (hep-ph/0506164)
 *The correction is (3.046/3)^(1/4), for N_eff = 3.046 */
#define TNU (T_CMB0 * pow(4 / 11., 1 / 3.) * 1.00381) /* Neutrino + antineutrino background temperature in Kelvin */
#endif // defined(KSPACE_NEUTRINOS_2) || defined(INCLUDE_RADIATION)

/* For convenience define OMEGAK*/
#define OMEGAK (1 - All.Omega0 - All.OmegaLambda)

#ifndef FOF_PRIMARY_LINK_TYPES
#define FOF_PRIMARY_LINK_TYPES 2
#endif // FOF_PRIMARY_LINK_TYPES

#ifndef FOF_SECONDARY_LINK_TYPES
#define FOF_SECONDARY_LINK_TYPES 1 + 16 + 32
#endif // FOF_SECONDARY_LINK_TYPES

/* some flags for the field "flag_ic_info" in the file header */
#define FLAG_ZELDOVICH_ICS 1
#define FLAG_SECOND_ORDER_ICS 2
#define FLAG_EVOLVED_ZELDOVICH 3
#define FLAG_EVOLVED_2LPT 4
#define FLAG_NORMALICS_2LPT 5

#define COND_TIMESTEP_PARAMETER 0.25
#define VISC_TIMESTEP_PARAMETER 0.25
#define MAXLEN_OUTPUTLIST 14000 /*!< maxmimum number of entries in output list */

enum
{
    /*!< length of the lookup table used to hold the drift and kick factors */
    DRIFT_TABLE_LENGTH = 1000
};

#define MAXITER 150

#ifndef LINKLENGTH
#define LINKLENGTH 0.16
#endif // LINKLENGTH

#ifndef FOF_GROUP_MIN_LEN
#define FOF_GROUP_MIN_LEN 32
#endif // FOF_GROUP_MIN_LEN

#define MINRESTFAC 0.05

#ifdef SUBFIND_DENSITY_AND_POTENTIAL /*!< activate needed options */
#define ONLY_PRODUCE_HSML_FILES
#define COMPUTE_POTENTIAL_ENERGY
#define SUBFIND_RESHUFFLE_AND_POTENTIAL
#define SUBFIND_RESHUFFLE_CATALOGUE
#endif // SUBFIND_DENSITY_AND_POTENTIAL
#ifdef ADAPTGRAVSOFT
#ifndef AGS_MAXSOFT
/*!< AGS_MAXSOFT gives the maximum allowed gravitational softening when using the TreePM method.
 *  The quantity is given in units of the scale used for the force split (ASMTH) */
#define AGS_MAXSOFT 0.5
#endif // AGS_MAXSOFT
#ifndef AGS_MINSOFT
/*!< AGS_MINSOFT gives the minimum allowed gravitational softening when using the TreePM method.
 *  The quantity is given in units of the scale used for the force split (ASMTH) */
#define AGS_MINSOFT 0.001
#endif // AGS_MINSOFT
#endif // ADAPTGRAVSOFT

#ifndef GDE_TYPES
#define GDE_TYPES 2
#endif // GDE_TYPES

struct unbind_data
{
    int index;
};

#ifdef FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG
extern MPI_Status mpistat;
#undef MPI_STATUS_IGNORE
#define MPI_STATUS_IGNORE &mpistat
#endif // FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG

#define GDE_ABS(x) (abs(x))
#define GDE_SQRT(x) (sqrt(x))
#define GDE_LOG(x) (log(x))

#define CPU_STRING_LEN 120

#ifndef QUINTIC_KERNEL
#if !defined(TWODIMS) && !defined(ONEDIM)
/*!< For 3D-normalized kernel */
#define KERNEL_COEFF_1 2.546479089470 /*!< Coefficients for SPH spline kernel and its derivative */
#define KERNEL_COEFF_2 15.278874536822
#define KERNEL_COEFF_3 45.836623610466
#define KERNEL_COEFF_4 30.557749073644
#define KERNEL_COEFF_5 5.092958178941
#define KERNEL_COEFF_6 (-15.278874536822)
#elif defined(TWODIMS) // !defined(TWODIMS) && !defined(ONEDIM)
/*!< For 2D-normalized kernel */
#define KERNEL_COEFF_1 (5.0 / 7 * 2.546479089470) /*!< Coefficients for SPH spline kernel and its derivative */
#define KERNEL_COEFF_2 (5.0 / 7 * 15.278874536822)
#define KERNEL_COEFF_3 (5.0 / 7 * 45.836623610466)
#define KERNEL_COEFF_4 (5.0 / 7 * 30.557749073644)
#define KERNEL_COEFF_5 (5.0 / 7 * 5.092958178941)
#define KERNEL_COEFF_6 (5.0 / 7 * (-15.278874536822))
#else // ONEDIM
#define KERNEL_COEFF_1 (4.0 / 3)
#define KERNEL_COEFF_2 (8.0)
#define KERNEL_COEFF_3 (24.0)
#define KERNEL_COEFF_4 (16.0)
#define KERNEL_COEFF_5 (8.0 / 3)
#define KERNEL_COEFF_6 (-8.0)
#endif // !defined(TWODIMS) && !defined(ONEDIM)
#endif // QUINTIC_KERNEL

#if !defined(TWODIMS) && !defined(ONEDIM)
#define NORM_COEFF 4.188790204786 /*!< Coefficient for kernel normalization. Note:  4.0/3 * PI = 4.188790204786 */
#elif defined(TWODIMS)
#define NORM_COEFF M_PI /*!< Coefficient for kernel normalization. */
#else // ONEDIM
#define NORM_COEFF 2.0 /*!< Coefficient for kernel normalization. */
#endif // !defined(TWODIMS) && !defined(ONEDIM)

const int default_kernel_ngb_number =
#if defined(WENDLAND_C8_KERNEL)
#if NUMDIMS == 1
    10;
#elif NUMDIMS == 2
    64;
#else // NUMDIMS
    384;
#endif // NUMDIMS
#elif defined(WENDLAND_C6_KERNEL)
#if NUMDIMS == 1
    8;
#elif NUMDIMS == 2
    54;
#else // NUMDIMS
    295;
#endif // NUMDIMS
#elif defined(WENDLAND_C4_KERNEL)
#if NUMDIMS == 1
    7;
#elif NUMDIMS == 2
    44;
#else // NUMDIMS
    216;
#endif // NUMDIMS
#elif defined(WENDLAND_C2_KERNEL)
#if NUMDIMS == 1
    5;
#elif NUMDIMS == 2
    19;
#else // NUMDIMS
    64;
#endif // NUMDIMS
#elif defined(QUINTIC_KERNEL)
#if NUMDIMS == 1
    7;
#elif NUMDIMS == 2
    41;
#else // NUMDIMS
    195;
#endif // NUMDIMS
#else // STANDARD_KERNEL
#if NUMDIMS == 1
    4;
#elif NUMDIMS == 2
    12;
#else // NUMDIMS
    32;
#endif // NUMDIMS
#endif // KERNEL

#ifdef PERIODIC
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern MyDouble boxSize, boxHalf;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern MyDouble inverse_boxSize;

#ifdef LONG_X
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern MyDouble boxSize_X, boxHalf_X;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern MyDouble inverse_boxSize_X;

#else // LONG_X
#define boxSize_X boxSize
#define boxHalf_X boxHalf
#define inverse_boxSize_X inverse_boxSize
#endif // LONG_X
#ifdef LONG_Y
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern MyDouble boxSize_Y, boxHalf_Y;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern MyDouble inverse_boxSize_Y;

#else // LONG_Y
#define boxSize_Y boxSize
#define boxHalf_Y boxHalf
#define inverse_boxSize_Y inverse_boxSize
#endif // LONG_Y
#ifdef LONG_Z
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern MyDouble boxSize_Z, boxHalf_Z, inverse_boxSize_Z;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern MyDouble inverse_boxSize_Z;

#else // LONG_Z
#define boxSize_Z boxSize
#define boxHalf_Z boxHalf
#define inverse_boxSize_Z inverse_boxSize
#endif // LONG_Z
#endif // PERIODIC

#ifdef PERIODIC
#define NGB_PERIODIC_LONG(x, box, hbox) ((abs(x) > hbox) ? (box - abs(x)) : abs(x))
#define NGB_PERIODIC_LONG_X(x) (xtmp = abs(x), (xtmp > boxHalf_X) ? (boxSize_X - xtmp) : xtmp)
#define NGB_PERIODIC_LONG_Y(x) (xtmp = abs(x), (xtmp > boxHalf_Y) ? (boxSize_Y - xtmp) : xtmp)
#define NGB_PERIODIC_LONG_Z(x) (xtmp = abs(x), (xtmp > boxHalf_Z) ? (boxSize_Z - xtmp) : xtmp)
#else // PERIODIC
#define NGB_PERIODIC_LONG(x, box, hbox) abs(x)
#define NGB_PERIODIC_LONG_X(x) abs(x)
#define NGB_PERIODIC_LONG_Y(x) abs(x)
#define NGB_PERIODIC_LONG_Z(x) abs(x)
#endif // PERIODIC

#define FACT1 0.366025403785 /* FACT1 = 0.5 * (sqrt(3)-1) */
#define FACT2 0.86602540 /* FACT2 = 0.5 * sqrt(3) */

/*********************************************************/
/*  Global variables                                     */
/*********************************************************/
#ifdef UM_SEN_H2_SOLOMON_PROCESS
#define JLWn 23
extern float JLW[JLWn];
extern float z_JLW[JLWn];
#endif // UM_SEN_H2_SOLOMON_PROCESS

#ifdef _OPENMP
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int NActivePart;
extern int *ActiveParticleList;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP
#ifdef LT_STELLAREVOLUTION
extern int NActivePartMerk;
#endif // LT_STELLAREVOLUTION
#endif // _OPENMP

extern int *NextActiveParticle;

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int FirstActiveParticle;
extern unsigned char *ProcessedFlag;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern int TimeBinCount[TIMEBINS];
extern int TimeBinCountSph[TIMEBINS];
extern int TimeBinActive[TIMEBINS];

extern int FirstInTimeBin[TIMEBINS];
extern int LastInTimeBin[TIMEBINS];
extern int *NextInTimeBin;
extern int *PrevInTimeBin;

#ifdef SFR
extern double TimeBinSfr[TIMEBINS];
#endif // SFR

#ifdef BLACK_HOLES
extern double TimeBin_BH_mass[TIMEBINS];
extern double TimeBin_BH_dynamicalmass[TIMEBINS];
extern double TimeBin_BH_Mdot[TIMEBINS];
extern double TimeBin_BH_Medd[TIMEBINS];
#endif // BLACK_HOLES

#ifdef KD_MONITOR_PERFORMANCE
extern int count_performance;
#endif // KD_MONITOR_PERFORMANCE

extern __thread int ThisThread; /* the Number of the local thread */
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int NThread; /* number of threads */
extern int ThisTask; /*!< the number of the local processor  */
extern int NTask; /*!< number of processors */
extern int PTask; /*!< note: NTask = 2^PTask */
extern MPI_Comm MYMPI_COMM_WORLD;
extern int MYMPI_MAX_TAG_UB;
extern int TakeLevel;
extern int *Send_offset, *Send_count, *Recv_count, *Recv_offset;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern double CPUThisRun; /*!< Sums CPU time of current process */

#ifdef RECOMPOSE_DOMAIN_FACTOR
extern int LastDomainUpdate;
#endif // RECOMPOSE_DOMAIN_FACTOR

extern int NumForceUpdate; /*!< number of active particles on local processor in current timestep  */
extern long long GlobNumForceUpdate;
extern int NumSphUpdate; /*!< number of active SPH particles on local processor in current timestep  */
extern int MaxTopNodes; /*!< Maximum number of nodes in the top-level tree used for domain decomposition */
extern int RestartFlag; /*!< taken from command line used to start code. 0 is normal start-up from
               initial conditions, 1 is resuming a run from a set of restart files, while 2
               marks a restart from a snapshot file. */
extern int RestartSnapNum;

extern double PostProcessDt; /*!< For the post-processing FKP the timestep is passed as a command line argument */
extern int SelRnd;

extern int *Exportflag; /*!< Buffer used for flagging whether a particle needs to be exported to another process */
extern int *Exportnodecount;
extern int *Exportindex;

#ifdef IMPORT_ALLtoALLv_FROM_G4
extern MPI_Win win;
#endif // IMPORT_ALLtoALLv_FROM_G4

extern size_t AllocatedBytes;
extern size_t HighMarkBytes;
extern size_t FreeBytes;

extern double WallclockTime; /*!< This holds the last wallclock time measurement for timings measurements */

extern int Flag_FullStep; /*!< Flag used to signal that the current step involves all particles */

extern size_t HighMark_run, HighMark_domain, HighMark_gravtree, HighMark_pmperiodic, HighMark_sphgrad,
    HighMark_pmnonperiodic, HighMark_sphhsml, HighMark_sphdensity, HighMark_sphhydro, HighMark_addSPH;
#if GADGET_HYDRO == HYDRO_MFM
extern size_t HighMark_mfmgrad, HighMark_mfmlim, HighMark_mfmflux;
#endif // GADGET_HYDRO == HYDRO_MFM


#ifdef ADAPTGRAVSOFT
extern size_t HighMark_agsdensity;
#endif // ADAPTGRAVSOFT

extern int TreeReconstructFlag;
extern int GlobFlag;
extern char DumpFlag;

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int NumPart; /*!< number of particles on the LOCAL processor */
extern int N_gas; /*!< number of gas particles on the LOCAL processor  */

#ifdef ACC_OPENMP
// Nitya - adding a variable here to keep a check of when to reallocate WalkLiteP
extern int NumPartOld;
extern int WalkLitePAlloc;
extern int GpuWalkLitePAlloc;

extern int LiteDInNumPartOld;
extern int GpuLiteDInAlloc;

extern int LiteDOutNumPartOld;
extern int GpuLiteDOutAlloc;

extern int GpuHSMLAlloc;
extern int HSMLNumPartOld;

extern int DAPLNumPartOld;
extern int GpuDAPLAlloc;

extern int APLNumPartOld;
extern int GpuAPLAlloc;
#pragma omp end declare target
#endif // ACC_OPENMP

extern int N_stars; /*!< number of star particles in the LOCAL processor */
extern int N_BHs;

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int N_ToWakeUp; /*!< number of particles to wake up */
extern int *ToWakeupList;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern long long Ntype[6]; /*!< total number of particles of each type */
extern int NtypeLocal[6]; /*!< local number of particles of each type */

#if defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM) || defined(PB_INITIALIZE_BH_PART)
extern int Particles_converted; /*!< current number of star/DM particles in gas particle block */
#endif // defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

#define d_fftw_real fftw_real

extern double TimeOfLastTreeConstruction; /*!< holds what it says */
extern int *Ngblist; /*!< Buffer to hold indices of neighbours retrieved by the neighbour search
            routines */
extern int *RecyclingBuffer;
extern int HasFisrtParticleBeenProcessed;
extern double *R2ngblist;
extern double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
extern int *DomainStartList, *DomainEndList;

extern double *DomainWork;
extern int *DomainCount;
extern int *DomainCountSph;

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int *DomainTask;
extern int *DomainNodeIndex;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern int *DomainList, DomainNumChanged;
extern peanokey *__restrict__ Key, *__restrict__ KeySorted;

#if GADGET_HYDRO == HYDRO_MFM && defined(DIVBCLEANING_DEDNER)
extern MyFloat FastestWaveSpeed; /*!< Fastest wavespeed across all particles */
#endif // GADGET_HYDRO == HYDRO_MFM && defined(DIVBCLEANING_DEDNER)

/**
 * @brief Represents the data structure for top-level nodes in a hierarchical tree.
 *
 * This structure contains information about a top-level node in a spatial hierarchy.
 *
 * @var Size     The size of the node, typically represented by a Peano-Hilbert key.
 * @var StartKey The starting Peano-Hilbert key for this node.
 * @var Count    The number of particles or elements within this node.
 * @var GravCost The gravitational cost associated with this node.
 * @var Daughter The index of the daughter node, or -1 if this node has no daughters (leaf).
 * @var Pstart   The starting index of particles in the node.
 * @var Blocks   The number of blocks associated with this node.
 * @var Leaf     Flag indicating if this node is a leaf (1) or not (0).
 */
extern struct topnode_data
{
    peanokey Size;
    peanokey StartKey;
    long long Count;
    MyFloat GravCost;
    int Daughter;
    int Pstart;
    int Blocks;
    int Leaf;
} *TopNodes;

extern int NTopnodes, NTopleaves;
extern double RndTable[RNDTABLE];

#ifdef SUBFIND
extern int GrNr;
extern int NumPartGroup;
#endif // SUBFIND

#ifdef WRITE_KEY_FILES
extern peanokey *KeyIndex;
extern int *NPartPerKey, *PartKeyOffset;
extern int NKeys[6];
extern long long NKeysTotal[6];
#endif // WRITE_KEY_FILES

/* variables for input/output , usually only used on process 0 */
extern char ParameterFile[100]; /*!< file name of parameterfile used for starting the simulation */

extern FILE *FdInfo, /*!< file handle for info.txt log-file. */
    *FdEnergy, /*!< file handle for energy.txt log-file. */
    *FdTimings, /*!< file handle for timings.txt log-file. */
    *FdTimebin, *FdParamChangeLog;

#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
extern FILE *FdPM; /*!< file handler for pm.txt log-file. */
#endif // PMGRID

#ifdef SFR
extern FILE *FdSfr; /*!< file handle for sfr.txt log-file. */
#endif // SFR


#ifdef BLACK_HOLES
extern FILE *FdBlackHoles; /*!< file handle for blackholes.txt log-file. */
extern FILE *FdBlackHolesDetails;
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
extern FILE *FdBlackHolesSpinDetails;
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
#if defined(PB_BH_HSML_PREVENT_HOLE) || defined(PB_OUTPUT_SINGLE_BH)
extern FILE *Fd_BlackHole_GasNgbs; /*!< file handle for BlackHole_GasNgbs.txt log-file. */
#endif // PB_BH_HSML_PREVENT_HOLE || PB_OUTPUT_SINGLE_BH
#endif // BLACK_HOLES

#ifdef FORCETEST
extern FILE *FdForceTest; /*!< file handle for forcetest.txt log-file. */
#endif // FORCETEST

#ifdef DARKENERGY
extern FILE *FdDE; /*!< file handle for darkenergy.txt log-file. */
#endif // DARKENERGY

#ifdef XXLINFO
extern FILE *FdXXL; /*!< file handle for xxl.txt log-file. */
#ifdef MAGNETIC
extern double MeanB;
extern double MaxDivB;
#endif // MAGNETIC
#ifdef TIME_DEP_ART_VISC
extern double MeanAlpha;
#endif // TIME_DEP_ART_VISC
#endif // XXLINFO
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
/*! table for the cosmological drift factors */
extern double DriftTable[DRIFT_TABLE_LENGTH];
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP
/*! table for the cosmological kick factor for gravitational forces */
extern double GravKickTable[DRIFT_TABLE_LENGTH];

/*! table for the cosmological kick factor for hydrodynmical forces */
extern double HydroKickTable[DRIFT_TABLE_LENGTH];

/*! table for the cosmological kick factor for induction equation */
extern double MagKickTable[DRIFT_TABLE_LENGTH];

#if defined(mSIDM) || defined(mIDM)
/*! table for the cosmological kick factor for mSIDM and mIDM */
extern double ScatteringKickTable[DRIFT_TABLE_LENGTH];
#endif // mSIDM || mIDM

extern void *CommBuffer; /*!< points to communication buffer, which is used at a few places */

/**
 * @brief Metadata for a single registered parameter (see setup_parameter() in begrun.cpp).
 *
 * type and restart_class are kept as plain int (rather than the Param_Type/Param_Restart
 * enums they mirror) so this header doesn't need to depend on those begrun.cpp-local enums.
 */
struct ParamInfo
{
    void *addr;
    int type;
    int restart_class;
    int min_expert_level; // for RESTART_DANGEROUS: minimum EXPERT_LEVEL required to allow the change
};

/**
 * @brief Holds the global data structure for all processes.
 *
 * This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure.
 */
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern struct global_data_all_processes
{
    long long TotNumPart; /*!<  total particle numbers (global value) */
    long long TotN_gas; /*!<  total gas particle number (global value) */

#ifdef SFR
    long long TotN_stars; /*!<  total star particle number (global value) */
#endif // SFR

#ifdef NEUTRINOS
    long long TotNumNeutrinos;
    int tree_nu_on;
#endif // NEUTRINOS

#ifdef BLACK_HOLES
    int TotBHs;
    int MaxPartBH;
#endif // BLACK_HOLES

    int MaxPart; /*!< This gives the maxmimum number of particles that can be stored on one
            processor. */
    int MaxPartSph; /*!< This gives the maxmimum number of SPH particles that can be stored on one
               processor. */
    int DoDynamicUpdate;

    int BunchSize; /*!< number of particles fitting into the buffer in the parallel tree algorithm  */

    double TreeAllocFactor; /*!< Each processor allocates a number of nodes which is TreeAllocFactor times
                   the maximum(!) number of particles.  Note: A typical local tree for N
                   particles needs usually about ~0.65*N nodes. */

    double TopNodeAllocFactor; /*!< Each processor allocates a number of nodes which is TreeAllocFactor times
                  the maximum(!) number of particles.  Note: A typical local tree for N
                  particles needs usually about ~0.65*N nodes. */

    double InitGasU; /*!< the same, but converted to thermal energy per unit mass */
    double MinEgySpec; /*!< the minimum allowed temperature expressed as energy per unit mass */

    /* some force counters  */
    long long TotNumOfForces; /*!< counts total number of force computations  */
    long long NumForcesSinceLastDomainDecomp; /*!< count particle updates since last domain decomposition */

    /* some variable for dynamic work-load adjustment based on CPU measurements */
    double cf_atime, cf_a2inv, cf_a3inv, cf_afac1, cf_afac2, cf_afac3,
        cf_hubble_a; /* various cosmological factors that are only a function of the current scale factor, and in
                        Newtonian runs are set to 1 */

    double UnitTime_in_s, /*!< factor to convert internal time unit to seconds/h */
        UnitPressure_in_cgs, /*!< factor to convert internal pressure unit to cgs units (little 'h' still
                    around!) */
        UnitDensity_in_cgs, /*!< factor to convert internal length unit to g/cm^3*h^2 */
        UnitCoolingRate_in_cgs, /*!< factor to convert internal cooling rate to cgs units */
        UnitEnergy_in_cgs, /*!< factor to convert internal energy to cgs units */
        UnitTime_in_Megayears; /*!< factor to convert internal time to megayears/h */
#ifdef MAGNETIC
    double UnitMagneticField_in_gauss; /*!< factor to convert internal magnetic field to gauss */
#endif // MAGNETIC

#ifdef MV_KRUMHOLZ_MOLECULAR_FRACTION
    double UnitSurfDens_in_Msunpc2; /*!< factor to convert internal surface density to Msun/pc^2 units */
#endif // MV_KRUMHOLZ_MOLECULAR_FRACTION
    double UnitDensity_in_Gev_per_cm3; /*!< factor to convert internal density unit to GeV/c^2 / cm^3 */
    /* Cosmology */

    double Hubble; /*!< Hubble-constant in internal units */

    int HighestActiveTimeBin;
    int HighestOccupiedTimeBin;

    /* parameters determining output frequency */
    int SnapshotFileCount; /*!< number of snapshot that is written next */

    double TimeLastRestartFile, /*!< cpu-time when last restart-file was written */
        TimeLastStatistics; /*!< simulation time when the energy statistics was computed the last time */

    int NumCurrentTiStep = 0; /*!< counts the number of system steps taken up to this point */

    /* Current time of the simulation, global step, and end of simulation */
    double Time, /*!< current time of the simulation */
        TimeStep; /*!< difference between current times of previous and current timestep */

    /* variables for organizing discrete timeline */

    double Timebase_interval; /*!< factor to convert from floating point time interval to integer timeline */
    integertime Ti_Current; /*!< current time on integer timeline */
    integertime Previous_Ti_Current;
    integertime Ti_nextoutput; /*!< next output time on integer timeline */
    integertime Ti_lastoutput;

#ifdef OUTPUT_LONGRANGE_POTENTIAL
    int dumped_after_snap; /* for whether the potential has been dumped after the snapshot! */
    integertime Ti_nextpotdump; /*!< next potential dump time on integer timeline */
    int PotentialFileCount; /*!< number of potential that is written next */
#endif // OUTPUT_LONGRANGE_POTENTIAL

#ifdef PMGRID
    integertime PM_Ti_endstep, PM_Ti_begstep;
    double Asmth[2], Rcut[2];
    double Corner[3][3], UpperCorner[3][3], Xmintot[2][3], Xmaxtot[2][3];
    double TotalMeshSize[3];
#endif // PMGRID

    integertime Ti_nextlineofsight;

    int CPU_TimeBinCountMeasurements[TIMEBINS];
    double CPU_TimeBinMeasurements[TIMEBINS][NUMBER_OF_MEASUREMENTS_TO_RECORD];

    int LevelToTimeBin[Gravcostlevels];

    /* gravitational and hydrodynamical softening lengths (given in terms of an `equivalent' Plummer softening
     * length)
     *
     * five groups of particles are supported 0=gas,1=halo,2=disk,3=bulge,4=stars
     */

    double MinGasHsml; /*!< minimum allowed SPH smoothing length */

    double SofteningTable[6]; /*!< current (comoving) gravitational softening lengths for each particle type */
    double ForceSoftening[6]; /*!< the same, but multiplied by a factor 2.8 - at that scale the force is Newtonian */

    /*! If particle masses are all equal for one type, the corresponding entry in MassTable is set to this
     *  value, * allowing the size of the snapshot files to be reduced
     */
    double MassTable[6];

    /*! table with desired output times */
    double OutputListTimes[MAXLEN_OUTPUTLIST];
    char OutputListFlag[MAXLEN_OUTPUTLIST];
    int OutputListLength; /*!< number of times stored in table of desired output times */

#ifdef ADAPTIVE_FORCE_ACC
    double ErrTolForceAccParam;
#endif // ADAPTIVE_FORCE_ACC

#ifdef SFR /* star formation and feedback sector */
    double OverDensThresh;
    double PhysDensThresh;
#ifdef LS_SF_RUNAWAY_TREATMENT
    double RunawayThresh;
#endif
#ifdef LT_STELLAREVOLUTION
    double OrigGasMass;
#endif // LT_STELLAREVOLUTION
    double EgySpecSN;
    double EgySpecCold;
    double FeedbackEnergy;
    double FactorForSofterEQS;
#endif // SFR

#if defined(SFR) || defined(BLACK_HOLES)
    unsigned int bits;
#endif // defined(SFR) || defined(BLACK_HOLES)

#ifdef CS_DECOUPLING
    double DensityTailThreshold;
#endif // CS_DECOUPLING

#ifdef DARKENERGY
#ifdef TIMEDEPGRAV
    double Gini;
#endif // TIMEDEPGRAV
#endif // DARKENERGY

#ifdef CONDUCTION
#ifdef CONDUCTION_SATURATION
    double ElectronFreePathFactor; /*!< Factor to get electron mean free path */
#endif // CONDUCTION_SATURATION

    integertime Conduction_Ti_begstep, Conduction_Ti_endstep;

#ifdef CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR
    double ConductionPerpendicularConstants;
#endif // CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR
#endif // CONDUCTION


#ifdef BLACK_HOLES
    double TimeNextOnTheFlyFoF; //
#endif // BLACK_HOLES

#ifdef REIONIZATION
    int not_yet_reionized; /*!< flag that makes sure that there is only one reionization */
#endif // REIONIZATION

#ifdef NAVIERSTOKES
    double NavierStokes_ShearViscosity;
#endif // NAVIERSTOKES

#ifdef NAVIERSTOKES_VISCOSITY_SATURATION
    double IonMeanFreePath;
#endif // NAVIERSTOKES_VISCOSITY_SATURATION


#ifdef SFR
    int MaxPartMet; /*!< This gives the maxmimum number of STAR particles that can be stored on one
               processor. */
#endif // SFR

#ifdef LT_STELLAREVOLUTION
    double Time_Age; /*!< current cosmic time in Gyrs */
    int StarBits;

    double LocalSpreadFactor;
    long double MetalsCheckSum[LT_NMetP] = {0};

    /* lifetimes */
    double inf_lifetime, mean_lifetime, sup_lifetime;
    int referenceZbin_SFTh;
#endif // LT_STELLAREVOLUTION


#ifdef ADAPTGRAVSOFT
#ifdef PMGRID
    double AGS_MaxSoft[2], AGS_MinSoft[2];
#endif // PMGRID
#endif // ADAPTGRAVSOFT

    int StartTiStep = 0;

    /*###################################*/
    /*    Parameters from param file     */
    /*###################################*/

    char InitCondFile[100];
    char OutputDir[100] = "./";
    char SnapshotFileBase[100] = "snap";
    char EnergyFile[100] = "energy.txt";
    char CpuFile[100] = "cpu.txt";
    char BalanceFile[100] = "balance.txt";
    char BalanceFileHead[100] = "balance-head.txt";
    char InfoFile[100] = "info.txt";
    char TimingsFile[100] = "timing.txt";
    char TimebinFile[100] = "timebin.txt";
    char RestartFile[100] = "restart";
    char OutputListFilename[100] = "outputs";
    int OutputListOn = 0; /*!< flags that output times are listed in a specified file */
    double Omega0 = 1.0, /*!< matter density in units of the critical density (at z=0) */
        OmegaLambda = 0.0, /*!< vaccum energy density relative to crictical density (at z=0) */
        OmegaBaryon = 0.0, /*!< baryon density in units of the critical density (at z=0) */
        HubbleParam = 1.0; /*!< little `h', i.e. Hubble constant in units of 100 km/s/Mpc.*/
    double BoxSize = 0.0; /*!< Boxsize in case periodic boundary conditions are used, 0 means use IC value */
    int PeriodicBoundariesOn = 0; /*!< flags that periodic boundaries are enabled */
    int MaxMemSize = 2000;
    double TimeOfFirstSnapshot = 0.0, /*!< simulation time of first snapshot files */
        CpuTimeBetRestartFile = 3600; /*!< cpu-time between regularly generated restart files */
    double TimeBetStatistics = 0.02; /*!< simulation time interval between computations of energy statistics */
    double TimeBegin = -1.0; /*!< time of initial conditions of the simulation, negative means use IC value */
    double TimeMax; /*!< marks the point of time until the simulation is to be evolved */
    double TimeBetSnapshot; /*!< simulation time interval between snapshot files */

    /* system of units  */
    double UnitVelocity_in_cm_per_s = 1e5; /*!< factor to convert internal velocity unit to cm/sec */
    double UnitLength_in_cm = CM_PER_KPC; /*!< factor to convert internal length unit to cm/h - 1.0 kpc /h */
    double UnitMass_in_g = (1e10 * SOLAR_MASS); /*!< factor to convert internal mass unit to grams/h - 1 Msol*/

    double ErrTolIntAccuracy = 0.05; /*!< accuracy tolerance parameter \f$ \eta \f$ for timestep criterion. The
           timesteps is \f$ \Delta t = \sqrt{\frac{2 \eta eps}{a}} \f$ */


    double ErrTolTheta = 0; /*!< BH tree opening angle for the BH cell-opening criterion (recommended value: 0.45).
                                 If it is zero, then the relative cell-opening criterion is used instead */

    int TimerReportLevel = 2;
    int TimerReportBalance = 2;

#ifdef fSIDM
#ifdef fSIDM_SIMPLE
    double FSIDM_CrossSectionPerMass_in_cgs; /*!< Read from parameter file */
    char FSIDM_CrossSectionDef[20] = "transfer";
#endif // fSIDM_SIMPLE
#ifdef mSIDM_TIMESTEP
    double FSIDM_tfact = 0.04;
#endif // mSIDM_TIMESTEP
#ifdef fSIDM_DISSIPATION
    double FSIDM_const_r_diss;
#endif // fSIDM_DISSIPATION
#endif // fSIDM

#ifdef rSIDM
#ifdef rSIDM_ISOTROPIC
    double RSIDM_CrossSectionPerMass_in_cgs; /*!< Read from parameter file */
    char RSIDM_CrossSectionDef[20] = "transfer";
#endif // rSIDM_ISOTROPIC
#ifdef mSIDM_TIMESTEP
    double RSIDM_tfact = 0.04;
#endif // mSIDM_TIMESTEP
#ifdef rSIDM_DISSIPATION
    double RSIDM_f_diss;
#endif // rSIDM_DISSIPATION
#endif // rSIDM


#ifdef mSIDM
    unsigned int MSIDM_SEED = 16; /*!< For random angles in scattering */
#if defined(mSIDM_VDEP0) || defined(mSIDM_VDEP1) || defined(mSIDM_VDEP_MV) || defined(mSIDM_VDEP_RV)
    double MSIDM_vdep_w;
#endif // mSIDM_VDEP0 || mSIDM_VDEP1 || mSIDM_VDEP_MV || mSIDM_VDEP_RV

#ifdef mSIDM_VDEP0
    double MSIDM_vdep_alpha;
    double MSIDM_vdep_beta;
#endif // mSIDM_VDEP0

#ifdef mSIDM_VDEP1
    double MSIDM_vdep_u;
#endif // mSIDM_VDEP1

#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
    char MSIDM_CrossSection[200];
    char MSIDM_DCS_params[200];
#if defined(rSIDM) && !defined(fSIDM)
    double MSIDM_thetaC_fwd = 0.0;
    double MSIDM_thetaC_bwd = 0.0;
#else // rSIDM && !fSIDM
    double MSIDM_thetaC_fwd = 0.1;
    double MSIDM_thetaC_bwd = 0.1;
#endif //  rSIDM && !fSIDM
    double MSIDM_v_min = 0.01;
    double MSIDM_v_max = 1000.0;
    double MSIDM_reserve_fact = 1.5;
    int MSIDM_len_xh = 256;
    int MSIDM_len_v = 2048;
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION

#ifdef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    unsigned int MSIDM_first_target_id;
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
#endif // mSIDM





#if defined(mSIDM) || defined(mIDM)
#ifndef ADAPTGRAVSOFT
    int DesNumNgb_pt1 = 48;
    double MaxNumNgbDeviation_pt1 = 2;
#endif // !ADAPTGRAVSOFT
#endif // mSIDM || mIDM


#ifdef SUBFIND
    int DesLinkNgb = 20;
    double ErrTolThetaSubfind = 0.45;
#endif // SUBFIND

    double ErrTolForceAcc = 0.005; /*!< parameter for relative opening criterion in tree walk */
    double MinGasHsmlFractional = 0.001; /*!< minimum allowed SPH smoothing length in units of SPH gravitational
           softening length */

#ifdef MAXHSML
    double MaxHsml;
#endif // MAXHSML

    double MinSizeTimestep = 0; /*!< minimum allowed timestep. Normally, the simulation terminates if the
              timestep determined by the timestep criteria falls below this limit. */
    double MaxSizeTimestep = 0.05; /*!< maximum allowed timestep */

    double MaxRMSDisplacementFac = 0.25; /*!< this determines a global timestep criterion for cosmological simulations
           in comoving coordinates.  To this end, the code computes the rms velocity
           of all particles, and limits the timestep such that the rms displacement
           is a fraction of the mean particle separation (determined from the
           particle mass and the cosmological parameters). This parameter specifies
           this fraction. */

#if GADGET_HYDRO == HYDRO_MFM
    double ArtBulkViscBeta = 0.0; /*!< Sets the parameter \f$\beta\f$ of the artificial viscosity */
    /*KD: Note that for time step reason this should not be set to zero*/
    double ArtBulkViscConst = 0.0; /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
#else // GADGET_HYDRO == HYDRO_MFM
    double ArtBulkViscBeta = 2.0; /*!< Sets the parameter \f$\beta\f$ of the artificial viscosity */
#ifdef TIME_DEP_ART_VISC
#if (TIME_DEP_ART_VISC == 0)
    double ArtBulkViscConst = 1.0; /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
#elif (TIME_DEP_ART_VISC == 3)
    double ArtBulkViscConst = 2.0; /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
#else // (TIME_DEP_ART_VISC == 1/2)
    double ArtBulkViscConst = 3.0; /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
#endif // (TIME_DEP_ART_VISC == 0)
#else // TIME_DEP_ART_VISC
#ifdef FIRST_ORDER_GRADIENTS
    double ArtBulkViscConst = 1.0; /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
#else // FIRST_ORDER_GRADIENTS
    double ArtBulkViscConst = 3.0; /*!< Sets the parameter \f$\alpha\f$ of the artificial viscosity */
#endif // FIRST_ORDER_GRADIENTS
#endif // TIME_DEP_ART_VISC
#endif // GADGET_HYDRO == HYDRO_MFM


#ifdef NAVIERSTOKES_TIMESTEP
    double AlphaViscTimestep = 0.1; /*!< dimensionless timestep parameter for viscosity */
#endif // NAVIERSTOKES_TIMESTEP

#ifdef ARTIFICIAL_CONDUCTIVITY
#ifdef TIME_DEP_ART_COND
    double ArtCondConstant = 0.4;
    double ArtCondMin = 0.04;
#else
    double ArtCondConstant = 0.2;
#endif // TIME_DEP_ART_COND
#endif // ARTIFICIAL_CONDUCTIVITY

    double CourantFac = 0.15; /*!< SPH-Courant factor */
    int DesNumNgb = default_kernel_ngb_number; /*!< Desired number of SPH neighbours */

#ifdef NEUTRINOS
    double Time_tree_on_nu;
#endif // NEUTRINOS



    double MaxNumNgbDeviationRel = 0.01; /*!< Maximum allowed deviation neighbour number */

    int ComovingIntegrationOn = 0; /*!< flags that comoving integration is enabled */
    int ICFormat = 2; /*!< selects different versions of IC file-format */
    int SnapFormat = 2; /*!< selects different versions of snapshot file-formats */
    int NumFilesPerSnapshot = 1; /*!< number of files in multi-file snapshot dumps */
    int NumFilesWrittenInParallel = 1; /*!< maximum number of files that may be written simultaneously when
               writing/reading restart-files, or when writing snapshot files */

    // on/off switches
    int SpectralCRsOn = 0;
    int BlackHolesOn = 0; // Switches black hole module on, see Springel&DiMatheo 2005
    int BlackHoleSubgridADOn = 0; // Main switch for AD model
    int CoolingOn = 0; /*!< flags that cooling is enabled */
    int StarformationOn = 0; /*!< flags that star formation is enabled */
    int MuppiOn = 0;

    int TypeOfTimestepCriterion = 0; /*!< gives type of timestep criterion (only 0 supported right now - unlike
           gadget-1.1) */

    /* variables that keep track of cumulative CPU consumption */
    double TimeLimitCPU = -1.0;

    double SofteningGas = 0, /*!< for type 0 */
        SofteningHalo = 0, /*!< for type 1 */
        SofteningDisk = 0, /*!< for type 2 */
        SofteningBulge = 0, /*!< for type 3 */
        SofteningStars = 0, /*!< for type 4 */
        SofteningBndry = 0; /*!< for type 5 */

    double SofteningGasMaxPhys = 0, /*!< for type 0 */
        SofteningHaloMaxPhys = 0, /*!< for type 1 */
        SofteningDiskMaxPhys = 0, /*!< for type 2 */
        SofteningBulgeMaxPhys = 0, /*!< for type 3 */
        SofteningStarsMaxPhys = 0, /*!< for type 4 */
        SofteningBndryMaxPhys = 0; /*!< for type 5 */

    double BufferSize = 20.0; /*!< size of communication buffer in MB */

    double PartAllocFactor; /*!< in order to maintain work-load balance, the particle load will usually
          NOT be balanced.  Each processor allocates memory for PartAllocFactor times
          the average number of particles to allow for that */

    double MaxTreeAllocFactor = 5; /*!<  Maximum tree allocation factor, used in forcetree.cpp */

    double GravityConstantInternal = 0; /*!< If set to zero in the parameterfile, the internal value of the
          gravitational constant is set to the Newtonian value based on the system of
          units specified. Otherwise the value provided is taken as internal gravity
          constant. */
    double InitGasTemp = 0.0; /*!< may be used to set the temperature in the IC's */
    double MinGasTemp = 0.0; /*!< may be used to set a floor for the gas temperature */

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)
    double ReferenceGasMass;
#endif // defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)

#ifdef NAVIERSTOKES
    double FractionSpitzerViscosity;
#endif // NAVIERSTOKES

#ifdef NAVIERSTOKES_CONSTANT
    double ShearViscosityTemperature;
#endif // NAVIERSTOKES_CONSTANT


#ifdef CONDUCTION
    double ConductionEfficiency; /*!< Thermal Conductivity */
    double MaxSizeConductionStep = 0.01;
#endif // CONDUCTION


#if defined(BLACK_HOLES) || defined(MV_KRUMHOLZ_MOLECULAR_FRACTION_CLUMPINESS)
    double massDMpart; // DM particle mass to use, unless a fixed mass is given in the header
#endif // defined(BLACK_HOLES) || defined(MV_KRUMHOLZ_MOLECULAR_FRACTION_CLUMPINESS)

#ifdef BLACK_HOLES
    int BlackHoleThermalFeedbackOn; // Switches on Thermal Feedback
    int BlackHoleKineticFeedbackOn = 0; // Switches on Kinetic Feedback
    double TimeBetOnTheFlyFoF = 1.03; // Frequency of Black Hole seeding
    double BlackHoleFoFMinMassForNewSeed; // Minmum FoF mass for seeding Black Holes (internal units)
    double SeedBlackHoleMass; // Mass of seeded Black Hole
    double BlackHoleSeedStarMassFraction = 0.02; // If > 0, seeding is on stellar body with reduced mass
    double BlackHoleSeedDMFraction = 0.05; // Star to DM fraction to exceed for seeding BH
    double BlackHoleSeedGasFraction = 0.1; // Gas fraction needede for seeding BH
    double BlackHoleFormationFactor = 0.01; // Maximum number of black holes per gas particle to form between restarts
    double BlackHoleNgbFactor = 1; // Boost of Black Hole neighbour number
    double BlackHoleMaxAccretionRadius = 100; // Maximum Black Hole accreation radius (internal units)
    int BlackHoleSwallowGasOn =
        2; // Removal of gas particles through black hole accretion (0=Off, 1=All, 2=Cold gas only)
    int BlackHoleIgnoreMomentum = 1; // Switches On/Off the transfer of momentum while swallowing
    int BlackHoleAccretionSlicesOn = 1; // Quantization for swallowing gas particles with All.Generations
    int BlackHoleDetails = 2; // Switches On/Off writing of black hole details by each MPI rank
    int BlackHoleRepositioningOn = 0; // Switches On/Off black hole repositioning, see Springel&DiMatheo 2005
    int BlackHoleHotColdAccretionOn = 1; // Switches On/Off hot/cold accreation, see Steinborn+ 2015
    double BlackHoleAccretionFactor; // Used as alpha Factor in Bondi for total (or hot gas) accreation
    double BlackHoleColdAccretionFactor; // Used for alpha for the cold gas accreation, values see Gaspari+ 2013
    int BlackHoleVelocityInBondiOn = 1; // Use the relative velocity in Bondi Formulae
    int BlackHoleVariableAccretionFactorOn = 0; // Switches on the scaled accreation factor, see Booth&Schaye 2013
    double BlackHoleVariableAccretionSlope = 1; // Slope for variable accreation factor
    int BlackHoleRadiativeEfficiencyTreatment =
        0; /* How to handle efficiencies. 0 = fixed (Springel & Di Matteo 2005), 1 = Steinborn+15 without outflow model,
              2 = Steinborn+15 with outflow model, 3 = model by L. Sala.
              If BlackHoleSubgridADOn = 0, the efficiency determined by the chosen model sets the radiative efficiency,
              and it also used in the computation of the Eddington accretion rate and the BH accretion rate (radiative
              efficiency = accretion efficiency).
              If BlackHoleSubgridADOn = 1, the spin-dependent efficiency is used as
              accretion efficiency as well as in the Eddington accretion rate. With treatment 0, it is also the
              radiative efficiency. With treatment 1/2/3, the radiative efficiency is set by the chosen model. */
    double BlackHoleRadiativeEfficiency; // Classical fixed radiative efficiency (Springel & Di Matteo 2005)
    double BlackHolePhysMdotUpperLimit = 50; // Maximum limit for the accretion rate in Msun/yr
    double BlackHoleEddingtonFactor = 1; // Maximum allowd accreation in terms of Eddington accreation

    double BlackHoleFeedbackFactor; // Fixed feedback factor for standard model
    double BlackHoleRadioModeFeedbackBoost; // Boosts for FeedbackFactor in radi mode, see Fabjan+ 2010
    double BlackHoleRadioModeTreshold; // Treshhold for switching to radio mode
    int BlackHoleLimitFeedbackOn; // Limits feedback (0=Off, 1=to hot particles only, 2=maximum pressure given by halo,
                                  // see Vogelsberger+ 2014, 3=1+2)
    double BlackHoleLimitMaxTemp = 5e8; // Limits feedback to heat individual particle to maximum temperature
    double BlackHoleColdGasTemperatureTresh =
        5e4; // Temperature Limit to seperate cold gas from hot particles (e.g. 5e4)

    int BlackHoleFrictionForceOn = 1; // Switches On/Off friction force for black holes, see Hirschmann+ 2014
    int BlackHoleMergerCondition = 1;
    double BlackHoleMergeDistFrac = 5; // Fraction of relative velocity compared to sound velocity for merging
    double BlackHoleMergeCsndFrac = 0.5; // Distance compared to softening length for merging, see Hirschmann+ 2014
    double BlackHoleMergeBindingFrac = -1; // Fraction of binding energy for merging, see Hirschmann+ 2014

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    double BlackHoleAccretionDiskAlpha =
        0.1; // Value for the viscosity prescription ("alpha"-prescription) as in Shakura&Sunyaev1973
    double SeedBlackHoleSpin = 0; // Dimensionless spin parameter of seeded Black Hole
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION


#endif // BLACK_HOLES

#if defined(UM_CHEMISTRY) && defined(UM_CHEMISTRY_INISET)
    /* used if read initial composition from the parameter file */
    double START_elec = 5.e-4;
    double START_HI = 0.926;
    double START_HII = 3.e-4;
    double START_HM = 7.e-12;
    double START_HeI = 0.073;
    double START_HeII = 0;
    double START_HeIII = 0;
    double START_H2I = 1.e-6;
    double START_H2II = 6.e-14;
    double START_HD = 7.e-10;
    double START_DI = 3.5e-5;
    double START_DII = 4.e-9;
    double START_HeHII = 4.e-15;
#endif // defined(UM_CHEMISTRY) && defined(UM_CHEMISTRY_INISET)

#ifdef SFR
    double CritOverDensity = 57.7;
    double CritPhysDensity = 0;
#ifndef LT_STELLAREVOLUTION
    double FactorSN = 0.1;
#endif // LT_STELLAREVOLUTION
    double FactorEVP = 1000;
    double TempSupernova = 1.0e8;
    double TempClouds = 1000;
    double MaxSfrTimescale = 1.5;

#ifdef WINDS
    double WindEfficiency = 2.0;
    double WindEnergyFraction = 0.5;
    double WindFreeTravelMaxTimeFactor =
        0.025; /* maximum free travel time in units of the Hubble time at the current simulation redshift */
    double WindFreeTravelDensFac = 0.5;
#endif // WINDS
#endif // SFR

#ifdef CS_DECOUPLING
    double DecouplingParam = 50.;
#endif // CS_DECOUPLING

#ifdef DARKENERGY
#ifdef TIMEDEPDE
    char DarkEnergyFile[100]; /*!< tabelized w for equation of state */
#else // TIMEDEPDE
    double DarkEnergyParam; /*!< fixed w for equation of state */
#endif // TIMEDEPDE
#endif // DARKENERGY

#ifdef RESCALEVINI
    double VelIniScale; /*!< Scale the initial velocities by this amount */
#endif // RESCALEVINI

#ifdef TIME_DEP_ART_VISC
#if (TIME_DEP_ART_VISC == 0)
    double ViscositySourceScaling = 5; /*!< Given sourceterm in viscosity evolution */
    double ViscosityDecayLength = 2; /*!< Number of h for the viscosity decay, Dolag+2005, after Eq. 11 */
    double ViscosityAlphaMin = 0.1; /*!< Minimum of allowed viscosity parameter */
#elif (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2)
    double ViscositySourceScaling = 3.0; /*!< Given sourceterm in viscosity evolution */
    double ViscosityDecayLength = 3.0; /*!< Number of h for the viscosity decay, Dolag+2005, after Eq. 11 */
    double ViscosityAlphaMin = 0.05; /*!< Minimum of allowed viscosity parameter */
#elif (TIME_DEP_ART_VISC == 3)
    double ViscositySourceScaling = 1.25; /*!< Given sourceterm in viscosity evolution */
    double ViscosityDecayLength = 1.5; /*!< Number of full Courant Timesteps to decay alpha */
    double ViscosityAlphaMin = 0.2; /*!< Minimum of allowed viscosity parameter */
#endif // (TIME_DEP_ART_VISC == 0,1,2,3)
#endif // TIME_DEP_ART_VISC

#ifdef ARTIFICIAL_MAGNETIC_DISSIPATION
    double ArtificialMagneticDissipationConstant = 0.1;
#ifdef TIME_DEP_MAGN_DISP
    double ArtificialMagneticDissipationMin = 0.02;
    double ArtificialMagneticDissipationSource = 15.0;
    double ArtificialMagneticDissipationDecaytime = 0.3;
#endif // TIME_DEP_MAGN_DISP
#endif // ARTIFICIAL_MAGNETIC_DISSIPATION

#ifdef DIVBCLEANING_TRICCO
    // DivBcleaningParabolicSigma has a fixed value for 2D/3D
#ifdef TWODIMS
    double DivBcleaningParabolicSigma = 0.3;
#else // TWODIMS
    double DivBcleaningParabolicSigma = 1.0;
#endif // TWODIMS
    // DivBOverCleanFac gives an additional timestep constraint for the cleaning of divB
    double DivBOverCleanFac = 2.0; // found empirically to be a good middle-ground between accuracy and speed
#endif // DIVBCLEANING_TRICCO


#ifdef LMB_MFM_ENERGY_CLOSURE_LIMITER
    /* Face-closure limiter on the rate of change of the internal energy. Bounds |du| over an
       interval by (dt c_sig / L) * max(TolMin, 1/(ClosureWeight * residual)) * u. */
    double MfmEnergyClosureMin = 0.001; /*!< closure residual sum|A_ij| / A_cell above which a cell is
                                             considered ill-conditioned and eligible for limiting */
    double MfmEnergyClosureWeight = 10.0; /*!< converts the closure residual into a tolerance; larger
                                               values tighten the limit on a given residual */
    double MfmEnergyTolMin = 0.025; /*!< floor on the relative du tolerance, reached once the closure
                                         residual is large; bounds how hard a cell can ever be throttled */
#endif // LMB_MFM_ENERGY_CLOSURE_LIMITER

#ifdef DIVBCLEANING_DEDNER
#if GADGET_HYDRO == HYDRO_MFM
    double DivBcleaningParabolicSigma = 0.1;
#else // GADGET_HYDRO == HYDRO_MFM
    double DivBcleaningParabolicSigma = 2.0;
#endif // GADGET_HYDRO == HYDRO_MFM
    double DivBcleaningHyperbolicSigma = 1.0;
#if GADGET_HYDRO != HYDRO_MFM
    double DivBcleaningQ = 0.5;
#endif // GADGET_HYDRO != HYDRO_MFM
#endif // DIVBCLEANING_DEDNER

#ifdef MAGNETIC
#ifdef BINISET
    /*!< Initial values for B */
    double BiniX = 0.0;
    double BiniY = 0.0;
    double BiniZ = 0.0;
#endif // BINISET

#ifdef MAGNETIC_SN_SEEDING
    double SnSeedRadius = 0.005; // SN remnant size in kpc/h
    double SnSeedBubble = 0.025; // SN bubble size in kpc/h
    double SnSeedField = 1e-5; // SN seed field in Gauss
    double SnSeedSoftening = 0.25; // softening in units of hsml
#endif // MAGNETIC_SN_SEEDING

#ifdef MAGNETIC_DIFFUSION
    double MagneticEta;
    /*!< Sets the parameter \f$\alpha\f$ of the magnetic diffusion
        For galaxies: 2.0 (~10^-27)
        For clusters: 20.0 (~10^-28) */
#endif // MAGNETIC_DIFFUSION
#ifdef MAGNETIC_DIFFUSION_LIMIT
    double MagneticDiffSpeed = 5.0; // [km/s]
#endif // MAGNETIC_DIFFUSION_LIMIT
#endif // MAGNETIC


#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    double Epsilon = 0.01;
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#ifdef SFR
    double SFfactor = 0.3; /*!< the expected maximum factor of proportionality between TotN_gas and TotN_star */
    int Generations = 2;
#endif // SFR

#ifdef LT_STELLAREVOLUTION
#ifdef LT_STOP_COOL_BELOW_Z
    double CoolStop_redshift;
#endif // LT_STOP_COOL_BELOW_Z
#ifdef LT_POPIII
    double PopIII_Zlimit = 1e-3;
    int PopIII_IMF_idx = 1;
#endif // LT_POPIII
#ifdef LT_STARBURSTS
    int StarBurstCondition;
    double SB_Density_Thresh;
    double SB_DEntropy_Thresh;
#endif // LT_STARBURSTS

#ifdef LT_SMOOTH_Z
#if defined(LT_SMOOTH_SIZE) && !defined(LT_SMOOTH_NGB)
    double SmoothRegionSize, SmoothRegionSizeMax;
#endif // defined(LT_SMOOTH_SIZE) && !defined(LT_SMOOTH_NGB)
#endif // LT_SMOOTH_Z

    int TestSuite = 0;
    int InfNeighNum; /*!< minimum number of neighbours for metal and egy spreading */
    int DesNumNgbSN = default_kernel_ngb_number; /*!< desired number of neighbours for sn spreading */
    double MaxNumNgbDevRelSN = 0.01; /*!< maximum deviation in number of neighbours for sn spreading */
    char SFsFileName[300], IMFsFileName[300];
    double SnII_InfMass = 8; /*!< SnII threshold (e.g. 8 Msun) */
    double MinBinSystemMass = 3; /*!< min bin. system mass */
    double MaxBinSystemMass = 16; /*!< max bin. system mass */
    double BinarySystemFrac = 0.1; /*!< bin. system fraction */
    double MinStarMass_inBinSystem = 0.8; /*!< min star mass in SnIa binary systems */

#ifdef LT_HOT_EJECTA
    double EgySpecEjecta;
#endif // LT_HOT_EJECTA

    char SnIaDataFile[300], SnIIdataFileName[300], AGBdataFileName[300];
    int SnIaDataSetNum, SnIIDataSetNum, AGBDataSetNum;
    double LongLiving_Step_Prec = 0.02;
    /* energy provided by sn explosions */
    double SnIaEnergy = 1e51;
    double SnIIEnergy = 1e51;
    /* define IRA range */
    double metIRAThMass = 0, egyIRAThMass = 0;
    double SnII_Step_Prec = 0.1;
    double MinSpreadLength = 0.05;
    double MinChemTimeStep = 3.1536e12;
    double MaxChemSpreadL = 150;
#endif // LT_STELLAREVOLUTION

#ifdef LT_METAL_COOLING_WAL
    char WalCoolTables_Path[200];
#endif // LT_METAL_COOLING_WAL

#ifdef LT_ADD_GAL_TO_SUB
    char BC_SED_Path[200];
#endif // LT_ADD_GAL_TO_SUB

#ifdef GENERATE_GAS_IN_ICS
#ifdef GENERATE_GAS_TG
    int GenGasRefFac;
#endif // GENERATE_GAS_TG
#endif // GENERATE_GAS_IN_ICS

#ifdef ADAPTGRAVSOFT
    int AGS_DesNumNgb;
    double AGS_MaxNumNgbDeviation;
#endif // ADAPTGRAVSOFT

#ifdef SPHERICAL_BOUNDARY
    struct
    {
        enum boundary_type
        {
            reflect = 0,
            stop = 1
        };
        double radius;
        int type;
    } spherical_boundary;
#endif // SPHERICAL_BOUNDARY

#if (GADGET_HYDRO == HYDRO_MFM) && !defined(MFM_CONSERVATIVE_TOTAL_ENERGY)
    double EkinSwitchFraction = 0.0;
    double EpotSwitchFraction = 0.0;
#endif // GADGET_HYDRO == HYDRO_MFM && !defined(MFM_CONSERVATIVE_TOTAL_ENERGY)

#ifdef ANALYTIC_POTENTIALS
    int AP_doHalo = 0;
    int AP_doBulge = 0;
    int AP_doThickDisk = 0;
    int AP_doThinDisk = 0;
    int AP_doDisk1 = 0;
    int AP_doDisk2 = 0;
    int AP_doKing = 0;
    double AP_HaloRho = 0.0;
    double AP_HaloR = 0.0;
    double AP_BulgeM = 0.0;
    double AP_BulgeA = 0.0;
    double AP_ThickDiskM = 0.0;
    double AP_ThickDiskA = 0.0;
    double AP_ThickDiskB = 0.0;
    double AP_ThinDiskM = 0.0;
    double AP_ThinDiskA = 0.0;
    double AP_ThinDiskB = 0.0;
    double AP_Disk1M = 0.0;
    double AP_Disk1A = 0.0;
    double AP_Disk1B = 0.0;
    double AP_Disk2M = 0.0;
    double AP_Disk2A = 0.0;
    double AP_Disk2B = 0.0;
    double AP_KingRho0 = 0.0;
    double AP_KingRc = 0.0;
#endif // ANALYTIC_POTENTIALS

#ifdef GT_DUST
    double DeltaVelDissipationConstant = 2.0;
#endif // GT_DUST

#ifdef DISK_SLICE
    double qDisk = 1.5; // default value for a Keplerian disk
    double OmegaDisk = 1;
#endif // DISK_SLICE

    int StopAfterNSteps = -1;
    double StopAtA = 9999;

    /*###################################*/
    /*  End Parameters from param file   */
    /*###################################*/
} All;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP


#ifdef mSIDM
extern class scatter::MSIDM msidm;
#endif // mSIDM


#if defined(mSIDM) || defined(mIDM)
extern class scatter::GaussLegendre gauss_legendre;
#endif // mSIDM || mIDM

#ifdef ANALYTIC_POTENTIALS
extern class ANALYTIC_POTENTIAL analytic_potentials;
#endif // ANALYTIC_POTENTIALS

/**
 * @brief This structure contains global data for all tasks, but which are allowed to change
 *
 * This structure contains data which is the SAME for all tasks as the All data structure
 * but its values are computed on the fly thus it can change in shape betwen restarts
 */
extern struct derived_global_data_all_processes
{
    int DoDomainDecompositionTBFull, DoDomainDecompositionTBDiffuse, DoDomainDecompositionTBFix;
} AllDerived;

/**
 * @brief Particle data structure
 *
 * This structure holds all the information that is
 * stored for each particle of the simulation.
 */
extern ALIGN(32) struct particle_data
{
    MyLongDouble Pos[3]; /*!< particle position at its current time */
    short int Type; /*!< flags particle type.  0=gas, 1=halo, 2=disk, 3=bulge, 4=stars, 5=bndry */
    short int TimeBin;
    MyIDType ID;

    integertime Ti_begstep; /*!< marks start of current timestep of particle on integer timeline */
    integertime Ti_current; /*!< current time of the particle */

    MyFloat Mass; /*!< particle mass */

    MyFloat Vel[3]; /*!< particle velocity at its current time */
    MyFloat dp[3];

    MyAtLeastDouble GravAccel[3]; /*!< particle acceleration due to gravity */
#ifdef PMGRID
    MyAtLeastDouble GravPM[3]; /*!< particle acceleration due to long-range PM gravity force */
#endif // PMGRID

#ifdef FORCETEST
    MyAtLeastDouble GravAccelDirect[3]; /*!< particle acceleration calculated by direct summation */
#endif // FORCETEST

#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL)
    MyFloat Potential; /*!< gravitational potential */
#endif // defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY) || defined(OUTPUTPOTENTIAL)

#ifdef AR_FIX_ASSIGN
    // particles with only ngb from other CPUs will
    // assign wrong values on non summed values in out2particle density (e.g. .BH_TimeBinGasNeighbor)
    int assigned;
#endif // AR_FIX_ASSIGN

    MyFloat OldAcc; /*!< magnitude of old gravitational force. Used in relative opening
           criterion */
#if defined(EVALPOTENTIAL) && defined(PMGRID)
    MyFloat PM_Potential;
#endif // defined(EVALPOTENTIAL) && defined(PMGRID)

#ifdef mSIDM_VMAX
    MyAtLeastDouble mSIDM_v_max;
#endif // mSIDM_VMAX

#if defined(mSIDM) && defined(SCATTERING_HMIN)
    MyFloat mSIDM_h_min;
#endif // mSIDM %% SCATTERING_HMIN




// for testing. Counts how many times particle has scattered
#ifdef SCATTERING_COUNT_INDI
    int scatterCount = 0;
#endif // SCATTERING_COUNT_INDI

    MyFloat Hsml;
    MyFloat NumNgb;

#ifdef PB_BH_HSML_PREVENT_HOLE
    MyFloat rGasNgb_Min;
    MyFloat hsml_At_rGasNgb_Min;
    MyFloat hsmlGasNgb_Min;
    MyFloat r_At_hsmlGasNgb_Min;
#endif // PB_BH_HSML_PREVENT_HOLE

#ifdef PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT
    short int Flag_AGN_On; /* 1 = AGN is currently active, 0 = Not active. */
    MyFloat time_AGN_Last; /* Time when AGN transitioned to last state (active or non-active). */
#endif // PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT

#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
    MyFloat GasMass_Cone_Up; /* Gas mass inside Upward-Cone around BH. */
    MyFloat GasMass_Cone_Dn; /* Gas mass inside Downward-Cone around BH. */
    int Np_Cone_Up;
    int Np_Cone_Dn;
    MyFloat UnitVector_AGN_Wind[3]; /* Direction for AGN Wind Kick. */
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE

#if defined(DISKPOT) && !defined(SUBFIND) && !defined(ORDER_SNAPSHOTS_BY_ID)
    int GrNr;
#endif // defined(DISKPOT) && !defined(SUBFIND) && !defined(ORDER_SNAPSHOTS_BY_ID)

#ifdef SUBFIND
    int GrNr;
    int SubNr;
    int DM_NumNgb;
#ifdef KD_MAIN_HALO
    int SubLevel;
#endif // KD_MAIN_HALO
    unsigned short targettask, origintask2;
    int origintask, submark, origindex;
    MyFloat DM_Hsml;
    union {
        MyFloat DM_Density;
        MyFloat DM_Potential;
    } u;
    union {
        MyFloat DM_VelDisp;
        MyFloat DM_BindingEnergy;
    } v;
#ifdef DENSITY_SPLIT_BY_TYPE
    union {
        MyFloat int_energy;
        MyFloat density_sum;
    } w;
#endif // DENSITY_SPLIT_BY_TYPE

#ifdef SAVE_HSML_IN_IC_ORDER
    MyIDType ID_ic_order;
#endif // SAVE_HSML_IN_IC_ORDER
#ifdef SUBFIND_ALTERNATIVE_COLLECTIVE
    peanokey Key;
#endif // SUBFIND_ALTERNATIVE_COLLECTIVE
#endif // SUBFIND

#if defined(ORDER_SNAPSHOTS_BY_ID) && !defined(SUBFIND)
    int GrNr;
    int SubNr;
#endif // defined(ORDER_SNAPSHOTS_BY_ID) && !defined(SUBFIND)

    float GravCost[Gravcostlevels]; /*!< weight factor used for balancing the work-load */
    int dt_step;

    union {
        unsigned int BHID;
        unsigned int MetID;
    } pt;

    int TrueNGB; /*!< Number of neighbours inside hsml */

#ifdef ADAPTGRAVSOFT
    MyFloat AGS_Density; /* !< (mass/number) density of particle */
#ifdef AGS_OUTPUTNGBS
    int AGS_REALNumNgb;
#endif // AGS_OUTPUTNGBS
    MyFloat AGS_NumNgb;
    MyFloat AGS_Hsml;
    MyFloat AGS_zeta; /*!< factor in the correction term */
    MyFloat AGS_omega; /*!< factor in the correction term */
#ifdef AGS_OUTPUTCORR
    MyFloat AGS_corr; /*!< factor in the correction term */
#endif // AGS_OUTPUTCORR
#endif // ADAPTGRAVSOFT

#if defined(FOF_EXTENDED_PROPERTIES) && !defined(SUBFIND)
    int GrNr;
    unsigned short targettask, origintask;
#endif // defined(FOF_EXTENDED_PROPERTIES) &&  !defined(SUBFIND)

#if defined(GM_STARDENSITY) || defined(GM_USE_STARDENS_AND_POTMIN)
    MyFloat StarHsml;

    MyFloat StarNumNgb;
    MyFloat StarDensity; /* SPH-smoothed stellar density of star particles */

#ifdef EVALPOTENTIAL
    MyFloat SmoothPot; /* SPH-smoothed grav potential of star  particles
              WARNING, smoothed on  neighbouring star particles ONLY */
#endif // EVALPOTENTIAL
#endif // defined(GM_STARDENSITY) || defined(GM_USE_STARDENS_AND_POTMIN)
} *__restrict__ P, /*!< holds particle data on local processor */
    *__restrict__ DomainPartBuf; /*!< buffer for particle data used in domain decomposition */

#ifndef GDE_LEAN
#define GDE_TIMEBEGIN(i) (P[i].a0)
#define GDE_VMATRIX(i, a, b) (P[i].V_matrix[a][b])
#define GDE_INITDENSITY(i) (P[i].init_density)
#else // GDE_LEAN
#define GDE_TIMEBEGIN(i) (All.TimeBegin)
#define GDE_VMATRIX(i, a, b) (0.0)
#define GDE_INITDENSITY(i) (All.GDEInitStreamDensity)
#endif // GDE_LEAN

#ifdef SFR
#define MPP(i) MetP[P[(i)].pt.MetID]
#endif // SFR

#ifdef BLACK_HOLES
#define BPP(i) BHP[P[(i)].pt.BHID]
/**
 * @brief Black Hole particle data structure
 *
 * This structure contains all the information that is necessary
 * for a black hole particle in the simulation.
 */
extern struct bh_particle_data
{
    unsigned int PID;
    MyFloat StellarAge; /*!< formation time of star particle */
    MyIDType SwallowID;
    MyFloat SwallowPot;
    int BH_CountProgs;
    MyAtLeastDouble BH_Mass;
    MyFloat BH_Mdot;
    int BH_TimeBinGasNeighbor;
    MyFloat BH_Density;
    MyFloat BH_Entropy;
    MyFloat BH_SurroundingGasVel[3];

    MyFloat BH_ColdDensity;
    MyFloat BH_ColdEntropy;
    MyFloat BH_SurroundingColdGasVel[3];

    MyFloat BH_HotDensity;
    MyFloat BH_HotEntropy;
    MyFloat BH_SurroundingHotGasVel[3];

#ifdef LB_PRESSURE_DEPENDENT_ACCRETION
    MyFloat BH_SurroundingMetallicity;
    MyFloat BH_SurroundingGasPressure;
    MyFloat BH_SurroundingGasMass;
    MyFloat BH_SurroundingTemperature;
    MyFloat BH_SfrThreshold;
#endif // LB_PRESSURE_DEPENDENT_ACCRETION

#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE_AD_MASSWEIGHTS
    MyFloat BH_TotalSurroundingMass;
#endif

    MyFloat BH_accreted_Mass;
    MyFloat BH_accreted_BHMass;

    MyFloat BH_accreted_momentum[3];

    int BH_flag_merged;
    MyAtLeastDouble BH_final_spin[3];

    MyFloat BH_SurroundingVel[3];
    MyFloat BH_SurroundingDensity;
    MyFloat BH_sigma;
    MyFloat BH_bmax;

    MyFloat BH_TotalFeedbackEfficiency;

    MyFloat mean_hsml;
    MyFloat mean_rho;

    MyLongDouble BH_MinPotPos[3];
    MyFloat BH_MinPot;

#if (defined(GM_MUPPI) && defined(GM_MV_ANGMOM)) ||                                                                    \
    defined(LS_BLACK_HOLES_SPIN_EVOLUTION) /* GM note: this will be used in BH+MUPPI. LS note: this is also used in    \
                                              for BH spin evolution. */
    MyFloat Vphi[3], VphiMod;
#endif // defined(GM_MUPPI) && defined(GM_MV_ANGMOM) || defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
#ifdef AD_DYNFRIC
    MyFloat BHDF_DynFric[3];
    int BHDF_ngb;
#endif // AD_DYNFRIC

#if defined(GM_REPOSITION_ON_STARDENSITY_MAX) || defined(GM_USE_STARDENS_AND_POTMIN) || defined(GM_STARDENSITY)
    MyLongDouble BH_StarDensPos[3];
    MyFloat BH_StarDens;
#ifdef GM_USE_VELS_ON_REPOSITIONING
    MyLongDouble BH_StarDensVel[3];
#endif // GM_USE_VELS_ON_REPOSITIONING
#endif // defined(GM_REPOSITION_ON_STARDENSITY_MAX) || defined(GM_USE_STARDENS_AND_POTMIN)

#ifdef PB_BH_HSML_PREVENT_HOLE
    MyFloat rGasNgb_Min;
    MyFloat hsml_At_rGasNgb_Min;
    MyFloat hsmlGasNgb_Min;
    MyFloat r_At_hsmlGasNgb_Min;
#endif // PB_BH_HSML_PREVENT_HOLE

#ifdef PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT
    short int Flag_AGN_On; /* 1 = AGN is currently active, 0 = Not active. */
    MyFloat time_AGN_Last; /* Time when AGN transitioned to last state (active or non-active). */
#endif // PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT

#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
    MyFloat GasMass_Cone_Up; /* Gas mass inside Upward-Cone around BH. */
    MyFloat GasMass_Cone_Dn; /* Gas mass inside Downward-Cone around BH. */
    int Np_Cone_Up;
    int Np_Cone_Dn;
    MyFloat UnitVector_AGN_Wind[3]; /* Direction for AGN Wind Kick. */
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    MyAtLeastDouble BH_Spin[3]; /* BH angular momentum vector (in internal units) */
    /* Intermediate quantities for the model */
    MyAtLeastDouble BH_EtaDisk; /* Radiative efficiency as computed consistently from the dimensionless BH spin
                                   parameter a = |J_BH|c/(G M_BH^2) */
    int BH_AccrCounter; /* Counter for the accretion episodes */
    MyAtLeastDouble BH_Mass_Sum; /* Cumulative BH mass over several accretion episodes (to be divided by the counter to
                                    obtain the average) (in internal units) */
    MyAtLeastDouble BH_accr_mass_Sum;
    MyAtLeastDouble BH_dt_Sum;
    MyAtLeastDouble Jgas_in_Kernel_Sum[3]; /* Cumulative AM of the gas in the BH kernel (to be divided by the counter to
                                              obtain the average) (in internal units) */
    int Mode; /* flag for self-gravitating regime */
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION

} *__restrict__ BHP, *__restrict__ DomainBHBuf;
#endif // BLACK_HOLES

/* [----------- start LT block ------------- ] */
#ifdef SFR /* for all versions of SFR define the structure which hosts the stellar data] */
extern struct met_particle_data
{
    MyFloat StellarAge; /*!< formation time of star particle */
    unsigned int PID;
#ifndef LT_STELLAREVOLUTION
    MyFloat Metallicity; /*!< metallicity of gas or star particle */
#else // LT_STELLAREVOLUTION
    float LastChemTime; /*!< the last and next next time of evolution for Ia and II */
    float iMass; /*!< initial mass of this SSP                               */
    float Metals[LT_NMetP]; /*!< the metal array (H is not stored here)                 */
    double weight; /*!< used in spreading                                      */

#ifdef LT_TRACK_CONTRIBUTES
    Contrib contrib;
#endif // LT_TRACK_CONTRIBUTES

#ifdef LT_ZAGE
    float ZAge;
#endif // LT_ZAGE

#ifdef LT_ZAGE_LLV
    float ZAge_llv;
#endif // LT_ZAGE_LLV

    int ChemTimeBin;

#ifdef LT_STARS_GUESSHSML
    MyFloat mean_hsml;
    MyFloat mean_rho;
#endif // LT_STARS_GUESSHSML

#ifdef LT_SMOOTH_Z
    float Zsmooth;
#endif // LT_SMOOTH_Z
#endif // LT_STELLAREVOLUTION
}

    *__restrict__ MetP, /*!< holds metal particle data on local processor */
        *__restrict__ DomainMetBuf; /*!< buffer for metal data in domain decomposition */
/* [----------- end LT block ------------- ]*/
#endif // SFR

/**
 * @brief SPH particle data structure
 *
 * This struture holds data that is stored for each SPH particle in addition to the collisionless
 * variables.
 */
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern struct sph_particle_data
{
    MyFloat Entropy; /*!< entropy (actually entropic function) of particle */
    MyFloat EntropyPred; /*!< predicted value of the entropy at the current time */
    MyFloat VelPred[3]; /*!< predicted SPH particle velocity at the current time */
    MyFloat MaxSignalVel; /*!< maximum signal velocity */

    MyFloat Density;
    MyFloat DtEntropy;

    MyFloat HydroAccel[3];
    MyFloat DivVel; /*!< local velocity divergence */

    MyFloat Pressure; /*!< current thermal pressure */
#if GADGET_HYDRO == HYDRO_SPH
    MyFloat DhsmlDensityFactor;
#elif GADGET_HYDRO == HYDRO_PESPH
    MyFloat NumDens; /* SPH number density */
    MyFloat DhsmlNumDensFactor; /*!< correction factor needed in Pressure-Entropy formulation of SPH */
    MyFloat DhsmlPressureFactor; /*!< correction factor needed in Pressure-Entropy formulation of SPH */
#elif GADGET_HYDRO == HYDRO_MFM
    MyFloat NumDens; /*!< SPH number density */
    MyFloat DhsmlNumDensFactor; /*!< correction factor needed in MFM */
    MyFloat DistNgbSqdMax; /*!< Maximum distance squared to all neighbours (Used in Scalar slope limiter) */
    MyFloat AlphaSlope[numvar]; /*!< Slope limiter weight factor */
    QFluidVector<mfm_numdims, MyAtLeastDouble> dQdt; /*!< Rate of change of conserved quantities */
    WFluidVector<mfm_numdims> Wprim; /*!< Primitive fluid vector */
    WFluidVector<mfm_numdims> Wmax; /*!< Max. values of W for all neighbours (used in slope limiter) */
    WFluidVector<mfm_numdims> Wmin; /*!< Min. values of W for all neighbours (used in slope limiter) */
    SymmetricMatrix<NUMDIMS> Bgrad; /*!< Matrix required for computing accurate gradients */
    Vector<mfm_numdims> Vel_smooth; /*!< averaged velocity over neighbours */
    Vector<NUMDIMS> gradW[numvar]; /*!< Gradients of primitive quantities */
    MyFloat ConditionNumber; /*!< Condition number of gradient calculation */
    MyFloat MaxVelSquareNgb; /*!< Maximum velocity squared (w.r.t. neighbours), used for energy/entropy switch */
    MyFloat InternalEnergy; /*!< internal energy of particle, used for evolution instead of entropy */
    MyFloat InternalEnergyPred; /*!< prediced internal energy at the current time */
    MyAtLeastDouble DtInternalEnergy;
    MyAtLeastDouble HydroAccelExtra[3];
    MyFloat DtEntropyExtra;
#ifdef MFM_FACE_CLOSURE_RESIDUAL
    MyFloat FaceAreaSum[3]; /*!< vector sum of the MFM effective face areas; the closure residual (ideally 0)
                                 used by the face-closure DtB and internal-energy limiters */
#endif // MFM_FACE_CLOSURE_RESIDUAL
#ifdef GT_DUST
    MyFloat AlphaSlope_dust[GT_NUM_DUST][numvar]; /*!< Slope limiter weight factor */
    std::array<QFluidVector<mfm_numdims>, GT_NUM_DUST> dQdt_dust; /*!< Rate of change of conserved quantities */
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wprim_dust; /*!< Primitive fluid vector */
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST>
        Wmax_dust; /*!< Max. values of W for all neighbours (used in slope limiter) */
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST>
        Wmin_dust; /*!< Min. values of W for all neighbours (used in slope limiter) */
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradW_dust; /*!< Gradients of primitive quantities */
    MyAtLeastDouble DtDeltaVelExtra[GT_NUM_DUST][3];
#endif // GT_DUST
#endif // GADGET_HYDRO

    MyFloat CurlVel; /*!< local velocity curl */
#ifdef OUTPUT_VORTICITY
    MyFloat Rot[3]; /*!< local velocity curl */
#endif // OUTPUT_VORTICITY

#ifdef NAVIERSTOKES
    MyFloat StressDiag[3];
    MyFloat StressOffDiag[3];
    MyFloat shear_visc_coeff;
#ifdef OUTPUT_ENERGY_VISC
    MyFloat visc_E;
#endif // OUTPUT_ENERGY_VISC
#ifdef NAVIERSTOKES_ANISO
    MyFloat P_aniso;
#endif // NAVIERSTOKES_ANISO
#endif // NAVIERSTOKES

#ifdef BLACK_HOLES
    MyIDType SwallowID;
    MyFloat Injected_BH_Energy;
#endif // BLACK_HOLES

#ifdef COOLING
    MyFloat elec; /*!< electron fraction, expressed as local electron number
         density normalized to the hydrogen number density. Gives
         indirectly ionization state and mean molecular weight. */
#endif // COOLING

#ifdef SFR
    MyFloat Sfr;
#endif // SFR
#ifdef WINDS
    MyFloat DelayTime; /*!< remaining maximum decoupling time of wind particle */
#endif // WINDS

#ifdef PB_BH_KIN_FEED_DECOUPLE_WIND
    MyFloat DecoupleTime_AGN; /*!< Remaining maximum decoupling time of AGN wind particle. */
#endif // PB_BH_KIN_FEED_DECOUPLE_WIND

#ifdef PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT
    MyFloat lastKick_Time_AGN; /*!< Time when gas particle is last kicked, check to prevent double-wind-kick. */
#endif // PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT

#if defined(JD_VTURB) || defined(KD_LIMIT_DIVV)
    MyFloat Vrms; /*!< RMS velocity inside kernel around Vbulk */
    MyFloat Vbulk[3]; /*!< Mean velocity inside kernel */
#endif // JD_VTURB || KD_LIMIT_DIVV
#ifdef JD_VTURB
#ifdef JD_DECOMPOSE_VTURB
    MyFloat Vtan; /*!< RMS of tangential component of velocity inside kernel around Vbulk */
    MyFloat Vrad; /*!< RMS of radial component of velocity inside kernel around Vbulk */
#endif // JD_DECOMPOSE_VTURB
#if (defined(JD_DPP) || defined(LMB_DPP))
    MyFloat Dpp; /*!< Reacceleration Coefficient as (Cassano+ '04) */
#endif // JD_DPP || LMB_DPP
#endif // JD_VTURB

#ifdef SHOCKFINDER
    MyFloat Shock_N1[3]; /*!< Parallel to shock > */
    MyFloat Shock_Up_V1[3]; /*!< Upstream parallel velocity > */
    MyFloat Shock_Down_V1[3]; /*!< Downstream parallel velocity > */
    MyFloat Shock_Up_Signal; /*!< Upstream signal velocity > */
    MyFloat Shock_Mach; /*!< Machnumber > */
    MyFloat Shock_Speed; /*!< Shock speed > */
    MyFloat Shock_Compress; /*!< Compression ratio > */
    MyFloat Shock_Up_Weight; /*!< Upstream weight > */
    MyFloat Shock_Down_Weight; /*!< Downstream weight > */
    MyFloat Shock_Up_Rho; /*!< Upstream density > */
    MyFloat Shock_Down_Rho; /*!< Downstream density > */
    MyFloat Shock_Up_Pressure; /*!< Upstream pressure > */
    MyFloat Shock_Down_Pressure; /*!< Downstream pressure > */
#ifndef SHOCKFINDER_SIMPLE
    MyFloat Shock_N2[3]; /*!< Perpendicular to shock > */
    MyFloat Shock_N3[3]; /*!< Perpendicular to shock > */
    MyFloat Shock_Up_V2[3]; /*!< Upstream perpendicular velocity > */
    MyFloat Shock_Down_V2[3]; /*!< Downstream perpendicular velocity > */
    MyFloat Shock_Up_V3[3]; /*!< Upstream perpendicular velocity > */
    MyFloat Shock_Down_V3[3]; /*!< Downstream perpendicular velocity > */
    MyFloat Shock_Up_Weight23[2]; /*!< Upstream perpendicular weights > */
    MyFloat Shock_Down_Weight23[2]; /*!< Downstream perpendicular weights > */
#endif // SHOCKFINDER_SIMPLE

#ifdef MAGNETIC
    MyFloat Shock_Mach_Alfven; /*!< Alfven Machnumber > */
    MyFloat Shock_Up_Alfven; /*!< Upstream Alfven velocity > */
    MyFloat Shock_Obliquity;
#endif // MAGNETIC
#endif // SHOCKFINDER


#ifdef TIME_DEP_ART_COND
    MyFloat Calpha, GradA[3];
#ifndef NOGRAVITY
#if (TIME_DEP_ART_COND > 0)
    MyFloat Climit;
#endif // (TIME_DEP_ART_COND > 0)
#ifdef TIME_DEP_ART_COND_GSMOOTH
    MyFloat Cgrav[3];
#endif // TIME_DEP_ART_COND_GSMOOTH
#endif // NOGRAVITY
#endif // TIME_DEP_ART_COND

#if defined(HIGHER_ORDER_INDUCTION) || defined(OUTPUT_GRADIENT_MATRICES)
    MyFloat dVelMatrix[3][3];
#endif
#ifdef OUTPUT_GRADIENT_MATRICES
    MyFloat DistanceMatrix[3][3];
#endif // OUTPUT_GRADIENT_MATRICES

#ifdef OUTPUT_GRADIENT_ORDER
    MyFloat GradientOrder;
#endif // OUTPUT_GRADIENT_ORDER

#ifdef MAGNETIC
    MyFloat B[3];
    MyFloat DtB[3];
    MyFloat divB;
    MyFloat magacc[3];
    MyFloat magcorr[3];

#ifdef WINDS
    MyFloat WindB[3];
#endif // WINDS

    MyFloat BPred[3];

#ifdef MAGNETIC_SN_SEEDING
    MyFloat MagSeedRate[3];
#endif // MAGNETIC_SN_SEEDING

#if defined(TIME_DEP_MAGN_DISP) || defined(DIVBCLEANING_DEDNER) || defined(DIVBCLEANING_TRICCO)
    MyFloat CleanVel;
#endif // TIME_DEP_MAGN_DISP || DIVBCLEANING_DEDNER || DIVBCLEANING_TRICCO
#ifdef TIME_DEP_MAGN_DISP
    MyFloat Balpha;
#if (TIME_DEP_MAGN_DISP == 0) || (TIME_DEP_MAGN_DISP == 1) || (TIME_DEP_MAGN_DISP == 3)
    MyFloat DtBalpha;
#endif // (TIME_DEP_MAGN_DISP==0) || (TIME_DEP_MAGN_DISP==1) || (TIME_DEP_MAGN_DISP==3)
#endif // TIME_DEP_MAGN_DISP
#ifdef DIVBCLEANING_TRICCO
    MyFloat Psi, PsiPred, DtPsi;
    MyFloat GradPsi[3];
#endif // DIVBCLEANING_TRICCO
#ifdef DIVBCLEANING_DEDNER
    MyFloat Phi, PhiPred, DtPhi;
    MyFloat GradPhi[3];
#endif // DIVBCLEANING_DEDNER

#if (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB) ||                              \
    defined(ROTB_VISCOSITY_LIMITER)
    MyFloat RotB[3];
#endif // (defined(TIME_DEP_MAGN_DISP) && (TIME_DEP_MAGN_DISP == 1)) || defined(OUTPUT_ROTB) ||
       // defined(ROTB_VISCOSITY_LIMITER)

#ifdef OUTPUT_MAGNETIC_CURVATURE
    MyFloat MagCurv[3];
#endif // OUTPUT_MAGNETIC_CURVATURE

#endif // MAGNETIC

    MyFloat alpha;
    MyFloat F_Balsara;
#ifdef TIME_DEP_ART_VISC
    MyFloat Dtalpha;
#if (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
    MyFloat visc_R, visc_OldDvel;
    integertime visc_Oldtime;
#ifdef ART_VISC_DEBUG
    MyFloat h2A;
#endif // ART_VISC_DEBUG
#endif // (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#endif // TIME_DEP_ART_VISC

#ifdef NAVIERSTOKES_TIMESTEP
    MyFloat ViscEntropyChange;
#endif // NAVIERSTOKES_TIMESTEP

#ifdef CONDUCTION_SATURATION
    MyFloat GradEntr[3];
#endif // CONDUCTION_SATURATION
#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
    MyFloat GradEnergy[3];
#endif // CONDUCTION_INCLUDEMAGNETIC || AA_OUTPUT_GRADS
#ifdef AA_OUTPUT_GRADS
    MyFloat GradPressure[3];
#endif // AA_OUTPUT_GRADS

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    MyFloat HI;
    MyFloat HII;

    MyFloat HeI;
    MyFloat HeII;
    MyFloat HeIII;

    MyFloat H2I;
    MyFloat H2II;

    MyFloat HM;

    MyFloat Gamma;
    MyFloat t_elec, t_cool;

#ifdef UM_CHEMISTRY
    MyFloat Um_MeanMolecularWeight;
    MyFloat HD;
    MyFloat DI;
    MyFloat DII;
    MyFloat HeHII;
#endif // UM_CHEMISTRY

#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

    // if we restart from a run w.o. soa, we still declare it in the all vars.
#if !defined(AR_SOA_LITE) || defined(AR_SOA_LITE_FROM_RESTART_AOS)
    short int wakeup; /*!< flag to wake up particle */
#endif // !defined(AR_SOA_LITE) || defined(AR_SOA_LITE_FROM_RESTART_AOS


#if (defined(SFR) && defined(MAGNETIC)) || defined(LT_STELLAREVOLUTION)
    float XColdCloud;
#endif // (defined(SFR) && defined(MAGNETIC)) || defined(LT_STELLAREVOLUTION)

#if defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)
    float Temperature;
#endif // defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)

#ifdef LT_STELLAREVOLUTION
    MyFloat MassRes;
    MyFloat EgyRes; /*!< external (Sn) energy resorvoir */
    float Metals[LT_NMetP]; /*!< H is not stored here */

#ifdef GL_CR_DUST
    float DustL[LT_NMetP]; /*!< the metal in large dust grains array (H is not stored here) */
    float DustS[LT_NMetP]; /*!< the metal in small dust grains array (H is not stored here) */
    float numSnIa, numSnII; /* number of supernovae from the last dust evolution calculation (spreaded!) */
#endif // GL_CR_DUST

    MyAtLeastDouble EgyStep; // GM: used to distribute SnIa egy with small chemsteps

#ifndef LT_LOCAL_IRA
    double mstar;
#endif // LT_LOCAL_IRA
#ifdef LT_TRACK_CONTRIBUTES
    Contrib contrib;
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_ZAGE
    MyFloat ZAge, ZAgeW;
#endif // LT_ZAGE
#ifdef LT_ZAGE_LLV
    MyFloat ZAge_llv, ZAgeW_llv;
#endif // LT_ZAGE_LLV
#ifdef LT_TRACK_WINDS
    MyFloat AvgHsml;
#endif // LT_TRACK_WINDS
#endif // LT_STELLAREVOLUTION

#ifdef LT_SMOOTH_Z
#ifdef LT_SMOOTH_ALLMETALS
    MyFloat Zsmooth[LT_NMetP];
#else // LT_SMOOTH_ALLMETALS
    MyFloat Zsmooth;
    MyFloat Zsmooth_a;
    MyFloat Zsmooth_b;
#ifdef LT_SMOOTH_SIZE
    MyFloat SmoothDens;
    MyFloat SmoothDens_b;
    int SmoothNgb;
#endif // LT_SMOOTH_SIZE
#endif // LT_SMOOTH_ALLMETALS
#endif // LT_SMOOTH_Z

#ifdef LT_SMOOTH_XCLD /* smooth the cloud fraction */
    float XCLDsmooth;
#endif // LT_SMOOTH_XCLD

#ifdef PHIDOT
    MyFloat PhiDot;
#endif // PHIDOT



#ifdef GM_DMVELS
    MyAtLeastDouble DmAveVel2, DmAveVel;
    int Ndm;
    MyAtLeastDouble Vwind, Etaw, LaunchTime;
#endif // GM_DMVELS

#if defined(GT_DUST) && defined(GT_NUM_DUST)
    MyAtLeastDouble DragCoefficient[GT_NUM_DUST];

    MyAtLeastDouble DeltaVel[GT_NUM_DUST][3];
    MyAtLeastDouble DtDeltaVel[GT_NUM_DUST][3];
    MyAtLeastDouble DeltaVelPred[GT_NUM_DUST][3];
    MyAtLeastDouble TotalDeltaVel[3];
    MyAtLeastDouble TotalDeltaVelPred[3];

    MyAtLeastDouble DtDustFrac[GT_NUM_DUST];
    MyAtLeastDouble DustFrac[GT_NUM_DUST];
    MyAtLeastDouble DustFracPred[GT_NUM_DUST];
    MyAtLeastDouble TotalDustFrac;
    MyAtLeastDouble TotalDustFracPred;

    MyAtLeastDouble InternalEnergySPH;
    MyAtLeastDouble DtInternalEnergySPH;
    MyAtLeastDouble InternalEnergySPHPred;

    MyAtLeastDouble StoppingTime;

    MyAtLeastDouble dust_fraction_smooth[GT_NUM_DUST];
    MyAtLeastDouble neigh_mass;

    MyAtLeastDouble DivVelMixture;

    MyAtLeastDouble DiffCoeff[GT_NUM_DUST];
#endif // GT_DUST && GT_NUM_DUST

#ifdef CS_DECOUPLING
    MyFloat CS_DensityOld;
#endif // CS_DECOUPLING

} *__restrict__ SphP, /*!< holds SPH particle data on local processor */
    *__restrict__ DomainSphBuf; /*!< buffer for SPH particle data in domain decomposition */
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#if defined(AR_PARALLEL_LOCAL_REFINE) || defined(AR_NODE_RECURSIVE_PARALLEL)
extern int
    NumUsedThreads; // when parallelise recursive region we check/set the number of free threads with atomic-capture
extern int NumberOfAllocatedNodeLocks;
#endif // defined(AR_PARALLEL_LOCAL_REFINE) || defined(AR_NODE_RECURSIVE_PARALLEL)

#ifdef AR_SOA_LITE
extern struct sph_soa
{
    short int *wakeup; /*!< flag to wake up particle */
} SphPSoa;

#define AR_ALLOC(name, typ, size, counter)                                                                             \
    {                                                                                                                  \
        int bytes;                                                                                                     \
        name = (typ *)mymalloc(#name, bytes = size * sizeof(typ));                                                     \
        counter += bytes;                                                                                              \
        bzero((void *)name, size * sizeof(typ));                                                                       \
    }
#define SPHP(i, prop) (SphPSoa.prop[i])
#else // AR_SOA_LITE
#define SPHP(i, prop) (SphP[i].prop)
#endif // AR_SOA_LITE

extern peanokey *DomainKeyBuf;

/**
 * @brief global state of system
 *
 * This structure contains the global state of the system, such as the total mass, energy, momentum, etc.
 */
extern struct state_of_system
{
    double Mass, EnergyKin, EnergyPot, EnergyInt,
#ifdef MAGNETIC_STATISTICS
        EnergyMag,
#endif // MAGNETIC_STATISTICS
        EnergyTot, Momentum[4], AngMomentum[4], CenterOfMass[4], MassComp[6], EnergyKinComp[6], EnergyPotComp[6],
        EnergyIntComp[6], EnergyTotComp[6], MomentumComp[6][4], AngMomentumComp[6][4], CenterOfMassComp[6][4];
#if defined(MAGNETIC_STATISTICS)
    double DivBerr;
#endif // defined(MAGNETIC_STATISTICS)
} SysState, SysStateAtStart, SysStateAtEnd;

/**
 * @brief Various structures for communication during the gravity computation.
 */
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern struct data_index
{
    int Task;
    int Index;
    int IndexGet;
} *DataIndexTable; /*!< the particles to be exported are grouped
              by task-number. This table allows the
              results to be disentangled again and to be
              assigned to the correct particle */

extern struct data_nodelist
{
    int NodeList[NODELISTLENGTH];
} *DataNodeList,
    *DataNodeListIn, /* ancestral modules (as FoF) will keep their DataNodeList in their exchange data structres */
    *DataNodeListGet; /* new modules (where nodelist is separated), for readability purposes will use DataNodeListIn and
                         DataNodeListGet */
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern struct gravdata_in
{
    MyLongDouble Pos[3];
    int Type;
#ifdef FOLLOW_ID
    MyIDType ID;
#endif // FOLLOW_ID
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat Soft;
#endif // ADAPTIVE_GRAVSOFT_FORGAS
    MyFloat OldAcc;
    int NodeList[NODELISTLENGTH];
#ifdef ADAPTGRAVSOFT
    MyFloat AGS_zeta, AGS_omega, AGS_Hsml, Mass;
#endif // ADAPTGRAVSOFT
} *GravDataIn, /*!< holds particle data to be exported to other processors */
    *GravDataGet; /*!< holds particle data imported from other processors */

extern struct gravdata_out
{
    MyAtLeastDouble Acc[3];
#ifdef EVALPOTENTIAL
    MyAtLeastDouble Potential;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
    MyFloat AGS_corr;
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
} *GravDataResult, /*!< holds the partial results computed for imported particles. Note: We use GravDataResult =
                      GravDataGet, such that the result replaces the imported data */
    *GravDataOut; /*!< holds partial results received from other processors. This will overwrite the GravDataIn array */

extern struct potdata_out
{
    MyLongDouble Potential;
} *PotDataResult, /*!< holds the partial results computed for imported particles. Note: We use GravDataResult =
                     GravDataGet, such that the result replaces the imported data */
    *PotDataOut; /*!< holds partial results received from other processors. This will overwrite the GravDataIn array */

extern struct phidotdata_out
{
    MyDouble PhiDot;
} *PhiDotDataResult, /*!< holds the partial results computed for imported particles. Note: We use GravDataResult =
                        GravDataGet, such that the result replaces the imported data */
    *PhiDotDataOut; /*!< holds partial results received from other processors. This will overwrite the GravDataIn array
                     */

extern struct info_block
{
    char label[4];
    char type[8];
    int ndim;
    int is_present[6];
} *InfoBlock;

/*! Header for the standard file format.
 */
extern struct io_header
{
    unsigned int npart[6]; /*!< number of particles of each type in this file */
    double mass[6]; /*!< mass of particles of each type. If 0, then the masses are explicitly
           stored in the mass-block of the snapshot file, otherwise they are omitted */
    double time; /*!< time of snapshot file */
    double redshift; /*!< redshift of snapshot file */
    int flag_sfr; /*!< flags whether the simulation was including star formation */
    int flag_feedback; /*!< flags whether feedback was included (obsolete) */
    unsigned int npartTotal[6]; /*!< total number of particles of each type in this snapshot. This can be
                   different from npart if one is dealing with a multi-file snapshot. */
    int flag_cooling; /*!< flags whether cooling was included  */
    int num_files; /*!< number of files in multi-file snapshot */
    double BoxSize; /*!< box-size of simulation in case periodic boundaries were used */
    double Omega0; /*!< matter density in units of critical density */
    double OmegaLambda; /*!< cosmological constant parameter */
    double HubbleParam; /*!< Hubble parameter in units of 100 km/sec/Mpc */
    int flag_stellarage; /*!< flags whether the file contains formation times of star particles */
    int flag_metals; /*!< flags whether the file contains metallicity values for gas and star
            particles */
    unsigned int npartTotalHighWord[6]; /*!< High word of the total number of particles of each type */
    int flag_entropy_instead_u; /*!< flags that IC-file contains entropy instead of u */
    int flag_doubleprecision; /*!< flags that snapshot contains double-precision instead of single precision */

    int flag_ic_info; /*!< flag to inform whether IC files are generated with ordinary Zeldovich approximation,
             or whether they ocontains 2nd order lagrangian perturbation theory initial conditions.
             For snapshots files, the value informs whether the simulation was evolved from
             Zeldoch or 2lpt ICs. Encoding is as follows:
             FLAG_ZELDOVICH_ICS     (1)   - IC file based on Zeldovich
             FLAG_SECOND_ORDER_ICS  (2)   - Special IC-file containing 2lpt masses
             FLAG_EVOLVED_ZELDOVICH (3)   - snapshot evolved from Zeldovich ICs
             FLAG_EVOLVED_2LPT      (4)   - snapshot evolved from 2lpt ICs
             FLAG_NORMALICS_2LPT    (5)   - standard gadget file format with 2lpt ICs
             All other values, including 0 are interpreted as "don't know" for backwards compatability.
           */
    float lpt_scalingfactor; /*!< scaling factor for 2lpt initial conditions */
    char fill[18]; /*!< fills to 256 Bytes */
    char names[15][2];
} header; /*!< holds header for snapshot files */

/*
 * Bit wise processed flag
 * ------------------
 */
enum ProcessedFlagFields : unsigned char
{
    ProcessedFlag_finished = 1,
    ProcessedFlag_communication = 1 << 1,
    // specific loops:
    ProcessedFlag_mfm_gradients = 1 << 2,
    ProcessedFlag_mfm_limiters = 1 << 3,
    ProcessedFlag_mfm_fluxes = 1 << 4,
    ProcessedFlag_shockfinder_mfm = 1 << 5,
    // no specific behavior in loop
    ProcessedFlag_null = 0
};

/*
 * Variables for Tree
 * ------------------
 */
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int Nexport, Nimport;
extern int BufferFullFlag;
extern int NextParticle;
extern int NextJ;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

extern int NextGroup;
extern int TimerFlag;

// note, the ALIGN(32) directive will effectively pad the structure size
// to a multiple of 32 bytes
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern ALIGN(32) struct NODE
{
    MyFloat center[3]; /*!< geometrical center of node */
    MyFloat len; /*!< sidelength of treenode */
#ifdef AR_XMAS_TREE_STRUCT
    unsigned char allocated;
#endif // AR_XMAS_TREE

    union {
        int suns[8]; /*!< temporary pointers to daughter nodes */
        struct
        {
            MyFloat s[3]; /*!< center of mass of node */
            MyFloat mass; /*!< mass of node */
            int sibling; /*!< this gives the next node in the walk in case the current node can be used */
            int nextnode; /*!< this gives the next node in case the current node needs to be opened */
            int father; /*!< this gives the parent node of each node (or -1 if we have the root node) */
            unsigned int bitflags; /*!< flags certain node properties */
        } d;
    } u;

    integertime Ti_current;
    double GravCost;

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
    MyFloat maxsoft; /*!< hold the maximum gravitational softening of particle in the
            node if either the ADAPTIVE_GRAVSOFT_FORGAS or the ADAPTGRAVSOFT option is selected */
#endif // defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)

#if defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
    int Flag;
#endif // defined(ADAPTGRAVSOFT) || defined(mSIDM) || defined(mIDM)
} *Nodes_base, /*!< points to the actual memory allocted for the nodes */
    *Nodes; /*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart]
           gives the first allocated node */
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef AR_TREEBUILD_PARALLEL
extern omp_lock_t *NodesLockList; // locks for the tree build
#endif // AR_TREEBUILD_PARALLEL

#ifdef ACC
/*! auxialiary variable used to set-up non-recursive walk */
// extern int last;
#define NTAB 1000
#define EN 64
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
/*! variables for short-range lookup table */
extern float shortrange_table[NTAB], shortrange_table_potential[NTAB];
extern double fac_intp;
extern MyFloat potcorr[EN + 1][EN + 1][EN + 1];
extern MyFloat fcorrx[EN + 1][EN + 1][EN + 1];
extern MyFloat fcorry[EN + 1][EN + 1][EN + 1];
extern MyFloat fcorrz[EN + 1][EN + 1][EN + 1];
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP
#endif // ACC_GRAVITY

extern struct extNODE
{
    MyLongDouble dp[3];
    MyFloat vs[3];
    MyFloat vmax;
    MyFloat hmax; /*!< maximum SPH smoothing length in node. Only used for gas particles */
    integertime Ti_lastkicked;
    int Flag;
} *Extnodes;

// Nitya declare target
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern struct extNODE *Extnodes_base;
extern int MaxNodes; /*!< maximum allowed number of internal nodes */
extern int Numnodestree; /*!< number of (internal) nodes in each tree */
extern int *Nextnode; /*!< gives next node in tree walk  (nodes array) */
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef GROUP_LEAVES
extern int *NextGroupedNode; /*!< gives next node in tree walk  (nodes array) */
#endif // GROUP_LEAVES

extern int *Father; /*!< gives parent node in tree (Prenodes array) */

#ifdef STATICNFW
extern double Rs, R200;
extern double Dc;
extern double RhoCrit, V200;
extern double fac;
#endif // STATICNFW

#ifdef UM_METAL_COOLING
/* --==[ link with LT_ stuffs]==-- */
extern float *um_ZsPoint, um_FillEl_mu, um_mass;
#endif // UM_METAL_COOLING

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
/* ----- chemistry part ------- */

// UM: H number fraction is 0.93, mass fraction is 0.76
#define H_number_fraction 0.76
#define He_number_fraction 0.06

/* ----- Tables ------- */
extern double T[N_T], J0_nu[N_nu], J_nu[N_nu], nu[N_nu];
extern double k1a[N_T], k2a[N_T], k3a[N_T], k4a[N_T], k5a[N_T], k6a[N_T], k7a[N_T], k8a[N_T], k9a[N_T], k10a[N_T],
    k11a[N_T];
extern double k12a[N_T], k13a[N_T], k14a[N_T], k15a[N_T], k16a[N_T], k17a[N_T], k18a[N_T], k19a[N_T], k20a[N_T],
    k21a[N_T];
extern double ciHIa[N_T], ciHeIa[N_T], ciHeIIa[N_T], ciHeISa[N_T], reHIIa[N_T], brema[N_T];
extern double ceHIa[N_T], ceHeIa[N_T], ceHeIIa[N_T], reHeII1a[N_T], reHeII2a[N_T], reHeIIIa[N_T];

/* cross-sections */
#ifndef UM_DISABLE_RADIATION
extern double sigma24[N_nu], sigma25[N_nu], sigma26[N_nu], sigma27[N_nu], sigma28[N_nu], sigma29[N_nu], sigma30[N_nu],
    sigma31[N_nu];
#endif // UM_DISABLE_RADIATION
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#ifdef UM_CHEMISTRY
extern double kHeHII1a[N_T], kHeHII2a[N_T], kHeHII3a[N_T];
extern double kHD1a[N_T], kHD2a[N_T], kHD3a[N_T], kHD4a[N_T], kHD5a[N_T], kHD6a[N_T], kHD7a[N_T];
extern double gJH0uvb;
extern double gJHe0uvb;
extern double gJHepuvb;
extern double epsH0uvb;
extern double epsHe0uvb;
extern double epsHepuvb;
extern double HIshielding;
#endif // UM_CHEMISTRY

#if defined(UM_CHEMISTRY) && defined(CHEMISTRY)
#error you cannot define both UM_CHEMISTRY and CHEMISTRY !
#endif // defined (UM_CHEMISTRY) && defined (CHEMISTRY)

extern int maxThreads;

#ifdef KD_ALTERNATIVE_GROUP_SORT
extern int MaxNgroups;
#endif // KD_ALTERNATIVE_GROUP_SORT

extern __thread int mythreadid;
extern __thread integertime last_time0_drift, last_time1_drift, last_time0_hydrokick, last_time1_hydrokick,
    last_time0_gravkick, last_time1_gravkick;
extern __thread double last_value_drift, last_value_hydrokick, last_value_gravkick;

extern __thread integertime last_time0_magkick, last_time1_magkick;
extern __thread double last_value_magkick;

#if defined(mSIDM) || defined(mIDM)
extern __thread integertime last_time0_scattering_kick, last_time1_scattering_kick;
extern __thread double last_value_scattering_kick;
#endif // mSIDM || mIDM

// #ifdef _OPENMP
// #pragma omp threadprivate(last_time0_drift, last_time1_drift, last_value_drift, last_time0_hydrokick,
// last_time1_hydrokick, last_value_hydrokick, last_time0_gravkick, last_time1_gravkick, last_value_gravkick) #ifdef
// MAGNETIC #pragma omp threadprivate(last_time0_magkick, last_time1_magkick, last_value_magkick) #endif #endif

extern int dofftw;
extern MyAtLeastDouble U_TO_TEMP_FAC;
extern MyAtLeastDouble U_TO_TEMP_FAC_WITHOUT_GAMMA;

extern MyAtLeastDouble av_source_scaling;
extern MyAtLeastDouble av_decay_scaling;

#if GADGET_HYDRO == HYDRO_MFM // has to be defined after SphP struct
#include "../Mfm/SlopeLimiter.hpp"
extern SlopeLimiter<mfm_numdims> *slopeLimiter;
extern PairwiseSlopeLimiter<mfm_numdims> *pairwiseSlopeLimiter;
#endif // GADGET_HYDRO==HYDRO_MFM

/* Include unit conversion struct */
#include "../CodeBase/units.hpp"
using namespace OpenGadget3;
extern Units *units;

#if defined(ACC) && !defined(ACC_DECLARED)
// since allvars.hpp can be included in many places multiple times, it happens that we
// redeclare openacc variables, with this ifdef we check that we run #pragma acc declare only once
#define ACC_DECLARED

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int AccErrorCode;
extern int NodesSize;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef ACC_OPENACC
#pragma acc declare create(All, TakeLevel, MaxNodes, Numnodestree, NActivePart, AccErrorCode, NumPart, N_gas,          \
                           ActiveParticleList, NodesSize)

#pragma acc declare create(ToWakeupList, N_ToWakeUp)
#endif // ACC_OPENACC

#if (defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
#define WalCool_n_Rho WalCool_n[0] /*!< number of density bins      */
#define WalCool_n_Ab WalCool_n[4] /*!< number of solar abundances  */

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
#ifdef ACC_GRADIENTS
extern int WalCool_n[6];
extern int *SpeciesIdx;
extern int *SpeciesPos;
extern int UseHeNumberRatio;
extern double *CoolZvalue;
#endif // ACC_GRADIENTS
extern int ZBins;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern int myHyd, myHel, HydPos, HelPos;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef ACC_OPENACC

#ifdef ACC_GRADIENTS
#pragma acc declare create(CoolZvalue, ZBins, WalCool_n, UseHeNumberRatio, myHyd, myHel, HydPos, HelPos, SpeciesIdx,   \
                           SpeciesPos)
#endif // ACC_GRADIENTS

#if defined(ACC_CONDUCTION) || defined(ACC_GRADIENTS)
#pragma acc declare create(SFs)
#endif // ACC_CONDUCTION || ACC_GRADIENTS

#endif // ACC_OPENACC

#endif // defined(LT_METAL_COOLING)|| defined(LT_METAL_COOLING_WAL)

#ifdef ACC_OPENACC

#ifdef PERIODIC
#pragma acc declare create(boxHalf, boxSize, fac_intp, fcorrx, fcorry, fcorrz, potcorr)
#ifdef LONG_X
#pragma acc declare create(boxHalf_X, boxSize_X)
#endif // LONG_X
#ifdef LONG_Y
#pragma acc declare create(boxHalf_Y, boxSize_Y)
#endif // LONG_Y
#ifdef LONG_Z
#pragma acc declare create(boxHalf_Z, boxSize_Z)
#endif // LONG_Z
#endif // PERIODIC

#pragma acc declare create(Nodes_base)
#pragma acc declare create(Nextnode)
#pragma acc declare create(Extnodes_base)
#pragma acc declare create(shortrange_table, shortrange_table_potential)
#pragma acc declare create(ThisTask)
#endif // ACC_OPENACC

#endif // defined(ACC) && !defined(ACC_DECLARED)

// Timer class content
#include "../SPACETimers/SPACE_timers.hpp"
TIMER_NEW_INSTANCE(GlobalTimer);
TIMER_NEW_INSTANCE(TimestepTimer);
TIMER_ALLVARS_HPP();

#ifdef TIMER_DOMAIN
extern class TimerHierarchy DomainTimer;
extern class TimerHierarchy DomainRecompositionTimer;
extern double DomainCritFractionFix;
#endif // TIMER_DOMAIN

#ifndef SWITCH_TIMER_OFF
extern std::ofstream FdCPU; /*!< file handle for cpu.txt log-file. */
extern std::ofstream FdBalance; /*!< file handle for balance.txt log-file. */
extern std::ofstream FdBalanceHead; /*!< file handle for balance.txt log-file. */

#ifdef TIMER_TIMESTAMP
extern std::ofstream FdTimersTimestamp; /*!< file handle for timestamps file. */
#endif // TIMER_TIMESTAMP

#ifdef TIMER_DETAILS
extern std::ofstream FdCPUTimestep; /* file to save timestep information */
#endif // TIMER_DETAILS

#endif // SWITCH_TIMER_OFF

#ifdef CONSTANT_GRAVITY_Y
extern const MyAtLeastDouble Constant_Gravity_Y;
#endif // CONSTANT_GRAVITY_Y

#ifdef DISK_SLICE
inline void wrap_velocity_disk_slice(const MyAtLeastDouble *pos_i, const MyAtLeastDouble *pos_j, MyFloat &dv_phi)
{
    MyAtLeastDouble dx = pos_i[0] - pos_j[0];
    dv_phi += All.qDisk * All.OmegaDisk * boxSize_X * ((dx > 0.5 * boxSize_X) ? (1) : ((dx < -boxHalf_X) ? (-1) : (0)));
}
#endif // DISK_SLICE

// Struct containing the cosmological parameters required in hydra
struct Cosmo
{
    MyAtLeastDouble hubble_a;
    MyAtLeastDouble hubble_a2;
    MyAtLeastDouble atime;
    MyAtLeastDouble fac_mu;
    MyAtLeastDouble fac_vsic_fix;
    MyAtLeastDouble a3inv;

    // Constructor to initialize values to zero
    Cosmo() : hubble_a(0), hubble_a2(0), atime(0), fac_mu(0), fac_vsic_fix(0), a3inv(0)
    {
    }
};

#if defined(ACC_OPENMP) && defined(ACC_HYDRA)
#pragma omp declare target
#endif // ACC_OPENMP && ACC_HYDRA
extern Cosmo cosmo_hydra;
#if defined(ACC_OPENMP) && defined(ACC_HYDRA)
#pragma omp end declare target
#endif // ACC_OPENMP && ACC_HYDRA

#if defined(ACC_OPENACC) && defined(ACC_HYDRA)
#pragma acc declare create(cosmo_hydra)
#endif // ACC_OPENMP && ACC_HYDRA

#endif // ALLVARS_HPP
