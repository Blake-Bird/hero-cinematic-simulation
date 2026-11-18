/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains default behaviour for switches for the code.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Geray S. Karademir and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef SWITCHES_HPP
#define SWITCHES_HPP

#include "../gadgetconfig.hpp"

#if defined(fSIDM) || defined(rSIDM) || defined(fIDM) || defined(rIDM)
#include "../mSIDM/switches.hpp"
#endif // defined(fSIDM) || defined(rSIDM) || defined(fIDM) || defined(rIDM)

#ifndef AR_GREEN_TREE_HSML_FRACTION
#define AR_GREEN_TREE_HSML_FRACTION 0.5
#endif // AR_GREEN_TREE_HSML_FRACTION

#ifdef AR_XMAS_TREE
#ifndef AR_XMAS_TREE_FACTOR
#define AR_XMAS_TREE_FACTOR 500
#ifndef AR_XMAS_TREE_STRUCT
// one may want to restart a AR_XMAS_TREE run with the legacy tree,
// since you need the same Nodes sturcture,
// do it disabling AR_XMAS_TREE and enabling AR_XMAS_TREE_STRUCT
#define AR_XMAS_TREE_STRUCT
#endif
#endif // AR_XMAS_TREE_FACTOR
#endif // AR_XMAS_TREE

#ifndef DENSITY_LESS_NGB_FACTOR
#define DENSITY_LESS_NGB_FACTOR 1
#endif // DENSITY_LESS_NGB_FACTOR

#ifndef DENSITY_LESS_NGB_FACTOR_SUB
#define DENSITY_LESS_NGB_FACTOR_SUB 1
#endif // DENSITY_LESS_NGB_FACTOR_SUB

#ifndef KD_BUFFER_MANAGEMENT
#define KD_BUFFER_MANAGEMENT 0.3
#endif // KD_BUFFER_MANAGEMENT

#ifndef TIMER_LEVEL
#define TIMER_LEVEL 1
#endif // TIMER_LEVEL
int constexpr Timer_Level = TIMER_LEVEL;

#ifndef TIMER_LIST_SYMBOL
#define TIMER_LIST_SYMBOL '*'
#endif // TIMER_LIST_SYMBOL
constexpr char Timer_List_Symbol = TIMER_LIST_SYMBOL;

#ifndef TIMER_INDENT_SYMBOL
#define TIMER_INDENT_SYMBOL '-'
#endif // TIMER_INDENT_SYMBOL
constexpr char Timer_Indent_Symbol = TIMER_INDENT_SYMBOL;

#ifdef RT_FLUID_TEST
#define CONSTANT_GRAVITY_Y -0.5
#endif // RT_FLUID_TEST

#ifndef WAKEUP
#define WAKEUP 3
#endif // WAKEUP

#ifndef EXPERT_LEVEL
#define EXPERT_LEVEL 0
#endif // EXPERT_LEVEL

#ifndef MPISENDRECV_SIZELIMIT
#define MPISENDRECV_SIZELIMIT 2047
#endif // MPISENDRECV_SIZELIMIT
long long constexpr MPISendrecv_Sizelimit = MPISENDRECV_SIZELIMIT;

#ifdef ACC_ASYNC_COPY
#define ACC_ASYNC_COPY1 async(1)
#else // ACC_ASYNC_COPY
#define ACC_ASYNC_COPY1
#endif // ACC_ASYNC_COPY

#ifndef ACC_ASYNC
#define ACC_ASYNC1
#endif // ACC_ASYNC

#ifndef ACC
#define ACC_DECLARE_ROUTINE_BEFORE
#define ACC_DECLARE_ROUTINE_AFTER
#endif // ACC

#ifdef ACC
#ifdef ACC_OPENACC

#define ACC_PRAGMA(...) ACC_APPLY_PRAGMA(acc __VA_ARGS__)
#define ACC_DECLARE_ROUTINE_BEFORE _Pragma("acc routine")
#define ACC_DECLARE_ROUTINE_AFTER

/* Any module whose device kernels call the neighbour tree walk needs the device build
 * of OpenACC/ngb.cpp; ACC_NGB is what puts the `acc routine` on its definition. Keep
 * this list in sync with the OpenACC/ngb.o filter in the Makefile. */
#if (defined(ACC_HYDRA) || defined(ACC_HSML) || defined(ACC_GRADIENTS) || defined(ACC_CONDUCTION) ||                   \
     defined(ACC_SPECTRAL_CRs_DIFFUSION)) &&                                                                           \
    !defined(ACC_NGB)
#define ACC_NGB
#endif // (ACC_HYDRO || ACC_HSML || ACC_GRADIENTS || ACC_CONDUCTION || ACC_SPECTRAL_CRs_DIFFUSION) && !ACC_NGB

#ifdef ACC_ASYNC
#define ACC_ASYNC1 async(1)
#endif // ACC_ASYNC

#elif defined(ACC_OPENMP)
#define ACC_PRAGMA(...) ACC_APPLY_PRAGMA(omp __VA_ARGS__)
#define ACC_DECLARE_ROUTINE_BEFORE _Pragma("omp declare target")
#define ACC_DECLARE_ROUTINE_AFTER _Pragma("omp end declare target")
#ifdef ACC_ASYNC
#define ACC_ASYNC1 nowait
#endif // ACC_ASYNC

#elif defined(ACC_NONE)
#define ACC_PRAGMA(...)
#define ACC_DECLARE_ROUTINE_BEFORE
#define ACC_DECLARE_ROUTINE_AFTER
#ifdef ACC_ASYNC
#define ACC_ASYNC1
#endif // ACC_ASYNC

#else // ACC_OPENACC, ACC_OPENMP, ACC_NONE
#error "please define ACC_OPENACC or ACC_NONE or ACC_OPENMP"
#endif // ACC_OPENACC

#define ACC_RUN
#ifndef ACC_ACTIVEPART_CPU
#define ACC_ACTIVEPART_CPU 1000
#endif // ACC_ACTIVEPART_CPU

#ifndef ACC_ACTIVEPART_II_CPU
#define ACC_ACTIVEPART_II_CPU ACC_ACTIVEPART_CPU
#endif // ACC_ACTIVEPART_II_CPU

#ifndef ACC_GPU_NGB
#define ACC_GPU_NGB 32
#endif // ACC_GPU_NGB

#ifndef ACC_GPU_NGB_CONDUCTION
#define ACC_GPU_NGB_CONDUCTION ACC_GPU_NGB
#endif // ACC_GPU_NGB_CONDUCTION

#ifndef ACC_GPU_NGB_CR_DIFFUSION
#define ACC_GPU_NGB_CR_DIFFUSION ACC_GPU_NGB
#endif // ACC_GPU_NGB_CR_DIFFUSION

#endif // ACC

#ifdef ONEDIM
#define NUMDIMS 1
#elif defined(TWODIMS)
#define NUMDIMS 2
#else // ONEDIM
#define NUMDIMS 3
#endif // ONEDIM

#define HYDRO_SPH 1
#define HYDRO_PESPH 2
#define HYDRO_MFM 3

#ifdef USE_MESHLESS_FINITE_MASS
#ifdef DISK_SLICE
#define DONT_BIN_MFM_FLUXES
#endif // DISK_SLICE

#define GADGET_HYDRO HYDRO_MFM
#ifdef MAGNETIC
#define MFM_MAGNETIC
constexpr int mfm_numdims = 3;
#else // MAGNETIC
constexpr int mfm_numdims = NUMDIMS;
#endif // MAGNETIC
/* SphP.FaceAreaSum, the vector sum of the effective face areas, is a purely geometric closure
   residual of the cell. The DtB limiter and the internal-energy limiter both consume it, so it is
   carried whenever either is active - independently of MAGNETIC, and not at all if neither is. */
#if (defined(MFM_MAGNETIC) && !defined(NO_MFM_DTB_LIMITER)) || defined(LMB_MFM_ENERGY_CLOSURE_LIMITER)
#define MFM_FACE_CLOSURE_RESIDUAL
#endif // (MFM_MAGNETIC && !NO_MFM_DTB_LIMITER) || LMB_MFM_ENERGY_CLOSURE_LIMITER
#elif defined(USE_PRESSURE_ENTROPY_SPH)
#define GADGET_HYDRO HYDRO_PESPH
#else // USE_MESHLESS_FINITE_MASS
#define GADGET_HYDRO HYDRO_SPH
#endif // USE_MESHLESS_FINITE_MASS

/* Shock-finder dispatch based on chosen hydro scheme */
#ifdef SHOCKFINDER
#if (GADGET_HYDRO == HYDRO_MFM)
#define SHOCKFINDER_MFM
#else // HYDRO_SPH or HYDRO_PESPH
#define SHOCKFINDER_SPH
#endif // GADGET_HYDRO
#endif // SHOCKFINDER

#define LIMITER_GIZMO 0
#define LIMITER_TVD_SCALAR 1
#define LIMITER_SCALAR 2
#define LIMITER_AREPO 3
#define LIMITER_NULL 4
#define LIMITER_ZERO_SLOPES 5

#ifdef SLOPE_LIMITER_ZERO_SLOPES
#define MFM_SLOPE_LIMITER LIMITER_ZERO_SLOPES
#elif defined(SLOPE_LIMITER_TVD_SCALAR)
#define MFM_SLOPE_LIMITER LIMITER_TVD_SCALAR
#elif defined(SLOPE_LIMITER_SCALAR)
#define MFM_SLOPE_LIMITER LIMITER_SCALAR
#elif defined(SLOPE_LIMITER_AREPO)
#define MFM_SLOPE_LIMITER LIMITER_AREPO
#elif defined(SLOPE_LIMITER_NULL)
#define MFM_SLOPE_LIMITER LIMITER_NULL
#else // SLOPE_LIMITER_ZERO
#define MFM_SLOPE_LIMITER LIMITER_GIZMO
#endif // SLOPE_LIMITER_ZERO

#ifndef SLOPE_LIMITER_TOLERANCE
#define SLOPE_LIMITER_TOLERANCE 1
#endif // SLOPE_LIMITER_TOLERANCE

#ifndef HLLC_RIEMANN_VERSION
#define HLLC_RIEMANN_VERSION 3
#endif // HLLC_RIEMANN_VERSION

#if (!defined(ROE_AVERAGE) && !defined(PVRS_APPROXIMATION) && !defined(TRRS_APPROXIMATION))
#define ROE_AVERAGE // other choice as standard???
#endif // (!defined(ROE_AVERAGE) && !defined(PVRS_APPROXIMATION) && !defined(TRRS_APPROXIMATION))

#ifndef MUSCL
#define NOMUSCL
#endif // MUSCL

#ifdef MAGNETIC
#ifndef DIVBFORCECORRECTION_FACTOR
#define DIVBFORCECORRECTION_FACTOR 1
#endif // DIVBFORCECORRECTION_FACTOR
#endif // MAGNETIC

#if (defined(LT_STELLAREVOLUTION) && !defined(LT_METAL_COOLING))
#define LT_METAL_COOLING_WAL
#endif

#if (defined(LT_STELLAREVOLUTION) && !defined(LT_MAX_TEMP_FEEDBACK))
#define LT_MAX_TEMP_FEEDBACK 5.0e8
#endif

#if (defined(LT_STELLAREVOLUTION) && !defined(LT_NMet))
#define LT_NMet 16
#endif

#if defined(LT_STELLAREVOLUTION)
#ifndef DO_NOT_USE_LT_STARS_GUESSHSML
#define LT_STARS_GUESSHSML
#endif
#ifndef DO_NOT_USE_LT_USE_TRUENGB
#define LT_USE_TRUENGB
#endif
#ifndef DO_NOT_USE_LS_SF_RUNAWAY_TREATMENT
#define LS_SF_RUNAWAY_TREATMENT
#endif
#endif

#define SFR_NONE 100
#define SFR_SPRINGEL 101
#define SFR_TORNATORE 102
#define SFR_MURANTE 103

#if   defined(LT_STELLAREVOLUTION)
#define GADGET_SFR SFR_TORNATORE
#elif defined(SFR)
#define GADGET_SFR SFR_SPRINGEL
#else // GM_MUPPI
#define GADGET_SFR SFR_NONE
#endif // GM_MUPPI

#define COOLING_KATZ 101
#define COOLING_SUTHERLAND 102
#define COOLING_WAL 103
#define COOLING_LUEDERS 104

#ifdef SL_COOLING_TABLES
#define GADGET_COOLING COOLING_LUEDERS
#elif defined(LT_METAL_COOLING_WAL)
#define GADGET_COOLING COOLING_WAL
#elif defined(LT_METAL_COOLING)
#define GADGET_COOLING COOLING_SUTHERLAND
#elif defined(COOLING)
#define GADGET_COOLING COOLING_KATZ
#endif // SL_COOLING_TABLES

#ifndef OLD_FILTERS
#define N_BINS_METAL 6
#define LOG_AGE_BINS 220
#else // OLD_FILTERS
#define N_BINS_METAL 7
#define LOG_AGE_BINS 219
#endif // OLD_FILTERS

int constexpr CB07_Tot_N_Filers = 90;

#ifdef STATICNFW
#ifndef NFW_M200
#define NFW_M200 100.
#endif // NFW_M200
double constexpr Nfw_M200 = NFW_M200;

#ifndef NFW_Eps
#define NFW_Eps 0.01
#endif // NFW_Eps
double constexpr Nfw_Eps = NFW_Eps;

#ifndef NFW_C
#define NFW_C 12
#endif // NFW_C
double constexpr Nfw_C = NFW_C;
#endif // STATICNFW

#ifdef STATICBRANDT
#ifndef BRANDT_OmegaBr
#define BRANDT_OmegaBr 0.5
#endif // BRANDT_OmegaBr
double constexpr Brandt_OmegaBr = BRANDT_OmegaBr;

#ifndef BRANDT_R0
#define BRANDT_R0 2.
#endif // BRANDT_R0
double constexpr Brandt_R0 = BRANDT_R0;
#endif // STATICBRANDT

#ifdef STATICISO
#ifndef ISO_R200
#define ISO_R200 160.
#endif // ISO_R200
double constexpr Iso_R200 = ISO_R200;

#ifndef ISO_M200
#define ISO_M200 95.21
#endif // ISO_M200
double constexpr Iso_M200 = ISO_M200;

#ifndef ISO_Eps
#define ISO_Eps 0.1
#endif // ISO_Eps
double constexpr Iso_Eps = ISO_Eps;
#endif // STATICISO

#ifdef STATICHQ
#ifndef HQ_M200
#define HQ_M200 95.2401
#endif // HQ_M200
double constexpr Hq_M200 = HQ_M200;

#ifndef HQ_C
#define HQ_C 9.
#endif // HQ_C
double constexpr Hq_C = HQ_C;
#endif // STATICHQ

#ifdef GM_STARDENSITY

#ifndef GM_STARDENSITY_NEIGH
#define GM_STARDENSITY_NEIGH 0.5
#endif // GM_STARDENSITY_NEIGH

#ifndef GM_STARDENSITY_THRESH
#define GM_STARDENSITY_THRESH 0.004
#endif // GM_STARDENSITY_THRESH

#endif // GM_STARDENSITY

#ifndef VERBOSE_LEVEL
#define VERBOSE_LEVEL 0
#endif // VERBOSE_LEVEL
constexpr int Verbose_Level = VERBOSE_LEVEL;


#ifndef GRAVCOSTLEVELS
#define GRAVCOSTLEVELS 6
#endif // GRAVCOSTLEVELS
int constexpr Gravcostlevels = GRAVCOSTLEVELS;

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
#ifdef UM_CHEMISTRY
#define T_SUP_INTERPOL_LIMIT 1.e4
#endif // UM_CHEMISTRY
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

/*With slightly relativistic massive neutrinos, for consistency we need to include radiation.
 * A note on normalisation (as of 08/02/2012):
 * CAMB appears to set Omega_Lambda + Omega_Matter+Omega_K = 1,
 * calculating Omega_K in the code and specifying Omega_Lambda and Omega_Matter in the paramfile.
 * This means that Omega_tot = 1+ Omega_r + Omega_g, effectively
 * making h0 (very) slightly larger than specified.
 */
#ifdef INCLUDE_RADIATION
/*Stefan-Boltzmann constant in cgs units*/
#define STEFAN_BOLTZMANN 5.670373e-5
/* Omega_g = 4 \sigma_B T_{CMB}^4 8 \pi G / (3 c^3 H^2)*/
#define OMEGAG                                                                                                         \
    (4 * STEFAN_BOLTZMANN * 8 * M_PI * GRAVITY / (3 * C * C * C * HUBBLE * HUBBLE) * pow(T_CMB0, 4) /                  \
     All.HubbleParam / All.HubbleParam)
#if (defined KSPACE_NEUTRINOS_2) || (defined NEUTRINOS)
/*Neutrinos are included elsewhere */
#define OMEGAR OMEGAG
#else // (defined KSPACE_NEUTRINOS_2) || (defined NEUTRINOS)
/*Neutrinos are included in the radiation */
/*For massless neutrinos, rho_nu/rho_g = 7/8 (T_nu/T_cmb)^4 *N_eff, but we absorbed N_eff into T_nu above */
#define OMEGANU (OMEGAG * 7 / 8. * pow(TNU / T_CMB0, 4) * 3)
/*With massless neutrinos only, add the neutrinos to the radiation */
#define OMEGAR (OMEGAG + OMEGANU)
#endif // (defined KSPACE_NEUTRINOS_2) || (defined NEUTRINOS)
#else // INCLUDE_RADIATION
/*Default is no radiation */
#define OMEGAR 0.
#endif // INCLUDE_RADIATION

/*! ASMTH gives the scale of the short-range/long-range force split in units of FFT-mesh cells */
#ifndef ASMTH
#define ASMTH 1.25
#endif // ASMTH
double constexpr Asmth = ASMTH;

/*! RCUT gives the maximum distance (in units of the scale used for the force split) out to which short-range
 * forces are evaluated in the short-range tree walk.
 */
#ifndef RCUT
#define RCUT 4.5
#endif // RCUT
float constexpr Rcut = RCUT;

#ifdef TIME_DEP_ART_VISC
#if (defined(TIME_DEP_ART_VISC) && ((TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3))) && defined(MAGNETIC) &&     \
    defined(DIVBCLEANING_TRICCO)
#ifndef DIVB_TO_ALPHA
#define DIVB_TO_ALPHA 1
#endif // !DIVB_TO_ALPHA
#endif //((TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)) && MAGNETIC && DIVBCLEANING_TRICCO
#endif // TIME_DEP_ART_VISC

#ifndef HIGH_ART_VISC_START
#if defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 1)
#define HIGH_ART_VISC_START 0.0
#else // TIME_DEP_ART_VISC == 1
#define HIGH_ART_VISC_START 1.0
#endif // TIME_DEP_ART_VISC == 1
#endif // HIGH_ART_VISC_START
float constexpr High_Art_Visc_Start = HIGH_ART_VISC_START;

#ifndef ART_VISC_BETA_MIN
#define ART_VISC_BETA_MIN 0.1
#endif // ART_VISC_BETA_MIN
float constexpr Art_Visc_Beta_Min = ART_VISC_BETA_MIN;

#ifndef AVERAGE_ART_VISC
#define AVERAGE_ART_VISC 0
#endif // AVERAGE_ART_VISC
float constexpr Average_Art_Visc = AVERAGE_ART_VISC;

#ifndef HIGH_MAGN_DISP_START
#define HIGH_MAGN_DISP_START 0.5
#endif // HIGH_MAGN_DISP_START
float constexpr High_Magn_Disp_Start = HIGH_MAGN_DISP_START;

#ifndef MULTIPLEDOMAINS
#define MULTIPLEDOMAINS 16
#endif // MULTIPLEDOMAINS
int constexpr Multipledomains = MULTIPLEDOMAINS;

#ifndef TOPNODEFACTOR
#define TOPNODEFACTOR (2 * Multipledomains)
#endif // TOPNODEFACTOR
int constexpr Topnodefactor = TOPNODEFACTOR;


#ifndef GAMMA
/* adiabatic index of simulated gas */
#define GAMMA (5.0 / 3.0)
#endif // GAMMA

#define GAMMA_INV (1. / GAMMA)
#define GAMMA_MINUS1 (GAMMA - 1)
#define GAMMA_MINUS1_INV (1. / (GAMMA - 1))


#if defined(FOF) && defined(SUBFIND)
#ifndef DENSITY_SPLIT_BY_TYPE
#define DENSITY_SPLIT_BY_TYPE 1 + 2 + 16 + 32
#endif // DENSITY_SPLIT_BY_TYPE
#define WRITE_SUB_IN_SNAP_FORMAT
#define WRITE_SUB_COLLECTIVELY
#define KD_ALTERNATIVE_GROUP_SORT
#define SO_BAR_INFO
#define SUBFIND_COUNT_BIG_HALOS 1e4
#define KD_MAIN_HALO
#ifdef LT_ADD_GAL_TO_SUB
#define INTERP_OBSERVER_FRAME 144
#define DUSTATT 11
#define OBSERVER_FRAME
#endif // LT_ADD_GAL_TO_SUB

#endif // defined(FOF) && defined(SUBFIND)

#endif // SWITCHES_HPP
