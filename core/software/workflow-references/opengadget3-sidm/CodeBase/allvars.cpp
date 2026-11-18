/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file declares all global variables.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "allvars.hpp"

#ifdef ACC_OPENACC
// nvc++ only emits a device code object for a translation unit that
// contains at least one offloadable construct. allvars.cpp has none of its
// own, so without this routine the device-side storage for every
// "#pragma acc declare create" global declared in allvars.hpp (All,
// ActiveParticleList, boxSize, Nodes_base, ...) never gets generated here,
// and nvlink fails with "undefined reference" from the files that use them.
// The routine must actually be called (a static initializer below does
// this), otherwise the compiler dead-code-eliminates it before it ever
// reaches OpenACC device codegen.
#pragma acc routine seq
void force_allvars_device_codegen()
{
}

namespace
{
struct ForceAllvarsDeviceCodegenInit
{
    ForceAllvarsDeviceCodegenInit()
    {
        force_allvars_device_codegen();
    }
} force_allvars_device_codegen_init;
} // namespace
#endif // ACC_OPENACC

#ifdef ACC_OPENMP
// Same nvc++ limitation as above (see the ACC_OPENACC block), but for the
// "#pragma omp declare target" globals in allvars.hpp.
#pragma omp declare target
void force_allvars_device_codegen()
{
}
#pragma omp end declare target

namespace
{
struct ForceAllvarsDeviceCodegenInit
{
    ForceAllvarsDeviceCodegenInit()
    {
        force_allvars_device_codegen();
    }
} force_allvars_device_codegen_init;
} // namespace
#endif // ACC_OPENMP

TimerHierarchy GlobalTimer("Global");
TimerHierarchy TimestepTimer("Timestep");

#ifdef TIMER_DOMAIN
TimerHierarchy DomainTimer("DomainFull");
TimerHierarchy DomainRecompositionTimer("DomainFix");
double DomainCritFractionFix = 0.3;
#endif // TIMER_DOMAIN

#ifndef SWITCH_TIMER_OFF
std::ofstream FdCPU; /*!< file handle for cpu.txt log-file. */
std::ofstream FdBalance; /*!< file handle for balance.txt log-file. */
std::ofstream FdBalanceHead; /*!< file handle for balance.txt log-file. */

#ifdef TIMER_TIMESTAMP
std::ofstream FdTimersTimestamp; /*!< file handle for timestamps file. */
#endif // TIMER_TIMESTAMP

#ifdef TIMER_DETAILS
/* file to save timestep information */
std::ofstream FdCPUTimestep;
#endif // TIMER_DETAILS

#endif // SWITCH_TIMER_OFF

#ifdef PERIODIC
MyDouble boxSize, boxHalf, inverse_boxSize;

#ifdef LONG_X
MyDouble boxSize_X, boxHalf_X, inverse_boxSize_X;
#endif // LONG_X

#ifdef LONG_Y
MyDouble boxSize_Y, boxHalf_Y, inverse_boxSize_Y;
#endif // LONG_Y

#ifdef LONG_Z
MyDouble boxSize_Z, boxHalf_Z, inverse_boxSize_Z;
#endif // LONG_Z
#endif // PERIODIC

#ifdef FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG
MPI_Status mpistat;
#endif // FIX_PATHSCALE_MPI_STATUS_IGNORE_BUG

/*********************************************************/
/*  Global variables                                     */
/*********************************************************/

#ifdef KD_MONITOR_PERFORMANCE
int count_performance = 0;
#endif // KD_MONITOR_PERFORMANCE

__thread int mythreadid;
__thread integertime last_time0_drift, last_time1_drift, last_time0_hydrokick, last_time1_hydrokick,
    last_time0_gravkick, last_time1_gravkick;
__thread double last_value_drift, last_value_hydrokick, last_value_gravkick;
__thread integertime last_time0_magkick, last_time1_magkick;
__thread double last_value_magkick;

#if defined(mSIDM) || defined(mIDM)
__thread integertime last_time0_scattering_kick, last_time1_scattering_kick;
__thread double last_value_scattering_kick;
#endif // mSIDM || mIDM

#ifdef AXION_DM
__thread integertime last_time0_axkick, last_time1_axkick;
__thread double last_value_axkick;
#endif // AXION_DM

__thread int ThisThread; /* the Number of the local thread */
int maxThreads = 1; /* number of threads */
int ThisTask; /*!< the number of the local processor  */
int NTask; /*!< number of processors */
int PTask; /*!< note: NTask = 2^PTask */
MPI_Comm MYMPI_COMM_WORLD;
int MYMPI_MAX_TAG_UB;

double CPUThisRun = 0; /*!< Sums CPU time of current process */

#ifdef RECOMPOSE_DOMAIN_FACTOR
int LastDomainUpdate = 0;
#endif // RECOMPOSE_DOMAIN_FACTOR

int NumForceUpdate; /*!< number of active particles on local processor in current timestep  */
long long GlobNumForceUpdate;
int NumSphUpdate; /*!< number of active SPH particles on local processor in current timestep  */
int MaxTopNodes; /*!< Maximum number of nodes in the top-level tree used for domain decomposition */
int RestartFlag; /*!< taken from command line used to start code. 0 is normal start-up from
            initial conditions, 1 is resuming a run from a set of restart files, while 2
            marks a restart from a snapshot file. */
int RestartSnapNum;
double PostProcessDt;
int SelRnd;

int *Exportflag; /*!< Buffer used for flagging whether a particle needs to be exported to another process */
int *Exportnodecount;
int *Exportindex;

int *Send_offset, *Send_count, *Recv_count, *Recv_offset, *Sendcount;

#ifdef IMPORT_ALLtoALLv_FROM_G4
MPI_Win win;
#endif // IMPORT_ALLtoALLv_FROM_G4

int TakeLevel;

#ifdef _OPENMP
int NActivePart;
int *ActiveParticleList;
#ifdef LT_STELLAREVOLUTION
int NActivePartMerk;
#endif // LT_STELLAREVOLUTION
#endif // _OPENMP

int FirstActiveParticle;
int *NextActiveParticle;
unsigned char *ProcessedFlag;

int TimeBinCount[TIMEBINS];
int TimeBinCountSph[TIMEBINS];
int TimeBinActive[TIMEBINS];

int FirstInTimeBin[TIMEBINS];
int LastInTimeBin[TIMEBINS];
int *NextInTimeBin;
int *PrevInTimeBin;

#ifdef AXION_DM
int TimeBinCountAx[TIMEBINS];
#endif // AXION_DM

size_t HighMark_run, HighMark_domain, HighMark_gravtree, HighMark_sphhsml, HighMark_sphgrad, HighMark_pmperiodic,
    HighMark_pmnonperiodic, HighMark_sphdensity, HighMark_sphhydro, HighMark_addSPH;
#if GADGET_HYDRO == HYDRO_MFM
size_t HighMark_mfmgrad, HighMark_mfmlim, HighMark_mfmflux;
#endif // GADGET_HYDRO == HYDRO_MFM


#ifdef ADAPTGRAVSOFT
size_t HighMark_agsdensity;
#endif // ADAPTGRAVSOFT

#ifdef SFR
double TimeBinSfr[TIMEBINS];
#endif // SFR

#ifdef BLACK_HOLES
double TimeBin_BH_mass[TIMEBINS];
double TimeBin_BH_dynamicalmass[TIMEBINS];
double TimeBin_BH_Mdot[TIMEBINS];
double TimeBin_BH_Medd[TIMEBINS];
#endif // BLACK_HOLES

char DumpFlag = 1;

size_t AllocatedBytes;
size_t HighMarkBytes;
size_t FreeBytes;

double WallclockTime; /*!< This holds the last wallclock time measurement for timings measurements */
int Flag_FullStep; /*!< Flag used to signal that the current step involves all particles */

int TreeReconstructFlag;
int GlobFlag;
int NumPart; /*!< number of particles on the LOCAL processor */
int N_gas; /*!< number of gas particles on the LOCAL processor  */
int N_stars; /*!< number of star particles in the LOCAL processor */
int N_BHs;

#ifdef ACC_OPENMP
// Nitya - adding a variable here to keep a check of when to reallocate WalkLiteP
int NumPartOld;
int WalkLitePAlloc;
int GpuWalkLitePAlloc;

int LiteDInNumPartOld;
int GpuLiteDInAlloc;

int LiteDOutNumPartOld;
int GpuLiteDOutAlloc;

int HSMLNumPartOld;
int GpuHSMLAlloc;

int DAPLNumPartOld;
int GpuDAPLAlloc;

int APLNumPartOld;
int GpuAPLAlloc;
#endif // ACC_OPENMP

int N_ToWakeUp; /*!< number of particles to wake up */
int *ToWakeupList;

long long Ntype[6]; /*!< total number of particles of each type */
int NtypeLocal[6]; /*!< local number of particles of each type */

#if defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM) || defined(PB_INITIALIZE_BH_PART)
int Particles_converted; /*!< current number of star/DM particles in gas particle block */
#endif // defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

double TimeOfLastTreeConstruction; /*!< holds what it says */
int *Ngblist; /*!< Buffer to hold indices of neighbours retrieved by the neighbour search
         routines */
int *RecyclingBuffer;
int HasFisrtParticleBeenProcessed;

double *R2ngblist;
double DomainCorner[3], DomainCenter[3], DomainLen, DomainFac;
int *DomainStartList, *DomainEndList;

double *DomainWork;
int *DomainCount;
int *DomainCountSph;
int *DomainTask;
int *DomainNodeIndex;
int *DomainList, DomainNumChanged;

peanokey *__restrict__ Key, *__restrict__ KeySorted;


struct topnode_data *TopNodes;
int NTopnodes, NTopleaves;
double RndTable[RNDTABLE];

#ifdef SUBFIND
int GrNr;
int NumPartGroup;
#endif // SUBFIND

#ifdef WRITE_KEY_FILES
peanokey *KeyIndex;
int *NPartPerKey, *PartKeyOffset;
int NKeys[6];
long long NKeysTotal[6];
#endif // WRITE_KEY_FILES

#ifdef LT_ADD_GAL_TO_SUB
float *tempiAS, *CB07, *Filters_Effective_Wavelenght;
#ifdef OBSERVER_FRAME
float *CB07obs;
#ifdef INTERP_OBSERVER_FRAME
float CB07obs_redshifts[INTERP_OBSERVER_FRAME];
int flag_readCB07obs_redshifts;
#endif // INTERP_OBSERVER_FRAME
#endif // OBSERVER_FRAME
#endif // LT_ADD_GAL_TO_SUB


/* variables for input/output , usually only used on process 0 */
char ParameterFile[100]; /*!< file name of parameterfile used for starting the simulation */

FILE *FdInfo, /*!< file handle for info.txt log-file. */
    *FdEnergy, /*!< file handle for energy.txt log-file. */
    *FdTimings, /*!< file handle for timings.txt log-file. */
    *FdTimebin, *FdParamChangeLog;

#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
FILE *FdPM; /*!< file handler for pm.txt log-file. */
#endif // PMGRID

#ifdef SFR
FILE *FdSfr; /*!< file handle for sfr.txt log-file. */
#endif // SFR

#ifdef BLACK_HOLES
FILE *FdBlackHoles; /*!< file handle for blackholes.txt log-file. */
FILE *FdBlackHolesDetails;
FILE *FdBlackHolesSpinDetails;
#if (defined(PB_BH_HSML_PREVENT_HOLE) || defined(PB_OUTPUT_SINGLE_BH))
FILE *Fd_BlackHole_GasNgbs;
#endif // PB_BH_HSML_PREVENT_HOLE || PB_OUTPUT_SINGLE_BH
#endif // BLACK_HOLES

#ifdef FORCETEST
FILE *FdForceTest; /*!< file handle for forcetest.txt log-file. */
#endif // FORCETEST

#ifdef DARKENERGY
FILE *FdDE; /*!< file handle for darkenergy.txt log-file. */
#endif // DARKENERGY

#ifdef XXLINFO
FILE *FdXXL; /*!< file handle for xxl.txt log-file. */

#ifdef MAGNETIC
double MeanB;
double MaxDivB;
#endif // MAGNETIC

#ifdef TIME_DEP_ART_VISC
double MeanAlpha;
#endif // TIME_DEP_ART_VISC
#endif // XXLINFO

/*! table for the cosmological drift factors */
double DriftTable[DRIFT_TABLE_LENGTH];

/*! table for the cosmological kick factor for gravitational forces */
double GravKickTable[DRIFT_TABLE_LENGTH];

/*! table for the cosmological kick factor for hydrodynmical forces */
double HydroKickTable[DRIFT_TABLE_LENGTH];

#ifdef AXION_DM
/*! table for the cosmological kick factor for axion hydrodynmical forces */
double AxHydroKickTable[DRIFT_TABLE_LENGTH];
#endif // AXION_DM

/*! table for the cosmological kick factor for induction equation */
double MagKickTable[DRIFT_TABLE_LENGTH];

#if defined(mSIDM) || defined(mIDM)
/*! table for the cosmological kick factor for mSIDM and mIDM */
double ScatteringKickTable[DRIFT_TABLE_LENGTH];
#endif //  mSIDM || mIDM

void *CommBuffer; /*!< points to communication buffer, which is used at a few places */

/*! This structure contains data which is the SAME for all tasks (mostly code parameters read from the
 * parameter file).  Holding this data in a structure is convenient for writing/reading the restart file, and
 * it allows the introduction of new global variables in a simple way. The only thing to do is to introduce
 * them into this structure. */
struct global_data_all_processes All;

// Struct containing the cosmo parameters needed for hydra
struct Cosmo cosmo_hydra;

/*! This structure contains data which is the SAME for all tasks as the All data structure
 * but its values are computed on the fly thus it can change in shape betwen restarts */
struct derived_global_data_all_processes AllDerived;

/*! Structure for global variables needed for self-interacting dark matter */
#ifdef mSIDM
scatter::MSIDM msidm;
#endif // mSIDM

/*! Structure for global variables needed for interacting dark matter */

#if defined(mSIDM) || defined(mIDM)
scatter::GaussLegendre gauss_legendre;
#endif // mSIDM || mIDM

#ifdef ANALYTIC_POTENTIALS
ANALYTIC_POTENTIAL analytic_potentials;
#endif // ANALYTIC_POTENTIALS

/*! This structure holds all the information that is
 * stored for each particle of the simulation. */
struct particle_data *__restrict__ P, /*!< holds particle data on local processor */
    *__restrict__ DomainPartBuf; /*!< buffer for particle data used in domain decomposition */

/* the following struture holds data that is stored for each SPH particle in addition to the collisionless
 * variables. */
struct sph_particle_data *__restrict__ SphP, /*!< holds SPH particle data on local processor */
    *__restrict__ DomainSphBuf; /*!< buffer for SPH particle data in domain decomposition */

#if defined(AR_PARALLEL_LOCAL_REFINE) || defined(AR_NODE_RECURSIVE_PARALLEL)
int NumUsedThreads; // when parallelise recursive region we check/set the number of free threads with atomic-capture
int NumberOfAllocatedNodeLocks;
#endif // defined(AR_PARALLEL_LOCAL_REFINE) || defined(AR_NODE_RECURSIVE_PARALLEL)

#ifdef AR_SOA_LITE
sph_soa SphPSoa;
#endif // AR_SOA_LITE

#ifdef AXION_DM
struct ax_particle_data *AxP, /*!< holds AX particle data on local processor */
    *__restrict__ DomainAxBuf; /*!< buffer for AX particle data in domain decomposition */
#endif // AXION_DM

#ifdef SFR
struct met_particle_data *__restrict__ MetP, *__restrict__ DomainMetBuf;
#endif // SFR

#ifdef BLACK_HOLES
struct bh_particle_data *__restrict__ BHP;
#endif // BLACK_HOLES

peanokey *DomainKeyBuf;

/* global state of system */
struct state_of_system SysState, SysStateAtStart, SysStateAtEnd;

/* Various structures for communication during the gravity computation. */
struct data_index *DataIndexTable; /*!< the particles to be exported are grouped
                      by task-number. This table allows the
                      results to be disentangled again and to be
                      assigned to the correct particle */

struct data_nodelist *DataNodeList, *DataNodeListIn, *DataNodeListGet;

struct gravdata_in *GravDataIn, /*!< holds particle data to be exported to other processors */
    *GravDataGet; /*!< holds particle data imported from other processors */

struct gravdata_out *GravDataResult, /*!< holds the partial results computed for imported particles. Note: We use
                                        GravDataResult = GravDataGet, such that the result replaces the imported data */
    *GravDataOut; /*!< holds partial results received from other processors. This will overwrite the GravDataIn array */

struct potdata_out *PotDataResult, /*!< holds the partial results computed for imported particles. Note: We use
                                      GravDataResult = GravDataGet, such that the result replaces the imported data */
    *PotDataOut; /*!< holds partial results received from other processors. This will overwrite the GravDataIn array */

struct phidotdata_out
    *PhiDotDataResult, /*!< holds the partial results computed for imported particles. Note: We use GravDataResult =
                          GravDataGet, such that the result replaces the imported data */
    *PhiDotDataOut; /*!< holds partial results received from other processors. This will overwrite the GravDataIn array
                     */

struct info_block *InfoBlock;

/* Header for the standard file format. */
struct io_header header; /*!< holds header for snapshot files */

/* struct for unit conversion and conversion from comoving to physical */
Units *units;

/* Variables for Tree */
int Nexport, Nimport;
int BufferFullFlag;
int NextParticle;
int NextGroup;
int NextJ;
int TimerFlag;

struct NODE *Nodes_base, /*!< points to the actual memory allocted for the nodes */
    *Nodes; /*!< this is a pointer used to access the nodes which is shifted such that Nodes[All.MaxPart]
           gives the first allocated node */

#ifdef AR_TREEBUILD_PARALLEL
omp_lock_t *NodesLockList; // locks for the tree build
#endif // AR_TREEBUILD_PARALLEL

struct extNODE *Extnodes, *Extnodes_base;

int MaxNodes; /*!< maximum allowed number of internal nodes */
int Numnodestree; /*!< number of (internal) nodes in each tree */

int *Nextnode; /*!< gives next node in tree walk  (nodes array) */
#ifdef GROUP_LEAVES
int *NextGroupedNode; /*!< gives next node in tree walk  (nodes array) */
#endif // GROUP_LEAVES

int *Father; /*!< gives parent node in tree (Prenodes array) */

#ifdef STATICNFW
double Rs, R200;
double Dc;
double RhoCrit, V200;
double fac;
#endif // STATICNFW

#if defined(UM_CHEMISTRY) || defined(UM_METAL_COOLING)
/* --==[ link with LT_ stuffs]==-- */
float *um_ZsPoint, um_FillEl_mu, um_mass;
/* char *PT_Symbols; */
/* double *PT_Masses; */
/* int NPT; */
/* double **II_AvgFillNDens, **IIShLv_AvgFillNDens, **Ia_AvgFillNDens, **AGB_AvgFillNDens; */
#endif // defined (UM_CHEMISTRY) || defined (UM_METAL_COOLING)

/* ----- chemistry part ------- */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#define H_number_fraction 0.76
#define He_number_fraction 0.06

/* ----- Tables ------- */
double T[N_T], J0_nu[N_nu], J_nu[N_nu], nu[N_nu];
double k1a[N_T], k2a[N_T], k3a[N_T], k4a[N_T], k5a[N_T], k6a[N_T], k7a[N_T], k8a[N_T], k9a[N_T], k10a[N_T], k11a[N_T];
double k12a[N_T], k13a[N_T], k14a[N_T], k15a[N_T], k16a[N_T], k17a[N_T], k18a[N_T], k19a[N_T], k20a[N_T], k21a[N_T];
double ciHIa[N_T], ciHeIa[N_T], ciHeIIa[N_T], ciHeISa[N_T], reHIIa[N_T], brema[N_T];
double ceHIa[N_T], ceHeIa[N_T], ceHeIIa[N_T], reHeII1a[N_T], reHeII2a[N_T], reHeIIIa[N_T];

/* cross-sections */
#ifndef UM_DISABLE_RADIATION
double sigma24[N_nu], sigma25[N_nu], sigma26[N_nu], sigma27[N_nu], sigma28[N_nu], sigma29[N_nu], sigma30[N_nu],
    sigma31[N_nu];
#endif // UM_DISABLE_RADIATION
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#ifdef UM_CHEMISTRY
double gJH0uvb;
double gJHe0uvb;
double gJHepuvb;
double epsH0uvb;
double epsHe0uvb;
double epsHepuvb;
double HIshielding;
double kHeHII1a[N_T], kHeHII2a[N_T], kHeHII3a[N_T];
double kHD1a[N_T], kHD2a[N_T], kHD3a[N_T], kHD4a[N_T], kHD5a[N_T], kHD6a[N_T], kHD7a[N_T];
#endif // UM_CHEMISTRY

#ifdef LT_STELLAREVOLUTION
#include "../CoolingSfr/Sfr_LT/lt.cpp"
#endif // LT_STELLAREVOLUTION

#ifdef KD_ALTERNATIVE_GROUP_SORT
int MaxNgroups;
#endif // KD_ALTERNATIVE_GROUP_SORT

int dofftw = 0;

MyAtLeastDouble U_TO_TEMP_FAC;
MyAtLeastDouble U_TO_TEMP_FAC_WITHOUT_GAMMA;

#if defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 0)
// Equation (21) in Morris & Monaghan 1997
MyAtLeastDouble av_source_scaling = (1.0 / log((GAMMA + 1) / (GAMMA - 1)));
// Equation (11) in Morris & Monaghan 1997
MyAtLeastDouble av_decay_scaling = (1.0 / sqrt((GAMMA - 1) / (2 * GAMMA)));
#endif // defined(TIME_DEP_ART_VISC) && (TIME_DEP_ART_VISC == 0)

TIMER_ALLVARS_CPP();

#ifdef CONSTANT_GRAVITY_Y
const MyAtLeastDouble Constant_Gravity_Y = CONSTANT_GRAVITY_Y;
#endif // CONSTANT_GRAVITY_Y

#if GADGET_HYDRO == HYDRO_MFM
Eos<mfm_numdims> *eos;
FluxSolver<mfm_numdims> *fluxSolver;
SlopeLimiter<mfm_numdims> *slopeLimiter;
PairwiseSlopeLimiter<mfm_numdims> *pairwiseSlopeLimiter;
#if !defined(DONT_BIN_MFM_FLUXES) && defined(_OPENMP)
omp_lock_t *MfmNgbLockList; // locks for the MFM neighbor flux
#endif // !defined(DONT_BIN_MFM_FLUXES) && defined(_OPENMP)
#endif // GADGET_HYDRO == HYDRO_MFM

#ifdef PXP_OPENMP
omp_lock_t *pxpPartLockList;
#endif // PXP_OPENMP
