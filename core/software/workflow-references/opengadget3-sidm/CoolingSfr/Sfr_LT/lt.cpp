/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Luca Tornatore.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "lt.hpp"

// This file is for inclusion in allvars.c

#if defined(LT_STELLAREVOLUTION)

int TimeBinCountStars[TIMEBINS];

FILE *FdSnInit, *FdWarn, *FdSn, *FdSnLost, *FdMetals, *FdSnDetails, *FdTrackW, *FdSfrDens, *FdSfrZ;

double cosmic_time;

int search_for_metalspread;

#if defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING)
int ignore_failure_in_convert_u;
#endif

int UseSnII, UseSnIa, UseAGB;

int integration_status;

lt_integrand F;

char *MetNames[LT_NMet];

double MetSolarValues[LT_NMet];

int Hyd, Hel, Oxygen, Iron, FillEl;
#ifdef UM_CHEMISTRY
int Carbon, Nitrogen, Neon, Magnesium, Silicon, Sulphur, Calcium, Sodium, Aluminium, Argon, Nickel;
#endif

double *PhysDensTh; /*!< array of density thresholds, one for each "SF type"             */

double *FEVP; /*!< array of evaporation factors, one for each "SF type"            */

int sfrrate_filenum;

double *sfrrates; /*!< array of star formation rates for every "SF type"               */
double *totsfrrates;

double *sfrrates_bydensity;
double *totsfrrates_bydensity;
double *sfrrates_byZ;
double *totsfrrates_byZ;

double *sum_sm, *sum_mass_stars, *total_sm, *total_sum_mass_stars; /*!< to collect sf rates data */

double MxSfrTimescale_rescale_by_densityth;

int IsThere_TimeDep_SF;

int IsThere_ZDep_SF;

SF_Type *SFs, SFu;

int SFs_dim;

FILE *FdIMFin, *FdIMF;

#ifdef LT_TRACK_CONTRIBUTES
unsigned int Packing_Factor;
unsigned int *Power_Factors, Max_Power, PowerBase;
unsigned int TrackMask;
float Max_Packed_Int, UnPacking_Factor, MaxError, MinPackableFraction, MaxRaisableFraction, PowerBaseLog_inv;

__thread float save_fractionII[LT_NMetP * LT_SFS_DIM], fractionII[LT_NMetP * LT_SFS_DIM],
    nfractionII[LT_NMetP * LT_SFS_DIM];
__thread float save_fractionIa[LT_NMetP * LT_SFS_DIM], fractionIa[LT_NMetP * LT_SFS_DIM],
    nfractionIa[LT_NMetP * LT_SFS_DIM];
__thread float save_fractionAGB[LT_NMetP * LT_SFS_DIM], fractionAGB[LT_NMetP * LT_SFS_DIM],
    nfractionAGB[LT_NMetP * LT_SFS_DIM];

__thread double dfractionII[LT_NMetP * LT_SFS_DIM];
__thread double dfractionIa[LT_NMetP * LT_SFS_DIM];
__thread double dfractionAGB[LT_NMetP * LT_SFS_DIM];

#endif /* LT_TRACK_CONTRIBUTES */

/* : ---------------------------------------------- : */
/* : Sn related                                     : */
/* : ---------------------------------------------- : */

struct SDtype SD;

double ***SNtimesteps;

int *LongLiv_Nsteps, *ShortLiv_Nsteps, *Nsteps;

double ***SnIaYields, **SnIaEj;

double **IaZbins, **IaMbins;

int *IaZbins_dim, *IaMbins_dim, *SnIaY_dim[LT_NMet];

int *NonProcOn_Ia;

double ***SnIIYields, **SnIIEj;

double ***SnII_ShortLiv_Yields;

double **IIZbins, **IIMbins;

int *IIZbins_dim, *IIMbins_dim, *SnIIY_dim[LT_NMet];

int *NonProcOn_II;

double ***AGBYields, **AGBEj;

double **AGBZbins, **AGBMbins;

int *AGBZbins_dim, *AGBMbins_dim, *AGB_dim[LT_NMet];

int *NonProcOn_AGB;

/* : ------------------------------- : */
/* :  SEvDbg                         : */
/* : ------------------------------- : */

#ifdef LT_SEvDbg
unsigned int FirstID;

int checkFirstID;

int *do_spread_dbg_list;
#endif

/* : ------------------------------- : */
/* :  SEv_INFO                       : */
/* : ------------------------------- : */
#ifdef LT_SEv_INFO

double *Zmass, *tot_Zmass;

double *SNdata, *tot_SNdata;

#ifdef LT_LOG_ENRICH_DETAILS
FILE *FdEnrichDetails_temp, *FdLogEnrichDetails;
#endif

#ifdef LT_SMOOTH_SIZE
int AvgSmoothN;

double AvgSmoothSize;

double MinSmoothSize;

double MaxSmoothSize;

int AvgSmoothNgb;

int MinSmoothNgb;

int MaxSmoothNgb;

/* extern float *HashCorrection;   */
/* extern int   *HashBlock, HashN; */
#endif

#ifdef LT_SEvDbg
FILE *FdMetSumCheck;
#endif

#ifdef WINDS
FILE *FdWinds;
#endif

#ifdef LT_EXTEGY_INFO
FILE *FdExtEgy;
#endif

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
FILE *FdSPinfo;
#endif

#endif /* close SEv_INFO */

#ifdef LT_SEv_INFO_DETAILS
double DetailsW[LT_NMet], DetailsWo[LT_NMet];
#endif

/* ------------------------------------------------ */

/* : ---------------------------------------- : */
/* :  METAL COOLING                           : */
/* : ---------------------------------------- : */

double *ThInst_onset;

int TBins, ZBins;
double TMin, TMax, ZMin, ZMax;
double *CoolTvalue, *CoolZvalue, **CoolingTables;

#endif
