/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains value checks for switches for the code.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Geray S. Karademir and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef SWITCHES_CHECK_HPP
#define SWITCHES_CHECK_HPP

#include <cstring>
#include "allvars.hpp"

static_assert(VERBOSE_LEVEL >= 0, "VERBOSE_LEVEL must be provided with a value larger than or equal to 0\n");

#ifdef LONG_X
static_assert(LONG_X >= 0, "LONG_X must be provided with a value larger than or equal to 0\n");
#endif // LONG_X

#ifdef LONG_Y
static_assert(LONG_Y >= 0, "LONG_Y must be provided with a value larger than or equal to 0\n");
#endif // LONG_Y

#ifdef LONG_Z
static_assert(LONG_Z >= 0, "LONG_Z must be provided with a value larger than or equal to 0\n");
#endif // LONG_Z

#if GADGET_HYDRO == HYDRO_MFM
#ifdef DOUBLEPRECISION
static_assert((DOUBLEPRECISION + 1) == 1 || (DOUBLEPRECISION + 1) == 2 || EXPERT_LEVEL >= 42,
              "DOUBLEPRECISION with MFM must be set to 1 or 2\n");
#else // DOUBLEPRECISION
static_assert(EXPERT_LEVEL >= 42, "DOUBLEPRECISION with MFM must be set to 1 or 2\n");
#endif // DOUBLEPRECISION
#endif // GADGET_HYDRO == HYDRO_MFM

#ifdef HIGH_ART_VISC_START
static_assert(High_Art_Visc_Start >= 0, "HIGH_ART_VISC_START must be provided a value larger or equal to 0\n");
#endif // HIGH_ART_VISC_START

#ifdef KD_RESTRICT_MAXVEL
static_assert(KD_RESTRICT_MAXVEL >= 0, "KD_RESTRICT_MAXVEL must be provided with a value larger than 0\n");
#endif // KD_RESTRICT_MAXVEL

#ifdef PMGRID
static_assert(PMGRID >= 0, "PMGRID must be provided with a value larger than 0\n");
#endif // PMGRID

#ifdef PLACEHIGHRESREGION
static_assert(PLACEHIGHRESREGION >= 0, "PLACEHIGHRESREGION must be provided with a value larger than 0");
#endif // PLACEHIGHRESREGION

#ifdef ENLARGEREGION
static_assert(ENLARGEREGION >= 0, "ENLARGEREGION must be provided with a value larger than 0\n");
#endif // ENLARGEREGION

#ifdef HIGHRES_PMGRID
static_assert(HIGHRES_PMGRID >= 0, "HIGHRES_PMGRID must be provided with a value larger than 0\n");
#endif // HIGHRES_PMGRID

#ifdef LT_MAX_TEMP_FEEDBACK
static_assert(LT_MAX_TEMP_FEEDBACK >= 0, "LT_MAX_TEMP_FEEDBACK must be provided with a value larger than 0\n");
#endif // LT_MAX_TEMP_FEEDBACK

#ifdef LT_NMet
static_assert(LT_NMet >= 0, "LT_NMet must be provided with a value larger than 0\n");
#endif // LT_NMet

#ifdef TIME_DEP_ART_VISC
static_assert((TIME_DEP_ART_VISC + 0) >= 0, "TIME_DEP_ART_VISC has to be set to 0, 1 ,2 or 3");
static_assert((TIME_DEP_ART_VISC + 0) <= 3, "TIME_DEP_ART_VISC has to be set to 0, 1 ,2 or 3");
#if !defined(FIRST_ORDER_GRADIENTS) && (TIME_DEP_ART_VISC == 0)
static_assert((EXPERT_LEVEL + 0) >= 2, "TIME_DEP_ART_VISC=0 was developed before higher order gradients in SPH where "
                                       "used. Either switch on FIRST_ORDER_GRADIENTS or retune the parameters !");
#endif // FIRST_ORDER_GRADIENTS && (TIME_DEP_ART_VISC == 0)
#if !defined(NO_BALSARA_LIMITER) && ((TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2))
static_assert((EXPERT_LEVEL + 0) >= 0,
              "TIME_DEP_ART_VISC=1/2 needs the NO_BALSARA_LIMITER to be set or to be retuned for its parameters !");
#endif // NO_BALSARA_LIMITER && (TIME_DEP_ART_VISC == 0)
#endif // TIME_DEP_ART_VISC

#ifdef AVERAGE_ART_VISC
static_assert(Average_Art_Visc >= 0, "AVERAGE_ART_VISC has to be set between 0 and 4");
static_assert(Average_Art_Visc <= 4, "AVERAGE_ART_VISC has to be set between 0 and 4");
#endif // AVERAGE_ART_VISC

#ifdef WAKEUP
static_assert(WAKEUP >= 0, "WAKEUP has to be set to a value >= 0");
#endif // WAKEUP

#ifdef TIME_DEP_ART_COND
static_assert(TIME_DEP_ART_COND >= 0, "TIME_DEP_ART_COND has to be set to a value >= 0");
#ifndef ARTIFICIAL_CONDUCTIVITY
static_assert(1 < 0, "TIME_DEP_ART_COND needs ARTIFICIAL_CONDUCTIVITY to be defined");
#endif // ARTIFICIAL_CONDUCTIVITY
#endif // TIME_DEP_ART_COND

#ifdef TIME_DEP_ART_COND_GSMOOTH
#ifndef ARTIFICIAL_CONDUCTIVITY
static_assert(1 < 0, "TIME_DEP_ART_COND_GSMOOTH needs ARTIFICIAL_CONDUCTIVITY to be defined");
#endif // ARTIFICIAL_CONDUCTIVITY
#ifndef TIME_DEP_ART_COND
static_assert(1 < 0, "TIME_DEP_ART_COND_GSMOOTH needs TIME_DEP_ART_COND to be defined");
#endif // TIME_DEP_ART_COND
#endif // TIME_DEP_ART_COND_GSMOOTH

#ifdef TIME_DEP_MAGN_DISP
static_assert(TIME_DEP_MAGN_DISP >= 0, "TIME_DEP_ART_COND has to be set to 0, 1, 2 or 3");
static_assert(TIME_DEP_MAGN_DISP <= 3, "TIME_DEP_ART_COND has to be set to 0, 1, 2 or 3");
#ifndef ARTIFICIAL_MAGNETIC_DISSIPATION
static_assert(1 < 0, "TIME_DEP_MAGN_DISP needs ARTIFICIAL_MAGNETIC_DISSIPATION to be defined");
#endif // ARTIFICIAL_MAGNETIC_DISSIPATION
#endif // TIME_DEP_MAGN_DISP

#ifdef MU0_UNITY
#ifndef MAGNETIC
static_assert(1 < 0, "MU0_UNITY needs MAGNETIC to be defined");
#endif // !MAGNETIC
#endif // MU0_UNITY

#ifdef DIVB_TO_ALPHA
#ifndef MAGNETIC
static_assert(1 < 0, "DIVB_TO_ALPHA needs MAGNETIC to be defined");
#else // MAGNETIC
#ifndef DIVBCLEANING_TRICCO
static_assert(1 < 0, "DIVB_TO_ALPHA is only allowed when DIVBCLEANING_TRICCO is used");
#else // DIVBCLEANING_TRICCO
#if !(defined(TIME_DEP_ART_VISC) && ((TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)))
static_assert(1 < 0, "DIVB_TO_ALPHA is only allowed when TIME_DEP_ART_VISC=2/3 is used");
#endif // TIME_DEP_ART_VISC == 2 || TIME_DEP_ART_VISC == 3
#endif // !DIVBCLEANING_TRICCO
#endif // !MAGNETIC
#endif // DIVB_TO_ALPHA

#if defined(DIVBCLEANING_DEDNER) && !defined(MAGNETIC)
static_assert(1 < 0, "DIVBCLEANING_DEDNER needs MAGNETIC to be defined");
#endif // DIVBCLEANING_DEDNER && !MAGNETIC
#if defined(DIVBCLEANING_TRICCO) && !defined(MAGNETIC)
static_assert(1 < 0, "DIVBCLEANING_TRICCO needs MAGNETIC to be defined");
#endif // DIVBCLEANING_TRICCO && !MAGNETIC
#ifdef LMB_MFM_ENERGY_CLOSURE_LIMITER
#if GADGET_HYDRO != HYDRO_MFM
static_assert(1 < 0, "LMB_MFM_ENERGY_CLOSURE_LIMITER needs USE_MESHLESS_FINITE_MASS to be defined");
#endif // GADGET_HYDRO != HYDRO_MFM
#endif // LMB_MFM_ENERGY_CLOSURE_LIMITER


#if defined(DIVBCLEANING_TRICCO) && defined(DIVBCLEANING_DEDNER)
static_assert(1 < 0, "DIVBCLEANING_TRICCO and DIVBCLEANING_DEDNER are mutually exclusive");
#endif // DIVBCLEANING_TRICCO && DIVBCLEANING_DEDNER

#ifdef LMB_MHD_ILLCOND_RESISTIVITY
#ifndef ARTIFICIAL_MAGNETIC_DISSIPATION
static_assert(1 < 0, "LMB_MHD_ILLCOND_RESISTIVITY needs ARTIFICIAL_MAGNETIC_DISSIPATION to be defined");
#endif // !ARTIFICIAL_MAGNETIC_DISSIPATION
#if GADGET_HYDRO != HYDRO_MFM
static_assert(1 < 0, "LMB_MHD_ILLCOND_RESISTIVITY needs MFM, the condition number is only defined there");
#endif // GADGET_HYDRO != HYDRO_MFM
#endif // LMB_MHD_ILLCOND_RESISTIVITY

#ifdef FOF
#ifndef DENSITY_SPLIT_BY_TYPE
static_assert(1 < 0, "FOF needs DENSITY_SPLIT_BY_TYPE to be defined");
#endif // DENSITY_SPLIT_BY_TYPE
#endif // FOF

#if (MULTIPLEDOMAINS < 16)
#warning "You are using a value for MULTIPLEDOMAINS that is smaller than 16. This is not recommended."
#endif // MULTIPLEDOMAINS


#if defined(GM_BH_EVAPORATE_COLD_CLOUDS) && !defined(BLACK_HOLES)
static_assert(1 < 0, "GM_BH_EVAPORATE_COLD_CLOUDS requires BLACK_HOLES");
#endif //  defined(MV_GM_AGNMUPPI_OUTPUT) && !defined(MV_GM_AGNMUPPI)

#if (defined(GM_SFEFF_TTHRESH) || defined(NO_SLOW_RELAX_TO_EOS)) && (!defined(SFR) || defined(GM_MUPPI))
static_assert(1 < 0, "GM_SFEFF_TTHRESH and NO_SLOW_RELAX_TO_EOS require SFR and cannot be used with GM_MUPPI");
#endif //  defined(MV_GM_AGNMUPPI_OUTPUT) && !defined(MV_GM_AGNMUPPI)

// Only one extension can be active at a time
#ifndef SWITCH_TIMER_OFF
#if (defined(TIMER_NVTX) ? 1 : 0) + (defined(TIMER_VTUNE) ? 1 : 0) + (defined(TIMER_ROCTX) ? 1 : 0) +                  \
        (defined(TIMER_OMNITRACE) ? 1 : 0) + (defined(TIMER_MERIC) ? 1 : 0) >                                          \
    1
static_assert(false, "Only one of TIMER_NVTX, TIMER_VTUNE, TIMER_OMNITRACE, TIMER_ROCTX, or TIMER_MERIC "
                     "may be defined at the same time.");
#endif // TIMER selection
#endif // SWITCH_TIMER_OFF

#if defined(CS_DO_NOT_DECOUPLE_HYDRO) && !defined(CS_DECOUPLING)
static_assert(1 < 0, "CS_DO_NOT_DECOUPLE_HYDRO needs CS_DECOUPLING to be defined");
#endif // CS_DECOUPLING && !CS_DO_NOT_DECOUPLE_HYDRO

#ifdef LMB_DPP_TTD_XMAX
static_assert(LMB_DPP_TTD_XMAX > 0.0, "LMB_DPP_TTD_XMAX must be provided with a value larger than 0\n");
#endif // LMB_DPP_TTD_XMAX

#ifdef LMB_LIMIT_DPP
static_assert(LMB_LIMIT_DPP > 0.0, "LMB_LIMIT_DPP must be provided with a value larger than 0 (in 1/s)\n");
#endif // LMB_LIMIT_DPP

#endif // SWITCHES_CHECK_HPP
