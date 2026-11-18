/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This header file sets dependent switches.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SWITCHES_HPP
#define MSIDM_SWITCHES_HPP

//////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(fSIDM) || defined(rSIDM)
#ifndef mSIDM
#define mSIDM
#endif // mSIDM

#if !defined(mSIDM_NOOPENMP) && !defined(mSIDM_OPENMP)
#define mSIDM_OPENMP
#endif // !mSIDM_NOOPENMP && !mSIDM_OPENMP

#if !defined(mSIDM_TIMESTEP) && !defined(mSIDM_NOTIMESTEPCRIT)
#define mSIDM_TIMESTEP
#endif // !mSIDM_TIMESTEP && !mSIDM_NOTIMESTEPCRIT

#if !defined(mSIDM_DOACTIVEONCE) && !defined(mSIDM_NODOACTIVEONCE)
#define mSIDM_DOACTIVEONCE
#endif // !mSIDM_DOACTIVEONCE && !mSIDM_NODOACTIVEONCE

#if defined(mSIDM_UNEQUALMASS)
#ifndef SCATTERING_INDI_MASS_RATIO
#define SCATTERING_INDI_MASS_RATIO
#endif // SCATTERING_INDI_MASS_RATIO
#ifndef SCATTERING_UNEQUAL_MASS
#define SCATTERING_UNEQUAL_MASS
#endif // SCATTERING_UNEQUAL_MASS
#endif // mSIDM_UNEQUALMASS

#if defined(mSIDM_TIMESTEP) || defined(mSIDM_DIFFERENTIAL_CROSS_SECTION)
#ifndef mSIDM_VMAX
#define mSIDM_VMAX
#endif // mSIDM_VMAX
#endif // mSIDM_TIMESTEP || mSIDM_DIFFERENTIAL_CROSS_SECTION

#if defined(mSIDM_VDEP0) || defined(mSIDM_VDEP1) || defined(mSIDM_VDEP_MV) || defined(mSIDM_VDEP_RV)
#define mSIDM_VDEP
#endif // mSIDM_VDEP0 || mSIDM_VDEP1 || mSIDM_VDEP_MV || mSIDM_VDEP_RV

#ifndef mSIDM_UNEQUALMASS
#ifndef mSIDM_EQUALMASS
#define mSIDM_EQUALMASS
#endif // !mSIDM_EQUALMASS
#endif // !mSIDM_UNEQUALMASS

#define mSIDM_INTERACTCRIT

#ifdef mSIDM_SSHI
#ifndef ANALYTIC_POTENTIALS
#error mSIDM_SSHI requieres a host halo potential.
#endif // ANALYTIC_POTENTIALS
#ifndef mSIDM_EQUALMASS
#error mSIDM_SSHI has only been implemented for equal particle masses.
#endif // !mSIDM_EQUALMASS
#endif // mSIDM_SSHI

#ifdef mSIDM_PRINTSCATTERS
#ifndef rSIDM_SCATTERCOUNT
#define rSIDM_SCATTERCOUNT
#endif // rSIDM_SCATTERCOUNT
#endif // mSIDM_PRINTSCATTERS

#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
#if (!defined(DCS_USE_GRID) && !defined(DCS_USE_ZSLICE)) || (defined(DCS_USE_GRID) && defined(DCS_USE_ZSLICE))
#warning DCS_USE_GRID xor DCS_USE_ZSLICE should be selected for the interpolation method. Choosing DCS_USE_ZSLICE for compilation.
#ifndef DCS_USE_ZSLICE
#define DCS_USE_ZSLICE
#endif // DCS_USE_ZSLICE
#ifdef DCS_USE_GRID
#undef DCS_USE_GRID
#endif // DCS_USE_GRID
#endif // (!DCS_USE_GRID && !DCS_USE_ZSLICE) || (DCS_USE_GRID && DCS_USE_ZSLICE)
#ifndef mSIDM_NOEXTEND_DCS_TABLE
#define mSIDM_EXTEND_DCS_TABLE
#endif // mSIDM_NOEXTEND_DCS_TABLE
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION

#if defined(rSIDM_SCATTER_ONCE) && !defined(rSIDM_SCATTERCOUNT_INDI)
#define rSIDM_SCATTERCOUNT_INDI
#endif // rSIDM_SCATTER_ONCE && !rSIDM_SCATTERCOUNT_INDI
#endif // fSIDM || rSIDM

#if defined(mSIDM_DISSIPATION) && defined(fSIDM)
#define SCATTERING_FREQ_DISSIPATION
#define fSIDM_DISSIPATION
#endif // mSIDM_DISSIPATION && fSIDM

#if defined(SCATTERING_FREQ_DISSIPATION) && defined(SCATTERING_UNEQUAL_MASS)
#error Dissipation is not implemented for unequal mass scattering.
#endif // SCATTERING_FREQ_DISSIPATION &&  SCATTERING_UNEQUAL_MASS

#if defined(mSIDM_DISSIPATION) && defined(rSIDM)
#define SCATTERING_RARE_DISSIPATION
#define rSIDM_DISSIPATION
#endif // mSIDM_DISSIPATION && rSIDM

//////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef rSIDM
#if !defined(mSIDM_DIFFERENTIAL_CROSS_SECTION)
#define rSIDM_ISOTROPIC
#endif // !defined(mSIDM_DIFFERENTIAL_CROSS_SECTION)
#endif // rSIDM

#ifdef fSIDM
#if !defined(mSIDM_DIFFERENTIAL_CROSS_SECTION)
#define fSIDM_SIMPLE
#endif // !mSIDM_DIFFERENTIAL_CROSS_SECTION
#endif // fSIDM



//////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(mSIDM) || defined(mIDM)

#if defined(mSIDM_VMAX) || defined(mIDM_VMAX)
#define SCATTERING_VMAX
#if defined(mSIDM) && !defined(mSIDM_VMAX)
#error VMAX is used for all scattering modules or none.
#endif // mSIDM && !mSIDM_VMAX
#endif // mSIDM_VMAX || mIDM_VMAX

#if defined(mSIDM_HMIN) || defined(mIDM_HMIN)
#define SCATTERING_HMIN
#endif // mSIDM_HMIN || mIDM_HMIN

#ifndef PXP
#define PXP
#endif // PXP

#if defined(mSIDM_OPENMP) || defined(mIDM_OPENMP)
#ifndef _OPENMP
#error _OPENMP is required for mSIDM_OPENMP and mIDM_OPENMP, you can switch of OpenMP with mSIDM_NOOPENMP or mIDM_NOOPENMP
#endif // _OPENMP
#define PXP_OPENMP
#define SCATTERING_OPENMP
#endif // mSIDM_OPENMP || mIDM_OPENMP
#if defined(mSIDM_DOACTIVEONCE) || defined(mIDM_DOACTIVEONCE)
#define PXP_DOACTIVEONCE
#define SCATTERING_DOACTIVEONCE
#if defined(mSIDM) && !defined(mSIDM_DOACTIVEONCE)
#error DOACTIVEONCE is used for all scattering modules or none.
#endif // mSIDM && !mSIDM_DOACTIVEONCE
#endif // mSIDM_DOACTIVEONCE || mIDM_DOACTIVEONCE
#if defined(mSIDM_DEBUG) || defined(mIDM_DEBUG)
#define PXP_DEBUG
#define SCATTERING_DEBUG
#define DCS_DEBUG
#endif // mSIDM_DEBUG || mIDM_DEBUG
#if defined(mSIDM_IGNOREWARNING) || defined(mIDM_IGNOREWARNING)
#define SCATTERING_IGNOREWARNING
#endif // mSIDM_IGNOREWARNING || mIDM_IGNOREWARNING
#if defined(mSIDM_KERNEL_WARNING) || defined(mIDM_KERNEL_WARNING)
#define SCATTERING_KERNEL_WARNING
#endif // mSIDM_KERNEL_WARNING || mIDM_KERNEL_WARNING
#if defined(mSIDM_ID0SPECIALTSTEP) || defined(mIDM_ID0SPECIALTSTEP)
#define SCATTERING_ID0SPECIALTSTEP
#endif // mSIDM_ID0SPECIALTSTEP || mIDM_ID0SPECIALTSTEP

#if defined(SCATTERING_ID0SPECIALTSTEP) || defined(PXP_DOACTIVEONCE) || defined(PXP_DEBUG) ||                          \
    defined(ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET) || defined(PXP_OPENMP) || defined(SCATTERING_DEBUG)
#ifndef SCATTERING_ID
#define SCATTERING_ID
#endif // SCATTERING_ID
#endif // SCATTERING_ID0SPECIALTSTEP || PXP_DOACTIVEONCE || PXP_DEBUG || ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET ||
       // PXP_OPENMP || SCATTERING_DEBUG
#if defined(mSIDM_PRINTSCATTERS) || defined(mIDM_PRINTSCATTERS)
#ifndef SCATTERING_ID
#define SCATTERING_ID
#endif // SCATTERING_ID
#endif // mSIDM_PRINTSCATTERS || mIDM_PRINTSCATTERS

#if (defined(rSIDM_SCATTERCOUNT_INDI) || defined(rIDM_SCATTERCOUNT_INDI)) && !defined(SCATTERING_COUNT_INDI)
#define SCATTERING_COUNT_INDI
#endif // (rSIDM_SCATTERCOUNT_INDI || rIDM_SCATTERCOUNT_INDI) && !SCATTERING_COUNT_INDI
#endif // mSIDM || mIDM

//////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(LT_STELLAREVOLUTION) || defined(KD_FRICTION_DYNAMIC) || defined(AXION_DM) || defined(BLACK_HOLES)
#define TYPE_FOR_DATASTRUCT
#endif // LT_STELLAREVOLUTION || KD_FRICTION_DYNAMIC || AXION_DM || BLACK_HOLES

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_SWITCHES_HPP */
