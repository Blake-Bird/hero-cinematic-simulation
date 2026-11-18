/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Template type definitions for the scattering class.
 *
 * Only the type definitions for the scattering class are given here.
 * The template specialization is part of scattering.cpp.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_TYPES_HPP
#define MSIDM_SCATTERING_TYPES_HPP

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_scattering
{
using namespace pxp_scattering;

// The switches in the following type definitions should be used to select
// the modules for scattering (don't use all of them).
#ifdef mSIDM
// This type definition should be the same as the template specialization
// given in scattering.cpp.
typedef Scattering<sct_opt_msidm, LambdaNumerical<kernel_msidm>
#ifdef mSIDM_DEBUG
                   ,
                   module_debug
#endif // mSIDM_DEBUG
#ifdef rSIDM_SCATTERCOUNT
                   ,
                   module_scattercount
#endif // rSIDM_SCATTERCOUNT
#ifdef rSIDM_SCATTERCOUNT_INDI
                   ,
                   module_scattercount_indi
#endif // rSIDM_SCATTERCOUNT_INDI
#ifdef rSIDM_SCATTER_ONCE
                   ,
                   module_scatter_once
#endif // rSIDM_SCATTER_ONCE
#ifdef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
                   ,
                   module_id_cut
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
#ifdef fSIDM
                   ,
                   module_freq
#endif // fSIDM
#ifdef rSIDM
                   ,
                   module_rare
#endif // rSIDM
#ifdef fSIDM_SIMPLE
                   ,
                   module_freq_simple
#endif // fSIDM_SIMPLE
#ifdef rSIDM_ISOTROPIC
                   ,
                   module_rare_isotropic
#endif // rSIDM_ISOTROPIC
#ifdef mSIDM_EQUALMASS
                   ,
                   module_equalmass
#endif // mSIDM_EQUALMASS
#ifdef mSIDM_UNEQUALMASS
                   ,
                   module_unequalmass
#endif // mSIDM_UNEQUALMASS
#ifdef mSIDM_VMAX
                   ,
                   module_vmax
#endif // mSIDM_VMAX
#ifdef mSIDM_HMIN
                   ,
                   module_hmin
#endif // mSIDM_HMIN
#ifdef mSIDM_VDEP0
                   ,
                   module_vdep0
#endif // mSIDM_VDEP0
#ifdef mSIDM_VDEP1
                   ,
                   module_vdep1
#endif // mSIDM_VDEP1
#ifdef mSIDM_VDEP_MV
                   ,
                   module_vdep_mv
#endif // mSIDM_VDEP_MV
#ifdef mSIDM_VDEP_RV
                   ,
                   module_vdep_rv
#endif // mSIDM_VDEP_RV
#ifdef rSIDM_DISSIPATION
                   ,
                   module_rare_dissipation
#endif // rSIDM_DISSIPATION
#ifdef fSIDM_DISSIPATION
                   ,
                   module_freq_dissipation
#endif // fSIDM_DISSIPATION
#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
                   ,
                   module_differential_cross_section
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION
                   >
    Scattering_msidm;
#endif // mSIDM

} // namespace scatter::pxp_scattering

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_SCATTERING_TYPES_HPP */
