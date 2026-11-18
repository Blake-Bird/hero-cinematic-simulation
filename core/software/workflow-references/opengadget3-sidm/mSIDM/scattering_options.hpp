/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Classes to specify the behaviour of the scattering class.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_SCATTERING_OTPIONS_HPP
#define MSIDM_SCATTERING_OTPIONS_HPP

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Scattering options for mSIDM.
 */
class sct_opt_msidm
{
  public:
    static const bool do_sidm = true;
    static const bool do_idm = false;
#ifdef rSIDM_ISOTROPIC
    static const bool do_isotropic = true;
#else // rSIDM_ISOTROPIC
    static const bool do_isotropic = false;
#endif // rSIDM_ISOTROPIC
#ifdef mSIDM_VDEP
    static const bool do_vdep = true;
#else // mSIDM_VDEP
    static const bool do_vdep = false;
#endif // mSIDM_VDEP
    static const bool do_vmax_param = false;
#ifdef mSIDM_HMIN
    static const bool do_hmin = true;
#else // mSIDM_HMIN
    static const bool do_hmin = false;
#endif // mSIDM_HMIN
#ifdef mSIDM_TIMESTEP
    static const bool do_timestep = true;
#else // mSIDM_TIMESTEP
    static const bool do_timestep = false;
#endif // mSIDM_TIMESTEP
#ifdef fSIDM_NORANDCOMP
    static const bool do_norandcomp = true;
#else // fSIDM_NORANDCOMP
    static const bool do_norandcomp = false;
#endif // fSIDM_NORANDCOMP
#ifdef rSIDM_SANGLE90
    static const bool do_sangle90 = true;
#else // rSIDM_SANGLE90
    static const bool do_sangle90 = false;
#endif // rSIDM_SANGLE90
#ifdef mSIDM_ID0SPECIALTSTEP
    static const bool do_id0specialtstep = true;
#else // mSIDM_ID0SPECIALTSTEP
    static const bool do_id0specialtstep = false;
#endif // mSIDM_ID0SPECIALTSTEP
#ifdef mSIDM_IGNOREWARNING
    static const bool do_ignorewarning = true;
#else // mSIDM_IGNOREWARNING
    static const bool do_ignorewarning = false;
#endif // mSIDM_IGNOREWARNING
#ifdef mSIDM_INTERACTCRIT
    static const bool do_interact_crit = true;
#else // mSIDM_INTERACTCRIT
    static const bool do_interact_crit = false;
#endif // mSIDM_INTERACTCRIT
    static const bool do_fastmode = false;
#ifdef mSIDM_PRINTSCATTERS
    static const bool do_printscatters = true;
#else // mSIDM_PRINTSCATTERS
    static const bool do_printscatters = false;
#endif // mSIDM_PRINTSCATTERS
#ifdef mSIDM_EXTEND_DCS_TABLE
    static const bool do_extend_dcs_table = true;
#else // mSIDM_EXTEND_DCS_TABLE
    static const bool do_extend_dcs_table = false;
#endif // mSIDM_EXTEND_DCS_TABLE
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////
#endif /* MSIDM_SCATTERING_OTPIONS_HPP */
