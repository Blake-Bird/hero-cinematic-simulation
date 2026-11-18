/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file with declarations for differential cross-section classes.
 *
 * This header file provides various methods that are needed to simulate full
 * differential cross-sections. It contains look-up tables for the cumulative
 * distribution function and the total and transfer cross-sections. Moreover, it
 * allows interpolating from those tables to obtain the quantities needed for
 * interactions, including the scattering angle.
 *
 * @author OpenGadget3-development team, in particular Marc Wiertel and Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, Klaus Dolag, and all contributing authors
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef DCS_CLASS_HPP
#define DCS_CLASS_HPP

#include "../CodeBase/precision.hpp"
#define JSON_ASSERT(x) // suppress redefinition of assert
#include <nlohmann/json.hpp>
#include "CrossSectionTable.hpp"

#ifdef SCATTERING_OPENMP
#include <omp.h>
#endif // SCATTERING_OPENMP

namespace scatter::pxp_scattering
{

using json = nlohmann::json;

/**
 * @brief Main class to handle full differential cross-sections.
 */
class diff_cross_section_class
{
  public:
    /** Constructor that takes care of an OpenMP lock. */
    diff_cross_section_class();
    /** Destructor that takes care of an OpenMP lock. */
    ~diff_cross_section_class();

    /** Initialization after the parameter file has been read. */
    void init(const std::string &name_in, const json &param);

    /** Method to check and extend the table if needed. */
    void extend_table_if_needed(const MyAtLeastDouble target_vel);

    /** Interpolation of the transfer cross-section for fSIDM. */
    template <bool do_ignorewarning> inline MyAtLeastDouble interp_sigmaT(const MyAtLeastDouble vel) const
    {
        return m_dataStructure.interp_sigmaT<do_ignorewarning>(vel);
    }

    /** Interpolation of the total cross-section for rSIDM. */
    template <bool do_ignorewarning> inline MyAtLeastDouble interp_sigma(const MyAtLeastDouble vel) const
    {
        return m_dataStructure.interp_sigma<do_ignorewarning>(vel);
    }

    /** Interpolation of the scattering angle. */
    template <bool do_ignorewarning>
    inline MyAtLeastDouble interp_theta(const double rnd_num, const MyAtLeastDouble vel) const
    {
        return m_dataStructure.interp_theta<do_ignorewarning>(rnd_num, vel);
    }

    /** Maximum of the transfer cross-section for fSIDM times velocity below vmax. */
    inline MyFloat sigmaDm_vel_f_ext(const MyAtLeastDouble vmax) const
    {
        return m_dataStructure.sigmaDm_vel_f_ext(vmax);
    }

    /** Maximum of the total cross-section for rSIDM times velocity below vmax. */
    inline MyFloat sigmaDm_vel_r_ext(const MyAtLeastDouble vmax) const
    {
        return m_dataStructure.sigmaDm_vel_r_ext(vmax);
    }

  private:
    std::string m_name;
    cross_section_table m_dataStructure;

#ifdef SCATTERING_OPENMP
    omp_lock_t lock_table;
#endif // SCATTERING_OPENMP
};

} // namespace scatter::pxp_scattering

#endif // DCS_CLASS_HPP
