/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file with definitions for differential cross-section classes.
 *
 * This source file provides various methods that are needed to simulate full
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

#include "dcs_class.hpp"
#include "tools.hpp"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>

using namespace scatter::pxp_scattering;
using namespace scatter::tools;

diff_cross_section_class::diff_cross_section_class()
{
#ifdef SCATTERING_OPENMP
    // initialize OpenMP lock
    omp_init_lock(&lock_table);
#endif
}

diff_cross_section_class::~diff_cross_section_class()
{
#ifdef SCATTERING_OPENMP
    // initialize OpenMP lock
    omp_destroy_lock(&lock_table);
#endif
}

void diff_cross_section_class::init(const std::string &name_in, const json &param)
{
    m_name = name_in;
    m_dataStructure.init(m_name, param);
}

void diff_cross_section_class::extend_table_if_needed(const MyAtLeastDouble target_vel)
{

    if (target_vel > m_dataStructure.max_velocity())
    {
        cout(info) << m_name << " DCS | INFO: Encountered velocity out of table. v = " << target_vel
                   << ". Extending table..." << endl;

#ifdef SCATTERING_OPENMP
        // set the lock, if already set => wait for release
        omp_set_lock(&lock_table);
#endif // SCATTERING_OPENMP

        MyAtLeastDouble new_velocity = m_dataStructure.extend_table(target_vel);

#ifdef SCATTERING_OPENMP
        omp_unset_lock(&lock_table);
#endif // SCATTERING_OPENMP

        cout(info) << m_name << " DCS | INFO: Table extension done. v_max = " << new_velocity << endl;
    }
}