/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Defines the integrator that is used during the computation of the dcs table.
 *
 * @author OpenGadget3-development team, in particular Marc Wiertel and Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#pragma once
#include "GslWrapper.hpp"
#include <memory>

namespace scatter::pxp_scattering
{

/**
 * @brief Class to integrate differential cross-sections over the scattering angle.
 */
class cross_section_integrator
{

    struct SolveStruct
    {
        double x;
        double vrel;
        double sigma_tot;
    };

    std::shared_ptr<GSLWrapperBase> m_diffWrapper;
    gsl_function m_diffFunc;

    const double m_cgsToCodeUnits;

    std::string m_name;
    json m_params;
    double m_tmin; // minimum theta, default: 0
    double m_tmax; // maximum theta, default: M_PI
    SolveStruct m_dsolve; // params for solve cdf

    static double solve_function(double tmax, void *data);

    /**
     * @brief Total cross section wrapper
     */
    static double total_cross_section(double t, void *data);

    /**
     * @brief Transfer cross section wrapper
     */
    static double transfer_cross_section(double t, void *data);

    /**
     * Computes the transfer cross section integral from `tmin` to `tmax` at given relative velocity
     *
     * @return transfer cross section in code units
     */
    double compute_trans_cross_section_integral(double tmin, double tmax, double vrel);

  public:
    cross_section_integrator();
    void init(std::string name, json params);

    /**
     * Computes total cross section at given relative velocity.
     *
     * @return Cross section in code units
     */
    double compute_total_cross_section(double vrel);

    /**
     * Computes total transfer cross section at given relative velocity
     */
    double compute_trans_cross_section(double vrel);

    /**
     * Solves the cumulative distribution function given by CDF(theta, vrel) = x for theta.
     *
     * @param x
     * @param vrel velocity
     * @param sigma_tot total cross section in code units
     * @return theta
     */
    double solve_cdf(double x, double vrel, double sigma_tot);

    /**
     * Computes the total cross section integral from `tmin` to `tmax` at given relative velocity.
     *
     * @return Cross section in code units
     */
    double compute_total_cross_section_integral(double tmin, double tmax, double vrel);
};

} // namespace scatter::pxp_scattering
