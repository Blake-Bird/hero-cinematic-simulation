/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Definition of functions for several differential cross-sections.
 *
 * The functions defined here are ment to return the differential cross-section in cgs units.
 * This implies, for example, that the parameter sigma_0 can be given in cm^2/g, while
 * velocity parameters can be in code units (their units typically cancel out).
 * A new differential cross-section (dsigma/dcos(theta)) can be implemented by simply adding
 * a new function and registering it as done for the ones already implemented. That is all
 * one needs to do! The first argument needs to be the scattering angle and if used the
 * argument for the scattering velocity has to be named vrel.
 *
 * @author OpenGadget3-development team, in particular Marc Wiertel and Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, Klaus Dolag, and all contributing authors
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_CROSS_SECTIONS_HPP
#define MSIDM_CROSS_SECTIONS_HPP
#include "GslWrapper.hpp"
#include "research_differential_table.hpp"

namespace scatter
{

//////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Differential cross-section (dsigma/dcos(theta)) for isotropic velocity-independent scattering.
 */
inline double isotropic_scattering(double theta, double sigma_0)
{
    return sigma_0; // the actual value of the total cross-section is two times sigma_0
}
REGISTER_FUNC(isotropic_scattering, theta, sigma_0);

//////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Differential cross-section (dsigma/dcos(theta)) for Rutherford scattering.
 */
inline double rutherford_scattering(double theta, double vrel, double sigma_0, double w)
{
    const double w2 = w * w;
    const double lower = w2 + ((vrel * vrel) / 2.0) * (1.0 - cos(theta));
    return sigma_0 * w2 * w2 / (2.0 * lower * lower);
}
REGISTER_FUNC(rutherford_scattering, theta, vrel, sigma_0, w);

//////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Differential cross-section (dsigma/dcos(theta)) for Moeller scattering.
 */
inline double moeller_scattering(double theta, double vrel, double sigma_0, double w)
{
    const double w2 = w * w;
    const double v2 = vrel * vrel;
    const double CosTheta2 = cos(theta) * cos(theta);
    const double lower = (1.0 - CosTheta2) * v2 * v2 + 4.0 * v2 * w2 + 4.0 * w2 * w2;
    const double _t = ((3.0 * CosTheta2 + 1.0) * v2 * v2 + 4.0 * v2 * w2 + 4.0 * w2 * w2) / (lower * lower);
    return sigma_0 * w2 * w2 * _t;
}
REGISTER_FUNC(moeller_scattering, theta, vrel, sigma_0, w);

//////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Differential cross-section (dsigma/dcos(theta)) for Kahlhoefer velocity-independent scattering.
 */
inline double kahlhoefer_velocity_independent_scattering(double theta, double sigma_0)
{
    const double cos2 = cos(theta) * cos(theta);
    return sigma_0 * ((1 + cos2) / (1 - cos2));
}
REGISTER_FUNC(kahlhoefer_velocity_independent_scattering, theta, sigma_0);

//////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace scatter

#endif /* MSIDM_CROSS_SECTIONS_HPP */
