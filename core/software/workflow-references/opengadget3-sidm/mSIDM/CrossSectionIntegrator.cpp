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

#include "CrossSectionIntegrator.hpp"
// #include <gsl/gsl_integration.h>
// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_roots.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include "cross_sections.hpp"
#include "tools.hpp"

namespace scatter::pxp_scattering
{

using namespace scatter::tools;

cross_section_integrator::cross_section_integrator()
    : m_cgsToCodeUnits(All.UnitLength_in_cm * All.UnitLength_in_cm / All.UnitMass_in_g)
{
    if (m_cgsToCodeUnits <= 0.0)
    {
        coutr(error) << "DCS | ERROR: Unit conversion factor is <= 0.0" << endl;
        std::abort();
    }
}

void cross_section_integrator::init(std::string name, json params)
{
    m_name = name;
    m_params = std::move(params);
    if (!m_params.contains("crosssection"))
    {
        coutr(error) << m_name << " DCS | ERROR: Field 'MSIDM_CrossSection' is not specified in parameter file" << endl;
        std::abort();
    }

    m_tmin = m_params.value<double>("thetaC_fwd", 0.0);
    m_tmax = M_PI - m_params.value<double>("thetaC_bwd", 0.0);

    auto fit = GslFactoryRegistry.find(m_params["crosssection"].get<std::string>());
    if (fit == GslFactoryRegistry.end())
    {
        coutr(error) << m_name << " DCS | ERROR: Could not find selected cross section '"
                     << m_params["crosssection"].get<std::string>() << "'. There are " << GslFactoryRegistry.size()
                     << " available cross sections:" << endl;
        for (const auto &[k, v] : GslFactoryRegistry)
        {
            coutr(error) << "      - " << k << endl;
        }
        std::abort();
    }
    m_diffFunc = fit->second.make(m_params);

    auto wit = GslWrapperRegistry.find(m_params["crosssection"].get<std::string>());
    if (wit == GslWrapperRegistry.end() || !wit->second)
    {
        coutr(error)
            << m_name
            << " DCS | ERROR: Something went wrong during the creation of the GslWrapper instance for cross section '"
            << m_params["crosssection"].get<std::string>() << "'. Available wrappers:" << endl;
        for (const auto &[k, v] : GslWrapperRegistry)
        {
            coutr(error) << "      - " << k << endl;
        }
        std::abort();
    }
    m_diffWrapper = wit->second;
}

double cross_section_integrator::compute_total_cross_section(double vrel)
{
    return compute_total_cross_section_integral(m_tmin, m_tmax, vrel);
}

double cross_section_integrator::compute_trans_cross_section(double vrel)
{
    double sigmaT = compute_trans_cross_section_integral(0.0, m_tmin, vrel);
    if (m_params["thetaC_bwd"].get<double>() > 0.0)
    {
        sigmaT += compute_trans_cross_section_integral(m_tmax, M_PI, vrel);
    }

    return sigmaT;
}

double cross_section_integrator::solve_function(double tmax, void *data)
{
    cross_section_integrator *inst = reinterpret_cast<cross_section_integrator *>(data);
    SolveStruct &dsolve = inst->m_dsolve;

    // avoid division by zero. Value of theta doesn't matter when sigma_tot = 0. Set to zero here
    if (dsolve.sigma_tot <= 0)
    {
        return 0;
    }

    // CDF = 1/sigma_tot * Int_{t_min}^{theta} (p(theta,params))
    double cdf = inst->compute_total_cross_section_integral(inst->m_tmin, tmax, dsolve.vrel) / dsolve.sigma_tot;

    // solve CDF(theta) - x = 0 for theta
    // should always have one root since CDF,x in [0,1]
    return (cdf - dsolve.x);
}

double cross_section_integrator::total_cross_section(double t, void *data)
{
    cross_section_integrator *inst = reinterpret_cast<cross_section_integrator *>(data);
    // directly call the gsl_function and modulate with factor
    return inst->m_diffFunc.function(t, inst->m_diffFunc.params) * sin(t);
}

double cross_section_integrator::transfer_cross_section(double t, void *data)
{
    cross_section_integrator *inst = reinterpret_cast<cross_section_integrator *>(data);
    const double int_fact = 1.0 - abs(cos(t));
    // directly call the gsl_function and modulate with factor
    return inst->m_diffFunc.function(t, inst->m_diffFunc.params) * int_fact * sin(t);
}

double cross_section_integrator::solve_cdf(double x, double vrel, double sigma_tot)
{
    m_dsolve.x = x;
    m_dsolve.vrel = vrel;
    m_dsolve.sigma_tot = sigma_tot;

    double tmin = m_tmin;
    double tmax = m_tmax;

    auto f = [this](double t) { return solve_function(t, this); };
    // matches GSL's gsl_root_test_interval(tmin, tmax, epsabs=0, epsrel=0.001)
    auto tol = [](double lo, double hi) { return std::fabs(hi - lo) < 0.001 * std::min(std::fabs(lo), std::fabs(hi)); };

    const std::uintmax_t iter_limit = 100;
    std::uintmax_t max_iter = iter_limit;
    const std::pair<double, double> bracket = boost::math::tools::toms748_solve(f, tmin, tmax, tol, max_iter);

#ifdef DCS_DEBUG
    if (max_iter >= iter_limit)
    {
        cout(error) << m_name << " DCS | ERROR: root solve did not converge within " << iter_limit << " iterations."
                    << std::endl;
    }
#endif // DCS_DEBUG

    return 0.5 * (bracket.first + bracket.second);
}

// double cross_section_integrator::solve_cdf(double x, double vrel, double sigma_tot)
// {
//     m_dsolve.x = x;
//     m_dsolve.vrel = vrel;
//     m_dsolve.sigma_tot = sigma_tot;

//     double tmin = m_tmin;
//     double tmax = m_tmax;

//     gsl_function F;
//     F.function = solve_function;
//     F.params = this;

//     gsl_root_fsolver *s = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
//     gsl_root_fsolver_set(s, &F, tmin, tmax);

//     int status = 0;
//     int iter = 0, max_iter = 100;
//     double r = 0;

//     do
//     {
//         ++iter;
//         status = gsl_root_fsolver_iterate(s);
//         r = gsl_root_fsolver_root(s);
//         tmin = gsl_root_fsolver_x_lower(s);
//         tmax = gsl_root_fsolver_x_upper(s);
//         status = gsl_root_test_interval(tmin, tmax, 0, 0.001);

// #ifdef DCS_DEBUG
//         if (status > 0)
//         {
//             cout(error) << m_name << " DCS | ERROR: GSL Error in solve. " << gsl_strerror(status) << std::endl;
//         }
// #endif // DCS_DEBUG
//     } while (status == GSL_CONTINUE && iter < max_iter);

//     gsl_root_fsolver_free(s);

//     return r;
// }

double cross_section_integrator::compute_total_cross_section_integral(double tmin, double tmax, double vrel)
{
    m_diffWrapper->set_velocity(vrel);

    // Use total cross section wrapper which uses internal gsl_function
    auto f = [this](double t) { return total_cross_section(t, this); };

    constexpr double epsrel = 1e-5; // the relative allowed error
    double error = DBL_MAX;
    boost::math::quadrature::tanh_sinh<double> integrator;
    const double result = integrator.integrate(f, tmin, tmax, epsrel, &error);

    if (error > epsrel * std::fabs(result) && result != 0.0)
    {
        cout(warning) << m_name << " DCS | WARNING: Cross Section Integral for total cross-section may be wrong."
                      << std::endl;
#ifndef DCS_IGNOREWARNING
        ENDRUN("If you are sure that DCS related warnings can be ignored turn on DCS_IGNOREWARNING to continue.");
#endif // DCS_IGNOREWARNING
        return 0.0;
    }
    else
    {
        return result / m_cgsToCodeUnits;
    }
}

// double cross_section_integrator::compute_total_cross_section_integral(double tmin, double tmax, double vrel)
// {
//     gsl_integration_cquad_workspace *workspace = gsl_integration_cquad_workspace_alloc(100);

//     m_diffWrapper->set_velocity(vrel);

//     // Use total cross section wrapper which uses internal gsl_function
//     gsl_function F;
//     F.function = &cross_section_integrator::total_cross_section;
//     F.params = this;

//     double result, error;
//     const double epsabs = 0; // use relative error
//     const double epsrel = 1e-5; // the relative allowed error
//     size_t nevals;

//     int status = gsl_integration_cquad(&F, tmin, tmax, epsabs, epsrel, workspace, &result, &error, &nevals);

//     if (status)
//     {
//         if (status == GSL_EROUND)
//         {
//             cout(warning) << m_name << " DCS | WARNING: Cross Section Integral may be wrong." << std::endl;
//             result = 0;
//         }
//         else
//         {
//             abort();
//         }
//     }

//     gsl_integration_cquad_workspace_free(workspace);
//     return result / m_cgsToCodeUnits;
// }
double cross_section_integrator::compute_trans_cross_section_integral(double tmin, double tmax, double vrel)
{
    m_diffWrapper->set_velocity(vrel);

    // Use transfer cross section wrapper which uses internal gsl_function
    auto f = [this](double t) { return transfer_cross_section(t, this); };

    constexpr double epsrel = 1e-5; // the relative allowed error
    double error = DBL_MAX;
    boost::math::quadrature::tanh_sinh<double> integrator;
    const double result = integrator.integrate(f, tmin, tmax, epsrel, &error);

    if (error > epsrel * std::fabs(result) && result != 0.0)
    {
        cout(warning) << m_name << " DCS | WARNING: Cross Section Integral for transfer cross-section may be wrong."
                      << std::endl;
#ifndef DCS_IGNOREWARNING
        ENDRUN("If you are sure that DCS related warnings can be ignored turn on DCS_IGNOREWARNING to continue.");
#endif // DCS_IGNOREWARNING
        return 0.0;
    }
    else
    {
        return result / m_cgsToCodeUnits;
    }
}

// double cross_section_integrator::compute_trans_cross_section_integral(double tmin, double tmax, double vrel)
// {
//     gsl_integration_cquad_workspace *workspace = gsl_integration_cquad_workspace_alloc(100);

//     m_diffWrapper->set_velocity(vrel);

//     // Use transfer cross section wrapper which uses internal gsl_function
//     gsl_function F;
//     F.function = &cross_section_integrator::transfer_cross_section;
//     F.params = this;

//     double result, error;
//     const double epsabs = 0; // use relative error
//     const double epsrel = 1e-5; // the relative allowed error
//     size_t nevals;

//     int status = gsl_integration_cquad(&F, tmin, tmax, epsabs, epsrel, workspace, &result, &error, &nevals);

//     if (status)
//     {
//         if (status == GSL_EROUND)
//         {
//             cout(warning) << m_name << " DCS | WARNING: Cross Section Integral may be wrong." << std::endl;
//             result = 0;
//         }
//         else
//         {
//             abort();
//         }
//     }

//     gsl_integration_cquad_workspace_free(workspace);
//     return result / m_cgsToCodeUnits;
// }
} // namespace scatter::pxp_scattering
