/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Handles the computation and extension of the dcs table, used to sample the scattering direction and strength
 * from the specified cross-section via interpolation.
 *
 * @author OpenGadget3-development team, in particular Marc Wiertel and Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "CrossSectionTable.hpp"
#include "../CodeBase/allvars.hpp"
#include "tools.hpp"
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <fstream>

namespace scatter::pxp_scattering
{

using namespace scatter::tools;

void cross_section_table::init(const std::string &name, const json &param)
{
    m_name = name;
    m_params = param;

    if (!m_params.contains("crosssection"))
    {
        coutr(error) << m_name << " DCS | ERROR: Field 'MSIDM_CrossSection' is not specified in parameter file" << endl;
        std::abort();
    }
    if (!m_params.contains("params") || !m_params["params"].is_string())
    {
        coutr(error) << m_name << " DCS | ERROR: Field 'MSIDM_DCS_params' is either not specified or not a string"
                     << endl;
        std::abort();
    }

    // Initialize the integrator
    {
        json ij{};
        ij["crosssection"] = m_params["crosssection"];
        ij["thetaC_fwd"] = m_params.value<double>("thetaC_fwd", 0.0);
        ij["thetaC_bwd"] = m_params.value<double>("thetaC_bwd", 0.0);
        ij["vrel"] = 0; // mandatory field
        if (!parse_model_parameters(m_params["params"], ij))
        {
            coutr(error) << m_name << " DCS | ERROR: Could not parse model parameters." << endl;
            std::abort();
        }

        m_integr.init(m_name, std::move(ij));
    }

    // TODO@MW: Could handle error values more rigorously
    m_lenxh = m_params.value<size_t>("lenxh", 256);
    m_lenv = m_params.value<size_t>("lenv", 256);
    m_vmin = m_params.value<double>("v_min", 0.0);
    m_vmax = m_params.value<double>("v_max", 0.0);
    m_reserveFact = m_params.value<double>("reserve_fact", 1.5);
    m_fwd = m_params.value<double>("thetaC_fwd", 0.0);
    m_bwd = m_params.value<double>("thetaC_bwd", 0.0);

    // TODO@MW: Add support for multiple config files
    m_dcsFile = name + "_DcsConfig.json";

    // Sanity checks
    if (m_lenv <= 1 || m_lenxh <= 1)
    {
        coutr(error) << m_name << " DCS | ERROR: lenv and lenxh have to be >= 2." << endl;
        std::abort();
    }
    if (m_vmin <= 0.0 || m_vmax <= 0.0 || m_vmin > m_vmax)
    {
        coutr(error) << m_name << " DCS | ERROR: vmin and vmax have to be greater than zero, with vmax > vmin." << endl;
        std::abort();
    }
    if (m_reserveFact <= 0.0)
    {
        coutr(error) << m_name
                     << " DCS | ERROR: reserveFact is a multiplicative factor that needs to be greater than zero."
                     << endl;
        std::abort();
    }

    setup_table();

    // compute log of v_min / v_max
    m_logVmin = log(m_vmin);
    m_logVmax = log(m_vmax);
    // remove rounding errors by reducing input variables to compatible number with exp/log operation
    m_vmin = exp(m_logVmin);
    m_vmax = exp(m_logVmax);
    // set current boundary vindex of table
    m_boundaryIndex = m_lenv - 1;

    if (!read_config(m_dcsFile))
    {
        compute_surface();
        export_config(m_dcsFile);
    }

    evaluate_sigmaDm_vel_max();
}

bool cross_section_table::parse_model_parameters(std::string strParam, json &out)
{
    size_t last = 0;
    size_t next = 0;
    std::vector<std::string> elems{};
    while ((next = strParam.find(";", last)) != std::string::npos)
    {
        elems.push_back(strParam.substr(last, next - last));
        last = next + 1;
    }
    elems.push_back(strParam.substr(last));

    try
    {
        std::string json_str = "{";
        next = 0;
        for (const auto &kv : elems)
        {
            if (!kv.empty() && (next = kv.find("=", 0)) != std::string::npos)
            {
                auto k = kv.substr(0, next);
                auto v = kv.substr(next + 1);
                // TODO@MW: To have support for non-integral values, the input format should be changed
                json_str += "\"" + k + "\":" + v + ",";
            }
        }
        json_str = json_str.substr(0, json_str.size() - 1) + "}";
        json other = json::parse(json_str);
        out.update(other);
    }
    catch (...)
    {
        return false;
    }

    return true;
}

void cross_section_table::setup_table()
{
    // Reserve more elements for easier extension
    size_t newLen = m_lenv * m_reserveFact;
    m_velocity.reserve(newLen);
    m_sigmaTot.reserve(newLen);
    m_sigmaTrans.reserve(newLen);
    m_sigmaTotMaxVel.reserve(newLen);
    m_sigmaTransMaxVel.reserve(newLen);
    m_table.reserve(newLen);

    // Initialize data structure
    m_velocity.resize(m_lenv);
    m_sigmaTot.resize(m_lenv);
    m_sigmaTrans.resize(m_lenv);
    m_sigmaTotMaxVel.resize(m_lenv);
    m_sigmaTransMaxVel.resize(m_lenv);
    m_axis.resize(m_lenv);

    m_table.resize(m_lenv);
    for (auto &x : m_table)
    {
        x.resize(m_lenxh);
    }
}

void cross_section_table::compute_surface()
{
    // set custom error handler. When using DCS_DEBUG, will print out error, else ignore them
    gsl_error_handler_t *old_handler = gsl_set_error_handler(&gsl_error_handler);

#ifdef DCS_USE_GRID
    coutr(info) << m_name << " DCS | INFO: Computing Theta table (REGULAR GRID) with params:" << endl
                << "    v_min = " << m_vmin << ", v_max = " << m_vmax << ", steps_v = " << m_lenv
                << ", steps_x = " << m_lenxh << endl;
#endif // DCS_USE_GRID

#ifdef DCS_USE_ZSLICE
    coutr(info) << m_name << " DCS | INFO: Computing Theta table (ZSLICE) with params:" << endl
                << "    v_min = " << m_vmin << ", v_max = " << m_vmax << ", steps_h = " << m_lenxh
                << ", steps_v = " << m_lenv << endl;
#endif // DCS_USE_ZSLICE

// initialize height array
#ifdef DCS_USE_ZSLICE
    double tbwd = m_params.value<double>("thetaC_bwd", 0.0);

    if (m_fwd > 0.0 || tbwd > 0.0)
    {
        const double t_range = M_PI - m_fwd - tbwd;
        for (size_t xn = 0; xn < m_lenxh; ++xn)
        {
            set_axis(xn, m_fwd + static_cast<double>(xn) / (m_lenxh - 1) * t_range);
        }
    }
    else
    {
        for (size_t xn = 0; xn < m_lenxh; ++xn)
        {
            set_axis(xn, static_cast<double>(xn) / (m_lenxh - 1) * M_PI);
        }
    }
#endif // DCS_USE_ZSLICE

    // Main loop over velocity
    for (size_t vn = 0; vn < m_lenv; ++vn)
    {
        const double vrel = get_velocity_step(vn, m_lenv);
        set_velocity(vn, vrel);

        const double totalCrossSection = m_integr.compute_total_cross_section(vrel);
        set_tot_sigma(vn, totalCrossSection);

        set_trans_sigma(vn, m_integr.compute_trans_cross_section(vrel));

        // loop over height
        for (size_t xn = 0; xn < m_lenxh; ++xn)
        {
#ifdef DCS_USE_GRID
            // grid in x is linear
            const double x = static_cast<double>(xn) / (m_lenxh - 1);
            set_axis(xn, x);

            // solve the equation CDF(theta) = x for theta to get a mapping between theta and x
            const double theta_res = m_integr.solve_cdf(x, vrel, totalCrossSection);
            set_table(xn, vn, theta_res);
#endif // DCS_USE_GRID

#ifdef DCS_USE_ZSLICE
            // compute the upper part of CDF
            const double partialCrossSection =
                m_integr.compute_total_cross_section_integral(m_fwd, get_axis_value(xn), vrel);
            set_table(xn, vn, partialCrossSection / totalCrossSection);
#endif // DCS_USE_ZSLICE
        }
    }

#ifdef DCS_USE_GRID
    coutr(info) << m_name << " DCS | INFO: Theta Regular Grid computation complete." << endl;
#endif // DCS_USE_GRID

#ifdef DCS_USE_ZSLICE
    coutr(info) << m_name << " DCS | INFO: ZSlice Computation complete." << endl;
#endif // DCS_USE_ZSLICE

    // switch back to old gsl error handler
    gsl_set_error_handler(old_handler);
}

void cross_section_table::extend_table_single_value(const double new_vel)
{
    // new index of last element in Velocity direction
    const size_t last_index = m_velocity.size();

    m_velocity.push_back(new_vel);
    size_t newLen = m_velocity.size();
    m_sigmaTot.resize(newLen);
    m_sigmaTrans.resize(newLen);
    m_sigmaTotMaxVel.resize(newLen);
    m_sigmaTransMaxVel.resize(newLen);

    m_table.push_back(std::vector<double>(m_lenxh, 0.0));

    // calculate and write total cross section in new slot
    const double total_crossSection = m_integr.compute_total_cross_section(new_vel);
    set_tot_sigma(last_index, total_crossSection);

    // calculate and write transfer cross section in new slot
    const double trans_crossSection = m_integr.compute_trans_cross_section(new_vel);
    set_trans_sigma(last_index, trans_crossSection);

    // add new maximum of cross-section times velocity
    set_tot_sigma_vel_max(last_index, std::max(static_cast<MyFloat>(total_crossSection * new_vel),
                                               get_tot_sigma_vel_max(last_index - 1)));
    set_trans_sigma_vel_max(last_index, std::max(static_cast<MyFloat>(trans_crossSection * new_vel),
                                                 get_trans_sigma_vel_max(last_index - 1)));

    // extend values of table
    for (size_t i = 0; i < m_lenxh; ++i)
    {
#ifdef DCS_USE_GRID
        set_table(i, last_index, m_integr.solve_cdf(m_axis[i], new_vel, total_crossSection));
#endif // DCS_USE_GRID

#ifdef DCS_USE_ZSLICE
        const double partialCrossSection = m_integr.compute_total_cross_section_integral(m_fwd, m_axis[i], new_vel);
        set_table(i, last_index, partialCrossSection / total_crossSection);
#endif // DCS_USE_ZSLICE
    }
}

MyAtLeastDouble cross_section_table::extend_table(const MyAtLeastDouble target_vrel)
{
    double new_velocity = max_velocity();
    while (target_vrel > new_velocity)
    {
        new_velocity = get_next_boundary_velocity();
        extend_table_single_value(new_velocity);
#ifdef SCATTERING_DEBUG
        cout(extra) << "    added v = " << new_velocity << std::endl;
#endif // SCATTERING_DEBUG
    }

    return static_cast<MyAtLeastDouble>(new_velocity);
}

double cross_section_table::get_next_boundary_velocity()
{
    return get_velocity_step(++m_boundaryIndex, m_lenv);
}

void cross_section_table::gsl_error_handler(const char *reason, const char *file, const int line, const int gsl_errno)
{
#ifdef DCS_DEBUG
    if (gsl_errno)
    {
        cout(error) << "DCS | GSL ERROR in " << file << "." << line << ": " << reason << endl;
    }
#endif // DCS_DEBUG
}

double cross_section_table::get_velocity_step(const int step, const int maxStep) const
{
    // first step returns zero velocity
    if (step == 0)
    {
        return 0;
    }
    else
    {
        return exp(m_logVmin + ((double)(step - 1) / (maxStep - 2)) * m_logVmax -
                   ((double)(step - 1) / (maxStep - 2)) * m_logVmin);
    }
}

void cross_section_table::evaluate_sigmaDm_vel_max()
{
    {
        const MyAtLeastDouble &vel = get_velocity(0);
        set_trans_sigma_vel_max(0, get_trans_sigma(0) * vel);
        set_tot_sigma_vel_max(0, get_tot_sigma(0) * vel);
    }

    for (size_t i = 1; i <= m_boundaryIndex; ++i)
    {
        const MyAtLeastDouble &vel = get_velocity(i);
        set_trans_sigma_vel_max(
            i, std::max(get_trans_sigma_vel_max(i - 1), static_cast<MyFloat>(get_trans_sigma(i) * vel)));
        set_tot_sigma_vel_max(i, std::max(get_tot_sigma_vel_max(i - 1), static_cast<MyFloat>(get_tot_sigma(i) * vel)));
    }
}

template <bool do_ignorewarning> MyAtLeastDouble cross_section_table::interp_sigmaT(const MyAtLeastDouble vel) const
{
    if (vel > m_velocity.back())
    {
        cout(warning) << m_name << " DCS | WARNING: Velocity is larger than max velocity in table. v: " << vel << endl;
        if constexpr (!do_ignorewarning)
        {
            ENDRUN("This can be resolved by switching off mSIDM_NOEXTEND_DCS_TABLE or choosing a larger value for "
                   "MSIDM_v_max.");
        }
    }
    // lowest index in Velocity array
    const unsigned int vindex = binary_search(m_velocity, vel, 0, m_velocity.size());

    // last index => return last sigma_tot value
    if (vindex >= (m_velocity.size() - 1))
    {
        return m_sigmaTrans.back();
    }

    // velocity low & high
    const double &vlo = m_velocity[vindex];
    const double &vhi = m_velocity[vindex + 1];
    // lambda in [0,1]
    const double lambda = (vel - vlo) / (vhi - vlo);
    // sigma low & high
    const double &slo = m_sigmaTrans[vindex];
    const double &shi = m_sigmaTrans[vindex + 1];

    return slo + lambda * (shi - slo);
}
template MyAtLeastDouble cross_section_table::interp_sigmaT<true>(const MyAtLeastDouble vel) const;
template MyAtLeastDouble cross_section_table::interp_sigmaT<false>(const MyAtLeastDouble vel) const;

template <bool do_ignorewarning> MyAtLeastDouble cross_section_table::interp_sigma(const MyAtLeastDouble vel) const
{
    if (vel > m_velocity.back())
    {
        cout(warning) << m_name << " DCS | WARNING: Velocity is larger than max velocity in table. v: " << vel << endl;
        if constexpr (!do_ignorewarning)
        {
            ENDRUN("This can be resolved by switching off mSIDM_NOEXTEND_DCS_TABLE or choosing a larger value for "
                   "MSIDM_v_max.");
        }
    }

    // lowest index in Velocity array
    const unsigned int vindex = binary_search(m_velocity, vel, 0, m_velocity.size());

    // last index => return last sigma_tot value
    if (vindex >= (m_velocity.size() - 1))
    {
        return m_sigmaTot.back();
    }

    // velocity low & high
    const double &vlo = m_velocity[vindex];
    const double &vhi = m_velocity[vindex + 1];
    // lambda in [0,1]
    const double lambda = (vel - vlo) / (vhi - vlo);
    // sigma low & high
    const double &slo = m_sigmaTot[vindex];
    const double &shi = m_sigmaTot[vindex + 1];

    return slo + lambda * (shi - slo);
}
template MyAtLeastDouble cross_section_table::interp_sigma<true>(const MyAtLeastDouble vel) const;
template MyAtLeastDouble cross_section_table::interp_sigma<false>(const MyAtLeastDouble vel) const;

template <bool do_ignorewarning>
MyAtLeastDouble cross_section_table::interp_theta(const double x, const MyAtLeastDouble vel) const
{
    if (vel > m_velocity.back())
    {
        cout(warning) << m_name << " DCS | WARNING: Velocity is larger than max velocity in table" << endl;
        if constexpr (!do_ignorewarning)
        {
            ENDRUN("This can be resolved by switching off mSIDM_NOEXTEND_DCS_TABLE or choosing a larger value for "
                   "MSIDM_v_max.");
        }
    }
#ifdef DCS_USE_GRID
    // lowest index in Velocity array
    const size_t vindex = binary_search(m_velocity, vel, 0, m_velocity.size());
    // lowest index in X array. Only possible because of homogeneous grid in x
    const size_t xindex = std::floor(x * (m_lenxh - 1));

    // point is at the edge of x, ie. x=1 => return max value for theta
    if (xindex >= (m_lenxh - 1))
    {
        return M_PI - m_bwd;
    }

    // x low & high values at lower velocity value (vlo)
    const double &xlo = m_axis[xindex];
    const double &xhi = m_axis[xindex + 1];
    // theta low & high values at lower velocity value (vlo)
    const double &tlo_vlo = m_table[vindex][xindex];
    const double &thi_vlo = m_table[vindex][xindex + 1];

    const double lambda = (x - xlo) / (xhi - xlo);

    // last index in velocity => interpolate in last column in theta
    if (vindex >= (m_velocity.size() - 1))
    {
        return tlo_vlo + lambda * (thi_vlo - tlo_vlo);
    }

    //--------------------------------------------------------------
    // point is now somewhere in the middle, all 4 edge points exist
    //--------------------------------------------------------------

    // velocity low & high values
    const double &vlo = m_velocity[vindex];
    const double &vhi = m_velocity[vindex + 1];
    // theta low & high values at higher velocity value (vhi)
    const double &tlo_vhi = m_table[vindex + 1][xindex];
    const double &thi_vhi = m_table[vindex + 1][xindex + 1];

    const double eta = (vel - vlo) / (vhi - vlo);

    const double theta_low = tlo_vlo + lambda * (thi_vlo - tlo_vlo);
    const double theta_high = tlo_vhi + lambda * (thi_vhi - tlo_vhi);

    return theta_low + eta * (theta_high - theta_low);
#endif // DCS_USE_GRID

#ifdef DCS_USE_ZSLICE
    /**  Naming convention for points
     *
     *  x1_l---------x1_r
     *    |        p  |
     *  x2_l---------x2_r
     *    |           |
     *  x3_l---------x3_r
     *    |           |
     *  x4_l---------x4_r
     *
     *  |vlo         |vhi
     */

    // index of vlo => vhi = vlo + 1
    const size_t vindex = binary_search(m_velocity, vel, 0, m_velocity.size());
    // index of x2_l
    const size_t xindex_left = binary_search(m_table[vindex], x, 0, m_lenxh);

    // last index => x is >= 1 => use last theta value
    if (xindex_left == m_lenxh - 1)
    {
        return m_axis.back();
    }

    //--------------------------------------------------------------
    // x is now 0 <= x < 1 => x1_l exists
    //--------------------------------------------------------------

    const double &x1_l = m_table[vindex][xindex_left + 1];
    const double &x2_l = m_table[vindex][xindex_left];
    // theta_xn_l = theta_xn_r
    const double &theta_x1 = m_axis[xindex_left + 1];
    const double &theta_x2 = m_axis[xindex_left];

    // last velocity index => only interpolate between x1_l & x2_l
    if (vindex == m_lenv - 1)
    {
        const double lambda = (x - x2_l) / (x1_l - x2_l);
        return theta_x2 + lambda * (theta_x1 - theta_x2);
    }

    //--------------------------------------------------------------
    // sample point is now in surface => all immediate points exist (x1/2_l/r)
    //--------------------------------------------------------------

    const double &vlo = m_velocity[vindex];
    const double &vhi = m_velocity[vindex + 1];

    const double &x1_r = m_table[vindex + 1][xindex_left + 1];
    const double &x2_r = m_table[vindex + 1][xindex_left];

    // lambda is interpolation factor in velocity
    const double lambda = (vel - vlo) / (vhi - vlo);

    // values for x1/2 on the line connecting left&right @ sample velocity
    const double x1_c = x1_l + lambda * (x1_r - x1_l);
    const double x2_c = x2_l + lambda * (x2_r - x2_l);

    // sample point is within Quadrilateral
    if ((x2_c <= x) && (x <= x1_c))
    {
        const double eta = (x - x2_c) / (x1_c - x2_c);
        return theta_x2 + eta * (theta_x1 - theta_x2);
    }

    //--------------------------------------------------------------
    // Miss 1st Order -- sample point is not in left Quadrilateral
    // Check Quadrilateral from right
    //--------------------------------------------------------------

    // index of x4_r
    const size_t xindex_right = binary_search(m_table[vindex + 1], x, 0, m_lenxh);

    const double &x3_l = m_table[vindex][xindex_right + 1];
    const double &x3_r = m_table[vindex + 1][xindex_right + 1];

    const double &x4_l = m_table[vindex][xindex_right];
    const double &x4_r = m_table[vindex + 1][xindex_right];

    const double &theta_x3 = m_axis[xindex_right + 1];
    const double &theta_x4 = m_axis[xindex_right];

    const double x3_c = x3_l + lambda * (x3_r - x3_l);
    const double x4_c = x4_l + lambda * (x4_r - x4_l);

    // sample point is within Quadrilateral
    if ((x4_c <= x) && (x <= x3_c))
    {
        const double eta = (x - x4_c) / (x3_c - x4_c);
        return theta_x4 + eta * (theta_x3 - theta_x4);
    }

    //--------------------------------------------------------------
    // Miss 2nd Order -- sample point is not in right Quadrilateral
    // Check every Quadrilateral between upper and lower Quadrilateral
    //--------------------------------------------------------------

    size_t loop_lo, loop_hi;

    if (xindex_left > xindex_right)
    {
        loop_lo = xindex_right + 1;
        loop_hi = xindex_left;
    }
    else
    {
        loop_lo = xindex_left + 1;
        loop_hi = xindex_right;
    }

    /**
     *  x1---xh_c---x2
     *  x3---xl_c---x4
     */

    for (; loop_lo < loop_hi; ++loop_lo)
    {
        const double &theta_x1 = m_axis[loop_lo + 1];
        const double &theta_x2 = m_axis[loop_lo];

        const double &x1 = m_table[vindex][loop_lo + 1];
        const double &x2 = m_table[vindex + 1][loop_lo + 1];

        const double &x3 = m_table[vindex][loop_lo];
        const double &x4 = m_table[vindex + 1][loop_lo];

        const double xh_c = x1 + lambda * (x2 - x1);
        const double xl_c = x3 + lambda * (x4 - x3);

        if ((xl_c <= x) && (x <= xh_c))
        {
            const double eta = (x - xl_c) / (xh_c - xl_c);
            return theta_x2 + eta * (theta_x1 - theta_x2);
        }
    }

    //--------------------------------------------------------------
    // Miss 3rd Order -- Could not find Quadrilateral
    //--------------------------------------------------------------

    cout(error) << m_name << " DCS | ERROR: Could not interpolate Point (x,v) = (" << x << ", " << vel << ")" << endl;
#endif // DCS_USE_ZSLICE

    return 0.0;
}
template MyAtLeastDouble cross_section_table::interp_theta<true>(const double x, const MyAtLeastDouble vel) const;
template MyAtLeastDouble cross_section_table::interp_theta<false>(const double x, const MyAtLeastDouble vel) const;

const MyFloat &cross_section_table::sigmaDm_vel_f_ext(const MyAtLeastDouble vmax) const
{
    const size_t ilo = binary_search(m_velocity, vmax, 0, m_velocity.size());
    const size_t ihi = std::min(ilo + 1, m_velocity.size() - 1);
    return std::max(get_trans_sigma_vel_max(ilo), get_trans_sigma_vel_max(ihi));
}

const MyFloat &cross_section_table::sigmaDm_vel_r_ext(const MyAtLeastDouble vmax) const
{
    const size_t ilo = binary_search(m_velocity, vmax, 0, m_velocity.size());
    const size_t ihi = std::min(ilo + 1, m_velocity.size() - 1);
    return std::max(get_tot_sigma_vel_max(ilo), get_tot_sigma_vel_max(ihi));
}

void cross_section_table::export_config(const std::string &path)
{
    if (ThisTask != 0)
        return;

    coutr(info) << m_name << " DCS | INFO: Export config to: " << path << endl;

    std::ofstream file{path};
    if (!file.is_open())
    {
        coutr(error) << m_name << " DCS | ERROR: File '" << path << "' could not be opened" << endl;
        return;
    }

    DcsFile data{};

    data.version = 1;

#ifdef DCS_USE_ZSLICE
    data.method = "zslice";
#endif
#ifdef DCS_USE_GRID
    data.method = "grid";
#endif

    data.len_v = m_lenv;
    data.len_xh = m_lenxh;
    data.reserveFact = m_reserveFact;
    data.v_min = m_vmin;
    data.v_max = m_vmax;
    data.tfwd = m_fwd;
    data.tbwd = m_bwd;

    data.crossFunction = m_params["crosssection"];
    data.crossParameters = m_params["params"];

    data.velocity = m_velocity;
    data.sigmaTot = m_sigmaTot;
    data.sigmaTrans = m_sigmaTrans;
    data.sigmaTotMaxVel = m_sigmaTotMaxVel;
    data.sigmaTransMaxVel = m_sigmaTransMaxVel;
    data.axis = m_axis;
    data.table = m_table;

    json output = data;
    file << output.dump(4);
    file.close();

    coutr(info) << m_name << " DCS | INFO: Export of config done." << endl;
}

bool cross_section_table::read_config(const std::string &path)
{

    cout(info) << m_name << " DCS | INFO: Reading config file: " << path << endl;

    std::ifstream file{path};
    if (!file.is_open())
    {
        coutr(warning) << m_name << " DCS | WARNING: File '" << path << "' could not be opened" << endl;
        return false;
    }

    DcsFile data{};
    try
    {
        json input{};
        file >> input;
        data = input.get<DcsFile>();
    }
    catch (const json::parse_error &e)
    {
        coutr(warning) << m_name << " DCS | WARNING: File content is not a json" << endl;
        return false;
    }
    catch (...)
    {
        coutr(warning) << m_name
                       << " DCS | WARNING: File content could not be parsed. Missing keys / invalid values detected"
                       << endl;
        return false;
    }

    file.close();

    if (!is_config_compatible(data))
    {
        coutr(warning) << m_name << " DCS | WARNING: Config is not compatible with current run" << endl;
        return false;
    }

    m_velocity = std::move(data.velocity);
    m_sigmaTot = std::move(data.sigmaTot);
    m_sigmaTrans = std::move(data.sigmaTrans);
    m_sigmaTotMaxVel = std::move(data.sigmaTotMaxVel);
    m_sigmaTransMaxVel = std::move(data.sigmaTransMaxVel);
    m_axis = std::move(data.axis);
    m_table = std::move(data.table);

    return true;
}

bool cross_section_table::is_config_compatible(const DcsFile &config)
{
    // Check data protocol version
    if (config.version != 1)
    {
        return false;
    }

    // Check for used method
#ifdef DCS_USE_GRID
    if (config.method != "grid")
    {
        return false;
    }
#endif
#ifdef DCS_USE_ZSLICE
    if (config.method != "zslice")
    {
        return false;
    }
#endif

    // Parameters must be unchanged for config to be loadable
    if (config.len_v != m_lenv || config.len_xh != m_lenxh || config.reserveFact != m_reserveFact ||
        config.v_min != m_vmin || config.v_max != m_vmax || config.crossFunction != m_params["crosssection"] ||
        config.crossParameters != m_params["params"] || config.tfwd != m_fwd || config.tbwd != m_bwd)
    {
        return false;
    }

    return true;
}

void cross_section_table::export_splines(const std::string &path, size_t steps) const
{
    export_splines(path, steps, steps);
}

void cross_section_table::export_splines(const std::string &path, size_t steps_v, size_t steps_xh) const
{
    if (ThisTask != 0)
        return;

    coutr(info) << m_name << " DCS | INFO: Export splines to: " << path << endl;

    std::ofstream file{path};
    if (!file.is_open())
    {
        coutr(error) << m_name << " DCS | ERROR: File '" << path << "' could not be opened" << endl;
        return;
    }

    json output{};

    std::vector<double> velocity{};
    std::vector<double> sigma{};
    std::vector<double> sigmaT{};
    std::vector<double> axis{};
    std::vector<std::vector<double>> theta{};

    for (size_t j = 0; j < steps_xh; ++j)
    {
        axis.push_back(static_cast<double>(j) / (steps_xh - 1));
    }

    for (size_t i = 0; i < steps_v; ++i)
    {
        double vrel = get_velocity_step(i, steps_v);
        velocity.push_back(vrel);

        sigma.push_back(interp_sigma<true>(vrel));
        sigmaT.push_back(interp_sigmaT<true>(vrel));

        std::vector<double> thetaPart{};
        for (size_t j = 0; j < steps_xh; ++j)
        {
            thetaPart.push_back(interp_theta<true>(axis[j], vrel));
        }
        theta.push_back(std::move(thetaPart));
    }

    output["velocity"] = velocity;
    output["sigma"] = sigma;
    output["sigmaT"] = sigmaT;
    output["theta"] = theta;
    output["x"] = axis;

#ifdef DCS_USE_ZSLICE
    output["method"] = "zslice";
#endif
#ifdef DCS_USE_GRID
    output["method"] = "grid";
#endif

    output["crosssection"] = m_params["crosssection"];
    output["params"] = m_params["params"];

    file << output.dump(4);
    file.close();

    coutr(info) << m_name << " DCS | INFO: Export of splines done." << endl;
}

void cross_section_table::sample_theta(const std::string &path, size_t N, double vrel)
{
    coutr(info) << m_name << " DCS | INFO: Sampling " << N << " times from theta pdf @ v = " << vrel << endl;

    std::ofstream file{path};
    if (!file.is_open())
    {
        coutr(error) << m_name << " DCS | ERROR: File '" << path << "' could not be opened" << endl;
        return;
    }

    json output{};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<double> theta{};

    for (size_t n = 0; n < N; ++n)
    {
        double rand_x = dis(gen);
        theta.push_back(interp_theta<false>(rand_x, vrel));
    }

    output["theta"] = theta;
    output["velocity"] = vrel;
    output["crosssection"] = m_params["crosssection"];
    output["params"] = m_params["params"];

#ifdef DCS_USE_ZSLICE
    output["method"] = "zslice";
#endif
#ifdef DCS_USE_GRID
    output["method"] = "grid";
#endif

    file << output.dump(4);
    file.close();
    coutr(info) << m_name << " DCS | INFO: Sampling done. Output stored at: " << path << endl;
}

} // namespace scatter::pxp_scattering
