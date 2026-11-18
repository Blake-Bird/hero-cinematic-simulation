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

#pragma once
#define JSON_ASSERT(x) // suppress redefinition of assert
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "../CodeBase/precision.hpp"
#include "CrossSectionIntegrator.hpp"
#include "DcsFile.hpp"

namespace scatter::pxp_scattering
{

using json = nlohmann::json;

/**
 * @brief Class that holds tables for angular-averaged cross-sections and the CDF.
 *
 * This class holds tables computed for a specific differential cross-section. It is used
 * to interpolate the angle-averaged cross-section and the cumulative distribution function.
 * The latter is employed when obtaining the scattering angle.
 */
class cross_section_table
{
    std::string m_name;

    json m_params;
    cross_section_integrator m_integr;

    size_t m_lenxh, m_lenv;
    size_t m_boundaryIndex;
    double m_reserveFact;
    double m_vmin, m_vmax;
    double m_logVmin, m_logVmax;
    double m_fwd, m_bwd;
    std::string m_dcsFile;

    std::vector<double> m_velocity;
    std::vector<double> m_sigmaTot;
    std::vector<double> m_sigmaTrans;
    std::vector<MyFloat> m_sigmaTotMaxVel;
    std::vector<MyFloat> m_sigmaTransMaxVel;
    std::vector<double> m_axis;
    std::vector<std::vector<double>> m_table;

    /** Parses parameters of the cross-section as given in the parameter file to json. */
    bool parse_model_parameters(std::string strParam, json &out);
    /** Method that sets up the array for the look-up tables. */
    void setup_table();
    /** Computes the CDF and the angle averaged cross-sections and writes it to the tables. */
    void compute_surface();
    /** Computes the velocities of the look-up tables. */
    double get_velocity_step(const int step, const int maxStep) const;
    /** Extends the tables by one entry. */
    void extend_table_single_value(double vrel);
    /** Next velocity that would be tabulated beyond the current boundary of the table. */
    double get_next_boundary_velocity();
    /** Exports cross-section configuration and tables to a file. */
    void export_config(const std::string &path);
    /** Read cross-section configuration and tables from a file. */
    bool read_config(const std::string &path);
    /** Checks if the passed configuration parameters are compatible with the current ones. */
    bool is_config_compatible(const DcsFile &config);

    /**
     * Returns the highest index i such that vec[i] < elem
     */
    template <typename T> static size_t binary_search(const std::vector<T> &vec, double elem, size_t low, size_t high)
    {
        size_t lo = low;
        size_t hi = high;
        while (hi > lo + 1)
        {
            size_t i = (hi + lo) / 2;
            if (vec[i] > elem)
            {
                hi = i;
            }
            else
            {
                lo = i;
            }
        }
        return lo;
    }

    /** gsl error handler */
    static void gsl_error_handler(const char *reason, const char *file, const int line, const int gsl_errno);

    // ----- Helper functions for setting values ----- //
    /** sets the velocity */
    inline void set_velocity(size_t index, double vel)
    {
        m_velocity[index] = vel;
    }
    /** sets the total cross-section */
    inline void set_tot_sigma(size_t index, double sigma)
    {
        m_sigmaTot[index] = sigma;
    }
    /** sets the transfer cross-section */
    inline void set_trans_sigma(size_t index, double sigmaT)
    {
        m_sigmaTrans[index] = sigmaT;
    }
    /** sets the maximum of the total cross-section times velocity */
    inline void set_tot_sigma_vel_max(size_t index, MyFloat sigma_vel)
    {
        m_sigmaTotMaxVel[index] = sigma_vel;
    }
    /** sets the maximum of the transfer cross-section times velocity */
    inline void set_trans_sigma_vel_max(size_t index, MyFloat sigmaT_vel)
    {
        m_sigmaTransMaxVel[index] = sigmaT_vel;
    }
    /** sets the axis */
    inline void set_axis(size_t index, double val)
    {
        m_axis[index] = val;
    }
    /** sets the values in the table used to obtain the scattering angle */
    inline void set_table(size_t xhi, size_t vi, double val)
    {
        m_table[vi][xhi] = val;
    }

    // ----- Helper functions for getting values ----- //
    /** gets the velocity */
    inline const double &get_velocity(size_t index) const
    {
        return m_velocity[index];
    }
    /** gets the transfer cross-section */
    inline const double &get_trans_sigma(size_t index) const
    {
        return m_sigmaTrans[index];
    }
    /** gets the total cross-section */
    inline const double &get_tot_sigma(size_t index) const
    {
        return m_sigmaTot[index];
    }
    /** gets the maximum of the transfer cross-section times velocity */
    inline const MyFloat &get_trans_sigma_vel_max(size_t index) const
    {
        return m_sigmaTransMaxVel[index];
    }
    /** gets the maximum of the total cross-section times velocity */
    inline const MyFloat &get_tot_sigma_vel_max(size_t index) const
    {
        return m_sigmaTotMaxVel[index];
    }
    /** gets the values in the table used to obtain the scattering angle */
    inline const double &get_table_value(size_t vi, size_t xhi) const
    {
        return m_table[vi][xhi];
    }
    /** gets the axis */
    inline const double &get_axis_value(size_t index) const
    {
        return m_axis[index];
    }

  public:
    cross_section_table() = default;

    /** Initializes the class after the parameter file has been read */
    void init(const std::string &name, const json &param);

    /**
     * Extends the table to at least the given velocity. Will add new log spaced velocities,
     * determined by v_min & v_max, until the max velocity is larger than the given velocity.
     * Computes cross sections and scattering angle.
     * Also exports the config and returns the maximum velocity
     */
    MyAtLeastDouble extend_table(const MyAtLeastDouble target_vrel);

    /**
     * Returns the maximum velocity within the table
     */
    inline MyAtLeastDouble max_velocity() const
    {
        return m_velocity.back();
    }

    /**
     * Export splines for total cross section, transfer cross section & theta with given
     * precision
     */
    void export_splines(const std::string &path, size_t steps) const;

    /**
     * Export splines for total cross section, transfer cross section & theta with given
     * precision
     */
    void export_splines(const std::string &path, size_t steps_v, size_t steps_xh) const;

    /**
     * Sample N times from the computed pdf for theta
     */
    void sample_theta(const std::string &path, size_t N, double vrel);

    /**
     * @brief Evaluates the maximum of the angle-averaged cross-sections times velocity.
     *
     * This is later used in the formulation of the time step criteria.
     */
    void evaluate_sigmaDm_vel_max();

    /** Interpolates the total cross-section for rSIDM. */
    template <bool do_ignorewarning> MyAtLeastDouble interp_sigma(const MyAtLeastDouble vel) const;

    /** Interpolates the transfer cross-section for fSIDM. */
    template <bool do_ignorewarning> MyAtLeastDouble interp_sigmaT(const MyAtLeastDouble vel) const;

    /** Interpolates the scattering angle. */
    template <bool do_ignorewarning> MyAtLeastDouble interp_theta(const double x, const MyAtLeastDouble vel) const;

    /** Returns the maximum of the transfer cross-section times velocity. */
    const MyFloat &sigmaDm_vel_f_ext(const MyAtLeastDouble vmax) const;

    /** Returns the maximum of the transfer cross-section times velocity. */
    const MyFloat &sigmaDm_vel_r_ext(const MyAtLeastDouble vmax) const;
};

} // namespace scatter::pxp_scattering
