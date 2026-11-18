/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file functions for the drifting and kicking of particles.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

static double logTimeBegin;
static double logTimeMax;

/**
 * Compute the integrand for drift.
 *
 * This function computes the integrand for drift using the given scale factor.
 *
 * @param a The scale factor.
 * @param param A pointer to additional parameters (unused in this implementation).
 * @return The value of the integrand.
 */
static double drift_integ(const double a, void *param)
{
    return 1.0 / (hubble_function(a) * a * a * a);
};

/**
 * Compute the integrand for gravitational kick.
 *
 * This function computes the integrand for gravitational kick using the given scale factor.
 *
 * @param a The scale factor.
 * @param param A pointer to additional parameters (unused in this implementation).
 * @return The value of the integrand.
 */
static double gravkick_integ(const double a, void *param)
{
    return 1.0 / (hubble_function(a) * a * a);
};

/**
 * @brief Compute the integrand for hydrokick.
 *
 * This function computes the integrand for hydrokick using the given scale factor.
 *
 * @param a The scale factor.
 * @param param A pointer to additional parameters (unused in this implementation).
 * @return The value of the integrand.
 */
static double hydrokick_integ(const double a, void *param)
{
    return 1.0 / (hubble_function(a) * pow(a, 3 * GAMMA_MINUS1) * a);
};

#ifdef AXION_DM
/**
 * @brief Computes the integrand for axion hydrokick.
 *
 * This function computes the integrand for axion hydrokick using the given scale factor.
 * Note: Integration differs for different terms of the axion equation.
 *       For QP, the integration term is a^-4.
 *       For P, the integration term is a^(-3 * gamma + 2) with gamma != 0.
 *
 * @param a The scale factor.
 * @param param A pointer to additional parameters (unused in this implementation).
 * @return The value of the integrand.
 */
static double ax_hydrokick_integ(const double a, void *param)
{
    return 1.0 / (hubble_function(a) * a * a);
}
#endif // AXION_DM

/**
 * @brief Compute the integrand for magnetic kick.
 *
 * This function computes the integrand for magnetic kick using the given scale factor.
 *
 * @param a The scale factor.
 * @param param A pointer to additional parameters (unused in this implementation).
 * @return The value of the integrand.
 */
static double magkick_integ(const double a, void *param)
{
    return 1.0 / (hubble_function(a) * a * a * a);
}

#if defined(mSIDM) || defined(mIDM)
/**
 * @brief Compute the integrand for scattering.
 *
 * This function computes the integrand for scattering using the given scale factor.
 *
 * @param a The scale factor.
 * @param param A pointer to additional parameters (unused in this implementation).
 * @return The value of the integrand.
 */
static double scattering_integ(const double a, void *param) /* !! */
{
    const double a2 = a * a;
    return 1.0 / (hubble_function(a) * a2 * a2 * a);
};
#endif // mSIDM || mIDM

/**
 * @brief Initialize the lookup tables for the cosmological drift and kick prefactors.
 *
 * This function initializes the lookup tables for the cosmological drift and kick prefactors
 * It calculates the values of these prefactors at various redshifts and stores them in global
 * arrays for later use in the simulation.
 *
 * @note This function requires the GSL (GNU Scientific Library) for numerical integration.
 *
 * The computed values of the prefactors are stored in global arrays `DriftTable`, `GravKickTable`,
 * HydroKickTable`, and optionally `MagKickTable` and `ScatteringKickTable`.
 *
 * @warning Ensure that the global parameters `All.TimeBegin` and `All.TimeMax` are properly
 * initialized before calling this function.
 */
void init_drift_table(void)
{
#ifdef DARKENERGY_DEBUG
    /*---------- DEBUG ! ------------*/
    FILE *FdDKfac;
    char mode[2], buf[200];

    strcpy(mode, "w");
    sprintf(buf, "%s%s", All.OutputDir, "driftkickfac.txt");
    FdDKfac = fopen(buf, mode);
    fprintf(FdDKfac, "i a drift GravKick HydroKick\n");
    /*---------- DEBUG ! ------------*/
#endif // DARKENERGY_DEBUG

    logTimeBegin = log(All.TimeBegin);
    logTimeMax = log(All.TimeMax);

    for (int i = 0; i < DRIFT_TABLE_LENGTH; i++)
    {
        double result, abserr;
        double a0 = exp(logTimeBegin);
        double a1 =
            exp(logTimeBegin + ((logTimeMax - logTimeBegin) / static_cast<double>(DRIFT_TABLE_LENGTH)) * (i + 1));

        result = boost::math::quadrature::gauss_kronrod<double, 41>::integrate(
            [](double a) { return drift_integ(a, nullptr); }, a0, a1, 15, 1.0e-8, &abserr);

        DriftTable[i] = result;

        result = boost::math::quadrature::gauss_kronrod<double, 41>::integrate(
            [](double a) { return gravkick_integ(a, nullptr); }, a0, a1, 15, 1.0e-8, &abserr);
        GravKickTable[i] = result;

        result = boost::math::quadrature::gauss_kronrod<double, 41>::integrate(
            [](double a) { return hydrokick_integ(a, nullptr); }, a0, a1, 15, 1.0e-8, &abserr);
        HydroKickTable[i] = result;

        result = boost::math::quadrature::gauss_kronrod<double, 41>::integrate(
            [](double a) { return magkick_integ(a, nullptr); }, a0, a1, 15, 1.0e-8, &abserr);
        MagKickTable[i] = result;

#if defined(mSIDM) || defined(mIDM)
        result = boost::math::quadrature::gauss_kronrod<double, 41>::integrate(
            [](double a) { return scattering_integ(a, nullptr); }, a0, a1, 15, 1.0e-8, &abserr);
        ScatteringKickTable[i] = result;
#endif // mSIDM || mIDM

#ifdef DARKENERGY_DEBUG
        /*---------- DEBUG ! ------------*/
        fprintf(FdDKfac, "%d %e %e %e %e \n", i,
                exp(logTimeBegin + ((logTimeMax - logTimeBegin) / DRIFT_TABLE_LENGTH) * (i + 1)), DriftTable[i],
                GravKickTable[i], HydroKickTable[i]);
        /*---------- DEBUG ! ------------*/
#endif // DARKENERGY_DEBUG
    }

#ifdef DARKENERGY_DEBUG
    /*---------- DEBUG ! ------------*/
    fclose(FdDKfac);
    /*---------- DEBUG ! ------------*/
#endif // DARKENERGY_DEBUG
}

/**
 * @brief Interpolate the lookup table for the cosmological drift or kick prefactors.
 *
 *  This function linearly interpolates between elements of the lookup tables
 *  for the cosmological drift or kick prefactors.
 *
 * @param ti The time index of the lookup table.
 * @param Table The lookup table.
 * @return The interpolated value.
 */
double interpolate_lookup_table(const integertime ti, const double *const Table)
{
    const double log_a = logTimeBegin + ti * All.Timebase_interval;
    const double u = (log_a - logTimeBegin) / (logTimeMax - logTimeBegin) * static_cast<double>(DRIFT_TABLE_LENGTH);
    size_t i = u;
    if (i >= DRIFT_TABLE_LENGTH)
    {
        i = DRIFT_TABLE_LENGTH - 1;
    }
    const double table_left = i == 0 ? 0 : Table[i - 1];
    const double table_right = Table[i];
    const double result = table_left + (table_right - table_left) * (u - i);
    return result;
}

/**
 * @brief Get the cosmological prefactor for a time step.
 *
 * This function integrates the cosmological prefactor (determined by the given
 * lookup table) for a time step between time0 and time1. It will only be called
 * for cosmological integration.
 *
 * @param time0 The initial time.
 * @param time1 The final time.
 * @param Table The lookup table.
 * @return The cosmological prefactor for the time step.
 */
static double get_cosmo_factor(const integertime time0, const integertime time1, const double *const Table)
{
    assert(All.ComovingIntegrationOn);
    assert(time0 >= 0 && time1 >= 0);
    if (time1 < time0)
    {
        VERBOSE_ALL(5, "DEBUG: Task %d: cosmo factor: %d %d ?? ", ThisTask, time0, time1);
    }
    assert(time1 >= time0);
    // if time0 and time1 are both 0, then the cosmological factor is 0
    if (time0 == 0 && time1 == 0)
    {
        return 0;
    }

    // get the cosmological prefactor at time0 and time1 from table
    const double cf0 = interpolate_lookup_table(time0, Table);
    const double cf1 = interpolate_lookup_table(time1, Table);
    // check that the cosmological factor is increasing
    if (cf1 < cf0)
    {
        VERBOSE_ALL(5, "DEBUG: Task %d: cosmo factor: %d %d -> %g < %g ?? ", ThisTask, time0, time1, cf0, cf1);
    }
    assert(cf1 >= cf0);
    return cf1 - cf0;
}

/**
 * @brief Get drift factor between two given times.
 *
 * This function integrates the cosmological prefactor for a drift
 * step between time0 and time1. The value returned is
 *  \f[ \int_{a_0}^{a_1} \frac{{\rm d}a}{H(a)} \f]. It uses
 * a lookup-table to speed up the calculation.
 *
 * @param time0 The starting time.
 * @param time1 The ending time.
 * @return The drift factor between the given times.
 */
double get_drift_factor(const integertime time0, const integertime time1)
{
    assert(time0 >= 0 && time1 >= 0);
    // Check if time1 is less than time0 and print debug message if true
    if (time1 < time0)
    {
        VERBOSE_ALL(5, "DEBUG: Task %d: cosmo factor: %d %d ?? ", ThisTask, time0, time1);
    }
    assert(time1 >= time0);

    // If the input times match the last computed times, return the cached result
    if (time0 == last_time0_drift && time1 == last_time1_drift)
    {
        return last_value_drift;
    }

    // Compute and cache the drift factor using get_cosmo_factor function
    last_time0_drift = time0;
    last_time1_drift = time1;
    return last_value_drift = get_cosmo_factor(time0, time1, DriftTable);
}

/**
 * @brief Computes the gravitational kick factor between two given times.
 *
 * This function calculates the gravitational kick factor between two given times.
 * If the second time is less than the first time, a debug message is printed.
 *
 * @param time0 The starting time.
 * @param time1 The ending time.
 * @return The gravitational kick factor between the given times.
 */
double get_gravkick_factor(const integertime time0, const integertime time1)
{
    // Check if time1 is less than time0 and print debug message if true
    if (time1 < time0)
    {
        VERBOSE_ALL(5, "DEBUG: Task %d: cosmo factor: %d %d ?? ", ThisTask, time0, time1);
    }
    assert(time1 >= time0);
    assert(time0 >= 0 && time1 >= 0);

    // If the input times match the last computed times, return the cached result
    if (time0 == last_time0_gravkick && time1 == last_time1_gravkick)
    {
        return last_value_gravkick;
    }

    // Compute and cache the gravitational kick factor using get_cosmo_factor function
    last_time0_gravkick = time0;
    last_time1_gravkick = time1;
    return last_value_gravkick = get_cosmo_factor(time0, time1, GravKickTable);
}

/**
 * @brief Get the hydrodynamical kick factor between two integration times.
 *
 * This function calculates the hydrodynamical kick factor between two integration times.
 * The hydrodynamical kick factor represents the contribution of hydrodynamical forces to the
 * acceleration of particles in a cosmological simulation. It is computed by interpolating
 * the precomputed values stored in the `HydroKickTable`.
 *
 * @param time0 The starting integration time.
 * @param time1 The ending integration time.
 *
 * @return The hydrodynamical kick factor between the specified integration times.
 *
 * @note This function assumes that the integration times `time0` and `time1` are non-negative
 * and that `time1` is greater than or equal to `time0`.
 */
double get_hydrokick_factor(const integertime time0, const integertime time1)
{
    // Check if time1 is less than time0 and print debug message if true
    if (time1 < time0)
    {
        VERBOSE_ALL(5, "DEBUG: Task %d: cosmo factor: %d %d ?? ", ThisTask, time0, time1);
    }
    assert(time1 >= time0);

    assert(time0 >= 0 && time1 >= 0);
    // Return the previously computed value if the integration times match the last queried times
    if (time0 == last_time0_hydrokick && time1 == last_time1_hydrokick)
    {
        return last_value_hydrokick;
    }

    // Compute and cache the hydro kick factor using get_cosmo_factor function
    last_time0_hydrokick = time0;
    last_time1_hydrokick = time1;
    return last_value_hydrokick = get_cosmo_factor(time0, time1, HydroKickTable);
}

/**
 * @brief Get the magnetic kick factor between two integration times.
 *
 * This function calculates the magnetic kick factor between two integration times.
 * The magnetic kick factor represents the contribution of magnetic forces to the
 * acceleration of particles. It is computed by interpolating
 * the precomputed values stored in the `MagKickTable`.
 *
 * @param time0 The starting integration time.
 * @param time1 The ending integration time.
 *
 * @return The magnetic kick factor between the specified integration times.
 *
 * @note This function assumes that the integration times `time0` and `time1` are non-negative.
 */
double get_magkick_factor(const integertime time0, const integertime time1)
{
    assert(time0 >= 0 && time1 >= 0);
    // Return the previously computed value if the integration times match the last queried times
    if (time0 == last_time0_magkick && time1 == last_time1_magkick)
    {
        return last_value_magkick;
    }
    // Update the last queried integration times
    last_time0_magkick = time0;
    last_time1_magkick = time1;
    // Compute and return the magnetic kick factor using linear interpolation
    return last_value_magkick = get_cosmo_factor(time0, time1, MagKickTable);
}

#if defined(mSIDM) || defined(mIDM)
/**
 * @brief Get the scattering kick factor between two integration times.
 *
 * This function calculates the scattering kick factor between two integration times.
 * The scattering kick factor represents the contribution of scattering forces to the
 * acceleration of particles. It is computed by interpolating
 * the precomputed values stored in the `ScatteringKickTable`.
 *
 * @param time0 The starting integration time.
 * @param time1 The ending integration time.
 *
 * @return The scattering kick factor between the specified integration times.
 */
double get_scattering_kick_factor(const integertime time0, const integertime time1)
{
    assert(time0 >= 0 && time1 >= 0);
    // Return the previously computed value if the integration times match the last queried times
    if (time0 == last_time0_scattering_kick && time1 == last_time1_scattering_kick)
    {
        return last_value_scattering_kick;
    }
    // Update cahced integration times
    last_time0_scattering_kick = time0;
    last_time1_scattering_kick = time1;
    // Compute and return the scattering kick factor using linear interpolation
    return last_value_scattering_kick = get_cosmo_factor(time0, time1, ScatteringKickTable);
}
#endif // mSIDM || mIDM
