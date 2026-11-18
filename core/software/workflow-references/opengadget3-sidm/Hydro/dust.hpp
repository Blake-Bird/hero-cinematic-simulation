/**
* @file
*
* This file is part of the N-body/SPH code OpenGadget3.
*
* @brief Helper functions for the One-Fluid model implementation of dust dynamics.

* @author OpenGadget3-development team, in particular Giovanni Tedeschi.
*
* @copyright Copyright (c) 2026, Klaus Dolag, and all contributing authors
* (see change logs). All rights reserved.
* This program is free software: you can redistribute it and/or modify it
* under the terms of the GNU General Public License v3 or later.
* See <https://www.gnu.org/licenses/>.
*/

#ifdef GT_DUST

/**
 * @brief Compute the physical dust fraction from the chosen parametrization.
 *
 * This function converts the dust fraction from the chosen parametrization
 * to the physical dust fraction.
 *
 * @param s Dust fraction in the chosen parametrization.
 * @return Physical dust fraction.
 */
static MyFloat dustfrac(MyFloat s)
{
#if ((GT_DUST_PARAM + 0) == 0)
    return s;
#elif ((GT_DUST_PARAM + 0) == 1)
    return pow(s, 2) / (1. + pow(s, 2));
#endif
}

/**
 * @brief Compute the parametrized dust fraction from the physical one.
 *
 * This function converts the physical dust fraction
 * to the parametrized dust fraction.
 *
 * @param eps Physical dust fraction.
 * @return Dust fraction in the chosen parametrization.
 */
static MyFloat invdustfrac(MyFloat eps)
{
#if ((GT_DUST_PARAM + 0) == 0)
    return eps;
#elif ((GT_DUST_PARAM + 0) == 1)
    return sqrt(eps / (1. - eps));
#endif
}

/**
 * @brief Computes the physical (1 - dust fraction)
 *
 * This function computes the physical (1 - dust fraction)
 * from the chosen parametrization.
 *
 * @param s Dust fraction in the chosen parametrization.
 * @return (1 - physical dust fraction)
 */
static MyFloat onem_dustfrac(MyFloat s)
{
#if ((GT_DUST_PARAM + 0) == 0)
    return 1. - s;
#elif ((GT_DUST_PARAM + 0) == 1)
    return 1. / (1. + pow(s, 2));
#endif
}

/**
 * @brief Computes the conversion factor for the time derivative of the parametrized dust fraction.
 *
 * This function computes the conversion factor for the time derivative of the parametrized dust fraction.
 *
 * @param s Dust fraction in the chosen parametrization.
 * @return Conversion factor for the time derivative of the parametrized dust fraction.
 */
static MyFloat dustfrac_conv_factor(MyFloat s)
{
#if GADGET_HYDRO == HYDRO_SPH
    return 1.0;
#elif GADGET_HYDRO == HYDRO_MFM
#if ((GT_DUST_PARAM + 0) == 0)
    return 1.0;
#elif ((GT_DUST_PARAM + 0) == 1)
    if (s > 1e-20)
    {
        return pow(1. + pow(s, 2), 2) / 2. / s;
    }
    else
    {
        return 0.0;
    }
#endif
#endif // GADGET_HYDRO
}

void compute_TotalDustFrac_and_TotalDeltaVel(int i);
void compute_TotalDustFracPred_and_TotalDeltaVelPred(int i);
void integrate_dust_drag(int i, integertime tstart, integertime tend);

#endif // GT_DUST