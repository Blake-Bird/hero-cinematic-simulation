/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Driver routines for computation of long-range gravitational PM force.
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

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef PMGRID

/**
 * @brief Call initialisation of periodic or/and non-periodic FFT routines.
 *
 * Driver routine to call initializiation of periodic or/and non-periodic FFT
 * routines.
 */
void long_range_init(void)
{
    pm_init();
}

/**
 * @brief Initializes the region size for long-range force calculation.
 *
 * This function initializes the region size for long-range force calculation,
 * taking into account periodic boundary conditions if enabled.
 * It also sets up the non-periodic kernel if required.
 */
void long_range_init_regionsize(void)
{
#ifdef PERIODIC
    if (RestartFlag != 1)
    {
        pm_init_regionsize();
    }
#ifdef PLACEHIGHRESREGION
    pm_setup_nonperiodic_kernel();
#endif // PLACEHIGHRESREGION
#else // PERIODIC
    if (RestartFlag != 1)
    {
        pm_init_regionsize();
    }
    pm_setup_nonperiodic_kernel();
#endif // PERIODIC
}

/**
 * @brief Computes long-range forces.
 *
 * This function computes long-range forces, initializing gravitational potential
 * and acceleration contributions for each particle. Depending on the compilation
 * flags and system configuration, it handles periodic boundary boundary conditions,
 * evaluate gravitational potential, and apply additional cosmological factors.
 */
void long_range_force(void)
{
    TIMER_PUSH(1, "PM");
    TIMER_PUSH(1, "PM_PREP");
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int ii = 0; ii < NumPart; ii++)
    {
        P[ii].GravPM[0] = P[ii].GravPM[1] = P[ii].GravPM[2] = 0;
#ifdef EVALPOTENTIAL
        P[ii].PM_Potential = 0;
#endif // EVALPOTENTIAL
    }
    TIMER_POP(1, "PM_PREP");

#ifdef NOGRAVITY
    return;
#endif // NOGRAVITY

#ifdef PERIODIC
    TIMER_PUSH(1, "PM_2");
    (void)pmforce(2);
    TIMER_POP(1, "PM_2");
#ifdef PLACEHIGHRESREGION
    TIMER_PUSH(1, "PM_1");
    int pm_i = pmforce(1);
    if (pm_i == 1) /* this is returned if a particle lied outside allowed range */
    {
        TIMER_POPPUSH(1, "PM_1", "PM_PREP");
        pm_init_regionsize();
        pm_setup_nonperiodic_kernel();
        TIMER_POPPUSH(1, "PM_PREP", "PM_1");
        /* try again */
        pm_i = pmforce(1);
    }
    TIMER_POP(1, "PM_1");
    ENDRUN_IF(pm_i == 1, "Error in long_range_force: pmforce failed");

#endif // PLACEHIGHRESREGION
#else // PERIODIC
    TIMER_PUSH(1, "PM_0");
    int pm_i = pmforce(0);

    if (pm_i == 1) /* this is returned if a particle lied outside allowed range */
    {
        TIMER_POPPUSH(1, "PM_0", "PM_PREP");
        pm_init_regionsize();
        pm_setup_nonperiodic_kernel();
        TIMER_POPPUSH(1, "PM_PREP", "PM_0");
        /* try again */
        pm_i = pmforce(0);
    }
    TIMER_POP(1, "PM_0");

    ENDRUN_IF(pm_i == 1, "Error in long_range_force: pmforce failed");

#ifdef PLACEHIGHRESREGION
    TIMER_PUSH(1, "PM_1");
    pm_i = pmforce(1);
    TIMER_POP(1, "PM_1");
    if (pm_i == 1) /* this is returned if a particle lied outside allowed range */
    {
        TIMER_PUSH(1, "PM_PREP");
        pm_init_regionsize();
        pm_setup_nonperiodic_kernel();
        /* try again */
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int ii = 0; ii < NumPart; ii++)
        {
            P[ii].GravPM[0] = P[ii].GravPM[1] = P[ii].GravPM[2] = 0;
        }
        TIMER_POPPUSH(1, "PM_PREP", "PM_0_1");
        pm_i = pmforce(0) + pmforce(1);
        TIMER_POP(1, "PM_0_1");
    }

    ENDRUN_IF(pm_i != 1, "Error in long_range_force: pmforce failed");

#endif // PLACEHIGHRESREGION
#endif // PERIODIC

#ifndef PERIODIC
    if (All.ComovingIntegrationOn)
    {
        TIMER_PUSH(1, "PM_COMPUTE_LAMBDA1");
        double fac = 0.5 * All.Hubble * All.Hubble * All.Omega0;

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int ii = 0; ii < NumPart; ii++)
        {
            for (int j = 0; j < 3; j++)
            {
                P[ii].GravPM[j] += fac * P[ii].Pos[j];
            }
        }
        TIMER_POP(1, "PM_COMPUTE_LAMBDA1");
    }

    /* Finally, the following factor allows a computation of cosmological simulation
       with vacuum energy in physical coordinates */

    if (All.ComovingIntegrationOn == 0)
    {
        TIMER_PUSH(1, "PM_COMPUTE_LAMBDA2");
        double fac = All.OmegaLambda * All.Hubble * All.Hubble;

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int ii = 0; ii < NumPart; ii++)
        {
            for (int j = 0; j < 3; j++)
            {
                P[ii].GravPM[j] += fac * P[ii].Pos[j];
            }
        }
        TIMER_POP(1, "PM_COMPUTE_LAMBDA2");
    }
#endif // PERIODIC

    TIMER_POP(1, "PM");
}
#endif // PMGRID
