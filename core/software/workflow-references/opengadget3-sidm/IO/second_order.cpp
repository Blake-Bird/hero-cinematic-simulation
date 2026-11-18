/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Produce actual ICs from special 2nd order lpt ICs.
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
#include "io.hpp"

/**
 * @brief Calculate the function F1(omega) for a given scale factor.
 *
 * This function calculates the function F1(omega) for a given scale factor 'a' using cosmological parameters.
 *
 * @param a The scale factor.
 * @return The value of F1(omega) for the given scale factor.
 */
double F1_Omega(double a)
{
    double omega_a = All.Omega0 / (All.Omega0 + a * OMEGAK + a * a * a * All.OmegaLambda + OMEGAR / a);
    return pow(omega_a, 5.0 / 9);
}

/**
 * @brief Calculate the function F2(omega) for a given scale factor.
 *
 * This function calculates the function F2(omega) for a given scale factor 'a' using cosmological parameters.
 *
 * @param a The scale factor.
 * @return The value of F2(omega) for the given scale factor.
 */
double F2_Omega(double a)
{
    double omega_a;
    omega_a = All.Omega0 /
              (All.Omega0 + a * (1 - All.Omega0 - All.OmegaLambda - OMEGAR) + a * a * a * All.OmegaLambda + OMEGAR / a);
    return 2.0 * pow(omega_a, 6.0 / 11);
}

/**
 * @brief Generate initial conditions using second-order Lagrangian Perturbation Theory (2LPT).
 *
 * This function generates initial conditions for the simulation using second-order Lagrangian Perturbation Theory
 * (2LPT). It calculates the displacements and velocity corrections for each particle based on the Zeldovich
 * approximation and 2LPT.
 */
void second_order_ics(void)
{
    VERBOSE(2, "\nNow producing ICs based on 2nd-order LPT \n");

    double a = All.TimeBegin;
    double hubble = hubble_function(a);
    double fac1 = 1.0 / (a * a * hubble * F1_Omega(a)); /* this factor converts the Zeldovich peculiar velocity
                               (expressed as  a^2*dX/dt  to comoving displacement */
    double fac2 = header.lpt_scalingfactor;
    double fac3 = fac2 * a * a * hubble * F2_Omega(a);

    VERBOSE(2, "fac1=%g  fac2=%g  fac3=%g ", fac1, fac2, fac3);

    for (int i = 0; i < NumPart; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i].Pos[j] += fac1 * P[i].Vel[j]; /* Zeldovich displacement */
        }

#ifdef PMGRID
        for (int j = 0; j < 3; j++)
        {
            P[i].Pos[j] += fac2 * (P[i].GravPM[j] + P[i].GravAccel[j]); /* second order lpt displacement */
        }

        for (int j = 0; j < 3; j++)
        {
            P[i].Vel[j] += fac3 * (P[i].GravPM[j] + P[i].GravAccel[j]); /* second order lpt velocity correction */
        }
#else // PMGRID
        for (int j = 0; j < 3; j++)
        {
            P[i].Pos[j] += fac2 * (P[i].GravAccel[j]); /* second order lpt displacement */
        }

        for (int j = 0; j < 3; j++)
        {
            P[i].Vel[j] += fac3 * (P[i].GravAccel[j]); /* second order lpt velocity correction */
        }
#endif // PMGRID
    }

    /* now set the masses correctly */
    for (int i = 0; i < NumPart; i++)
    {
        P[i].Mass = P[i].OldAcc;

        if (All.MassTable[P[i].Type] != 0)
        {
            P[i].Mass = All.MassTable[P[i].Type];
        }
    }

    if (All.ComovingIntegrationOn)
    {
        if (All.PeriodicBoundariesOn == 1)
        {
            check_omega(); /* check again whether we have plausible mass density */
        }
    }

    /* recompute long range force */

    All.DoDynamicUpdate = 1;

    domain_Decomposition(1, 0); /* redo domain decomposition because particles may have shifted */

#ifdef PMGRID
    long_range_force();
#endif // PMGRID

    gravity_tree();

    for (int i = 0; i < NumPart; i++)
    {
        double ax, ay, az;
#ifdef PMGRID
        ax = (P[i].GravAccel[0] + P[i].GravPM[0]) / All.GravityConstantInternal;
        ay = (P[i].GravAccel[1] + P[i].GravPM[1]) / All.GravityConstantInternal;
        az = (P[i].GravAccel[2] + P[i].GravPM[2]) / All.GravityConstantInternal;
#else // PMGRID
        ax = P[i].GravAccel[0] / All.GravityConstantInternal;
        ay = P[i].GravAccel[1] / All.GravityConstantInternal;
        az = P[i].GravAccel[2] / All.GravityConstantInternal;
#endif // PMGRID
        P[i].OldAcc = sqrt(ax * ax + ay * ay + az * az);
    }

    if (All.ErrTolTheta == 0)
    {
        /* Using the relative opening criterion */
        gravity_tree();
    }
}
