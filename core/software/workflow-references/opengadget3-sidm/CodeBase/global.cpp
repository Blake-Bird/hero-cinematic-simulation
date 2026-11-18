/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Compute global properties of the particle distribution.
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

#include "allvars.hpp"
#include "proto.hpp"

/**
 * @brief Compute global properties of the particle distribution.
 *
 * This routine computes various global properties of the particle
 * distribution and stores the result in the struct `SysState'.
 * Currently, not all the information that's computed here is
 * actually used (e.g. momentum is not really used anywhere),
 * just the energies are written to a log-file every once in a while.
 */
void compute_global_quantities_of_system(void)
{
    int dt_step;
    struct state_of_system sys;
#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY)
    double a1;
#endif // EVALPOTENTIAL || COMPUTE_POTENTIAL_ENERGY
    double a2, a3;
    double entr = 0, egyspec, vel[3];

    VERBOSE(2, "Start global statistics");

#ifdef MAGNETIC_STATISTICS
    double dt_mag, mag[3];
    mag[0] = mag[1] = mag[2] = 0.0;
    sys.EnergyMag = 0.0;
    sys.DivBerr = 0;
#endif // MAGNETIC_STATISTICS
#if defined(MAGNETIC_STATISTICS)
    double volweight = 0, massweight = 0, Volweight = 0, Massweight = 0;
#endif // defined(MAGNETIC_STATISTICS)

    if (All.ComovingIntegrationOn)
    {
#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY)
        a1 = All.Time;
#endif // EVALPOTENTIAL || COMPUTE_POTENTIAL_ENERGY
        a2 = All.Time * All.Time;
        a3 = All.Time * All.Time * All.Time;
    }
    else
    {
#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY)
        a1 = 1;
#endif // EVALPOTENTIAL || COMPUTE_POTENTIAL_ENERGY
        a2 = a3 = 1;
    }

    for (int n = 0; n < 6; n++)
    {
        sys.MassComp[n] = sys.EnergyKinComp[n] = sys.EnergyPotComp[n] = sys.EnergyIntComp[n] = 0;

        for (int j = 0; j < 4; j++)
        {
            sys.CenterOfMassComp[n][j] = sys.MomentumComp[n][j] = sys.AngMomentumComp[n][j] = 0;
        }
    }

    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Mass > 0)
        {
            double dt_entr, dt_gravkick, dt_hydrokick;
            sys.MassComp[P[i].Type] += P[i].Mass;

#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY)
            sys.EnergyPotComp[P[i].Type] += 0.5 * P[i].Mass * P[i].Potential / a1;
#endif /* / /defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY) */

            dt_step = P[i].TimeBin ? (((integertime)1) << P[i].TimeBin) : 0;

            if (All.ComovingIntegrationOn)
            {
                dt_entr = (All.Ti_Current - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;
                if (P[i].Ti_begstep > All.Ti_Current)
                {
#ifdef DEBUG
                    VERBOSE_ALL(2, "Task %d: GSTAT -> Ti_Current(%d) < P[%d].Ti_begstep(%d) ! (dt_step/2=%d)\n",
                                ThisTask, All.Ti_Current, i, P[i].Ti_begstep, dt_step / 2);
#endif // DEBUG
                    dt_gravkick = -get_gravkick_factor(All.Ti_Current, P[i].Ti_begstep);
                    dt_hydrokick = get_hydrokick_factor(All.Ti_Current, P[i].Ti_begstep);
#ifdef MAGNETIC_STATISTICS
                    dt_mag = get_magkick_factor(All.Ti_Current, P[i].Ti_begstep);
#endif // MAGNETIC_STATISTICS
                }
                else
                {
                    dt_gravkick = get_gravkick_factor(P[i].Ti_begstep, All.Ti_Current);
                    dt_hydrokick = get_hydrokick_factor(P[i].Ti_begstep, All.Ti_Current);
#ifdef MAGNETIC_STATISTICS
                    dt_mag = get_magkick_factor(P[i].Ti_begstep, All.Ti_Current);
#endif // MAGNETIC_STATISTICS
                }
                dt_gravkick -= get_gravkick_factor(P[i].Ti_begstep, P[i].Ti_begstep + dt_step / 2);
                dt_hydrokick -= get_hydrokick_factor(P[i].Ti_begstep, P[i].Ti_begstep + dt_step / 2);
#ifdef MAGNETIC_STATISTICS
                dt_mag -= get_magkick_factor(P[i].Ti_begstep, P[i].Ti_begstep + dt_step / 2);
#endif // MAGNETIC_STATISTICS
            }
            else
            {
                dt_entr = dt_gravkick = dt_hydrokick =
#ifdef MAGNETIC_STATISTICS
                    dt_mag =
#endif // MAGNETIC_STATISTICS
                        (All.Ti_Current - (P[i].Ti_begstep + dt_step / 2)) * All.Timebase_interval;
            }

            for (int j = 0; j < 3; j++)
            {
                vel[j] = P[i].Vel[j] + P[i].GravAccel[j] * dt_gravkick;
                if (P[i].Type == type_gas)
                {
                    vel[j] += SphP[i].HydroAccel[j] * dt_hydrokick;
                }
            }
            if (P[i].Type == type_gas)
            {
                entr = SphP[i].Entropy + SphP[i].DtEntropy * dt_entr;
#ifdef MAGNETIC_STATISTICS
                for (int j = 0; j < 3; j++)
                {
                    mag[j] = SphP[i].b2.BPred[j] + SphP[i].DtB[j] * dt_mag;
                }
#endif // MAGNETIC_STATISTICS
            }
#ifdef PMGRID
            if (All.ComovingIntegrationOn)
            {
                dt_gravkick = get_gravkick_factor(All.PM_Ti_begstep, All.Ti_Current) -
                              get_gravkick_factor(All.PM_Ti_begstep, (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2);
            }
            else
            {
                dt_gravkick = (All.Ti_Current - (All.PM_Ti_begstep + All.PM_Ti_endstep) / 2) * All.Timebase_interval;
            }

            for (int j = 0; j < 3; j++)
            {
                vel[j] += P[i].GravPM[j] * dt_gravkick;
            }
#endif // PMGRID

            sys.EnergyKinComp[P[i].Type] +=
                0.5 * P[i].Mass * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]) / a2;

            if (P[i].Type == type_gas)
            {
                egyspec = entr / (GAMMA_MINUS1)*pow(SphP[i].Density / a3, GAMMA_MINUS1);

#ifdef MAGNETIC_STATISTICS
                sys.EnergyMag +=
                    P[i].Mass / SphP[i].Density * mu0_inv / 2.0 * (mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2]);
#endif // MAGNETIC_STATISTICS

#if defined(MAGNETIC_STATISTICS)
                if (P[i].ID != 0 &&
                    (SphP[i].b2.BPred[0] * SphP[i].b2.BPred[0] + SphP[i].b2.BPred[1] * SphP[i].b2.BPred[1] +
                     SphP[i].b2.BPred[2] * SphP[i].b2.BPred[2]) != 0)
                {
                    sys.DivBerr +=
                        abs(SphP[i].divB) * P[i].Hsml * P[i].Mass /
                        (SphP[i].b2.BPred[0] * SphP[i].b2.BPred[0] + SphP[i].b2.BPred[1] * SphP[i].b2.BPred[1] +
                         SphP[i].b2.BPred[2] * SphP[i].b2.BPred[2] + 1E-2) /
                        SphP[i].Density;
                }
#endif // defined(MAGNETIC_STATISTICS)

#if defined(MAGNETIC_STATISTICS)
                volweight += P[i].Mass / SphP[i].Density;
                massweight += P[i].Mass;
#endif // defined(MAGNETIC_STATISTICS)
                sys.EnergyIntComp[0] += P[i].Mass * egyspec;
            }

            for (int j = 0; j < 3; j++)
            {
                sys.MomentumComp[P[i].Type][j] += P[i].Mass * vel[j];
                sys.CenterOfMassComp[P[i].Type][j] += P[i].Mass * P[i].Pos[j];
            }

            sys.AngMomentumComp[P[i].Type][0] += P[i].Mass * (P[i].Pos[1] * vel[2] - P[i].Pos[2] * vel[1]);
            sys.AngMomentumComp[P[i].Type][1] += P[i].Mass * (P[i].Pos[2] * vel[0] - P[i].Pos[0] * vel[2]);
            sys.AngMomentumComp[P[i].Type][2] += P[i].Mass * (P[i].Pos[0] * vel[1] - P[i].Pos[1] * vel[0]);
        } // Mass > 0
    } // loop i 0,NumPart

    /* some the stuff over all processors */
    MPI_Reduce(&sys.MassComp[0], &SysState.MassComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyPotComp[0], &SysState.EnergyPotComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyIntComp[0], &SysState.EnergyIntComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sys.EnergyKinComp[0], &SysState.EnergyKinComp[0], 6, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sys.MomentumComp[0][0], &SysState.MomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sys.AngMomentumComp[0][0], &SysState.AngMomentumComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
               MYMPI_COMM_WORLD);
    MPI_Reduce(&sys.CenterOfMassComp[0][0], &SysState.CenterOfMassComp[0][0], 6 * 4, MPI_DOUBLE, MPI_SUM, 0,
               MYMPI_COMM_WORLD);

#ifdef MAGNETIC_STATISTICS
    MPI_Reduce(&sys.EnergyMag, &SysState.EnergyMag, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sys.DivBerr, &SysState.DivBerr, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&volweight, &Volweight, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&massweight, &Massweight, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
#endif

    if (ThisTask == 0)
    {

#ifdef MAGNETIC_STATISTICS
        SysState.DivBerr /= Volweight;
#endif // (defined(MAGNETIC_STATISTICS)

        for (int i = 0; i < 6; i++)
        {
            SysState.EnergyTotComp[i] =
                SysState.EnergyKinComp[i] + SysState.EnergyPotComp[i] + SysState.EnergyIntComp[i];
        }

        SysState.Mass = SysState.EnergyKin = SysState.EnergyPot = SysState.EnergyInt = SysState.EnergyTot = 0;

        for (int j = 0; j < 3; j++)
        {
            SysState.Momentum[j] = SysState.AngMomentum[j] = SysState.CenterOfMass[j] = 0;
        }

        for (int i = 0; i < 6; i++)
        {
            SysState.Mass += SysState.MassComp[i];
            SysState.EnergyKin += SysState.EnergyKinComp[i];
            SysState.EnergyPot += SysState.EnergyPotComp[i];
            SysState.EnergyInt += SysState.EnergyIntComp[i];
            SysState.EnergyTot += SysState.EnergyTotComp[i];

            for (int j = 0; j < 3; j++)
            {
                SysState.Momentum[j] += SysState.MomentumComp[i][j];
                SysState.AngMomentum[j] += SysState.AngMomentumComp[i][j];
                SysState.CenterOfMass[j] += SysState.CenterOfMassComp[i][j];
            }
        }

        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (SysState.MassComp[i] > 0)
                {
                    SysState.CenterOfMassComp[i][j] /= SysState.MassComp[i];
                }
            }
        }

        for (int j = 0; j < 3; j++)
        {
            if (SysState.Mass > 0)
            {
                SysState.CenterOfMass[j] /= SysState.Mass;
            }
        }

        for (int i = 0; i < 6; i++)
        {
            SysState.CenterOfMassComp[i][3] = SysState.MomentumComp[i][3] = SysState.AngMomentumComp[i][3] = 0;
            for (int j = 0; j < 3; j++)
            {
                SysState.CenterOfMassComp[i][3] += SysState.CenterOfMassComp[i][j] * SysState.CenterOfMassComp[i][j];
                SysState.MomentumComp[i][3] += SysState.MomentumComp[i][j] * SysState.MomentumComp[i][j];
                SysState.AngMomentumComp[i][3] += SysState.AngMomentumComp[i][j] * SysState.AngMomentumComp[i][j];
            }
            SysState.CenterOfMassComp[i][3] = sqrt(SysState.CenterOfMassComp[i][3]);
            SysState.MomentumComp[i][3] = sqrt(SysState.MomentumComp[i][3]);
            SysState.AngMomentumComp[i][3] = sqrt(SysState.AngMomentumComp[i][3]);
        }

        SysState.CenterOfMass[3] = SysState.Momentum[3] = SysState.AngMomentum[3] = 0;

        for (int j = 0; j < 3; j++)
        {
            SysState.CenterOfMass[3] += SysState.CenterOfMass[j] * SysState.CenterOfMass[j];
            SysState.Momentum[3] += SysState.Momentum[j] * SysState.Momentum[j];
            SysState.AngMomentum[3] += SysState.AngMomentum[j] * SysState.AngMomentum[j];
        }

        SysState.CenterOfMass[3] = sqrt(SysState.CenterOfMass[3]);
        SysState.Momentum[3] = sqrt(SysState.Momentum[3]);
        SysState.AngMomentum[3] = sqrt(SysState.AngMomentum[3]);
    }

    /* give everyone the result, maybe they want to do something with it */
    MPI_Bcast(&SysState, sizeof(struct state_of_system), MPI_BYTE, 0, MYMPI_COMM_WORLD);
    VERBOSE(2, "Finished global statistics");
}
