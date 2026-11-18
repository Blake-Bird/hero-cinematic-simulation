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
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#ifdef BLACK_HOLES
#include "../Blackholes/blackhole.hpp"
#endif // BLACK_HOLES

/**
 * @brief Reconstruct time bins and update related particle information.
 *
 * This function reconstructs time bins and updates related particle information based on the current state
 * of the simulation. It initializes arrays for each time bin and loops over all particles to assign them to
 * appropriate time bins. Additionally, it updates counts and aggregates information related to different types
 * of particles, such as gas, black holes, and stars.
 *
 * After reconstructing the time bins and updating particle information, it creates a linked list of active particles
 * and their respective active time bins. Finally, it counts and verifies particles, ensuring index consistency
 * and summing up the number of particles requiring force updates globally.
 */
void reconstruct_timebins(void)
{
    TIMER_PUSH(1, "RECONSTRUCT_TIMEBINS");
    VERBOSE(3, "Reconstruct timebins. ");
    TIMER_PUSH(3, "rect_fill_bins");

    // Initialize time bin related arrays
    for (int i = 0; i < TIMEBINS; i++)
    {
        // Initialize arrays for each time bin
        TimeBinCount[i] = 0;
        TimeBinCountSph[i] = 0;
        FirstInTimeBin[i] = -1;
        LastInTimeBin[i] = -1;
#ifdef SFR
        TimeBinSfr[i] = 0;
#endif // SFR
#ifdef BLACK_HOLES
        TimeBin_BH_mass[i] = 0;
        TimeBin_BH_dynamicalmass[i] = 0;
        TimeBin_BH_Mdot[i] = 0;
        TimeBin_BH_Medd[i] = 0;
#endif // BLACK_HOLES
#ifdef LT_STELLAREVOLUTION
        TimeBinCountStars[i] = 0;
#endif // LT_STELLAREVOLUTION
    }

    // Loop over particle data
    for (int i = 0; i < NumPart; i++)
    {
        int bin = P[i].TimeBin;

        assert(bin >= 0);
        assert(bin < TIMEBINS);

        // Update Time Bin Information
        if (TimeBinCount[bin] > 0)
        {
            // If the time bin already contains particles, link the current particle
            PrevInTimeBin[i] = LastInTimeBin[bin];
            NextInTimeBin[i] = -1;
            assert(LastInTimeBin[bin] < NumPart);
            NextInTimeBin[LastInTimeBin[bin]] = i;
            LastInTimeBin[bin] = i;
        }
        else
        {
            // If the time bin is empty, the current particle becomes the first and last
            FirstInTimeBin[bin] = LastInTimeBin[bin] = i;
            PrevInTimeBin[i] = NextInTimeBin[i] = -1;
        }
        TimeBinCount[bin]++;
        if (P[i].Type == type_gas)
        {
            TimeBinCountSph[bin]++;
        }

#ifdef SFR
        // Additional processing for gas particles (Type == type_gas)
        if (P[i].Type == type_gas)
        {
            TimeBinSfr[bin] += SphP[i].Sfr;
        }
#endif // SFR
#ifdef BLACK_HOLES
        // Additional processing for black hole particles (Type == type_bh)
        if (P[i].Type == type_bh)
        {
            TimeBin_BH_mass[bin] += BPP(i).BH_Mass;
            TimeBin_BH_dynamicalmass[bin] += P[i].Mass;
            TimeBin_BH_Mdot[bin] += BPP(i).BH_Mdot;
            double temp_meddington = Mdot_Eddington(BPP(i).BH_Mass);
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
            if (All.BlackHoleSubgridADOn == 1)
            {
                temp_meddington = temp_meddington / (BPP(i).BH_EtaDisk / 0.1);
            }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
            TimeBin_BH_Medd[bin] += BPP(i).BH_Mdot / temp_meddington;
        }
#endif // BLACK_HOLES
#ifdef LT_STELLAREVOLUTION
        // Additional processing for stellar particles (Type == type_star)
        if (P[i].Type == type_star)
        {
            TimeBinCountStars[MetP[P[i].pt.MetID].ChemTimeBin]++;
        }
#endif // LT_STELLAREVOLUTION
    }
    TIMER_POP(3, "rect_fill_bins");

    // Create a linked list of active particles and their active time bin
    make_list_of_active_particles();

    TIMER_PUSH(3, "rect_comp_force_update");

    // Count and verify particles
    NumForceUpdate = 0;
    long long glob_sum;
#ifdef _OPENMP
// Parallelized global count
#pragma omp parallel for reduction(+ : NumForceUpdate) // if(NActivePart>1000)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    // serial loop over active particles
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        NumForceUpdate++;
        // Verify particle index consistency
        if (i >= NumPart)
        {
            VERBOSE_ALL(0, "Bummer i=%d ", i);
            ENDRUN_MAIN("inconsistent list");
        }
    }

    // Sum up the number of particles requiring force updates globally
    sumup_large_ints(1, &NumForceUpdate, &glob_sum);

    GlobNumForceUpdate = glob_sum;
    TIMER_POP(3, "rect_comp_force_update");
    TIMER_POP(1, "RECONSTRUCT_TIMEBINS");
}

/**
 * @brief Drift particle according to leapfrog integration.
 *
 * Drift a particle to its new position and updates associated properties.
 * This function advances the position of a particle based on its velocity and
 * updates various properties such as velocity predictions, entropy, density, and pressure.
 *
 * @param i Index of the particle.
 * @param time1 New time of the particle.
 */
void drift_particle(int i, integertime time1)
{
    // VERBOSE(3, "Drift particle. ");
    double dt_drift;

    integertime time0 = P[i].Ti_current;
    // Check for valid time range
    if (time1 < time0)
    {
        VERBOSE_ALL(0, "DRIFT PARTICLE, Tasl=%d, i=%d time0=%d time1=%d ", ThisTask, i, (int)time0, (int)time1);
        ENDRUN_MAIN("no prediction into past allowed");
    }

    // No drift needed if time1 = time0
    if (time1 == time0)
    {
        return;
    }

    // Compute drift factor based on comoving integration
    if (All.ComovingIntegrationOn)
    {
        dt_drift = get_drift_factor(time0, time1);
    }
    else
    {
        assert(time1 >= time0);
        dt_drift = (time1 - time0) * All.Timebase_interval;
    }

    // Update particle position
#if defined(SPH_BND_PARTICLES) && !defined(SPH_BND_PARTICLES_ALLOW_DRIFT)
    if (P[i].ID != 0)
#endif // defined(SPH_BND_PARTICLES) && !defined(SPH_BND_PARTICLES_ALLOW_DRIFT)
    {
        for (int j = 0; j < NUMDIMS; j++)
        {
            P[i].Pos[j] += P[i].Vel[j] * dt_drift;
        }
    }
#if defined(FG_MAKE_OUTSIDE_HIGHRES_BND) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM)
    if (P[i].Type == type_gas)
    {
        int outside_highres = 0;
        for (int j = 0; j < NUMDIMS; j++)
        {
            MyFloat center = 0.5 * (All.Corner[1][j] + All.UpperCorner[1][j]);
            MyFloat width = All.UpperCorner[1][j] - All.Corner[1][j];
#ifdef ENLARGEREGION
            width *= (0.1 + 0.9 * ENLARGEREGION) /
                     ENLARGEREGION; // remove gas particles already if they come close to the boundary, in general
                                    // should be more confined to the center.
#endif // ENLARGEREGION
       // compare the position of the gas particle to the highres PM region
            if ((center - 0.5 * width > P[i].Pos[j]) || (center + 0.5 * width < P[i].Pos[j]))
            {
                outside_highres = 1;
            } // if ( (center - 0.5*width > P[i].Pos[j]) || (center + 0.5*width < P[i].Pos[j]) )
        } // for (int j = 0; j < NUMDIMS; j++)
        if (outside_highres == 1)
        {
#ifdef FG_MAKE_OUTSIDE_HIGHRES_BND
            P[i].ID = 0;
#elif defined(FG_MAKE_OUTSIDE_HIGHRES_DM)
            P[i].Type = type_dm;
#ifdef _OPENMP
#pragma omp critical(_Particle_converted_)
#endif // _OPENMP
            {
                Particles_converted++;
                TimeBinCountSph[P[i].TimeBin]--;
            }
#endif // FG_MAKE_OUTSIDE_HIGHRES_...
        } // (outside_highres == 1)
    } // if (P[i].Type == type_gas)
#endif // FG_MAKE_OUTSIDE_HIGHRES_...

    // Update properties for active SPH particles
    if (IS_SPH_ALIVE(P[i]))
    {
        double dt_gravkick, dt_hydrokick, dt_entr;

        // Compute various kick factors based on comoving integration
        if (All.ComovingIntegrationOn)
        {
            dt_entr = (time1 - time0) * All.Timebase_interval;
            dt_gravkick = get_gravkick_factor(time0, time1);
            dt_hydrokick = get_hydrokick_factor(time0, time1);
        }
        else
        {
            dt_entr = dt_gravkick = dt_hydrokick = (time1 - time0) * All.Timebase_interval;
        }

// Additional handling for specific hydrodynamics methods (e.g., MFM)
#if GADGET_HYDRO == HYDRO_MFM
        if (P[i].Type == type_gas)
        {
            mfm_fluxes_to_code(i, time0, time1);
        }
#endif // if GADGET_HYDRO == HYDRO_MFM

#ifdef SPH_BND_PARTICLES
        if (P[i].ID != 0)
#endif // SPH_BND_PARTICLES
        {
            for (int j = 0; j < 3; j++)
            {
#ifdef PMGRID
                const MyAtLeastDouble dvg = (P[i].GravAccel[j] + P[i].GravPM[j]) * dt_gravkick;
#else // PMGRID
                const MyAtLeastDouble dvg = P[i].GravAccel[j] * dt_gravkick;
#endif // PMGRID
                const MyAtLeastDouble dvh = SphP[i].HydroAccel[j] * dt_hydrokick;
                SphP[i].VelPred[j] += dvg + dvh;
            }
#ifdef KD_RESTRICT_MAXVEL
            restrict_maxvel(SphP[i].VelPred, i);
#endif // KD_RESTRICT_MAXVEL

            // Update entropy and internal energy predictions
            MyAtLeastDouble dEntr = SphP[i].DtEntropy * dt_entr;
            // the maximum change per kick os by a factor 0.5, so in total over 2 kicks by a factor 0.25
            SphP[i].EntropyPred = max(SphP[i].EntropyPred + dEntr, (MyAtLeastDouble)0.25 * SphP[i].EntropyPred);
#if GADGET_HYDRO == HYDRO_MFM
            MyAtLeastDouble dEner = SphP[i].DtInternalEnergy * dt_entr;
       // similar limit as for the entropy also to the internal energy, which is the primary variable for MFM
            SphP[i].InternalEnergyPred = max(SphP[i].InternalEnergyPred + dEner, 0.25 * SphP[i].InternalEnergyPred);
            mfm_synchronize_entropy(i);
#endif // GADGET_HYDRO == HYDRO_MFM
        }


        // Update density and pressure
        {
#ifndef GT_DUST
            MyAtLeastDouble density_evol_fak = exp(-SphP[i].DivVel * dt_drift);
#else // GT_DUST
            MyAtLeastDouble density_evol_fak = exp(-SphP[i].DivVelMixture * dt_drift);
#endif // GT_DUST
            if (density_evol_fak > (1 + 3 * All.CourantFac))
            {
#ifdef DEBUG
                VERBOSE_ALL(0,
                            "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange density evol value in "
                            "predict: %e",
                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].DivVel, dt_drift,
                            density_evol_fak);
#endif // DEBUG
                density_evol_fak = 1 + 3 * All.CourantFac;
            }
            if (density_evol_fak < 1. / (1 + 3 * All.CourantFac))
            {
#ifdef DEBUG
                VERBOSE_ALL(0,
                            "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange density evol value in "
                            "predict: %e",
                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].DivVel, dt_drift,
                            density_evol_fak);
#endif // DEBUG
                density_evol_fak = 1. / (1 + 3 * All.CourantFac);
            }
            SphP[i].Density *= density_evol_fak;
#if GADGET_HYDRO == HYDRO_MFM
            SphP[i].NumDens *= density_evol_fak;
#endif // GADGET_HYDRO == HYDRO_MFM

#ifdef DEBUG
            if (SphP[i].Density < 0 || SphP[i].Density > 1e5)
            {
                VERBOSE_ALL(2,
                            "task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange Density value after predict: %e",
                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].DivVel, dt_drift,
                            (double)SphP[i].Density);
#ifdef KD_WAKEUP_ON_STRANGE_DENSITY
                int min_occupied_bin = TIMEBINS;
                // Determine the maximum and minimum occupied timebins
                for (int j = 1; j < TIMEBINS; j++)
                    if (TimeBinCount[j] > 0)
                        if (j < min_occupied_bin)
                            min_occupied_bin = j;
                SPHP(i, wakeup) = min_occupied_bin;
                int n_towakeup;
#pragma omp atomic capture
                n_towakeup = N_ToWakeUp++;
                ToWakeupList[n_towakeup] = i;
#endif // KD_WAKEUP_ON_STRANGE_DENSITY
                SphP[i].Density = 1e5;
#if GADGET_HYDRO == HYDRO_MFM
                SphP[i].NumDens = 1e5 / P[i].Mass;
#endif // GADGET_HYDRO == HYDRO_MFM
            }
#endif // DEBUG
        }

        // Update pressure
        SphP[i].Pressure = get_pressure(i);

        {
            // Adjust smoothing length to restrict the number of neighbors
#ifndef GT_DUST
            MyAtLeastDouble hsml_evol_fak = exp(0.333333333333 * SphP[i].DivVel * dt_drift);
#else // GT_DUST
            MyAtLeastDouble hsml_evol_fak = exp(SphP[i].DivVelMixture * dt_drift / NUMDIMS);
#endif // GT_DUST

            if (hsml_evol_fak > (1 + All.CourantFac))
            {
#ifdef DEBUG
                VERBOSE_ALL(
                    0,
                    "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange hsml evol value in predict: %e",
                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].DivVel, dt_drift, hsml_evol_fak);
#endif // DEBUG
                hsml_evol_fak = 1 + All.CourantFac;
            }
            if (hsml_evol_fak < 1. / (1 + All.CourantFac))
            {
#ifdef DEBUG
                VERBOSE_ALL(
                    0,
                    "WARNING: task=%d: Gas Particle id=%llu,m=%e,divV=%e,dt=%e strange hsml evol value in predict: %e",
                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, SphP[i].DivVel, dt_drift, hsml_evol_fak);
#endif // DEBUG
                hsml_evol_fak = 1. / (1 + All.CourantFac);
            }
#ifdef KD_RESTRICT_NEIGHBOURS /* prevent particles from getting too many true neighbours */
            if (hsml_evol_fak > 1 && P[i].TrueNGB > All.DesNumNgb * 3)
            {
                hsml_evol_fak = 1;
            }
            if (hsml_evol_fak < 1 && P[i].TrueNGB < All.DesNumNgb * 0.4)
            {
                hsml_evol_fak = 1;
            }
#endif // KD_RESTRICT_NEIGHBOURS
            P[i].Hsml *= hsml_evol_fak;
        }

        // Check and enforce minimum gas smoothing length
        if (P[i].Hsml < All.MinGasHsml)
        {
            P[i].Hsml = All.MinGasHsml;
        }

#ifdef MAXHSML
        // Check and enforce maximum smoothing length
        if (P[i].Hsml > All.MaxHsml)
        {
            P[i].Hsml = All.MaxHsml;
        }
#endif // MAXHSML

        // Additional physics for SPH particles
        drift_sph_extra_physics(i, time0, time1, dt_entr);
    }

    // Update the current time of the particle
    P[i].Ti_current = time1;
}

#ifdef KD_RESTRICT_MAXVEL
/**
 * @brief Restrict the maximum velocity of a particle if necessary.
 *
 * This function restricts the maximum velocity of a particle if it exceeds a certain threshold. If the velocity
 * magnitude of the particle is greater than the specified maximum velocity, the velocity components
 * are scaled down to ensure that the magnitude does not exceed the maximum velocity. Additionally, this function may
 * update particle properties based on defined conditions, such as changing the particle type or setting its ID to
 * a specific value.
 *
 * @param Vel Pointer to the array containing the velocity components of the particle.
 * @param i Index of the particle.
 */
void restrict_maxvel(MyFloat *Vel, int i)
{
    // Calculate the velocity magnitude
    MyFloat vv = sqrt((MyAtLeastDouble)Vel[0] * Vel[0] + Vel[1] * Vel[1] + Vel[2] * Vel[2]);

    MyFloat max_vel = KD_RESTRICT_MAXVEL;
#ifdef FG_RESTRICT_MAXVEL_DISTANCE
    MyLongDouble distance_from_center2 = 0.0;
    for (int k = 0; k < 3; k++)
    {
        // Use information from HIGHRES PMGRID
        distance_from_center2 += (P[i].Pos[k] - 0.5 * (All.UpperCorner[1][k] + All.Corner[1][k])) *
                                 (P[i].Pos[k] - 0.5 * (All.UpperCorner[1][k] + All.Corner[1][k]));
    }

    // Check if the particle is outside the maximum velocity distance
    if (distance_from_center2 > FG_RESTRICT_MAXVEL_DISTANCE * FG_RESTRICT_MAXVEL_DISTANCE)
    {
        MyLongDouble distance_from_center = sqrt(distance_from_center2);
#ifdef FG_RESTRICT_MAXVEL_DISTANCE_LINEAR
        if (distance_from_center > 2 * FG_RESTRICT_MAXVEL_DISTANCE)
        {
            max_vel *= 0.2;
        }
        else
        {
            max_vel *= 1.0 - 0.8 * (distance_from_center - FG_RESTRICT_MAXVEL_DISTANCE) / FG_RESTRICT_MAXVEL_DISTANCE;
        }
#else // FG_RESTRICT_MAXVEL_DISTANCE_LINEAR
        max_vel *=
            0.1 + 0.9 * exp(-10 * (distance_from_center - FG_RESTRICT_MAXVEL_DISTANCE) / FG_RESTRICT_MAXVEL_DISTANCE);
#endif // FG_RESTRICT_MAXVEL_DISTANCE_LINEAR
    }
#endif // FG_RESTRICT_MAXVEL_DISTANCE

    // Check if the velocity magnitude exceeds the maximum velocity
    if (vv > max_vel)
    {
        for (int j = 0; j < 3; j++)
        {
            Vel[j] = Vel[j] / vv * max_vel;
        }
#ifdef FG_MAKE_MAXVEL_BND
        P[i].ID = 0;
#elif defined(FG_MAKE_MAXVEL_DM)
        if (P[i].Type == type_gas)
        {
            P[i].Type = type_dm;
#ifdef _OPENMP
#pragma omp critical(_Particle_converted_)
#endif // _OPENMP
            {
                Particles_converted++;
                TimeBinCountSph[P[i].TimeBin]--;
            }
        }
#endif // FG_MAKE_MAXVEL_...
    }
}
#endif // KD_RESTRICT_MAXVEL

/**
 * @brief Checks a particle for the minimum allowed temperature/entropy and updates it if necessary.
 *
 * This function examines gas particles to ensure that its entropy or temperature does not fall
 * below the minimum allowed value specified in the simulation parameters. If the particle's entropy
 * (or specific internal energy in the case of certain hydrodynamics methods like MFM) is found to be
 * below the minimum threshold, it is updated to the minimum value, and the corresponding time derivative
 * is reset to zero.
 *
 * @param i The index of the particle to be checked.
 */
void check_particle_for_temperature_minimum(int i)
{
    // VERBOSE(3, "Check particle for temperature minimum. ");
    double minentropy;
    // Check for minimum entropy and update if necessary
    if (All.MinEgySpec)
    {
        minentropy = All.MinEgySpec * GAMMA_MINUS1 / pow(SphP[i].Density * All.cf_a3inv, GAMMA_MINUS1);
        if (SphP[i].Entropy < minentropy)
        {
            SphP[i].Entropy = minentropy;
            SphP[i].DtEntropy = 0;
        }
// Additional handling for specific hydrodynamics methods (e.g., MFM)
#if GADGET_HYDRO == HYDRO_MFM
        if (SphP[i].InternalEnergy < All.MinEgySpec)
        {
            SphP[i].InternalEnergy = All.MinEgySpec;
            if (All.MinEgySpec != 0)
            {
                MyFloat div_vel = limited_divvel(i);
                SphP[i].dQdt.GetEnergy() = eos->GetDtQFromDtEntropy(
                    (MyFloat)0.0, SphP[i].Entropy, SphP[i].InternalEnergy, P[i].Mass, div_vel, All.cf_hubble_a);
                SphP[i].DtInternalEnergy =
                    eos->GetDtUFromDtEntropy((MyFloat)0.0, SphP[i].Entropy, SphP[i].InternalEnergy, div_vel,
                                             All.cf_hubble_a, All.ComovingIntegrationOn);
            }
            else
            {
                SphP[i].dQdt.GetEnergy() = 0.0;
                SphP[i].DtInternalEnergy = 0.0;
            }
        }
#endif // GADGET_HYDRO == HYDRO_MFM
    }
}

/**
 * @brief Returns the limited divergence of velocity (DivVel) to numerically meaningful values.
 *
 * This function calculates and returns the divergence of velocity for a given particle, limited
 * to ensure numerical stability and accuracy in the simulation. The divergence of velocity can become
 * unphysically large due to numerical errors, which can lead to instability and inaccuracies in the
 * simulation results. This function restricts the divergence of velocity to within a specified range,
 * ensuring that it remains within physically meaningful bounds.
 *
 * @param i The index of the particle for which the divergence of velocity is calculated.
 * @return The limited divergence of velocity
 */
MyFloat limited_divvel(int i)
{
    double dt;
    integertime ti_step = P[i].TimeBin ? (((integertime)1) << P[i].TimeBin) : 0;
    integertime tstart_step = P[i].Ti_begstep; /* start of step */
    integertime tend_step = P[i].Ti_begstep + ti_step; /* end of step */
    // Compute drift factor based on comoving integration
    if (All.ComovingIntegrationOn)
    {
        dt = get_drift_factor(tstart_step, tend_step);
    }
    else
    {
        dt = (tend_step - tstart_step) * All.Timebase_interval;
    }
    MyFloat div_vel_max = 3 * All.CourantFac / dt;
    // Limit the divergence of velocity to within a specified range
    if (SphP[i].DivVel > div_vel_max)
    {
        return div_vel_max;
    }
    else if (SphP[i].DivVel < -div_vel_max)
    {
        return -div_vel_max;
    }
    else
    {
        return SphP[i].DivVel;
    }

#ifdef GT_DUST
    if (SphP[i].DivVelMixture > div_vel_max)
    {
        return div_vel_max;
    }
    else if (SphP[i].DivVelMixture < -div_vel_max)
    {
        return -div_vel_max;
    }
    else
    {
        return SphP[i].DivVelMixture;
    }
#endif // GT_DUST
}

/**
 * @brief Move particles to their new positions for the given time.
 *
 * This function moves all particles to their new positions for the given time.
 * Wihtin it iterates over all particles to move them.
 *
 * @param time1 New time of the particles.
 */
void move_particles(integertime time1)
{
    VERBOSE(3, "Moving particles. ");
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    // Iterate over all particles and update their positions
    for (int i = 0; i < NumPart; i++)
    {
        drift_particle(i, time1);
    }
}

/**
 * @brief Get gamma value for particle or global GAMMA value
 *
 * This function returns the gamma value for a given particle or the global GAMMA value if the
 * UM_CHEMISTRY flag is not set.
 *
 * @param i Index of the particle.
 * @return The gamma value for the particle or the global GAMMA value.
 */
double get_gamma(int i)
{
#ifndef UM_CHEMISTRY
    return GAMMA;
#else // UM_CHEMISTRY
    return SphP[i].Gamma;
#endif // UM_CHEMISTRY
}

/**
 * @bried Get gamma value - 1 for particle or global GAMMA_MINUS1 value
 *
 * This function returns the gamma value - 1 for a given particle or the global GAMMA_MINUS1 value if the
 * UM_CHEMISTRY flag is not set.
 *
 * @param i Index of the particle.
 * @return The gamma value - 1 for the particle or the global GAMMA_MINUS1 value.
 */
double get_gamma_minus1(int i)
{
#ifndef UM_CHEMISTRY
    return GAMMA_MINUS1;
#else // UM_CHEMISTRY
    return SphP[i].Gamma - 1.0;
#endif // UM_CHEMISTRY
}

/**
 * @brief Calculate the pressure of particle i based on its predicted entropy and density.
 *
 * This function calculates the pressure of a particle based on its predicted entropy and density.
 *
 * The pressure is calculated using the formula: \f$ P_i = \text{EntropyPred}_i \times \text{Density}_i^{\gamma_i} \f$,
 * where \f$ \text{EntropyPred}_i \f$ is the predicted entropy of particle \f$ i \f$, \f$ \text{Density}_i \f$ is the
 * density of particle \f$ i \f$, and \f$ \gamma_i \f$ is the adiabatic index (ratio of specific heats) of particle \f$
 * i \f$.
 *
 * @param i Index of the particle.
 * @return The pressure of the particle.
 */
double get_pressure(int i)
{
    MyFloat press = SphP[i].EntropyPred * pow(SphP[i].Density, get_gamma(i));

#ifdef GT_DUST
    press *= pow(1. - SphP[i].TotalDustFracPred, get_gamma(i));
#endif // GT_DUST


    return press;
}

/**
 * @brief Drift additional SPH physics for a particle.
 *
 * This function drifts additional SPH physics for a particle, such as magnetic fields and dust-related properties.
 *
 * @param i Index of the particle.
 * @param tstart Start time of the particle.
 * @param tend End time of the particle.
 * @param dt_entr Time step for the particle.
 */
void drift_sph_extra_physics(int i, integertime tstart, integertime tend, double dt_entr)
{
#ifdef MAGNETIC
    double dt_mag;
    // Compute magnetic kick factor based on comoving integration
    if (All.ComovingIntegrationOn)
    {
        dt_mag = get_magkick_factor(tstart, tend);
    }
    else
    {
        assert(tend >= tstart);
        dt_mag = (tend - tstart) * All.Timebase_interval;
    }

#ifdef DIVBCLEANING_TRICCO
    // Update magnetic potential for Tricco cleaning
    SphP[i].PsiPred += SphP[i].DtPsi * dt_mag;
#endif
#ifdef DIVBCLEANING_DEDNER
    // Update magnetic potential for Dedner's cleaning
    SphP[i].PhiPred += SphP[i].DtPhi * dt_mag;
#endif // DIVBCLEANING_DEDNER
    for (int j = 0; j < 3; j++)
    {
        SphP[i].BPred[j] += SphP[i].DtB[j] * dt_mag;
    }
#endif // MAGNETIC

#ifdef GT_DUST
    MyFloat dt_dust = (tend - tstart) * All.Timebase_interval;

#if GADGET_HYDRO == HYDRO_SPH
    SphP[i].InternalEnergySPHPred += dt_dust * SphP[i].DtInternalEnergySPH;
    SphP[i].EntropyPred = SphP[i].InternalEnergySPHPred * get_gamma_minus1(i) /
                          pow(SphP[i].Density * (1. - SphP[i].TotalDustFracPred), get_gamma_minus1(i));
#endif // GADGET_HYDRO

    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        for (int j = 0; j < NUMDIMS; j++)
        {
            SphP[i].DeltaVelPred[idust][j] += dt_dust * SphP[i].DtDeltaVel[idust][j];
        }
        SphP[i].DustFracPred[idust] +=
            SphP[i].DtDustFrac[idust] * dt_dust * dustfrac_conv_factor(SphP[i].DustFracPred[idust]);
    }

    compute_TotalDustFracPred_and_TotalDeltaVelPred(i);

#endif // GT_DUST
}

#ifdef PERIODIC
/**
 * @brief Do box wrapping for all particles.
 *
 *  This function makes sure that all particle coordinates (Pos) are
 *  periodically mapped onto the interval [0, BoxSize].  After this function
 *  has been called, a new domain decomposition should be done, which will
 *  also force a new tree construction.
 */
void do_box_wrapping(void)
{
    VERBOSE(3, "Do box wrapping. ");
    double boxsize[3];
    // Set the box size based on the simulation parameters
    for (int j = 0; j < 3; j++)
    {
        boxsize[j] = All.BoxSize;
    }
#ifdef LONG_X
    boxsize[0] *= LONG_X;
#endif // LONG_X
#ifdef LONG_Y
    boxsize[1] *= LONG_Y;
#endif // LONG_Y
#ifdef LONG_Z
    boxsize[2] *= LONG_Z;
#endif // LONG_Z

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        // Wrap particle coordinates to the box size
        for (int j = 0; j < 3; j++)
        {
            while (P[i].Pos[j] < 0)
            {
                P[i].Pos[j] += boxsize[j];
#ifdef DISK_SLICE
                if (j == 0)
                {
                    P[i].Vel[2] -= All.qDisk * All.OmegaDisk * boxSize_X;
                    if (P[i].Type == 0)
                    {
                        SphP[i].VelPred[2] -= All.qDisk * All.OmegaDisk * boxSize_X;
                    }
                }
#endif // DISK_SLICE
            }
            while (P[i].Pos[j] >= boxsize[j])
            {
                P[i].Pos[j] -= boxsize[j];
#ifdef DISK_SLICE
                if (j == 0)
                {
                    P[i].Vel[2] += All.qDisk * All.OmegaDisk * boxSize_X;
                    if (P[i].Type == 0)
                    {
                        SphP[i].VelPred[2] += All.qDisk * All.OmegaDisk * boxSize_X;
                    }
                }
#endif // DISK_SLICE
            }
        }
    }
}
#endif // PERIODIC

#ifdef GT_DUST

#endif // GT_DUST
