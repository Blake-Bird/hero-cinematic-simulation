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


#ifdef PMGRID
/**
 * @brief Apply long-range gravitational kick to collisionless particles.
 *
 * This function applies a long-range gravitational kick to collisionless particles. The kick is
 * calculated over the time interval from `tstart` to `tend`. It updates the velocities and momentum of the particles
 * based on the gravitational acceleration they experience from the long-range force.
 *
 * @param tstart Starting integration time.
 * @param tend Ending integration time.
 */
void apply_long_range_kick(integertime tstart, integertime tend)
{
    double dt_gravkick;
    if (All.ComovingIntegrationOn)
    {
        dt_gravkick = get_gravkick_factor(tstart, tend);
    }
    else
    {
        dt_gravkick = (tend - tstart) * All.Timebase_interval;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
#ifdef BLACK_HOLES
        if (P[i].Mass > 0)
#endif // BLACK_HOLES
            for (int j = 0; j < 3; j++) /* do the kick, only collisionless particles */
            {
                double dvel = P[i].GravPM[j] * dt_gravkick;
                P[i].Vel[j] += dvel;
                P[i].dp[j] += P[i].Mass * dvel;
            }
#ifdef STATICBRANDT_KICK
        P[i].Vel[0] = -OmegaR(i, BRANDT_MODE) * (P[i].Pos[1] - LONG_Y / 2.);
        P[i].Vel[1] = OmegaR(i, BRANDT_MODE) * (P[i].Pos[0] - LONG_X / 2.);
        P[i].Vel[2] = 0.0;
        P[i].dp[0] = P[i].Vel[0] * P[i].Mass;
        P[i].dp[1] = P[i].Vel[1] * P[i].Mass;
        P[i].dp[2] = P[i].Vel[2] * P[i].Mass;
#endif // STATICBRANDT_KICK
    }
}
#endif // PMGRID

/**
 * @brief Perform the first kick for particle dynamics.
 *
 * This function applies the first half-step kick for particle dynamics, including gravitational and hydrodynamical
 * interactions. It also handles additional physical processes such as long-range gravitational kicks and kinetic energy
 * contributions from specific physical models.
 */
void do_first_halfstep_kick(void)
{
    VERBOSE(3, "Do first halfstep kick.");
    TIMER_PUSH(1, "FIRST_HALF_KICK");

    integertime ti_step, tstart, tend;

#ifdef PMGRID
    if (All.PM_Ti_begstep == All.Ti_Current) /* need to do long-range kick */
    {
        ti_step = All.PM_Ti_endstep - All.PM_Ti_begstep;
        tstart = All.PM_Ti_begstep;
        tend = tstart + ti_step / 2;

        apply_long_range_kick(tstart, tend);
    }
#endif // PMGRID

#ifdef _OPENMP
#pragma omp parallel for private(ti_step, tstart, tend) if (NActivePart > 1000)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP

        ti_step = P[i].TimeBin ? (((integertime)1) << P[i].TimeBin) : 0;

        tstart = P[i].Ti_begstep; /* beginning of step */
        tend = P[i].Ti_begstep + ti_step / 2; /* midpoint of step */

#ifdef GT_DUST
        integrate_dust_drag(i, tstart, tend);
#endif // GT_DUST

#if GADGET_HYDRO == HYDRO_MFM
        if (P[i].Type == type_gas)
        {
            mfm_fluxes_to_code(i, tstart, tend);
        }
#endif // if GADGET_HYDRO == HYDRO_MFM

#ifdef BLACK_HOLES
        if (P[i].Mass > 0)
#endif // BLACK_HOLES
        {
            do_the_kick(i, tstart, tend, P[i].Ti_current);
        }

    } /* closes for(il=0... */

    TIMER_POP(1, "FIRST_HALF_KICK");
}

/**
 * @brief Perform the second kick for particle dynamics.
 *
 * This function applies the second half-step kick for particle dynamics, including gravitational and hydrodynamical
 * interactions. It also handles additional physical processes such as long-range gravitational kicks and kinetic energy
 * contributions from specific physical models.
 */
void do_second_halfstep_kick(void)
{
    VERBOSE(3, "Do second halfstep kick.");
    TIMER_PUSH(1, "SECOND_HALF_KICK");

    integertime ti_step, tstart, tend;
#ifdef PMGRID
    if (All.PM_Ti_endstep == All.Ti_Current) /* need to do long-range kick */
    {
        ti_step = All.PM_Ti_endstep - All.PM_Ti_begstep;
        tstart = All.PM_Ti_begstep + ti_step / 2;
        tend = tstart + ti_step / 2;

        apply_long_range_kick(tstart, tend);
    }
#endif // PMGRID

#ifdef _OPENMP
#pragma omp parallel for private(ti_step, tstart, tend)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        ti_step = P[i].TimeBin ? (((integertime)1) << P[i].TimeBin) : 0;
        tstart = P[i].Ti_begstep + ti_step / 2; /* midpoint of step */
        tend = P[i].Ti_begstep + ti_step; /* end of step */

#if GADGET_HYDRO == HYDRO_MFM
        if (P[i].Type == type_gas)
        {
            mfm_fluxes_to_code(i, tstart, tend);
        }
#endif // if GADGET_HYDRO == HYDRO_MFM

#ifdef BLACK_HOLES
        if (P[i].Mass > 0)
#endif // BLACK_HOLES
            do_the_kick(i, tstart, tend, P[i].Ti_current);

        /* after completion of a full step, set the predication values of SPH quantities
         * to the current values. They will then predicted along in drift operations
         */

        if (IS_SPH_ALIVE(P[i]))
        {
            SphP[i].Pressure = get_pressure(i);

            for (int j = 0; j < 3; j++)
            {
                SphP[i].VelPred[j] = P[i].Vel[j];
            }

            SphP[i].EntropyPred = SphP[i].Entropy;
#if GADGET_HYDRO == HYDRO_MFM
            SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
            mfm_synchronize_entropy(i);
#endif // if GADGET_HYDRO == HYDRO_MFM

            set_predicted_sph_quantities_for_extra_physics(i);
        }

#ifdef AXION_DM
        if (P[i].Type == AXION_TYPE)
        {
            for (int j = 0; j < 3; j++)
            {
                APP(i).VelPred[j] = P[i].Vel[j];
            }
#if GAMMA_AXION != 0
            APP(i).EntropyPred = APP(i).Entropy;
            APP(i).Pressure = get_pressure(i);
#endif //  if GAMMA_AXION!=0
        }
#endif // AXION_DM

#ifdef GT_DUST
        integrate_dust_drag(i, tstart, tend);
#endif // GT_DUST
    } // for

    TIMER_POP(1, "SECOND_HALF_KICK");
}

/**
 * @brief Perform particle kicks for various physical processes.
 *
 * This function computes and applies particle kicks for gravitational, hydrodynamical, and additional physical
 * processes such as magnetic field updates and dust particle interactions.
 *
 * @param i Index of the particle.
 * @param tstart Starting time of the integration.
 * @param tend Ending time of the integration.
 * @param tcurrent Current time of the integration.
 */
void do_the_kick(int i, integertime tstart, integertime tend, integertime tcurrent)
{
    double dt_entr, dt_gravkick, dt_hydrokick;
#ifdef KD_USE_IPOT_FOR_BH_MERGER
    double ascale;
#endif // KD_USE_IPOT_FOR_BH_MERGER
#ifdef AXION_DM
    double dt_ax_hydrokick;
#endif // AXION_DM

    // Compute the time steps for the various physical processes, else set them to the time interval.
    if (All.ComovingIntegrationOn)
    {
        dt_entr = (tend - tstart) * All.Timebase_interval;
        dt_gravkick = get_gravkick_factor(tstart, tend);
        dt_hydrokick = get_hydrokick_factor(tstart, tend);
#ifdef AXION_DM
        dt_ax_hydrokick = get_ax_hydrokick_factor(tstart, tend);
#endif // AXION_DM
#ifdef KD_USE_IPOT_FOR_BH_MERGER
        ascale = All.Time;
#endif // KD_USE_IPOT_FOR_BH_MERGER
    }
    else
    {
        assert(tend >= tstart);
#ifdef AXION_DM
        dt_ax_hydrokick =
#endif // AXION_DM
            dt_entr = dt_gravkick = dt_hydrokick = (tend - tstart) * All.Timebase_interval;
#ifdef KD_USE_IPOT_FOR_BH_MERGER
        ascale = 1;
#endif // KD_USE_IPOT_FOR_BH_MERGER
    }

    /* do the kick */
    double dv[3];
    for (int j = 0; j < 3; j++)
    {
        dv[j] = P[i].GravAccel[j] * dt_gravkick;
#ifdef RELAXOBJECT
        dv[j] -= P[i].Vel[j] * All.RelaxFac * dt_gravkick;
#endif // RELAXOBJECT
        P[i].Vel[j] += dv[j];
        P[i].dp[j] += P[i].Mass * dv[j];
    }
#ifdef KD_USE_IPOT_FOR_BH_MERGER
    if (P[i].IPotential > P[i].Potential / ascale)
    {
        P[i].IPotential = P[i].Potential / ascale;
    }
#endif // KD_USE_IPOT_FOR_BH_MERGER

#ifdef STATICBRANDT_KICK
    P[i].Vel[0] = -OmegaR(i, BRANDT_MODE) * (P[i].Pos[1] - LONG_Y / 2.);
    P[i].Vel[1] = OmegaR(i, BRANDT_MODE) * (P[i].Pos[0] - LONG_X / 2.);
    P[i].Vel[2] = 0.0;
    P[i].dp[0] = P[i].Vel[0] * P[i].Mass;
    P[i].dp[1] = P[i].Vel[1] * P[i].Mass;
    P[i].dp[2] = P[i].Vel[2] * P[i].Mass;
#endif // STATICBRANDT_KICK

    if (P[i].Type == type_gas) /* kick for SPH quantities */
    {
#ifdef SPH_BND_PARTICLES
        if (P[i].ID != 0)
#endif // SPH_BND_PARTICLES
        {
            for (int j = 0; j < 3; j++)
            {
                dv[j] = SphP[i].HydroAccel[j] * dt_hydrokick;
#ifdef RT_RAD_PRESSURE
                dv[j] += SphP[i].RadAccel[j] * dt_hydrokick;
#endif
#ifndef STATICBRANDT_KICK
                P[i].Vel[j] += dv[j];
                P[i].dp[j] += P[i].Mass * dv[j];
#endif // STATICBRANDT_KICK
            }
#ifdef KD_RESTRICT_MAXVEL
            restrict_maxvel(P[i].Vel, i);
#endif // KD_RESTRICT_MAXVEL
#ifdef STATICBRANDT_KICK
            P[i].Vel[0] = -OmegaR(i, BRANDT_MODE) * (P[i].Pos[1] - LONG_Y / 2.);
            P[i].Vel[1] = OmegaR(i, BRANDT_MODE) * (P[i].Pos[0] - LONG_X / 2.);
            P[i].Vel[2] = 0.0;
            P[i].dp[0] = P[i].Vel[0] * P[i].Mass;
            P[i].dp[1] = P[i].Vel[1] * P[i].Mass;
            P[i].dp[2] = P[i].Vel[2] * P[i].Mass;
#endif // STATICBRANDT_KICK

            MyAtLeastDouble dEntr = SphP[i].DtEntropy * dt_entr;
#ifndef GT_DUST
            SphP[i].Entropy = std::max(SphP[i].Entropy + dEntr, (MyAtLeastDouble)0.5 * SphP[i].Entropy);
#endif // GT_DUST
#if GADGET_HYDRO == HYDRO_MFM
            MyAtLeastDouble dEner = SphP[i].DtInternalEnergy * dt_entr;
            SphP[i].InternalEnergy = std::max(SphP[i].InternalEnergy + dEner, 0.5 * SphP[i].InternalEnergy);
            mfm_synchronize_entropy(i);
#endif // if GADGET_HYDRO == HYDRO_MFM

#ifdef JD_RELAX_CLUSTERS
            SphP[i].Entropy = GAMMA_MINUS1 * SphP[i].U / pow(SphP[i].Density, GAMMA_MINUS1);
#endif // JD_RELAX_CLUSTERS
            check_particle_for_temperature_minimum(i);

            do_sph_kick_for_extra_physics(i, tstart, tend, dt_entr);
        }
    }

#ifdef AXION_DM
    if (P[i].Type == AXION_TYPE) /* kick for axion SPH quantities */
    {
#ifdef SPH_BND_PARTICLES
        if (P[i].ID != 0)
#endif // SPH_BND_PARTICLES
        {
            for (int j = 0; j < 3; j++)
            {
                dv[j] = APP(i).HydroAccel[j] * dt_ax_hydrokick;
                P[i].Vel[j] += dv[j];
                P[i].dp[j] += P[i].Mass * dv[j];
            }

#if GAMMA_AXION != 0
            double dEntr = APP(i).DtEntropy * dt_entr;
            APP(i).Entropy = std::(APP(i).Entropy + dEntr, 0.5 * APP(i).Entropy);

            // MN check this in case:
            // check_particle_for_temperature_minimum(i);

#endif // #if GAMMA_AXION!=0
        }
    }
#endif // AXION_DM

#ifdef SPHERICAL_BOUNDARY // enforce a boundary at a given radius, that either reflects or stops the particles
    apply_spherical_boundary(i);
#endif // SPHERICAL_BOUNDARY
}

/**
 * @brief Set predicted SPH quantities for extra physics.
 *
 * This function sets predicted SPH quantities for extra physics, such as magnetic field predictions,
 * divergence cleaning predictions, and dust particle predictions.
 *
 * @param i Index of the particle.
 */
void set_predicted_sph_quantities_for_extra_physics(int i)
{
#ifdef MAGNETIC
    for (int j = 0; j < 3; j++)
    {
        SphP[i].BPred[j] = SphP[i].B[j];
    }
#endif // MAGNETIC
#if defined(MAGNETIC) && defined(DIVBCLEANING_TRICCO)
    SphP[i].PsiPred = SphP[i].Psi;
#endif // if defined(MAGNETIC) && defined(DIVBCLEANING_TRICCO)
#if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER)
    SphP[i].PhiPred = SphP[i].Phi;
#endif // if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER)

#ifdef GT_DUST
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        for (int j = 0; j < NUMDIMS; j++)
        {
            SphP[i].DeltaVelPred[idust][j] = SphP[i].DeltaVel[idust][j];
        }
        SphP[i].DustFracPred[idust] = SphP[i].DustFrac[idust];
        SphP[i].InternalEnergySPHPred = SphP[i].InternalEnergySPH;
    }
    SphP[i].TotalDustFracPred = SphP[i].TotalDustFrac;
    for (int j = 0; j < NUMDIMS; j++)
    {
        SphP[i].TotalDeltaVelPred[j] = SphP[i].TotalDeltaVel[j];
    }

#endif // GT_DUST
}

/**
 * @brief Perform kicks for extra physics.
 *
 * This function performs sph kicks for extra physics, such as magnetic field updates,
 * divergence cleaning, time-dependent artificial viscosity, time-dependent magnetic dissipation,
 * and chemical abundance updates.
 *
 * @param i Index of the particle.
 * @param tstart Starting time of the integration.
 * @param tend Ending time of the integration.
 * @param dt_entr Time step for entropy update.
 */
void do_sph_kick_for_extra_physics(int i, integertime tstart, integertime tend, double dt_entr)
{
    double dt_mag;
    if (All.ComovingIntegrationOn)
    {
        dt_mag = get_magkick_factor(tstart, tend);
    }
    else
    {
        dt_mag = (tend - tstart) * All.Timebase_interval;
    }

    for (int j = 0; j < 3; j++)
    {
#ifdef MAGNETIC
        SphP[i].B[j] += SphP[i].DtB[j] * dt_mag;
#endif // MAGNETIC
    }
#if defined(MAGNETIC) && defined(DIVBCLEANING_TRICCO)
    SphP[i].Psi += SphP[i].DtPsi * dt_mag;
#endif // if defined(MAGNETIC) && defined(DIVBCLEANING_TRICCO)
#if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER)
    SphP[i].Phi += SphP[i].DtPhi * dt_mag;
#endif // if defined(MAGNETIC) && defined(DIVBCLEANING_DEDNER)
#ifdef TIME_DEP_ART_VISC
#if (TIME_DEP_ART_VISC == 0) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
    SphP[i].alpha += SphP[i].Dtalpha * dt_mag;
    SphP[i].alpha = std::min(SphP[i].alpha, (MyFloat)All.ArtBulkViscConst);
#if (TIME_DEP_ART_VISC == 3)
    SphP[i].alpha = std::max(SphP[i].alpha, (MyFloat)(All.ViscosityAlphaMin * SphP[i].F_Balsara));
#else
    SphP[i].alpha = std::max(SphP[i].alpha, (MyFloat)All.ViscosityAlphaMin);
#endif
#endif // (TIME_DEP_ART_VISC == 0) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#endif // TIME_DEP_ART_VISC
#ifdef TIME_DEP_MAGN_DISP
#if (TIME_DEP_MAGN_DISP == 0) || (TIME_DEP_MAGN_DISP == 1) || (TIME_DEP_MAGN_DISP == 3)
    SphP[i].Balpha += SphP[i].DtBalpha * dt_mag;
    SphP[i].Balpha = std::min(SphP[i].Balpha, (MyFloat)All.ArtificialMagneticDissipationConstant);
    SphP[i].Balpha = std::max(SphP[i].Balpha, (MyFloat)All.ArtificialMagneticDissipationMin);
#endif // (TIME_DEP_MAGN_DISP == 0) || (TIME_DEP_MAGN_DISP == 1) || (TIME_DEP_MAGN_DISP == 3)
#endif // if TIME_DEP_MAGN_DISP

#ifdef CHEMISTRY
    /* update the chemical abundances for the new density and temperature */
    double a_start = All.TimeBegin * exp(tstart * All.Timebase_interval);
    double a_end = All.TimeBegin * exp(tend * All.Timebase_interval);
    int mode;
    /* time in cosmic expansion parameter */
    compute_abundances(mode = 1, i, a_start, a_end);
#endif // CHEMISTRY

#ifdef GT_DUST
    MyFloat dt_dust = (tend - tstart) * All.Timebase_interval;

#if GADGET_HYDRO == HYDRO_SPH
    SphP[i].InternalEnergySPH += dt_dust * SphP[i].DtInternalEnergySPH;
    SphP[i].Entropy = SphP[i].InternalEnergySPH * get_gamma_minus1(i) /
                      pow(SphP[i].Density * (1. - SphP[i].TotalDustFrac), get_gamma_minus1(i));
#endif // GADGET_HYDRO

    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        for (int j = 0; j < NUMDIMS; j++)
        {
            SphP[i].DeltaVel[idust][j] += dt_dust * SphP[i].DtDeltaVel[idust][j];
        }
        SphP[i].DustFrac[idust] += SphP[i].DtDustFrac[idust] * dt_dust * dustfrac_conv_factor(SphP[i].DustFrac[idust]);
    }

    compute_TotalDustFrac_and_TotalDeltaVel(i);

#endif // GT_DUST
}

#ifdef SPHERICAL_BOUNDARY // enforce a boundary at a given radius, that either reflects or stops the particles
void apply_spherical_boundary(const int i)
{
    auto &sb = All.spherical_boundary;
    float dist = 0.0;
    for (short loopc = 0; loopc < 3; ++loopc)
    {
        dist += P[i].Pos[loopc] * P[i].Pos[loopc];
    }
    dist = sqrt(dist);
    if (dist > sb.radius)
    {
        float sp = 0.0;
        for (short loopc = 0; loopc < 3; ++loopc)
        {
            sp += P[i].Pos[loopc] * P[i].Vel[loopc];
        }
        if (sp > 0.0) // check whether the particle is heading outwards
        {
            if (sb.type == sb.reflect)
            {
                for (short loopc = 0; loopc < 3; ++loopc)
                {
                    P[i].Vel[loopc] *= -1.0;
                }
            }
            else if (sb.type == sb.stop)
            {
                for (short loopc = 0; loopc < 3; ++loopc)
                {
                    P[i].Vel[loopc] = 0.0;
                }
            }
        }
    }
}
#endif // SPHERICAL_BOUNDARY
