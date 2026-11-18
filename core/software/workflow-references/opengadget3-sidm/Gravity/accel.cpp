/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Driver routines to carry out gravitational force computation.
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

#ifdef CS_DECOUPLING
#include "../CoolingSfr/cs_model/cs_model.hpp"
#endif // CS_DECOUPLING

#ifdef DEBUG_PARTICLE_GRAVITY_ACCELERATION
/**
 * @brief Debug the properties of a specific particle.
 *
 * This function debugs the properties of a particle with a specific ID. If the ID matches
 * the value defined in the configuration, the properties of the particle, such as its gravitational
 * acceleration, cost total, and other relevant information, are printed.
 *
 * @param Costtotal The total cost associated with the particle.
 */
void debug_particle(double Costtotal)
{
    /* for instance: add in Config.sh: DEBUG_PARTICLE_GRAVITY_ACCELERATION=717 and the particle 717 will be tracked */
    {
        int PARTICELLONA, mycounter;
        for (PARTICELLONA = 0; PARTICELLONA < NumPart; PARTICELLONA++)
        {
            if (P[PARTICELLONA].ID == DEBUG_PARTICLE_GRAVITY_ACCELERATION)
            {
                VERBOSE_ALL(2,
                            "\n\t Debug:true, Time:%f, Task:%d, ID=%d, Acc:[%lg,%lg,%lg], CostTotal:%e, P.GravCost:%e, "
                            "NActivePart:%d ",
                            All.Time, ThisTask, DEBUG_PARTICLE_GRAVITY_ACCELERATION, P[PARTICELLONA].g.dGravAccel[0],
                            P[PARTICELLONA].g.dGravAccel[1], P[PARTICELLONA].g.dGravAccel[2], Costtotal,
                            P[PARTICELLONA].GravCost[TakeLevel], NActivePart);
            }
        }
    }
}
#endif // DEBUG_PARTICLE_GRAVITY_ACCELERATION

/**
 * @brief This routine computes the gravitational forces for all active particles.
 *
 * This routine computes the accelerations for all active particles.  First, the gravitational forces are
 * computed. This also reconstructs the tree, if needed, otherwise the drift/kick operations have updated the
 * tree to make it fullu usable at the current time.
 *
 * If gas particles are presented, the `interior' of the local domain is determined. This region is guaranteed
 * to contain only particles local to the processor. This information will be used to reduce communication in
 * the hydro part.  The density for active SPH particles is computed next. If the number of neighbours should
 * be outside the allowed bounds, it will be readjusted by the function ensure_neighbours(), and for those
 * particle, the densities are recomputed accordingly. Finally, the hydrodynamical forces are added.
 */
void compute_grav_accelerations(void)
{
    VERBOSE(1, "ACCEL: Start force computation...");

#ifdef PMGRID
    if (All.PM_Ti_endstep == All.Ti_Current)
    {
        long_range_force();
    }
#endif // PMGRID

#ifdef ACC_RUN
    on_beginning_of_timestep();
#endif // ACC_RUN

#ifndef ONLY_PM
    gravity_tree(); /* computes gravity accel. on CPU */
#ifdef DEBUG_PARTICLE_GRAVITY_ACCELERATION
    debug_particle(0);
#endif // DEBUG_PARTICLE_GRAVITY_ACCELERATION

    /* For the first timestep, we redo it
     * to allow usage of relative opening
     * criterion for consistent accuracy.
     */
    if (All.ErrTolTheta == 0 && All.Ti_Current == 0)
    {
        gravity_tree(); /* computes gravity accel. */
#ifdef DEBUG_PARTICLE_GRAVITY_ACCELERATION
        debug_particle(0);
#endif // DEBUG_PARTICLE_GRAVITY_ACCELERATION
    }
#endif // ONLY_PM

    if (All.Ti_Current == 0 && RestartFlag == 0 && header.flag_ic_info == FLAG_SECOND_ORDER_ICS)
    {
        second_order_ics(); /* produces the actual ICs from the special second order IC file */
    }

#ifdef FORCETEST
    gravity_forcetest();
#endif // FORCETEST

    VERBOSE(1, "ACCEL: force computation done.");
}

/**
 * @brief Compute densities for various particle types.
 *
 * This function computes densities for different types of particles.
 * It includes computations such as calculating the smoothing length (hsml), density,
 * pressure, and other related quantities for the particles. Depending on compilation
 * flags and simulation parameters, specific computations may be included or excluded.
 *
 * The function also handles specific computations for gas particles, such as marking
 * star particles affected by supernovae (if enabled), computing smoothed values,
 * and performing supernova and dust evolution calculations. Additionally, it handles
 * specific computations for axion dark matter particles if enabled.
 */
void compute_densities(void)
{
#if (defined(mSIDM) || defined(mIDM)) && !defined(ADAPTGRAVSOFT)
    VERBOSE(2, "Start density computation for PartType == 1\n");
    scatter::pt1_find_hsml(); /* compute hsml for PartType==1, used for mSIDM */
    scatter::pt1_force_update_hmax();
#endif // (defined(mSIDM) || defined(mIDM)) && !defined(ADAPTGRAVSOFT)

    if (All.TotN_gas > 0)
    {
#ifdef CS_DECOUPLING
        TIMER_PUSH(1, "CS_DECOUPLE");
        VERBOSE(1, "HYDRO / CS_DECOUPLING: copy densities/find low rho tail...");
        cs_copy_densities();
        cs_find_low_density_tail();
        VERBOSE(1, "HYDRO / CS_DECOUPLING: copy densities/find low rho tail done.");
        TIMER_POP(1, "CS_DECOUPLE");
#endif // CS_DECOUPLING

        VERBOSE(1, "HYDRO: start hsml computation...");
        find_hsml(); /* computes hsml for all particles, computes density and pressure for SPH */
        VERBOSE(1, "HYDRO: hsml done, start gradients computation...");
        compute_unified_gradients(); /* computes gradients and other additional values, density for stars and BHs */
        VERBOSE(1, "HYDRO: gradients done.");

#if (defined(LT_SMOOTH_Z) || defined(LT_SMOOTH_XCLD) || defined(LT_TRACK_WINDS)) || defined(GM_DMVELS)
        {
            TIMER_PUSH(1, "SMOOTHING");
            smoothed_values();
            TIMER_POP(1, "SMOOTHING");
        }
#endif // ( defined(LT_SMOOTH_Z) || defined(LT_SMOOTH_XCLD) || defined(LT_TRACK_WINDS) ) || defined(GM_DMVELS)

#ifdef LT_STELLAREVOLUTION
        {
            VERBOSE(1, "SFR: start supernovae computation...");

            evolve_SN();
        }
#endif // LT_STELLAREVOLUTION

#ifdef GL_CR_DUST
        {
            VERBOSE(1, "DUST: start dust computation...");

            TIMER_PUSH(1, "DUST_EVOLVE");
            evolve_dust();
            TIMER_POP(1, "DUST_EVOLVE");
        }
#endif // GL_CR_DUST
    }

#ifdef AXION_DM
    if (All.TotN_axion > 0)
    {
        VERBOSE(1, "Start axion density computation: first cycle...");
        ax_density(0);
        VERBOSE(1, "Start axion density computation: second cycle...");
        // MN: MPI_BARRIER
        ax_density(1);
    }
#endif // AXION_DM
}

#if GADGET_HYDRO == HYDRO_SPH || GADGET_HYDRO == HYDRO_PESPH
/**
 * @brief Compute hydrodynamic accelerations for gas particles.
 *
 * This function computes hydrodynamic accelerations for gas particles.
 * It first checks if there are any gas particles present. If gas particles exist, it performs
 * various computations related to hydrodynamics, including updating smoothing lengths in the
 * tree, adding hydrodynamical accelerations, and computing the rate of change of specific
 * internal energy (du/dt).
 *
 * Additionally, if wake-up procedures are enabled, it processes wake-up events for gas particles.
 * If the simulation includes wind tunnel physics, it sets injection accelerations accordingly.
 *
 * If axion dark matter particles are present in the simulation, it also computes hydrodynamic
 * accelerations specific to axion particles.
 */
void compute_hydro_accelerations(void)
{
    TIMER_PUSH(1, "HYDRO_ACCEL");

    if (All.TotN_gas > 0)
    {
        VERBOSE(1, "HYDRO: start hydro-force computation...");

        /* it's not needed to exit the GPU data region:
         * force update hmax do not change any array size, we can just
         * update Nodes_base on hydra if we need to go on GPU
    #ifdef ACC
            on_end_of_timestep();
    #endif // ACC
        */
        force_update_hmax(); /* update smoothing lengths in tree */

        /*
    #ifdef ACC
            on_beginning_of_timestep();
    #endif // ACC
        */
        hydro_force(); /* adds hydrodynamical accelerations  and computes du/dt  */

        if (WAKEUP > 0)
        {
            process_wake_ups_new();
        }

        VERBOSE(1, "HYDRO: hydro-force computation done.");
    }

#ifdef AXION_DM
    if (All.TotN_axion > 0)
    {
        VERBOSE(2, "Start axion hydro-force computation...\n");

        ax_force_update_hmax(); /* update smoothing lengths in tree */
        ax_hydro_force(); /* adds hydrodynamical accelerations  and computes du/dt  */

        VERBOSE(2, "hydro axion force computation done.\n");
    }
#endif // AXION_DM

    TIMER_POP(1, "HYDRO_ACCEL");
}

#elif GADGET_HYDRO == HYDRO_MFM
/**
 * @brief Compute fluxes for gas particles.
 *
 * This function computes fluxes for gas particles. It first checks if there
 * are any gas particles present. If gas particles exist, it computes Godunov fluxes.
 *
 * Additionally, if wake-up procedures are enabled, it processes wake-up events for gas particles.
 */
void compute_fluxes(void)
{
    TIMER_PUSH(1, "MFM_FLUXES");

    if (All.TotN_gas > 0)
    {
        VERBOSE(2, "Start flux computation...\n");
        mfm_fluxes(); // computes godunov fluxes
        VERBOSE(2, "flux computation done.\n");
    }

    if (WAKEUP > 0)
    {
        process_wake_ups_new();
    }

    TIMER_POP(1, "MFM_FLUXES");
}

/**
 * @brief Compute slope limiters for gas particles.
 *
 * This function computes slope limiters for gas particles. It first checks
 * if there are any gas particles present. If gas particles exist, it computes slope limiter terms.
 */
void compute_limiters(void)
{
    TIMER_PUSH(1, "MFM_LIMITERS");

    if (All.TotN_gas > 0)
    {
        VERBOSE(2, "Start slope limiter computation...\n");
        mfm_limiters(); // computes slope limiter terms
        VERBOSE(2, "slope limiter computation done.\n");
    }

    TIMER_POP(1, "MFM_LIMITERS");
}
#endif // GADGET_HYDRO == HYDRO_MFM
