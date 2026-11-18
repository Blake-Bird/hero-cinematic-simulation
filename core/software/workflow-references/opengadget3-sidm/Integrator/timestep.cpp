/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file functions for the time stepping.
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

static double dt_displacement = 0;
// static double modgrav_timestep = 0;

#if defined(LT_STELLAREVOLUTION) && defined(LT_CONSTRAIN_DYNAMICAL_TIMESTEP)
static double time_convert_factor;
#endif // LT_STELLAREVOLUTION && LT_CONSTRAIN_DYNAMICAL_TIMESTEP

/**
 * @brief Set cosmological factors for the current time.
 *
 * This function sets cosmological factors used in simulations for the current time. These factors
 * are dependent on whether comoving integration is enabled.
 *
 * If comoving integration is enabled, the following factors are computed:
 * - All.cf_atime: Scale factor at the current time.
 * - All.cf_a2inv: Inverse square of the scale factor.
 * - All.cf_a3inv: Inverse cube of the scale factor.
 * - All.cf_afac1: Power of the scale factor raised to the power of 3 times GAMMA_MINUS1.
 * - All.cf_afac2: Inverse power of the scale factor raised to the power of 3 times GAMMA minus 2.
 * - All.cf_afac3: Power of the scale factor raised to the power of 3 times (1 - GAMMA) / 2.0.
 * - All.cf_hubble_a: Hubble function at the current time.
 *
 * If comoving integration is not enabled, all cosmological factors are set to 1.
 */
void set_cosmo_factors_for_current_time(void)
{
    if (All.ComovingIntegrationOn)
    {
        All.cf_atime = All.Time;
        All.cf_a2inv = 1 / (All.Time * All.Time);
        All.cf_a3inv = 1 / (All.Time * All.Time * All.Time);
        All.cf_afac1 = pow(All.Time, 3 * GAMMA_MINUS1);
        All.cf_afac2 = 1 / pow(All.Time, 3 * GAMMA - 2);
        All.cf_afac3 = pow(All.Time, 3 * (1 - GAMMA) / 2.0);
        All.cf_hubble_a = hubble_function(All.Time);
    }
    else
    {
        All.cf_atime = 1;
        All.cf_a2inv = 1;
        All.cf_a3inv = 1;
        All.cf_afac1 = 1;
        All.cf_afac2 = 1;
        All.cf_afac3 = 1;
        All.cf_hubble_a = 1;
    }
    units->init_cosmo_values(All.Time);
}

/**
 * @brief Find the maximum allowed timestep for a particle.
 *
 * This function advances the system in momentum space, i.e. it does apply the 'kick' operation after the
 * forces have been computed. Additionally, it assigns new timesteps to particles. At start-up, a
 * half-timestep is carried out, as well as at the end of the simulation. In between, the half-step kick that
 * ends the previous timestep and the half-step kick for the new timestep are combined into one operation.
 */
void find_timesteps(void)
{
    TIMER_PUSH(1, "TIMELINE");

    // Set cosmological factors for the current time
    if (All.HighestActiveTimeBin == All.HighestOccupiedTimeBin || dt_displacement == 0)
    {
        find_dt_displacement_constraint(All.cf_hubble_a * All.cf_atime * All.cf_atime);
    }

#if defined(LT_STELLAREVOLUTION) && defined(LT_CONSTRAIN_DYNAMICAL_TIMESTEP)
    time_convert_factor = (1000 * SEC_PER_MEGAYEAR) / /* chemical time is in Gyr */
                          All.UnitTime_in_s * /* convert in code units   */
                          All.HubbleParam;
#endif // LT_STELLAREVOLUTION && LT_CONSTRAIN_DYNAMICAL_TIMESTEP


#ifdef MAKEGLASS
    do_glass_making_step();
#endif // MAKEGLASS

#ifdef FORCE_EQUAL_TIMESTEPS
    integertime ti_min_glob;
    {
        integertime ti_min = TIMEBASE;
        integertime ti_step;
        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            if (P[i].Type == type_gas && P[i].Mass != 0)
            {
                double aphys;
                ti_step = get_timestep(i, &aphys, 0);

                ti_step = std::min(ti_step, ti_min);
            }
        }

        ti_min = std::min(ti_min, (dt_displacement / All.Timebase_interval));

        ti_step = TIMEBASE;
        while (ti_step > ti_min)
        {
            ti_step >>= 1;
        }

#ifdef ENLARGE_DYNAMIC_RANGE_IN_TIME
        minimum_large_ints(1, &ti_step, &ti_min_glob);
#else // ENLARGE_DYNAMIC_RANGE_IN_TIME
        MPI_Allreduce(&ti_step, &ti_min_glob, 1, MPI_INT, MPI_MIN, MYMPI_COMM_WORLD);
#endif // ENLARGE_DYNAMIC_RANGE_IN_TIME
    }
#endif // if defined(FORCE_EQUAL_TIMESTEPS)

#ifdef RELAXOBJECT
    determine_relaxfac();
#endif // RELAXOBJECT

    int same = 0, changed = 0;
    /* Now assign new timesteps  */
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : same, changed) // if(NActivePart>500)
    for (int il = 0; il < NActivePart; il++)
    {
        int bin = 0, binold = 0, prev, next;
        integertime ti_step;
        int i = ActiveParticleList[il];
#else /* _OPENMP */
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        int bin = 0, binold = 0, prev, next;
        integertime ti_step;
#endif // _OPENMP

        if (WAKEUP > 0)
            if (P[i].Type == type_gas)
                if (SPHP(i, wakeup) != TIMEBINS)
                {
                    bin = binold = SPHP(i, wakeup);
                    SPHP(i, wakeup) = TIMEBINS;
                    ti_step = bin ? (((integertime)1) << bin) : 0;
                }

        if (bin == 0)
        {
#ifdef FORCE_EQUAL_TIMESTEPS
            ti_step = ti_min_glob;
#else // FORCE_EQUAL_TIMESTEPS
            double aphys;
            ti_step = get_timestep(i, &aphys, 0);
#endif // FORCE_EQUAL_TIMESTEPS

            /* make it a power 2 subdivision */
            integertime ti_min = TIMEBASE;
            // for(ti_min; ti_min > ti_step;)
            while (ti_min > ti_step)
            {
                ti_min >>= 1;
            }

            /* old version
               ti_min = TIMEBASE;
               while(ti_min > ti_step)
               ti_min >>= 1;
             */

            ti_step = ti_min;

            bin = get_timestep_bin(ti_step);
            binold = P[i].TimeBin;
        }

        assert(bin < TIMEBINS);
        assert(binold < TIMEBINS);

        if (bin > binold) /* timestep wants to increase */
        {
            while (TimeBinActive[bin] == 0 && bin > binold)
            { /* make sure the new step is synchronized */
                bin--;
            }

            ti_step = bin ? (((integertime)1) << bin) : 0;
        }

        if (All.Ti_Current >= TIMEBASE) /* we here finish the last timestep. */
        {
            ti_step = 0;
            bin = 0;
        }

        if ((TIMEBASE - All.Ti_Current) < ti_step) /* check that we don't run beyond the end */
        {
            ENDRUN_MAIN("we are beyond the end of the timeline"); /* should not happen */

            for (integertime ti_min = TIMEBASE; ti_min > TIMEBASE - All.Ti_Current;)
            {
                ti_min >>= 1;
                ti_step = ti_min;
            }

            /* old version
               ti_step = TIMEBASE - All.Ti_Current;
               ti_step = ti_min;
               while(ti_min > ti_step)
               ti_min >>= 1;
             */
        }
        if (bin != binold)
        {
            changed++;
            if (P[i].Type == type_gas)
            {
#pragma omp critical(_fix_timebins_sph_)
                {
#ifdef SFR
                    TimeBinSfr[binold] -= SphP[i].Sfr;
                    TimeBinSfr[bin] += SphP[i].Sfr;
#endif // SFR
                }
            }
#ifdef FOLLOW_ID
            if (bin == 20 && (P[i].ID == FOLLOW_ID || FOLLOW_ID == -1))
            {
                printf("Task %d: Type=%d bin=%d acc=(%g,%g,%g) %llu \n", ThisTask, P[i].Type, bin, P[i].GravAccel[0],
                       P[i].GravAccel[1], P[i].GravAccel[2], (unsigned long long)P[i].ID);
            }
#endif // FOLLOW_ID
#ifdef BLACK_HOLES
            if (P[i].Type == type_bh)
            {
#pragma omp critical(_fix_timebins_bhs_)
                {
                    double temp_meddington = Mdot_Eddington(BPP(i).BH_Mass);
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                    if (All.BlackHoleSubgridADOn == 1)
                    {
                        temp_meddington = temp_meddington / (BPP(i).BH_EtaDisk / 0.1);
                    }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
                    TimeBin_BH_mass[binold] -= BPP(i).BH_Mass;
                    TimeBin_BH_dynamicalmass[binold] -= P[i].Mass;
                    TimeBin_BH_Mdot[binold] -= BPP(i).BH_Mdot;
                    if (BPP(i).BH_Mass > 0)
                        TimeBin_BH_Medd[binold] -= BPP(i).BH_Mdot / temp_meddington;
                    TimeBin_BH_mass[bin] += BPP(i).BH_Mass;
                    TimeBin_BH_dynamicalmass[bin] += P[i].Mass;
                    TimeBin_BH_Mdot[bin] += BPP(i).BH_Mdot;
                    if (BPP(i).BH_Mass > 0)
                        TimeBin_BH_Medd[bin] += BPP(i).BH_Mdot / temp_meddington;
                }
            }
#endif // BLACK_HOLES

#pragma omp critical(_fix_timebins_)
            {
                TimeBinCount[binold]--;
                prev = PrevInTimeBin[i];
                next = NextInTimeBin[i];

                if (FirstInTimeBin[binold] == i)
                {
                    FirstInTimeBin[binold] = next;
                }
                if (LastInTimeBin[binold] == i)
                {
                    LastInTimeBin[binold] = prev;
                }
                if (prev >= 0)
                {
                    NextInTimeBin[prev] = next;
                }
                if (next >= 0)
                {
                    PrevInTimeBin[next] = prev;
                }

                if (TimeBinCount[bin] > 0)
                {
                    PrevInTimeBin[i] = LastInTimeBin[bin];
                    NextInTimeBin[LastInTimeBin[bin]] = i;
                    NextInTimeBin[i] = -1;
                    LastInTimeBin[bin] = i;
                }
                else
                {
                    FirstInTimeBin[bin] = LastInTimeBin[bin] = i;
                    PrevInTimeBin[i] = NextInTimeBin[i] = -1;
                }
                TimeBinCount[bin]++;
                if (P[i].Type == type_gas)
                {
                    TimeBinCountSph[binold]--;
                    TimeBinCountSph[bin]++;
                }
            }
            P[i].TimeBin = bin;
        }
        else
        {
            same++;
        }

        /* This works only if particles are not woken up.
       It should be save to force the starting point of the integration
       cycle to be now, as we have only active particles here. So now the
           old code is preplace accordingly.
        */
        // integertime ti_step_old;
        // ti_step_old = binold ? (((integertime) 1) << binold) : 0;
        // P[i].Ti_begstep += ti_step_old;
        P[i].Ti_begstep = All.Ti_Current;
    }

#ifdef CONDUCTION
    if (All.Conduction_Ti_endstep == All.Ti_Current)
    {
        integertime ti_step = TIMEBASE;
        while (ti_step > (All.MaxSizeConductionStep / All.Timebase_interval))
        {
            ti_step >>= 1;
        }
        while (ti_step > (All.MaxSizeTimestep / All.Timebase_interval))
        {
            ti_step >>= 1;
        }

        if (ti_step > (All.Conduction_Ti_endstep - All.Conduction_Ti_begstep)) /* PM-timestep wants to increase */
        {
            /* we only increase if an integer number of steps will bring us to the end */
            if (((TIMEBASE - All.Conduction_Ti_endstep) % ti_step) > 0)
            {
                ti_step = All.Conduction_Ti_endstep - All.Conduction_Ti_begstep; /* leave at old step */
            }
        }

        if (All.Ti_Current == TIMEBASE)
        { /* we here finish the last timestep. */
            ti_step = 0;
        }

        All.Conduction_Ti_begstep = All.Conduction_Ti_endstep;
        All.Conduction_Ti_endstep = All.Conduction_Ti_begstep + ti_step;
    }
#endif // CONDUCTION


#ifdef PMGRID
    if (All.PM_Ti_endstep == All.Ti_Current) /* need to do long-range kick */
    {
        integertime ti_step = TIMEBASE;
        while (ti_step > (dt_displacement / All.Timebase_interval))
        {
            ti_step >>= 1;
        }

        if (ti_step > (All.PM_Ti_endstep - All.PM_Ti_begstep)) /* PM-timestep wants to increase */
        {
            integertime bin = get_timestep_bin(ti_step);
            integertime binold = get_timestep_bin(All.PM_Ti_endstep - All.PM_Ti_begstep);

            while (TimeBinActive[bin] == 0 && bin > binold)
            { /* make sure the new step is synchronized */
                bin--;
            }

            ti_step = bin ? (((integertime)1) << bin) : 0;
        }

        if (All.Ti_Current == TIMEBASE)
        { /* we here finish the last timestep. */
            ti_step = 0;
        }

        All.PM_Ti_begstep = All.PM_Ti_endstep;
        All.PM_Ti_endstep = All.PM_Ti_begstep + ti_step;
    }
#endif // PMGRID

    VERBOSE(3, "Find Next Timestep: same=%d changed=%d", same, changed);

    TIMER_POP(1, "TIMELINE");
}

/**
 * @brief Get the maximum allowed timestep of a particle.
 *
 * This function normally (for flag==0) returns the maximum allowed timestep of a particle, expressed in
 * terms of the integer mapping that is used to represent the total simulated timespan. The physical
 * acceleration is returned in aphys. The latter is used in conjunction with the PSEUDOSYMMETRIC integration
 * option, which also makes of the second function of get_timestep. When it is called with a finite timestep
 * for flag, it returns the physical acceleration that would lead to this timestep, assuming timestep
 * criterion 0.
 *
 * @param p Particle index.
 * @param aphys Physical acceleration in physical units.
 * @param flag Flag to indicate normal operation (0) or finite timestep to get corresponding aphys.
 * @return Maximum allowed timestep of the particle.
 */
integertime get_timestep(int p, double *aphys, int flag)
{
#if defined(mSIDM) && defined(mSIDM_TIMESTEP)
    double dt_msidm;
    if (P[p].Type == type_dm)
    {
        dt_msidm = msidm.calc_dt(P[p]);
    }
#endif // defined(mSIDM) && defined(mSIDM_TIMESTEP)


    double ac;
    if (flag == 0)
    {
        double ax, ay, az;
        ax = All.cf_a2inv * P[p].GravAccel[0];
        ay = All.cf_a2inv * P[p].GravAccel[1];
        az = All.cf_a2inv * P[p].GravAccel[2];
#ifdef PMGRID
        ax += All.cf_a2inv * P[p].GravPM[0];
        ay += All.cf_a2inv * P[p].GravPM[1];
        az += All.cf_a2inv * P[p].GravPM[2];
#endif // PMGRID

#ifdef DEBUG
        if (P[p].Type == type_bh)
        {
            double grav_ac = sqrt(P[p].GravAccel[0] * P[p].GravAccel[0] + P[p].GravAccel[1] * P[p].GravAccel[1] +
                                  P[p].GravAccel[2] * P[p].GravAccel[2]);
            double grav_pm = 0;
            double GravPM[3] = {0, 0, 0};
#ifdef PMGRID
            grav_pm = sqrt(P[p].GravPM[0] * P[p].GravPM[0] + P[p].GravPM[1] * P[p].GravPM[1] +
                           P[p].GravPM[1] * P[p].GravPM[2]);
            GravPM[0] = P[p].GravPM[0];
            GravPM[1] = P[p].GravPM[1];
            GravPM[2] = P[p].GravPM[2];
#endif // PMGRID
            if (grav_pm > grav_ac || grav_ac > 1e5)
                VERBOSE_ALL(0, "WARNING BH accel strange: Part-ID=%llu  acc=(%g,%g,%g) pm=(%g,%g,%g)",
                            (unsigned long long)P[p].ID, P[p].GravAccel[0], P[p].GravAccel[1], P[p].GravAccel[2],
                            GravPM[0], GravPM[1], GravPM[2]);
        }
#endif // DEBUG

#if defined(AB_TURB) || defined(TURB_DRIVING)
        if (P[p].Type == type_gas)
        {
            ax += SphP[p].TurbAccel[0];
            ay += SphP[p].TurbAccel[1];
            az += SphP[p].TurbAccel[2];
        }
#endif // if defined(AB_TURB) || defined(TURB_DRIVING)


#ifdef SPH_BND_PARTICLES
        if (P[p].ID != 0)
#endif // SPH_BND_PARTICLES
            if (P[p].Type == type_gas)
            {
                ax += All.cf_afac2 * SphP[p].HydroAccel[0];
                ay += All.cf_afac2 * SphP[p].HydroAccel[1];
                az += All.cf_afac2 * SphP[p].HydroAccel[2];
            }

        ac = sqrt(ax * ax + ay * ay + az * az); /* this is now the physical acceleration */
        *aphys = ac;
    }
    else
    {
        ac = *aphys;
    }

    if (ac == 0)
    {
        ac = 1.0e-30;
    }
    const double ac_soft = ac;

    double dt = 0, dt_acc = 0;
    switch (All.TypeOfTimestepCriterion)
    {
    case 0:
        if (flag > 0)
        {
            dt = flag * All.Timebase_interval;

            dt /= All.cf_hubble_a; /* convert dloga to physical timestep  */
#ifdef ADAPTGRAVSOFT
            ac = 2 * All.ErrTolIntAccuracy * All.cf_atime * P[p].AGS_Hsml / 2.8 / (dt * dt);
#else // ADAPTGRAVSOFT
            ac = 2 * All.ErrTolIntAccuracy * All.cf_atime * All.SofteningTable[P[p].Type] / (dt * dt);
#endif // ADAPTGRAVSOFT
            *aphys = ac;
            return flag;
        }
#ifdef ADAPTGRAVSOFT
        dt_acc = sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime * P[p].AGS_Hsml / 2.8 / ac_soft);
#else // ADAPTGRAVSOFT
        dt_acc = sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime * All.SofteningTable[P[p].Type] / ac_soft);
#endif // ADAPTGRAVSOFT

#ifdef ADAPTIVE_GRAVSOFT_FORGAS
#ifdef ADAPTIVE_GRAVSOFT_FORGAS_HSML
        if (P[p].Type == type_gas)
            dt_acc = sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime * P[p].Hsml / 2.8 / ac_soft);
#else // ADAPTIVE_GRAVSOFT_FORGAS_HSML
        if (P[p].Type == type_gas)
            dt_acc = sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime * All.SofteningTable[P[p].Type] *
                          pow(P[p].Mass / All.ReferenceGasMass, 1.0 / 3) / ac_soft);
#endif // ADAPTIVE_GRAVSOFT_FORGAS_HSML
#endif // ADAPTIVE_GRAVSOFT_FORGAS
        break;

    default:
        ENDRUN_MAIN("All.TypeOfTimestepCriterion != 0. Stopping here.");
        break;
    }

    dt = dt_acc;
    double csnd = 0, dt_divv = 0, dt_courant = 0;

#ifdef NAVIERSTOKES_TIMESTEP
    double dt_NS = 0.0;
#endif // NAVIERSTOKES_TIMESTEP

    if (IS_SPH_ALIVE(P[p]))
    {
#if GADGET_HYDRO == HYDRO_SPH
        csnd = sqrt(GAMMA * SphP[p].Pressure / SphP[p].Density);

#elif GADGET_HYDRO == HYDRO_PESPH
        csnd = sqrt(GAMMA * SphP[p].Pressure / SphP[p].Density);
#elif GADGET_HYDRO == HYDRO_MFM
        csnd = eos->SoundSpeed(SphP[p].InternalEnergyPred);
#endif //  GADGET_HYDRO == HYDRO_SPH

#if GADGET_HYDRO == HYDRO_MFM
        MyFloat particle_size = get_mfm_cell_size(p);
#else // GADGET_HYDRO == HYDRO_MFM
        MyFloat particle_size = P[p].Hsml;
#endif // GADGET_HYDRO == HYDRO_MFM
        if (All.ComovingIntegrationOn)
        {
            dt_courant =
                2 * All.CourantFac * units->length_to_physical * particle_size / (units->afac3 * SphP[p].MaxSignalVel);
        }
        else
        {
            dt_courant = 2 * All.CourantFac * particle_size / SphP[p].MaxSignalVel;
        }
        // #ifdef AR_MAX_DELAYTIME_AFFECTS_SIGNALVEL
        // if(SphP[p].DelayTime<=AR_MAX_DELAYTIME_AFFECTS_SIGNALVEL) //if winds are decoupled why their timesteps should
        // be influenced #endif
#ifdef NOGRAVITY
        dt = dt_courant; // Without self gravity we do not want the timestep to be drive by an artificial, unused
                         // softening parameter
        dt_acc = -1; // Mark that dt_acc was not used. Avoids confusion in case there are problems with zero timestep
#else // NOGRAVITY
        dt = std::min(dt, dt_courant);
#endif // NOGRAVITY


#ifdef DEDNER_TIMESTEP
        double dt_ded;
        if (SphP[p].divB != 0)
        {
            dt_ded = sqrt(SphP[p].Density / (mu0_inv * SphP[p].divB * SphP[p].divB));
            if (P[p].ID != 0)
            {
                if (P[p].TimeBin > 10 && TimeBinCountSph[P[p].TimeBin] > All.TotN_gas / 100.)
                {
                    dt = dt_ded < dt ? dt_ded : dt;
                }
            }
        }
#endif // DEDNER_TIMESTEP

        /* make sure that the velocity divergence does not imply a too large change of density or smoothing length in
         * the step */
#ifndef GT_DUST
        if (SphP[p].DivVel != 0)
#else // GT_DUST
        if (SphP[p].DivVelMixture != 0)
#endif // GT_DUST
        {
#ifndef GT_DUST
            dt_divv = 1.5 / abs(All.cf_a2inv * SphP[p].DivVel);
#else // GT_DUST
            dt_divv = 1.5 / abs(All.cf_a2inv * SphP[p].DivVelMixture);
#endif // GT_DUST
#ifndef PATCH_DIVV_TIMESTEP
            // allow divv to constzrain any timestep, even if very small due to unusual large divv.
            dt = std::min(dt, dt_divv);
#else // !PATCH_DIVV_TIMESTEP
            if (dt_divv < dt / 2)
            {
                VERBOSE_ALL(2, "WARNING divv timestep: Part-ID=%llu  dt=%g dta=%g dtc=%g dtv=%g  hsml=%g",
                            (unsigned long long)P[p].ID, dt, dt_acc, dt_courant, dt_divv, (float)P[p].Hsml);
            }
            // small reduction (by maximum 2 timebins) due to divv should be always allowed
            if (dt_divv < dt && dt_divv > dt / 4)
            {
                dt = dt_divv;
            }
            else
            {
                // divv would ask to very small timesteps, so limit it to going two timebins down
                if (dt_divv < dt)
                {
                    dt = dt / 4;
                }
            }
#endif // !PATCH_DIVV_TIMESTEP
        }
        else
        {
            dt_divv = -1; // in case of problems (zero timestep) this indicates that it was not used
        }

#ifdef LMB_SPECTRAL_CRs_TIMESTEP
        /*  Function in CosmicRays/bp_cr_timestep.h  */
        // MyAtLeastDouble dt_cr = compute_bp_cr_timestep_from_cooling_times(p);

        if (SphP[p].DivVel != 0)
        {
            // The factor 1.4e6 comes from tests
            MyAtLeastDouble dt_cr = abs(SphP[p].DivVel) / (All.cf_a2inv * 1.4e6);
            dt = std::min(dt, dt_cr);
        }

#endif // LMB_SPECTRAL_CRs_TIMESTEP

#ifdef NAVIERSTOKES_TIMESTEP
        if (abs(SphP[p].ViscEntropyChange))
        {
            dt_NS = All.AlphaViscTimestep * SphP[p].Entropy / SphP[p].ViscEntropyChange / All.cf_hubble_a;

            dt = std::min(dt, dt_NS);
        }
#endif // NAVIERSTOKES_TIMESTEP
    }

#ifdef BLACK_HOLES
    double dt_accr;
    if (P[p].Type == type_bh)
    {
        if (BPP(p).BH_Mdot > 0 && BPP(p).BH_Mass > 0)
        {
            dt_accr = 0.25 * BPP(p).BH_Mass / BPP(p).BH_Mdot;
            if (dt_accr < dt)
            {
#ifdef DEBUG
                if (dt_accr < dt / 4)
                {
                    VERBOSE_ALL(4, "WARNING BH accreation timestep: Part-ID=%llu  dt=%g dt_acc=%g  hsml=%g",
                                (unsigned long long)P[p].ID, dt, dt_accr, (float)P[p].Hsml);
                }
#endif
                dt = dt_accr;
            }
        }

        double dt_ngbs = (BPP(p).BH_TimeBinGasNeighbor ? (1 << BPP(p).BH_TimeBinGasNeighbor) : 0) *
                         All.Timebase_interval / All.cf_hubble_a;

        if (dt > dt_ngbs && dt_ngbs > 0)
        {
#ifdef DEBUG
            if (dt_ngbs < dt / 4)
            {
                VERBOSE_ALL(4, "WARNING BH gas neighbours timestep: Part-ID=%llu  dt=%g dt_ngbs=%g  hsml=%g",
                            (unsigned long long)P[p].ID, dt, dt_ngbs, (float)P[p].Hsml);
            }
#endif
            dt = 1.01 * dt_ngbs;
        }
    }
#endif // BLACK_HOLES

#ifdef UM_CHEMISTRY
    double dt_cool, dt_elec;
    /* another criterion given by the local cooling time */
    if (P[p].Type == type_gas)
    {
        dt_cool = abs(SphP[p].t_cool); /* still in yrs */
        dt_cool *= SEC_PER_YEAR; /* in seconds */
        dt_cool /= All.UnitTime_in_s;
        dt_cool *= All.HubbleParam; /* internal units */

        dt_cool = All.Epsilon * dt_cool;

#ifdef DONOT_EXCLUDE_WINDS_FROM_COOLING
        if (dt_cool > 0 && dt_cool < dt)
        {
            dt = dt_cool;
        }
#else // DONOT_EXCLUDE_WINDS_FROM_COOLING
#ifdef WINDS
        if (dt_cool > 0 && dt_cool < dt && SphP[p].DelayTime < 0)
        {
            dt = dt_cool;
        }
#else // WINDS
        if (dt_cool > 0 && dt_cool < dt)
        {
            dt = dt_cool;
        }
#endif // WINDS
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING

        /* yet another criterion given by the electron number density change */
        dt_elec = abs(SphP[p].t_elec); /* still in yrs */
        dt_elec *= SEC_PER_YEAR; /* in seconds */
        dt_elec /= All.UnitTime_in_s;
        dt_elec *= All.HubbleParam; /* internal units */
        dt_elec = All.Epsilon * dt_elec;

#ifdef DONOT_EXCLUDE_WINDS_FROM_COOLING
        if (dt_elec > 0 && dt_elec < dt)
        {
            dt = dt_elec;
        }
#else // DONOT_EXCLUDE_WINDS_FROM_COOLING
        if (dt_elec > 0 && dt_elec < dt && SphP[p].DelayTime < 0)
        {
            dt = dt_elec;
        }
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING
    }
#endif // UM_CHEMISTRY

#if defined(LT_STELLAREVOLUTION) && defined(LT_CONSTRAIN_DYNAMICAL_TIMESTEP)
    double dt_chem;
    if (P[i].Type == type_star)
    {
        dt_chem = MetP[P[i].MetID].NextChemTime - All.Time_Age; /* this is the next chemical time in Gyr */
        dt_chem *= time_convert_factor; /* convert in code physical units        */

        if (dt_chem > 0 && dt_chem < dt)
        {
            dt = dt_chem;
        }
    }
#endif // if defined(LT_STELLAREVOLUTION) && defined(LT_CONSTRAIN_DYNAMICAL_TIMESTEP)

#if defined(mSIDM) && defined(mSIDM_TIMESTEP)
    if (P[p].Type == type_dm)
    {
        dt = std::min(dt, dt_msidm);
    }
#endif // defined(mSIDM) && defined(mSIDM_TIMESTEP)


#ifdef GT_DUST
    MyFloat dv_max = 0.0;
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        dv_max = std::max(dv_max, sqrt(pow(SphP[p].DeltaVelPred[idust][0], 2) + pow(SphP[p].DeltaVelPred[idust][1], 2) +
                                       pow(SphP[p].DeltaVelPred[idust][2], 2)));
    }
    MyFloat sqrtv2 = sqrt(pow(SphP[p].VelPred[0], 2) + pow(SphP[p].VelPred[1], 2) + pow(SphP[p].VelPred[2], 2));
    MyFloat dt_dust = 0.5 * All.CourantFac * P[p].Hsml / (sqrtv2 + dv_max + csnd);
    dt = std::min(dt, dt_dust);
#endif // GT_DUST

    /* convert the physical timestep to dloga if needed. Note: If comoving integration has not been selected,
       All.cf_hubble_a=1.
     */
    dt *= All.cf_hubble_a;

#ifdef ONLY_PM
    dt = All.MaxSizeTimestep;
#endif // ONLY_PM

    dt = std::min(dt, All.MaxSizeTimestep);

    dt = std::min(dt, dt_displacement);

#ifdef CONDUCTION
    if (P[p].Type == type_gas)
    {
        dt = std::min(dt, All.MaxSizeConductionStep);
    }
#endif // CONDUCTION

    if (dt < All.MinSizeTimestep)
    {
        VERBOSE_ALL(2, "WARNING: Timestep wants to be below the limit `MinSizeTimestep'");

        if (P[p].Type == type_gas)
        {
            VERBOSE_ALL(2,
                        "Part-ID=%llu  dt=%g dta=%g dtc=%g dtv=%g dtdis=%g ac=%g xyz=(%g|%g|%g)  hsml=%g  maxcsnd=%g "
                        "dt0=%g eps=%g",
                        (unsigned long long)P[p].ID, dt, dt_acc, dt_courant * All.cf_hubble_a, dt_divv, dt_displacement,
                        ac, (double)P[p].Pos[0], (double)P[p].Pos[1], (double)P[p].Pos[2], (float)P[p].Hsml, csnd,
                        sqrt(2 * All.ErrTolIntAccuracy * All.cf_atime * All.SofteningTable[P[p].Type] / ac) *
                            All.cf_hubble_a,
                        All.SofteningTable[P[p].Type]);
#ifdef NAVIERSTOKES_TIMESTEP
#ifndef NAVIERSTOKES_CONSTANT
            VERBOSE_ALL(2,
                        "NAVIERSTOKES:  dt_NS=%g  ShearViscCoeff=%g A=%g  rho=%g  dotAvisc=%g  dtold=%g, meanpath=%g",
                        dt_NS * All.cf_hubble_a, SphP[p].shear_visc_coeff, SphP[p].Entropy, SphP[p].Density,
                        SphP[p].ViscEntropyChange, (P[p].TimeBin ? (1 << P[p].TimeBin) : 0) * All.Timebase_interval,
                        All.IonMeanFreePath);
            VERBOSE_ALL(2, "    Stressd=(%g|%g|%g) ", SphP[p].StressDiag[0], SphP[p].StressDiag[1],
                        SphP[p].StressDiag[2]);
            VERBOSE_ALL(2, "    Stressoffd=(%g|%g|%g) ", SphP[p].StressOffDiag[0], SphP[p].StressOffDiag[1],
                        SphP[p].StressOffDiag[2]);
#endif // NAVIERSTOKES_CONSTANT
#endif // NAVIERSTOKES_TIMESTEP
        }
        else
        {
            VERBOSE_ALL(2, "Part-ID=%llu  dt=%g ac=%g xyz=(%g|%g|%g)", (unsigned long long)P[p].ID, dt, ac,
                        (double)P[p].Pos[0], (double)P[p].Pos[1], (double)P[p].Pos[2]);
        }
#ifdef NOSTOP_WHEN_BELOW_MINTIMESTEP
        dt = All.MinSizeTimestep;
#endif // NOSTOP_WHEN_BELOW_MINTIMESTEP
    }

    integertime ti_step = (integertime)(dt / All.Timebase_interval);

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    if (ti_step == 0)
    {
        ENDRUN("Error: A timestep of size zero was assigned on the integer timeline!\n"
               "We better stop.\n"
               "Task=%d Part-ID=%llu dt=%g dt_elec=%g dt_cool=%g tibase=%g ti_step=%d ac=%g xyz=(%g|%g|%g) ",
               ThisTask, (unsigned long long)P[p].ID, dt, (float)SphP[p].t_elec, (float)SphP[p].t_cool,
               All.Timebase_interval, (int)ti_step, ac, (double)P[p].Pos[0], (double)P[p].Pos[1], (double)P[p].Pos[2]);
    }

#ifdef GM_STARDENSITY
    if (P[p].Type == type_star)
    {
        VERBOSE_ALL(5, "    Stellar density, hsml: %g %g", P[p].StarDensity, P[p].StarHsml);
    }
#endif // GM_STARDENSITY

#endif // if defined(CHEMISTRY) || defined(UM_CHEMISTRY)

    if (!(ti_step > 0 && ti_step < TIMEBASE))
    {
        VERBOSE_ALL(
            0,
            "\nError: A timestep of size zero was assigned on the integer timeline!\n"
            "We better stop.\n"
            "Task=%d Part-ID=%llu Type=%d mass=%g dt=%g dta=%g dtc=%g dtv=%g dtdis=%g tibase=%g ti_step=%d ac=%g "
            "xyz=(%g|%g|%g) tree=(%g|%g|%g)",
            ThisTask, (unsigned long long)P[p].ID, P[p].Type, (float)P[p].Mass, dt, dt_acc, dt_courant, dt_divv,
            dt_displacement, All.Timebase_interval, ti_step, ac, (double)P[p].Pos[0], (double)P[p].Pos[1],
            (double)P[p].Pos[2], (double)P[p].GravAccel[0], (double)P[p].GravAccel[1], (double)P[p].GravAccel[2]);
#ifdef PMGRID
        VERBOSE_ALL(0, "pm_force=(%g|%g|%g)", (double)P[p].GravPM[0], (double)P[p].GravPM[1], (double)P[p].GravPM[2]);
#endif // PMGRID

        if (P[p].Type == type_gas)
        {
            VERBOSE_ALL(0, "hydro-frc=(%g|%g|%g) dens=%g hsml=%g  dTdE=%g", (double)SphP[p].HydroAccel[0],
                        (double)SphP[p].HydroAccel[1], (double)SphP[p].HydroAccel[2], (double)SphP[p].Density,
                        (float)P[p].Hsml, (float)SphP[p].DtEntropy);

        }

#ifdef GM_STARDENSITY
        if (P[p].Type == type_star)
        {
            VERBOSE_ALL(0, "    Stellar density, hsml: %g %g", P[p].StarDensity, P[p].StarHsml);
        }
#endif // GM_STARDENSITY

#ifdef BLACK_HOLES
        if (P[p].Type == type_bh)
        {
            VERBOSE_ALL(0,
                        "BH: mass = %g, md = %g, prog=%d, dens=%g entr=%g v=(%g,%g,%g) macc=%g mBHacc=%e p=(%g,%g,%g)",
                        BPP(p).BH_Mass, BPP(p).BH_Mdot, BPP(p).BH_CountProgs, BPP(p).BH_Density, BPP(p).BH_Entropy,
                        BPP(p).BH_SurroundingGasVel[0], BPP(p).BH_SurroundingGasVel[1], BPP(p).BH_SurroundingGasVel[2],
                        BPP(p).BH_accreted_Mass, BPP(p).BH_accreted_BHMass, BPP(p).BH_accreted_momentum[0],
                        BPP(p).BH_accreted_momentum[1], BPP(p).BH_accreted_momentum[2]);
#ifdef KD_FRICTION_DYNAMIC
            VERBOSE_ALL(0, "BH: v=(%g,%g,%g) dens=%g sigma=%g, bmax=%g", BPP(p).BH_SurroundingVel[0],
                        BPP(p).BH_SurroundingVel[1], BPP(p).BH_SurroundingVel[2], BPP(p).BH_SurroundingDensity,
                        BPP(p).BH_sigma, BPP(p).BH_bmax);
#endif // KD_FRICTION_DYNAMIC
        }
#endif // BLACK_HOLES

#ifdef MAGNETIC
        if (P[p].Type == type_gas)
        {
            VERBOSE_ALL(0, "mhd-frc=(%g|%g|%g) mhd-corr=(%g|%g|%g)", (float)SphP[p].magacc[0], (float)SphP[p].magacc[1],
                        (float)SphP[p].magacc[2], (float)SphP[p].magcorr[0], (float)SphP[p].magcorr[1],
                        (float)SphP[p].magcorr[2]);
        }
#endif // MAGNETIC
        ENDRUN(" Timestep zero !!!");
    }

    return ti_step;
}

/**
 * @brief Find the maximum allowed timestep of the system.
 *
 * This function computes an upper limit ('dt_displacement') to the global timestep of the system based on
 * the rms velocities of particles. For cosmological simulations, the criterion used is that the rms
 * displacement should be at most a fraction MaxRMSDisplacementFac of the mean particle separation. Note that
 * the latter is estimated using the assigned particle masses, separately for each particle type. If comoving
 * integration is not used, the function imposes no constraint on the timestep.
 *
 * @param hfac Should be a^2*H(a).
 */
void find_dt_displacement_constraint(double hfac)
{

    dt_displacement = All.MaxSizeTimestep;

    if (All.ComovingIntegrationOn)
    {
        int count[6];
        long long count_sum[6];
        double v[6], v_sum[6], mim[6], min_mass[6];
        for (int type = 0; type < 6; type++)
        {
            count[type] = 0;
            v[type] = 0;
            mim[type] = 1.0e30;
        }

        // Iterate over all particles to compute the mean particle separation and the rms velocity.
        for (int i = 0; i < NumPart; i++)
        {
            v[P[i].Type] += P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] + P[i].Vel[2] * P[i].Vel[2];
            if (P[i].Mass > 0)
            {
                mim[P[i].Type] = std::min(mim[P[i].Type], double(P[i].Mass));
            }
            count[P[i].Type]++;
        }
        MPI_Allreduce(v, v_sum, 6, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
        MPI_Allreduce(mim, min_mass, 6, MPI_DOUBLE, MPI_MIN, MYMPI_COMM_WORLD);
        sumup_large_ints(6, count, count_sum);

#ifdef SFR
        /* add star and gas particles together to treat them on equal footing, using the original gas particle
           spacing. */
        v_sum[0] += v_sum[4];
        count_sum[0] += count_sum[4];
        v_sum[4] = v_sum[0];
        count_sum[4] = count_sum[0];
#ifdef BLACK_HOLES
        v_sum[0] += v_sum[5];
        count_sum[0] += count_sum[5];
        v_sum[5] = v_sum[0];
        count_sum[5] = count_sum[0];
        min_mass[5] = min_mass[0];
#endif // BLACK_HOLES
#endif // SFR

        for (int type = 0; type < 6; type++)
        {
            double dmean;
            if (count_sum[type] > 0)
            {
                if (type == type_gas || (type == type_star && All.StarformationOn))
                {
                    dmean = pow(min_mass[type] / (All.OmegaBaryon * 3 * All.Hubble * All.Hubble /
                                                  (8 * M_PI * All.GravityConstantInternal)),
                                1.0 / 3);
                }
                else
                {
                    dmean = pow(min_mass[type] / ((All.Omega0 - All.OmegaBaryon) * 3 * All.Hubble * All.Hubble /
                                                  (8 * M_PI * All.GravityConstantInternal)),
                                1.0 / 3);
                }

#ifdef BLACK_HOLES
                if (type == type_bh)
                {
                    dmean = pow(min_mass[type] / (All.OmegaBaryon * 3 * All.Hubble * All.Hubble /
                                                  (8 * M_PI * All.GravityConstantInternal)),
                                1.0 / 3);
                }
#endif // BLACK_HOLES
                double dt = All.MaxRMSDisplacementFac * hfac * dmean / sqrt(v_sum[type] / count_sum[type]);

                double asmth = 0;
#ifdef PMGRID
                asmth = All.Asmth[0];
#ifdef PLACEHIGHRESREGION
                if (((1 << type) & (PLACEHIGHRESREGION)))
                {
                    asmth = All.Asmth[1];
                }
#endif // PLACEHIGHRESREGION
                if (asmth < dmean)
                {
                    dt = All.MaxRMSDisplacementFac * hfac * asmth / sqrt(v_sum[type] / count_sum[type]);
                }
#endif // PMGRID

                VERBOSE(2, "type=%d  dmean=%g asmth=%g minmass=%g a=%g  sqrt(<p^2>)=%g  dlogmax=%g", type, dmean, asmth,
                        min_mass[type], All.Time, sqrt(v_sum[type] / count_sum[type]), dt);

#ifdef NEUTRINOS
                if (type != 2) /* don't constrain the step to the neutrinos */
#endif // NEUTRINOS
                    dt_displacement = std::min(dt_displacement, dt);
            }
        }
        VERBOSE(2, "INTEGRATOR: displacement time constraint: %g  (%g)", dt_displacement, All.MaxSizeTimestep);
    }
}

/**
 * @brief Determine the time-step bin for a given integer time step.
 *
 * This function determines the time-step bin for a given integer time step.
 * Time-step bins are used to categorize time steps for computational purposes.
 *
 * @param ti_step The integer time step.
 * @return The time-step bin for the given time step.
 */
int get_timestep_bin(integertime ti_step)
{
    if (ti_step == 0)
    {
        return 0;
    }

    if (ti_step == 1)
    {
        ENDRUN_MAIN("time-step of integer size 1 not allowed ");
    }

    int bin = -1;
    while (ti_step)
    {
        bin++;
        ti_step >>= 1;
    }
    return bin;
}

#ifdef MAKEGLASS
/**
 * @brief Perform a step in the process of generating a glass distribution of particles.
 *
 * The function calculates the maximum displacement and the sum of squared displacements for all particles,
 * and then adjusts particle positions and velocities accordingly.
 */
void do_glass_making_step(void)
{
    int dispmax = 0, disp2sum = 0;
    for (int i = 0; i < NumPart; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i].GravAccel[j] *= -1;
#ifdef PMGRID
            P[i].GravPM[j] *= -1;
            P[i].GravAccel[j] += P[i].GravPM[j];
            P[i].GravPM[j] = 0;
#endif // PMGRID
        }

        disp = sqrt(P[i].GravAccel[0] * P[i].GravAccel[0] + P[i].GravAccel[1] * P[i].GravAccel[1] +
                    P[i].GravAccel[2] * P[i].GravAccel[2]);

        disp *= 2.0 / (3 * All.Hubble * All.Hubble);

        disp2sum += disp * disp;

        dispmax = std::max(dispmax, disp);
    }

    MPI_Allreduce(&dispmax, &globmax, 1, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);
    MPI_Allreduce(&disp2sum, &globdisp2sum, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    dmean =
        pow(P[0].Mass / (All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal)), 1.0 / 3);

    if (globmax > dmean)
    {
        fac = dmean / globmax;
    }
    else
    {
        fac = 1.0;
    }

    VERBOSE(1, "\nglass-making:  dmean= %g  global disp-maximum= %g  rms= %g ", dmean, globmax,
            sqrt(globdisp2sum / All.TotNumPart));

    for (int i = 0; i < NumPart; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i].Vel[j] = 0;
            P[i].Pos[j] += fac * P[i].GravAccel[j] * 2.0 / (3 * All.Hubble * All.Hubble);
            P[i].GravAccel[j] = 0;
        }
    }
}
#endif // MAKEGLASS

#ifdef RELAXOBJECT
/**
 * @brief Determine the relaxation factor based on the current simulation time.
 *
 * This function determines the relaxation factor (All.RelaxFac) based on the current simulation time (All.Time).
 * The relaxation factor is used in certain simulation calculations to control relaxation effects.
 *
 * The relaxation factor is determined as follows:
 * - If the current time (All.Time) is less than 0.2 times the maximum time (All.TimeMax), the relaxation factor is set
 * to 1 divided by All.RelaxBaseFac.
 * - If the current time is greater than 0.8 times the maximum time, the relaxation factor is set to 0.
 * - For other times, the relaxation factor is calculated using a power-law function with exponential scaling.
 */
void determine_relaxfac(void)
{
    if (All.Time < 0.2 * All.TimeMax)
    {
        All.RelaxFac = 1. / All.RelaxBaseFac;
    }
    else
    {
        if (All.Time > 0.8 * All.TimeMax)
        {
            All.RelaxFac = 0.;
        }
        else
        {
            All.RelaxFac =
                1. / (All.RelaxBaseFac * pow(10., (All.Time - 0.2 * All.TimeMax) / (0.6 * All.TimeMax) * 3.));
        }
    }
}
#endif // RELAXOBJECT

/**
 * @brief Process wake-up of particles based on wake-up list.
 *
 * This function processes the wake-up of particles based on the wake-up list.
 * It adjusts particle properties and updates time bins and active particle list accordingly.
 */
void process_wake_ups_new(void)
{
    TIMER_PUSH(1, "WAKEUP");
    long long ntowakeup = 0;
    sumup_large_ints(1, &N_ToWakeUp, &ntowakeup);
    VERBOSE(2, "%lld particles might need to be woken up.", ntowakeup);

    int n = 0;
    //  VERBOSE(5,"ThisTask%d wakup ",ThisTask);
#if defined(LT_STELLAREVOLUTION) && defined(LT_SN_DEBUG)
    double sfr_local = 0;
#endif

    // This part will get obsolete as soon as we switch generally to the active particle list
    int LastActive, prev;
    // int appended_particles = 0;
    // int alreadyactive_particles = 0;

    LastActive = prev = FirstActiveParticle;
    while (LastActive >= 0)
    {
        prev = LastActive;
        LastActive = NextActiveParticle[LastActive];
    }

    LastActive = prev;

    // This loop can not be OpenMP paralelized, as particles could appear multiple times in the list
    // We also need to take the latest apperance in the list
    for (int iwup = N_ToWakeUp - 1; iwup >= 0; iwup--)
    {
        int i = ToWakeupList[iwup];
#ifdef ACC_HYDRA
        if (i == -1) // the ACC_HYDRA masks ToWakeupList to -1 when copying the array from GPU to CPU if a particle was
                     // woken up from CPU already
            continue;
#endif
        assert(P[i].Type == type_gas); // Only Gas particles should be woken up
        if (SPHP(i, wakeup) == TIMEBINS)
            continue;
        if (!IS_SPH_ALIVE(P[i]))
        {
            if (P[i].Type == type_gas)
            {
                SPHP(i, wakeup) = TIMEBINS;
            }
            continue;
        }
        else
        {
            assert(P[i].Type == type_gas); // Paranoic check ??
        }

#ifdef WINDS // wind particles do not wakeup, yet they should stay in the hydroloop to keep their MaxWindVelocity
             // updated for when they'll wake up
        assert(P[i].Type == type_gas); // Paranoic check ??
        if (SphP[i].DelayTime > 0.)
        {
            SPHP(i, wakeup) = TIMEBINS; // Do not wake up wid particles
            continue;
        }
#endif // WINDS

        int binold = P[i].TimeBin;
        int binnew = SPHP(i, wakeup);
        assert(TimeBinActive[binnew]); // We should have been always woken up by active particle !

#if defined(LT_STELLAREVOLUTION) && defined(LT_SN_DEBUG)
#pragma omp atomic
        sfr_local += SphP[i].Sfr;
#endif // LT_STELLAREVOLUTION

        // We keep the particle marked as woken up, so that in find timestep we do not compute a new one
        P[i].TimeBin = binnew;
        P[i].Ti_begstep = P[i].Ti_current;
        VERBOSE_ALL(5, "Task %d (WAKEUP DEBUG): old=%d new=%d", ThisTask, binold, binnew);

#pragma omp critical(_wakeup_)
        {
            TimeBinCount[binold]--;
            TimeBinCountSph[binold]--;

#ifdef LT_STELLAREVOLUTION
            TimeBinSfr[binold] -= SphP[i].Sfr;
            TimeBinSfr[binnew] += SphP[i].Sfr;
            // GM: this fix the double-count for the Sfr of the woken-up particles
#endif // LT_STELLAREVOLUTION

            // Now fix the timebin linked list
            int prev = PrevInTimeBin[i];
            int next = NextInTimeBin[i];

            if (FirstInTimeBin[binold] == i)
            {
                FirstInTimeBin[binold] = next;
            }
            if (LastInTimeBin[binold] == i)
            {
                LastInTimeBin[binold] = prev;
            }
            if (prev >= 0)
            {
                NextInTimeBin[prev] = next;
            }
            if (next >= 0)
            {
                PrevInTimeBin[next] = prev;
            }
            if (TimeBinCount[binnew] > 0)
            {
                PrevInTimeBin[i] = LastInTimeBin[binnew];
                NextInTimeBin[LastInTimeBin[binnew]] = i;
                NextInTimeBin[i] = -1;
                LastInTimeBin[binnew] = i;
            }
            else
            {
                FirstInTimeBin[binnew] = LastInTimeBin[binnew] = i;
                PrevInTimeBin[i] = NextInTimeBin[i] = -1;
            }
            TimeBinCount[binnew]++;
            TimeBinCountSph[binnew]++;

            NumForceUpdate++;

            // Need to fix particles in linked list as well as in active list
            if (!TimeBinActive[binold])
            {

#ifdef _OPENMP
                ActiveParticleList[NActivePart] = i;
                NActivePart++;
#endif
                if (LastActive > 0)
                {
                    NextActiveParticle[LastActive] = i;
                    LastActive = i;
                    NextActiveParticle[LastActive] = -1;
                }
                else
                {
                    FirstActiveParticle = LastActive = i;
                    NextActiveParticle[i] = -1;
                }
            }

        } // omp critical

#ifdef SPH_BND_PARTICLES
        if (P[i].ID != 0) // this should not be necessary
#endif // SPH_BND_PARTICLES
        {
            // Now correct al kicked values
            for (int k = 0; k < 3; k++)
            {
                P[i].Vel[k] = SphP[i].VelPred[k];
#ifdef MAGNETIC
                SphP[i].B[k] = SphP[i].BPred[k];
#endif // MAGNETIC
            }
            SphP[i].Entropy = SphP[i].EntropyPred;
#if GADGET_HYDRO == HYDRO_MFM
            SphP[i].InternalEnergy = SphP[i].InternalEnergyPred;
#endif // GADGET_HYDRO == HYDRO_MFM
            SphP[i].Pressure = get_pressure(i);
        }
#pragma omp atomic
        n++;
    }

    long long ntot = 0;
    sumup_large_ints(1, &n, &ntot);

#if defined(LT_STELLAREVOLUTION) && defined(LT_SN_DEBUG)
    double sfr_total;
    MPI_Allreduce(&sfr_local, &sfr_total, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    VERBOSE(2, "%d%09d particles woken up, total sfr = %g.", (int)(ntot / 1000000000), (int)(ntot % 1000000000),
            sfr_total);

#else // LT_STELLAREVOLUTION
    VERBOSE(2, "%lld particles woken up.", ntot);
#endif // LT_STELLAREVOLUTION

    N_ToWakeUp = 0;
    TIMER_POP(1, "WAKEUP");
}
