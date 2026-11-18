/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains functions related to the different star formation modes.
 *
 * @author OpenGadget3-development team, in particular Cecilia Scannapieco, Giuseppe Murante, Volker Springel and Luca
 * Tornatore.
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
#ifdef MAGNETIC_SN_SEEDING
#include "../Hydro/mhd_seeding.hpp"
#endif // MAGNETIC_SN_SEEDING

#if !defined(LT_STELLAREVOLUTION) && !defined(GM_MUPPI)
#ifdef COOLING

/*
 * This routine does cooling and star formation for
 * the effective multi-phase model.
 */

#ifdef SFR

double cold_function(double p, double delta)
{
    float g;
    g = 1 + sqrt(p) / 2 / delta - sqrt(sqrt(p) / delta + p / (4 * delta * delta)) - p;
    return g;
}

double find_root_x(double delta)
{
    double a, b, fa, fb, xx, yy, root;

    a = 0.;
    b = 1.;
    int mycount = 0;
    fa = cold_function(a, delta);
    fb = cold_function(b, delta);
    if (fa * fb >= 0) /* No root, should never happen ! */
    {
        if (abs(fa) < 1e-3)
        {
            VERBOSE(3, "Task=%d: WARNING, root at beginning of range: a=%.5f, b=%.5f, fa=%.5f, fb=%.5f ", ThisTask, a,
                    b, fa, fb);
            b = a;
        }
        else if (abs(fb) < 1e-3)
        {
            VERBOSE(3, "Task=%d: WARNING, root at end of range: a=%.5f, b=%.5f, fa=%.5f, fb=%.5f ", ThisTask, a, b, fa,
                    fb);
            a = b;
        }
        else
        {
            ENDRUN_MAIN("Task=%d: ERROR, no root possible: a=%.5f, b=%.5f, fa=%.5f, fb=%.5f ", ThisTask, a, b, fa, fb);
        }
    }

    while (abs(a - b) > 1e-5 && mycount <= 1000)
    {
        xx = (a + b) / 2.;
        yy = cold_function(xx, delta);
        if ((yy >= 0 && fa >= 0) || (yy <= 0 && fa <= 0))
        {
            a = xx;
            fa = yy;
        }
        else
        {
            b = xx;
            fb = yy;
        }
        mycount++;
    }
    root = (a + b) / 2.;
    return root;
}

double calculate_dens_threshold(int i)
{
    double ne, tcool, egyhot, meanweight, u4, x, A0, PhysDensThresh, P4, f_mol4;
    double Po, t_dyn_int, dens, lambda;
    double f_star = 0.1;

    A0 = All.FactorEVP;
    egyhot = All.EgySpecSN / A0;
    ne = SphP[i].elec;

    dens = 1.0e5 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);
    tcool = GetCoolingTimeSTD(egyhot, dens, &ne);
    lambda = egyhot / tcool / dens;

    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4; // cgs
    P4 = u4 * dens * GAMMA_MINUS1 * All.UnitDensity_in_cgs;
    u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
    Po = 35000 * BOLTZMANN;
    f_mol4 = 1 / (1 + Po / P4);

    x = (egyhot - u4) / (egyhot - All.EgySpecCold);
    t_dyn_int = 1 / sqrt(x) * sqrt(3 * M_PI / 32 / All.GravityConstantInternal / dens);

    PhysDensThresh = x / pow(1 - x, 2) * f_star * f_mol4 *
                     (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold) / t_dyn_int / lambda;

    return PhysDensThresh;
}

void cooling_and_starformation(void)
/* cooling routine when star formation is enabled */
{
    int i, flag, stars_spawned, tot_spawned, stars_converted, tot_converted, number_of_stars_generated;
    unsigned int bits;
    double dt, dtime, hubble_a = 0, a3inv, ne = 1;
    double unew, mass_of_star;
    double sum_sm, total_sm, sm, rate, sum_mass_stars, total_sum_mass_stars;
    double p, prob;
    double cloudmass;
    double factorEVP;
    double tsfr, trelax;
    double egyhot, egyeff, egycurrent, tcool, x, y, rate_in_msunperyear;
    double totsfrrate;

#ifdef WINDS
    int j;
    double v;
    double norm, dir[3];

#ifdef ISOTROPICWINDS
    double theta, phi;
#endif // ISOTROPICWINDS
#endif // WINDS

#ifdef METALS
    double w;
#endif // METALS

#if defined(QUICK_LYALPHA) || defined(BLACK_HOLES)
    double temp;
#endif // QUICK_LYALPHA || BLACK_HOLES
    double u_to_temp_fac;

    u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * GAMMA_MINUS1 *
                    All.UnitEnergy_in_cgs / All.UnitMass_in_g;

    for (int bin = 0; bin < TIMEBINS; bin++)
    {
        if (TimeBinActive[bin])
        {
            TimeBinSfr[bin] = 0;
        }
    }

    if (All.ComovingIntegrationOn)
    {
        /* Factors for comoving integration of hydro */
        a3inv = 1 / (All.Time * All.Time * All.Time);
        hubble_a = hubble_function(All.Time);
    }
    else
    {
        a3inv = hubble_a = 1;
    }

    stars_spawned = stars_converted = 0;
    sum_sm = sum_mass_stars = 0;

    for (bits = 0; All.Generations > (1 << bits); bits++)
        ;

    for (i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if (P[i].Type == 0)
        {
            dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
            /*  the actual time-step */

            if (All.ComovingIntegrationOn)
            {
                dtime = dt * units->timestep_to_physical;
            }
            else
            {
                dtime = dt;
            }

            /* check whether conditions for star formation are fulfilled.
             *
             * f=1  normal cooling
             * f=0  star formation
             */
            flag = 1; /* default is normal cooling */

            if (SphP[i].Density * a3inv >= All.PhysDensThresh)
            {
                flag = 0;
            }

            if (All.ComovingIntegrationOn)
            {
                if (SphP[i].Density < All.OverDensThresh)
                {
                    flag = 1;
                }
            }

#ifdef BLACK_HOLES
            if (P[i].Mass == 0)
            {
                flag = 1;
            }
#endif // BLACK_HOLES

#ifdef WINDS
            if (SphP[i].DelayTime > 0)
            {
                flag = 1; /* only normal cooling for particles in the wind */
            }
            if (SphP[i].DelayTime > 0)
            {
                SphP[i].DelayTime -= dtime;
            }

            if (SphP[i].DelayTime > 0)
            {
#ifdef QUICK_LYALPHA
                if (SphP[i].Density < All.WindFreeTravelDensFac * All.OverDensThresh)
                {
                    SphP[i].DelayTime = 0;
#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING
                    // start only in next round with cooling/sfr as particle just left the wind phase
                    continue;
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING
                }
#else // QUICK_LYALPHA
                if (SphP[i].Density * a3inv < All.WindFreeTravelDensFac * All.PhysDensThresh)
                {
                    SphP[i].DelayTime = 0;
#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING
                    // start only in next round with cooling/sfr as particle just left the wind phase
                    continue;
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING
                }
#ifdef LS_SF_RUNAWAY_TREATMENT
                if (SphP[i].Density * a3inv > All.RunawayThresh)
                {
                    SphP[i].DelayTime = 0;
#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING
                    // start only in next round with cooling/sfr as particle just left the wind phase
                    continue;
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING
                }
#endif
#endif // QUICK_LYALPHA
            }
            if (SphP[i].DelayTime < 0)
            {
                SphP[i].DelayTime = 0;
#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING
                // start only in next round with cooling/sfr as particle just left the wind phase
                continue;
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING
            }

#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING
            if (SphP[i].DelayTime > 0)
                continue; /* neglect cooling in winds */
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING

#endif // WINDS

#ifdef QUICK_LYALPHA
            temp = u_to_temp_fac * (SphP[i].Entropy) / GAMMA_MINUS1 * pow(SphP[i].Density * a3inv, GAMMA_MINUS1);

            if (SphP[i].Density > All.OverDensThresh && temp < 1.0e5)
            {
                flag = 0;
            }
            else
            {
                flag = 1;
            }
#ifdef WINDS
            if (SphP[i].DelayTime > 0)
            {
                flag = 1; /* only normal cooling for particles in the wind */
            }
#endif // WINDS
#endif // QUICK_LYALPHA

#ifdef MAGNETIC
            x = 0.;
#endif // MAGNETIC

#ifdef MAGNETIC_SN_SEEDING
            SphP[i].MagSeedRate[0] = SphP[i].MagSeedRate[1] = SphP[i].MagSeedRate[2] = 0.0;
#endif // MAGNETIC_SN_SEEDING




#if !defined(NOISMPRESSURE) && !defined(QUICK_LYALPHA)
            if (flag == 1) /* normal implicit isochoric cooling */
#endif // !defined(NOISMPRESSURE) && !defined(QUICK_LYALPHA)
            {
                SphP[i].Sfr = 0;
                ne = SphP[i].elec; /* electron abundance (gives ionization state and mean molecular weight) */
                double fac_entr_to_u = pow(SphP[i].Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
                double uold = max(All.MinEgySpec, SphP[i].Entropy * fac_entr_to_u);

#ifndef RT_COOLING_PHOTOHEATING
                unew = DoCoolingSTD(uold, SphP[i].Density * a3inv, dtime, &ne);
#else // RT_COOLING_PHOTOHEATING
                unew = uold + dt * fac_entr_to_u * (rt_DoHeating(i, dt) + rt_DoCooling(i, dt));
#endif // RT_COOLING_PHOTOHEATING

#ifdef UM_CHEMISTRY
                unew = Um_DoCooling(uold, SphP[i].Density * a3inv, dtime, &ne, i, flag);
#endif // UM_CHEMISTRY

#ifdef BLACK_HOLES
                if (All.BlackHoleThermalFeedbackOn == 1 || All.BlackHoleKineticFeedbackOn == 1)
                    if (SphP[i].Injected_BH_Energy)
                    {
                        if (P[i].Mass == 0)
                        {
                            SphP[i].Injected_BH_Energy = 0;
                        }
                        else
                        {
                            unew += SphP[i].Injected_BH_Energy / P[i].Mass;
                        }

                        temp = u_to_temp_fac * unew;

                        if (temp > 5.0e9)
                        {
                            unew = 5.0e9 / u_to_temp_fac;
                        }

                        SphP[i].Injected_BH_Energy = 0;
                    }
#endif // BLACK_HOLES

#if GADGET_HYDRO == HYDRO_MFM
                SphP[i].InternalEnergy = unew;
#endif // GADGET_HYDRO == HYDRO_MFM
                SphP[i].Entropy = unew / fac_entr_to_u;
                SphP[i].elec = ne;

#if GADGET_HYDRO == HYDRO_MFM
                SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
                mfm_synchronize_entropy(i);
#endif // GADGET_HYDRO == HYDRO_MFM
                SphP[i].EntropyPred = SphP[i].Entropy;
                SphP[i].Pressure = get_pressure(i);
            }

            if (flag == 0) /* active star formation */
            {
#ifndef QUICK_LYALPHA
                factorEVP = pow(SphP[i].Density * a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;
                egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

                ne = SphP[i].elec;
#ifndef UM_CHEMISTRY
                tcool = GetCoolingTimeSTD(egyhot, SphP[i].Density * a3inv, &ne);
#else // UM_CHEMISTRY
                tcool = Um_GetCoolingTime(egyhot, SphP[i].Density * a3inv, &ne, i);
#endif // UM_CHEMISTRY

                SphP[i].elec = ne;

                tsfr = sqrt(All.PhysDensThresh / (SphP[i].Density * a3inv)) * All.MaxSfrTimescale;
                y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);

                x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

                egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

#ifdef MAGNETIC_SFR_EGY
                egyeff += (SphP[i].BPred[0] * SphP[i].BPred[0] + SphP[i].BPred[1] * SphP[i].BPred[1] +
                           SphP[i].BPred[2] * SphP[i].BPred[2]) /
                          (2. * mu0) * x;
#endif // MAGNETIC_SFR_EGY

                cloudmass = x * P[i].Mass;

                if (tsfr < dtime)
                {
                    tsfr = dtime;
                }
                sm = (1 - All.FactorSN) * dtime / tsfr * cloudmass; /* amount of stars expect to form */
                p = sm / P[i].Mass;
                sum_sm += P[i].Mass * (1 - exp(-p));

                if (dt > 0)
                {
                    if (P[i].TimeBin) /* upon start-up, we need to protect against dt==0 */
                    {
                        trelax = tsfr * (1 - x) / x / (All.FactorSN * (1 + factorEVP));
                        egycurrent = SphP[i].Entropy * pow(SphP[i].Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;

#ifdef BLACK_HOLES
                        if (All.BlackHoleThermalFeedbackOn == 1 || All.BlackHoleKineticFeedbackOn == 1)
                            if (SphP[i].Injected_BH_Energy > 0)
                            {
                                egycurrent += SphP[i].Injected_BH_Energy / P[i].Mass;
                                temp = u_to_temp_fac * egycurrent;

                                if (temp > 5.0e9)
                                {
                                    egycurrent = 5.0e9 / u_to_temp_fac;
                                }

                                if (egycurrent > egyeff)
                                {
                                    tcool = GetCoolingTimeSTD(egycurrent, SphP[i].Density * a3inv, &ne);

                                    if (tcool < trelax && tcool > 0)
                                    {
                                        trelax = tcool;
                                    }
                                }

                                SphP[i].Injected_BH_Energy = 0;
                            }
#endif // BLACK_HOLES

#ifdef MAGNETIC_SN_SEEDING

                            // supernova rate per timestep
                        double sn_eff = sqrt(0.008 * cloudmass * All.UnitMass_in_g / SOLAR_MASS * dtime / tsfr);

                        if (sn_eff < 1.0)
                        {
                            if (get_random_number(P[i].ID + 6) <= sn_eff * sn_eff)
                            {
                                sn_eff = 1.0;
                            }
                            else
                            {
                                sn_eff = 0.0;
                            }
                        }

                        // compute magnetic field contribution from supernova seeding (in mhd_seeding.hpp)
                        calculate_magseedrate_sn(i, sn_eff, dtime);


#endif // MAGNETIC_SN_SEEDING

#ifndef NOISMPRESSURE

#if GADGET_HYDRO == HYDRO_MFM
                        SphP[i].InternalEnergy = egyeff + (egycurrent - egyeff) * exp(-dtime / trelax);
#endif // GADGET_HYDRO == HYDRO_MFM
#ifdef NO_SLOW_RELAX_TO_EOS
                        SphP[i].Entropy = egyeff * GAMMA_MINUS1 / pow(SphP[i].Density * a3inv, GAMMA_MINUS1);
#else // NO_SLOW_RELAX_TO_EOS
                        SphP[i].Entropy = (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax)) * GAMMA_MINUS1 /
                                          pow(SphP[i].Density * a3inv, GAMMA_MINUS1);
#endif // NO_SLOW_RELAX_TO_EOS

#if GADGET_HYDRO == HYDRO_MFM
                        MyFloat div_vel = limited_divvel(i);
                        SphP[i].dQdt[NUMDIMS + 1] = eos->GetDtQFromDtEntropy(
                            (MyFloat)0.0, SphP[i].Entropy, SphP[i].InternalEnergy, P[i].Mass, div_vel, All.cf_hubble_a);
                        SphP[i].DtInternalEnergy =
                            eos->GetDtUFromDtEntropy((MyFloat)0.0, SphP[i].Entropy, SphP[i].InternalEnergy, div_vel,
                                                     All.cf_hubble_a, All.ComovingIntegrationOn);
#endif // GADGET_HYDRO == HYDRO_MFM
                        SphP[i].DtEntropy = 0;
#endif // NOISMPRESSURE

#if GADGET_HYDRO == HYDRO_MFM
                        SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
                        mfm_synchronize_entropy(i);
#endif // GADGET_HYDRO == HYDRO_MFM
                        SphP[i].EntropyPred = SphP[i].Entropy;
                        SphP[i].Pressure = get_pressure(i);
                    }
                }

                /* the upper bits of the gas particle ID store how man stars this gas
                   particle gas already generated */

                if (bits == 0)
                {
                    number_of_stars_generated = 0;
                }
                else
                {
                    number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType) * 8 - bits));
                }

                mass_of_star = P[i].Mass / (All.Generations - number_of_stars_generated);

                SphP[i].Sfr = (1 - All.FactorSN) * cloudmass / tsfr * (All.UnitMass_in_g / SOLAR_MASS) /
                              (All.UnitTime_in_s / SEC_PER_YEAR);

                TimeBinSfr[P[i].TimeBin] += SphP[i].Sfr;
                double w = get_random_number(P[i].ID);
                MPP(i).Metallicity += w * METAL_YIELD * (1 - exp(-p));

                prob = P[i].Mass / mass_of_star * (1 - exp(-p));

#ifdef LS_SF_RUNAWAY_TREATMENT
                if ((SphP[i].Density * a3inv >= All.RunawayThresh) && (SphP[i].DelayTime <= 0))
                {
                    prob = 2.0;
                }
#endif // LS_SF_RUNAWAY_TREATMENT

#else /* belongs to ifndef(QUICK_LYALPHA) */

                prob = 2.0; /* this will always cause a star creation event */
                if (bits == 0)
                {
                    number_of_stars_generated = 0;
                }
                else
                {
                    number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType) * 8 - bits));
                }

                mass_of_star = P[i].Mass / (All.Generations - number_of_stars_generated);

                SphP[i].Sfr = 0;

#ifdef QUICK_LYALPHA_FEEDBACK
                egyhot = All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) * mass_of_star;
                sm = mass_of_star;
#endif // QUICK_LYALPHA_FEEDBACK

#endif //  QUICK_LYALPHA

#ifdef DO_NOT_CREATE_STAR_PARTICLES
                prob = -1;
#endif // DO_NOT_CREATE_STAR_PARTICLES

                if (get_random_number(P[i].ID + 1) < prob) /* ok, make a star */
                {
                    if (number_of_stars_generated == (All.Generations - 1))
                    {
                        /* here we turn the gas particle itself into a star */
                        Particles_converted++;
                        stars_converted++;

                        sum_mass_stars += P[i].Mass;

                        P[i].Type = 4;

                        P[i].pt.MetID = N_stars;
                        MetP[N_stars].PID = i;

                        TimeBinCountSph[P[i].TimeBin]--;
                        TimeBinSfr[P[i].TimeBin] -= SphP[i].Sfr;
                        MPP(i).StellarAge = All.Time;

                        N_stars++;
                    }
                    else
                    {
                        /* here we spawn a new star particle */
                        ENDRUN_IF(NumPart + stars_spawned >= All.MaxPart,
                                  "On Task=%d with NumPart=%d we try to spawn %d particles. Sorry, no space "
                                  "left...(All.MaxPart=%d)",
                                  ThisTask, NumPart, stars_spawned, All.MaxPart);

                        P[NumPart + stars_spawned] = P[i];
                        P[NumPart + stars_spawned].Type = 4;

                        NextActiveParticle[NumPart + stars_spawned] = FirstActiveParticle;
                        FirstActiveParticle = NumPart + stars_spawned;
                        NumForceUpdate++;

                        TimeBinCount[P[NumPart + stars_spawned].TimeBin]++;

                        PrevInTimeBin[NumPart + stars_spawned] = i;
                        NextInTimeBin[NumPart + stars_spawned] = NextInTimeBin[i];
                        if (NextInTimeBin[i] >= 0)
                        {
                            PrevInTimeBin[NextInTimeBin[i]] = NumPart + stars_spawned;
                        }
                        NextInTimeBin[i] = NumPart + stars_spawned;
                        if (LastInTimeBin[P[i].TimeBin] == i)
                        {
                            LastInTimeBin[P[i].TimeBin] = NumPart + stars_spawned;
                        }

                        P[i].ID += ((MyIDType)1 << (sizeof(MyIDType) * 8 - bits));

                        P[NumPart + stars_spawned].Mass = mass_of_star;
#if GADGET_HYDRO == HYDRO_MFM
                        mfm_add_mass(i, -P[NumPart + stars_spawned].Mass);
#endif // GADGET_HYDRO == HYDRO_MFM
                        P[i].Mass -= P[NumPart + stars_spawned].Mass;
                        sum_mass_stars += P[NumPart + stars_spawned].Mass;

                        P[NumPart + stars_spawned].pt.MetID = N_stars;
                        MetP[N_stars].PID = NumPart + stars_spawned;

                        MPP(NumPart + stars_spawned).StellarAge = All.Time;

                        N_stars++;

                        force_add_star_to_tree(i, NumPart + stars_spawned);
                        stars_spawned++;
                    }
                }

                if (P[i].Type == 0) /* to protect using a particle that has been turned into a star */
                {
                    MPP(i).Metallicity += (1 - w) * METAL_YIELD * (1 - exp(-p));
                }

#ifdef WINDS
                /* Here comes the wind model */

                if (P[i].Type == 0) /* to protect using a particle that has been turned into a star */
                {
                    p = All.WindEfficiency * sm / P[i].Mass;

                    prob = 1 - exp(-p);

                    if (get_random_number(P[i].ID + 2) < prob) /* ok, make the particle go into the wind */
                    {
#if WINDS + 0 > 0
                        v = WINDS;
#else
                        v = sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) /
                                 All.WindEfficiency);
#endif

#ifdef ISOTROPICWINDS
                        theta = acos(2 * get_random_number(P[i].ID + 3) - 1);
                        phi = 2 * M_PI * get_random_number(P[i].ID + 4);

                        dir[0] = sin(theta) * cos(phi);
                        dir[1] = sin(theta) * sin(phi);
                        dir[2] = cos(theta);
#else // ISOTROPICWINDS
                        dir[0] = P[i].GravAccel[1] * P[i].Vel[2] - P[i].GravAccel[2] * P[i].Vel[1];
                        dir[1] = P[i].GravAccel[2] * P[i].Vel[0] - P[i].GravAccel[0] * P[i].Vel[2];
                        dir[2] = P[i].GravAccel[0] * P[i].Vel[1] - P[i].GravAccel[1] * P[i].Vel[0];
#endif // ISOTROPICWINDS

                        for (j = 0, norm = 0; j < 3; j++)
                        {
                            norm += dir[j] * dir[j];
                        }

                        norm = sqrt(norm);
                        if (get_random_number(P[i].ID + 5) < 0.5)
                        {
                            norm = -norm;
                        }

                        if (norm != 0)
                        {
                            for (int j = 0; j < 3; j++)
                            {
                                dir[j] /= norm;
                            }

                            for (int j = 0; j < 3; j++)
                            {
                                P[i].Vel[j] += v * units->velocity_to_comoving * dir[j];
                                SphP[i].VelPred[j] += v * units->velocity_to_comoving * dir[j];
                            }

                            SphP[i].DelayTime = All.WindFreeTravelMaxTimeFactor / hubble_a;
                        }
#ifdef MAGNETIC
                        x = 0.;
#endif // MAGNETIC
                    }
                }
#endif // WINDS
            }
#ifdef MAGNETIC
            SphP[i].XColdCloud = x;
#endif // MAGNETIC
        } /* End of If Type = 0 */
    } /* end of main loop over active particles */

    MPI_Allreduce(&stars_spawned, &tot_spawned, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&stars_converted, &tot_converted, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    if (tot_spawned > 0 || tot_converted > 0)
    {
        VERBOSE(2, "SFR: spawned %d stars, converted %d gas particles into stars", tot_spawned, tot_converted);

        All.TotNumPart += tot_spawned;
        All.TotN_gas -= tot_converted;
        NumPart += stars_spawned;
        All.TotN_stars += tot_spawned + tot_converted;
        /* Note: N_gas is only reduced once rearrange_particle_sequence is called */

        /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    }

#ifdef _OPENMP
    if (stars_spawned > 0)
    {
        build_active_particle_list();
    }
#endif // _OPENMP

    double sfrrate = 0;
    for (int bin = 0; bin < TIMEBINS; bin++)
    {
        if (TimeBinCount[bin])
        {
            sfrrate += TimeBinSfr[bin];
        }
    }

    MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    MPI_Reduce(&sum_sm, &total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars, &total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    if (ThisTask == 0)
    {
        if (All.TimeStep > 0)
        {
            rate = total_sm / (All.TimeStep * units->timestep_to_physical);
        }
        else
        {
            rate = 0;
        }

        /* convert to solar masses per yr */

        rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        fprintf(FdSfr, "%g %g %g %g %g\n", All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars);
        fflush(FdSfr);
    }
}

double get_starformation_rate(int i)
{
    double rateOfSF;
    double a3inv;
    int flag;
    double tsfr;
    double factorEVP, egyhot, ne, tcool, y, x, cloudmass;

#ifdef WINDS
    if (SphP[i].DelayTime > 0)
    {
        return 0;
    }
#endif // WINDS

    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1;
    }

    flag = 1; /* default is normal cooling */

    if (SphP[i].Density * a3inv >= All.PhysDensThresh)
    {
        flag = 0;
    }

    if (All.ComovingIntegrationOn)
    {
        if (SphP[i].Density < All.OverDensThresh)
        {
            flag = 1;
        }
    }

    if (flag == 1)
    {
        return 0;
    }

    factorEVP = pow(SphP[i].Density * a3inv / All.PhysDensThresh, -0.8) * All.FactorEVP;

    egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

    ne = SphP[i].elec;
#ifndef UM_CHEMISTRY
    tcool = GetCoolingTimeSTD(egyhot, SphP[i].Density * a3inv, &ne);
#else // UM_CHEMISTRY
    tcool = Um_GetCoolingTime(egyhot, SphP[i].Density * a3inv, &ne, i);
#endif // UM_CHEMISTRY

    tsfr = sqrt(All.PhysDensThresh / (SphP[i].Density * a3inv)) * All.MaxSfrTimescale;
    y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    cloudmass = x * P[i].Mass;

    rateOfSF = (1 - All.FactorSN) * cloudmass / tsfr;

    /* convert to solar masses per yr */
    rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
    return rateOfSF;
}
#endif // SFR

#ifdef SFR
void init_clouds(void)
{
    double A0, dens, tcool, ne, coolrate, egyhot, x, u4, meanweight;
    double tsfr, y, peff, fac, neff, egyeff, factorEVP, sigma, thresholdStarburst;
    double e_unit, f_mol_eff, Po_int;

    if (All.PhysDensThresh == 0)
    {
        A0 = All.FactorEVP;

        egyhot = All.EgySpecSN / A0;

        meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)); /* note: assuming FULL ionization */

        u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
        u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
        e_unit = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);

        dens = 1.0e6 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);

        if (All.ComovingIntegrationOn)
        {
            All.Time = 1.0; /* to be guaranteed to get z=0 rate */
            set_cosmo_factors_for_current_time();
            IonizeParams();
        }

        ne = 1.0;
        SetZeroIonization();
        tcool = GetCoolingTimeSTD(egyhot, dens, &ne);

        coolrate = egyhot / tcool / dens;

        x = (egyhot - u4) / (egyhot - All.EgySpecCold);
        All.PhysDensThresh = x / pow(1 - x, 2) * (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold) /
                             (All.MaxSfrTimescale * coolrate);

        VERBOSE(3, "\nA0= %g  ", A0);
        VERBOSE(3, "Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3", All.PhysDensThresh,
                All.PhysDensThresh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs));
        VERBOSE(3, "EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n", x);
        VERBOSE(3, "tcool=%g dens=%g egyhot=%g", tcool, dens, egyhot);

        dens = All.PhysDensThresh * 10;

        do
        {
            tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
            factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
            egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

            ne = 0.5;
            tcool = GetCoolingTimeSTD(egyhot, dens, &ne);
            y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

            egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

            peff = GAMMA_MINUS1 * dens * egyeff;

            fac = 1 / (log(dens * 1.025) - log(dens));
            dens *= 1.025;

            neff = -log(peff) * fac;

            tsfr = sqrt(All.PhysDensThresh / (dens)) * All.MaxSfrTimescale;
            factorEVP = pow(dens / All.PhysDensThresh, -0.8) * All.FactorEVP;
            egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;
            f_mol_eff = 1 / (1 + Po_int / peff);

            ne = 0.5;
            tcool = GetCoolingTimeSTD(egyhot, dens, &ne);
            y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

            egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

            peff = GAMMA_MINUS1 * dens * egyeff;

            neff += log(peff) * fac;
        } while (neff > 4.0 / 3);

        thresholdStarburst = dens;
#ifdef LS_SF_RUNAWAY_TREATMENT
        All.RunawayThresh = thresholdStarburst;
#endif

        VERBOSE(3, "Run-away sets in for dens=%g", thresholdStarburst);
        VERBOSE(3, "Dynamic range for quiescent star formation= %g", thresholdStarburst / All.PhysDensThresh);

        integrate_sfr();

        if (ThisTask == 0)
        {
            sigma = 10.0 / All.Hubble * 1.0e-10 / pow(1.0e-3, 2);

            VERBOSE(3, "Isotherm sheet central density: %g   z0=%g\n",
                    M_PI * All.GravityConstantInternal * sigma * sigma / (2 * GAMMA_MINUS1) / u4,
                    GAMMA_MINUS1 * u4 / (2 * M_PI * All.GravityConstantInternal * sigma));
        }

        if (All.ComovingIntegrationOn)
        {
            All.Time = All.TimeBegin;
            set_cosmo_factors_for_current_time();
            IonizeParams();
        }

#ifdef WINDS
        if (All.WindEfficiency > 0)
        {
            VERBOSE(3, "Windspeed: %g\n",
                    sqrt(2 * All.WindEnergyFraction * All.FactorSN * All.EgySpecSN / (1 - All.FactorSN) /
                         All.WindEfficiency));
        }
#endif // WINDS
    }
}

void integrate_sfr(void)
{
    double rho0, rho, rho2, q, dz, gam, sigma = 0, sigma_u4, sigmasfr = 0, ne, P1;
    double x = 0, y, P, P2, x2, y2, tsfr2, factorEVP2, egyhot2, tcool2, drho, dq;
    double meanweight, u4, z, tsfr, tcool, egyhot, factorEVP, egyeff, egyeff2;
    double Po, f_mol_eff, Po_int, e_unit, P4, f_mol4, dens;
    FILE *fd;

    dens = 1.0e6 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);

    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)); /* note: assuming FULL ionization */
    u4 = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
    u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
    Po = 35000 * BOLTZMANN;
    e_unit = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);
    Po_int = Po / e_unit * pow(All.UnitLength_in_cm, 3) / pow(All.HubbleParam, 2);
    P4 = u4 * dens * GAMMA_MINUS1 * e_unit * All.UnitDensity_in_cgs * pow(All.HubbleParam, 2);
    f_mol4 = 1 / (1 + Po / P4);


    if (All.ComovingIntegrationOn)
    {
        All.Time = 1.0; /* to be guaranteed to get z=0 rate */
        set_cosmo_factors_for_current_time();
        IonizeParams();
    }

    if (ThisTask == 0)
    {
        fd = fopen("eos.txt", "w");
    }
    else
    {
        fd = 0;
    }

    for (rho = All.PhysDensThresh; rho <= 1000 * All.PhysDensThresh; rho *= 1.1)
    {
        tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;

        factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;

        egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

        ne = 1.0;
        tcool = GetCoolingTimeSTD(egyhot, rho, &ne);
        y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
        x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

        egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

        P = GAMMA_MINUS1 * rho * egyeff;
        f_mol_eff = 1 / (1 + Po_int / P);

        if (ThisTask == 0)
        {
            fprintf(fd, "%g %g\n", rho, P);
        }
    }

    if (ThisTask == 0)
    {
        fclose(fd);
    }

    if (ThisTask == 0)
    {
        fd = fopen("sfrrate.txt", "w");
    }
    else
    {
        fd = 0;
    }

    for (rho0 = All.PhysDensThresh; rho0 <= 10000 * All.PhysDensThresh; rho0 *= 1.02)
    {
        z = 0;
        rho = rho0;
        q = 0;
        dz = 0.001;

        sigma = sigmasfr = sigma_u4 = 0;

        while (rho > 0.0001 * rho0)
        {
            if (rho > All.PhysDensThresh)
            {
                tsfr = sqrt(All.PhysDensThresh / rho) * All.MaxSfrTimescale;

                factorEVP = pow(rho / All.PhysDensThresh, -0.8) * All.FactorEVP;

                egyhot = All.EgySpecSN / (1 + factorEVP) + All.EgySpecCold;

                ne = 1.0;
                tcool = GetCoolingTimeSTD(egyhot, rho, &ne);
                y = tsfr / tcool * egyhot / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
                x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
                egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

                P = P1 = GAMMA_MINUS1 * rho * egyeff;
                f_mol_eff = 1 / (1 + Po_int / P);

                rho2 = 1.1 * rho;
                tsfr2 = sqrt(All.PhysDensThresh / rho2) * All.MaxSfrTimescale;
                factorEVP2 = pow(rho2 / All.PhysDensThresh, -0.8) * All.FactorEVP;
                egyhot2 = All.EgySpecSN / (1 + factorEVP2) + All.EgySpecCold;
                tcool2 = GetCoolingTimeSTD(egyhot2, rho2, &ne);

                y2 = tsfr2 / tcool2 * egyhot2 / (All.FactorSN * All.EgySpecSN - (1 - All.FactorSN) * All.EgySpecCold);
                x2 = 1 + 1 / (2 * y2) - sqrt(1 / y2 + 1 / (4 * y2 * y2));
                egyeff2 = egyhot2 * (1 - x2) + All.EgySpecCold * x2;
                P2 = GAMMA_MINUS1 * rho2 * egyeff2;

                gam = log(P2 / P1) / log(rho2 / rho);
            }
            else
            {
                tsfr = 0;
                P = GAMMA_MINUS1 * rho * u4;
                gam = 1.0;
                sigma_u4 += rho * dz;
            }

            drho = q;
            dq = -(gam - 2) / rho * q * q - 4 * M_PI * All.GravityConstantInternal / (gam * P) * rho * rho * rho;

            sigma += rho * dz;
            if (tsfr > 0)
            {
                sigmasfr += (1 - All.FactorSN) * rho * x / tsfr * dz;
            }

            rho += drho * dz;
            q += dq * dz;
        }

        sigma *= 2; /* to include the other side */
        sigmasfr *= 2;
        sigma_u4 *= 2;

        if (ThisTask == 0)
        {
            fprintf(fd, "%g %g %g %g\n", rho0, sigma, sigmasfr, sigma_u4);
        }
    }

    if (All.ComovingIntegrationOn)
    {
        All.Time = All.TimeBegin;
        set_cosmo_factors_for_current_time();
        IonizeParams();
    }

    if (ThisTask == 0)
    {
        fclose(fd);
    }
}

#endif // SFR

#ifdef SFR
void set_units_sfr(void)
{
    double meanweight;

    All.OverDensThresh =
        All.CritOverDensity * All.OmegaBaryon * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);

    All.PhysDensThresh = All.CritPhysDensity * PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs;

    meanweight = 4 / (1 + 3 * HYDROGEN_MASSFRAC); /* note: assuming NEUTRAL GAS */

    All.EgySpecCold = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempClouds;
    All.EgySpecCold *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)); /* note: assuming FULL ionization */

    All.EgySpecSN = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.TempSupernova;
    All.EgySpecSN *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
}

#endif // SFR

#endif // COOLING

#endif // LT_STELLAREVOLUTION
