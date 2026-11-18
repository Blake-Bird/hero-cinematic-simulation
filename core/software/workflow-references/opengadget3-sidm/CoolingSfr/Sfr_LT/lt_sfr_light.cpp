/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Giuseppe Murante, Luca Tornatore and Stefano Borgani.
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
#include <ctype.h>
#include "../../CodeBase/allvars.hpp"
#include "../../CodeBase/proto.hpp"
#include "../../Gravity/forcetree.hpp"
#ifdef MAGNETIC_SN_SEEDING
#include "../../Hydro/mhd_seeding.hpp"
#endif // MAGNETIC_SN_SEEDING
#include "lt_wal_cooling.hpp"

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

#if !defined(LT_SEv_INFO) && !defined(LT_ZAGE)
#ifdef LT_STELLAREVOLUTION

#ifdef LT_METAL_COOLING_WAL
extern double DZ, Redshift;
#ifdef LS_SF_RUNAWAY_TREATMENT
double RunawayDensityThreshold;
#endif // LS_SF_RUNAWAY_TREATMENT
#ifdef GL_DUST_COOLING
static double DL, DS; // particle's dust to gas ration for small and big grains
#endif // GL_DUST_COOLING
#endif

#ifdef LT_LOCAL_IRA
#error "module not compatible with LT_LOCAL_IRA. Use lt_sfr.c instead."
#endif

/*
 * This routine does cooling and star formation for
 * the effective multi-phase model.
 */

#ifdef WIND_DEBUG
unsigned long long trace_id = 0;
#endif // WIND_DEBUG

#define KD_I_WILL_MAKE_A_STAR 64
#define KD_I_WILL_BE_A_STAR 128

void cooling_and_starformation(void)
/* cooling routine when star formation is enabled */
{
    int i, j, bin, stars_spawned, tot_spawned, stars_converted, tot_converted, number_of_stars_generated;
    double hubble_a = 0, a3inv;
    double Sum_sm, Total_sm, rate, Sum_mass_stars, Total_sum_mass_stars, factor;
    double sfrrate, totsfrrate, mass_of_star;

    //  double Z;
    double NextChemTime;
    double mstar;
    int index_of_star, chem_step;

    MyIDType GENERATIONS_BUNCH;

#if defined(GM_BH_EVAPORATE_COLD_CLOUDS) || defined(GM_SFEFF_TTHRESH)
    double Temperature_eff;
#endif

    VERBOSE(2, "LT_SFR: Start computation of cooling and sfr");

    for (bin = 0; bin < TIMEBINS; bin++)
        if (TimeBinActive[bin])
            TimeBinSfr[bin] = 0;

    if (All.ComovingIntegrationOn)
    {
        /* Factors for comoving integration of hydro */
        a3inv = 1 / (All.Time * All.Time * All.Time);
        hubble_a = hubble_function(All.Time);
    }
    else
    {
        a3inv = 1;
    }

#ifdef LT_METAL_COOLING_WAL
    if (All.ComovingIntegrationOn)
    {
        Redshift = 1.0 / units->atime - 1;
    }
    else
    {
        Redshift = 0.0;
    }
    get_cool_redshift(Redshift, &DZ);

#if defined(LDF_HYDROGEN_FRACTIONS)
    if (ThisTask == 0)
    {
        double Metallicities[LT_NMet];
#ifdef DEBUG
        set_metallicities(0, &Metallicities[0], a3inv);
        printf("[WalCool] Met=[");
        for (int i = 0; i < LT_NMet - 1; i++)
            printf("%g ", Metallicities[i]);
        printf("]\n");
#endif
        for (int i = 0; i < LT_NMet - 1; i++)
            Metallicities[i] = 0;
        Metallicities[LT_NMet - 2] = 0.24;
        Metallicities[LT_NMet - 3] = 0.00168795;
        double temp = 1e5;
        Hfraction Hfrac = AbundanceRatios_fromCoolingTables(temp, Redshift, DZ, &Metallicities[0]);
        VERBOSE(3, "[WalCool] WcoolZIndex=%d z=%g DZ=%g temp=%g HI_frac=%g ", WalCool_redshift_index, Redshift, DZ,
                temp, Hfrac.HI_frac);
    }
#else
    VERBOSE(3, "[WalCool] WcoolZIndex=%d z=%g DZ=%g ", WalCool_redshift_index, Redshift, DZ);
#endif

    ENDRUN_IF(LT_NMet < get_cool_n_el(), "1009");
    WalCool_tables_load(Redshift);
#endif // LT_METAL_COOLING_WAL

    for (i = 0; i < SFs_dim; i++)
    {
        sum_sm[i] = sum_mass_stars[i] = 0;
        total_sm[i] = total_sum_mass_stars[i] = 0;
        Sum_sm = Sum_mass_stars = Total_sm = Total_sum_mass_stars = 0;
        sfrrates[i] = totsfrrates[i] = 0;
    }
    for (i = 0; i < NBINS_SFR_by_DENSITY; i++)
    {
        sfrrates_bydensity[i] = totsfrrates_bydensity[i] = 0;
    }
    for (i = 0; i < NBINS_SFR_by_Z; i++)
    {
        sfrrates_byZ[i] = totsfrrates_byZ[i] = 0;
    }

    stars_spawned = stars_converted = 0;

#ifdef _OPENMP
    int il;
#pragma omp parallel for private(i, j, GENERATIONS_BUNCH, number_of_stars_generated, mass_of_star)
    for (il = 0; il < NActivePart; il++)
    {
        i = ActiveParticleList[il];
#else /* _OPENMP */
    for (i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif

        int flag;
        double myFactorEVP, myPhysDensThresh, myFactorSN, myEgySpecSN, myMaxSfrTimescale;
        double Zcool, dt, dtime, SNEgy = 0, egyhot, egyeff, egycurrent, tcool, x, y;
        double p, prob, sm;
        double cloudmass;
        double factorEVP;
        double tsfr, trelax;
        double Temperature, ne = 1, unew;

        // SphP[i].XColdCloud = 0;

#ifdef WINDS
        double v;
        double norm, dir[3];
#endif // WINDS

#ifdef LT_METAL_COOLING_WAL
        double Metallicities[LT_NMet];
#endif // LT_METAL_COOLING_WAL

#ifdef BLACK_HOLES
        if (P[i].Type == 0 && P[i].Mass > 0)
#else // BLACK_HOLES
        if (P[i].Type == 0)
#endif // BLACK_HOLES
        {
            // computed once per particle instead of on every use, since Density and a3inv/gamma don't
            // change while this particle is being processed below
            const double rho_phys = SphP[i].Density * a3inv;
            const double gm1 = get_gamma_minus1(i);

            dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;
            /*  the actual time-step */

            if (All.ComovingIntegrationOn)
                dtime = dt * units->timestep_to_physical;
            else
                dtime = dt;

#ifndef LT_LOCAL_IRA
            SphP[i].mstar = 0;
#endif

#ifdef LT_METAL_COOLING_WAL
            set_metallicities(i, &Metallicities[0], a3inv); /* note: sets the physical density too! */
            Zcool = get_metallicity_solarunits(get_metallicity(i, Iron));
#else
            Zcool = get_metallicity_solarunits(get_metallicity(i, Iron));
#endif
            int myIMFi, mySFi;

            get_SF_index(i, &mySFi, &myIMFi);

            GENERATIONS_BUNCH = All.Generations / SFs[mySFi].Generations;

            myFactorSN = SFs[mySFi].FactorSN;
            myEgySpecSN = SFs[mySFi].EgySpecSN;
            myMaxSfrTimescale = SFs[mySFi].MaxSfrTimescale;

            if (SFs[mySFi].SFTh_Zdep)
            {
                /*
                   if the effective model is allowed to depend on metallicity,
                   thresholds and evaporation factors differ for different Z.
                 */
                getindex(&CoolZvalue[0], 0, ZBins - 1, &Zcool, &flag);

                if (flag == 0 || flag == ZBins - 1)
                {
                    myFactorEVP = SFs[mySFi].FEVP[flag];
                    myPhysDensThresh = SFs[mySFi].PhysDensThresh[flag];
                }
                else
                {
                    p = (Zcool - CoolZvalue[flag + 1]) / (CoolZvalue[flag + 1] - CoolZvalue[flag]);
                    /* interpolate */
                    myFactorEVP = SFs[mySFi].FEVP[flag] * (1 - p) + SFs[mySFi].FEVP[flag + 1] * p;
                    myPhysDensThresh =
                        SFs[mySFi].PhysDensThresh[flag] * (1 - p) + SFs[mySFi].PhysDensThresh[flag + 1] * p;
                }
            }
            else
            {
                myFactorEVP = SFs[mySFi].FEVP[0];
                myPhysDensThresh = SFs[mySFi].PhysDensThresh[0];
            }

            /* check whether conditions for star formation are fulfilled.
             *
             * f=1  normal cooling
             * f=0  star formation
             */

            /* default is normal cooling */
            flag = 1;

            double fac_entr_to_u = pow(rho_phys, gm1) / gm1;
            double uold = max(All.MinEgySpec, (double)SphP[i].Entropy * fac_entr_to_u);

#ifdef LT_MAX_TEMP_FEEDBACK
            // As we want to imply a maximum temperature cut, we can savely assume full ionization
            // and ignore all effects of additional physics
            {
                MyAtLeastDouble temp_test = uold * U_TO_TEMP_FAC;
                if (temp_test > LT_MAX_TEMP_FEEDBACK)
                {
#ifdef DEBUG
                    VERBOSE_ALL(
                        2,
                        "task=%d: Particle id=%llu,m=%e too large temperature before cooling/sfr %e>%e, correcting !",
                        ThisTask, (unsigned long long)P[i].ID, (double)P[i].Mass, (double)temp_test,
                        (double)LT_MAX_TEMP_FEEDBACK);
#endif
                    uold = LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC;
                    temp_test = LT_MAX_TEMP_FEEDBACK;
#ifdef LT_STELLAREVOLUTION
                    SphP[i].EgyRes = 0;
#endif // LT_STELLAREVOLUTION
                }
#ifdef DEBUG
                if (!(temp_test >= 0 && temp_test <= LT_MAX_TEMP_FEEDBACK))
                {
                    VERBOSE_ALL(2, "task=%d: Particle id=%llu,m=%e strange temperature before cooling/sfr %e>%e !",
                                ThisTask, (unsigned long long)P[i].ID, (double)P[i].Mass, (double)temp_test,
                                (double)LT_MAX_TEMP_FEEDBACK);
                }
                if (temp_test == 0)
                {
                    VERBOSE_ALL(2,
                                "task=%d: WARNING Particle id=%llu,m=%e,uold=%e,rho=%e got zero tempersture before "
                                "cooling/sfr (uold=%e) !",
                                ThisTask, (unsigned long long)P[i].ID, (double)P[i].Mass, (double)uold,
                                (double)SphP[i].Density, (double)uold);
                }
#endif
            }
#endif
            if (rho_phys >= myPhysDensThresh)
            {
                flag = 0;
#if defined(GM_BH_EVAPORATE_COLD_CLOUDS) || defined(GM_SFEFF_TTHRESH)
                factorEVP = pow(rho_phys / myPhysDensThresh, -0.8) * myFactorEVP;
                egyhot = myEgySpecSN / (1 + factorEVP) + All.EgySpecCold;
                tsfr = sqrt(myPhysDensThresh / rho_phys) * myMaxSfrTimescale;
                tcool = GetCoolingTimeMET(egyhot, rho_phys, Redshift, DZ, &Metallicities[0], &Temperature);
                y = tsfr / tcool * egyhot / (myFactorSN * myEgySpecSN - (1 - myFactorSN) * All.EgySpecCold);
                x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
                egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

#if defined(GM_BH_EVAPORATE_COLD_CLOUDS) && defined(BLACK_HOLES)
                double uc_new;
                /* All BH energy goes to heat cold gas, can be changed... */
                uc_new = All.EgySpecCold * x + SphP[i].Injected_BH_Energy / (x * P[i].Mass);

                /* if specific energy of COLD gas is larger than average specific energy of particle,
                   all cold gas has been evaporated and particle is not MP anymore */
                if (uc_new > uold)
                    flag = 1;
#endif

                double ne_guess = ne;
// GM: ...thus I set a temp thresh to avoid that heated gas re-enters mp the timestep after being heated
#if !defined(LT_METAL_COOLING_WAL)
                Temperature = convert_u_to_tempSTD(
                    uold * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                    SphP[i].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam * a3inv, &ne_guess);
                Temperature_eff = convert_u_to_tempSTD(
                    egyeff * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                    SphP[i].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam * a3inv, &ne_guess);
#else
                Temperature = convert_u_to_tempMET(uold * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs, Redshift,
                                                   DZ, &Metallicities[0]);
                Temperature_eff = convert_u_to_tempMET(egyeff * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                                                       Redshift, DZ, &Metallicities[0]);
#endif

#if defined(GM_BH_EVAPORATE_COLD_CLOUDS) && !defined(GM_SFEFF_TTHRESH) // in this case there is noo threshold on T
                if (Temperature > GM_BH_EVAPORATE_COLD_CLOUDS *
                                      Temperature_eff) // value taken from phase diagrams of one test cluster:250000K
                {
                    flag = 1;
                }

#endif

#ifdef GM_SFEFF_TTHRESH
                if (Temperature > GM_SFEFF_TTHRESH * Temperature_eff)
                {
                    flag = 1;
                }
#endif // GM_SFEFF_TTHRESH

#endif // GM_BH_EVAPORATE_COLD_CLOUDS  || GM_SFEFF_TTHRESH
            }

            x = 0;

            if (All.ComovingIntegrationOn)
                if (SphP[i].Density < All.OverDensThresh)
                    flag = 1;

#ifdef BLACK_HOLES
            if (P[i].Mass == 0)
            {
                flag = 1;
            }
#endif

            if (P[i].Mass > 0 && SphP[i].EgyStep > 0.)
            {
                SNEgy = (dtime > SphP[i].EgyStep ? SphP[i].EgyRes : SphP[i].EgyRes * dtime / SphP[i].EgyStep);
                SphP[i].EgyRes -= SNEgy;
                SNEgy /= P[i].Mass;
            }
            else
            {
                SNEgy = 0;
                SphP[i].EgyRes = 0;
            }

#ifdef WINDS
            if (SphP[i].DelayTime > 0)
            {
                flag = 1; /* only normal cooling for particles in the wind */
                SphP[i].DelayTime -= dtime;
                MyAtLeastDouble WindStopDensity = 0.5 * All.WindFreeTravelDensFac * SFs[mySFi].PhysDensThresh[0];
#ifdef KD_ACCOUNT_FOR_INCREASED_THRESHOLD_IN_WIND_TRAVEL
                WindStopDensity *= 0.000831051 / SFs[mySFi].PhysDensThresh[0];
#endif
#ifndef LS_SF_RUNAWAY_TREATMENT
                if ((SphP[i].DelayTime < 0) || ((SphP[i].DelayTime > 0) && (rho_phys < WindStopDensity)))
#else // LS_SF_RUNAWAY_TREATMENT
                /*
                We assume that beyond the runaway threshold predicted by SH03
                gas particles should not remain wind, but converted back into
                normal gas particles.
                */
                if ((SphP[i].DelayTime < 0) || ((SphP[i].DelayTime > 0) && (rho_phys < WindStopDensity)) ||
                    (rho_phys > RunawayDensityThreshold))
#endif // LS_SF_RUNAWAY_TREATMENT
                {
                    SphP[i].DelayTime = 0;
#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING
                    // start only in next round with cooling/sfr as particle just left the wind phase
                    continue;
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING
                }
            }

#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING
            if (SphP[i].DelayTime > 0)
            {
#ifdef BLACK_HOLES
                if (All.BlackHoleThermalFeedbackOn == 1 || All.BlackHoleKineticFeedbackOn == 1)
                    SphP[i].Injected_BH_Energy = 0;
#endif
                continue; /* neglect cooling in winds */
                /* update any further process before continue! */
            }
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING

#endif

#ifdef QUICK_LYALPHA
            {
                MyAtLeastDouble temp_test =
                    U_TO_TEMP_FAC_WITHOUT_GAMMA * gm1 * SphP[i].Entropy / gm1 * pow(rho_phys, gm1);

                if (SphP[i].Density > All.OverDensThresh && temp_test < 1.0e5)
                    flag = 0;
                else
                    flag = 1;
            }
#endif

#ifdef MAGNETIC_SN_SEEDING
            SphP[i].MagSeedRate[0] = SphP[i].MagSeedRate[1] = SphP[i].MagSeedRate[2] = 0.0;
#endif // MAGNETIC_SN_SEEDING


#if !defined(NOISMPRESSURE) && !defined(QUICK_LYALPHA)
            if (flag == 1) /* normal implicit isochoric cooling */
#endif
            {
                SphP[i].Sfr = 0;
                ne = SphP[i].elec;

#ifdef LT_METAL_COOLING_WAL
                //          WalCool_set_PID(P[i].ID);
#ifndef UM_CHEMISTRY
#ifdef GL_DUST_COOLING
                if (P[i].Mass > 0)
                {
                    DL = get_metalmass(SphP[i].DustL) / P[i].Mass;
                    DS = get_metalmass(SphP[i].DustS) / P[i].Mass;
                }
                else
                {
                    DL = 0;
                    DS = 0;
                }
                unew = DoCoolingMET(uold, rho_phys, &Metallicities[0], DL, DS, Redshift, DZ, dtime, &Temperature);
#else // GL_DUST_COOLING
                unew = DoCoolingMET(uold, rho_phys, &Metallicities[0], Redshift, DZ, dtime, &Temperature);
#ifdef WIND_DEBUG
                if (trace_id == 0 && ThisTask == 0)
                {
                    if (SphP[i].DelayTime > 0.054)
                    {
                        VERBOSE(2, "  COOLING(id=%llu,Dt=%e)  change of u %e -> %e for rho=%e",
                                (unsigned long long)P[i].ID, SphP[i].DelayTime, uold, unew, SphP[i].Density);
                        trace_id = P[i].ID;
                    }
                }
                if (trace_id != 0)
                {
                    if (trace_id == P[i].ID)
                        VERBOSE_ALL(2, " %d %llu %e  %e %e %e %e %e %e %e", ThisTask, (unsigned long long)P[i].ID,
                                    SphP[i].DelayTime, uold, unew, SphP[i].Density, SNEgy, SphP[i].HydroAccel[0],
                                    SphP[i].HydroAccel[1], SphP[i].HydroAccel[2]);
                }
#endif // WIND_DEBUG
#endif // GL_DUST_COOLING

#else // UM_CHEMISTRY
                unew = Um_DoCooling(uold, rho_phys, dtime, &ne, &Metallicities[0], Redshift, DZ, i, flag);
#endif // UM_CHEMISTRY
#else // LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
#ifdef UM_CHEMISTRY
                unew = Um_DoCooling(uold, rho_phys, dtime, &ne, Zcool, i, flag);
#else
                unew = DoCoolingZ(uold, rho_phys, dtime, &ne, Zcool, &Temperature);
#endif
#else
                unew = DoCoolingSTD(uold, rho_phys, dtime, &ne);
#endif /* closes LT_METAL_COOLING */
#endif /* closes LT_METAL_COOLING_WAL */

#ifdef LT_MAX_TEMP_FEEDBACK
                // As we want to imply a maximum temperature cut, we can savely assume full ionization
                // and ignore all effects of additional physics
                {
                    MyAtLeastDouble temp_test = U_TO_TEMP_FAC * unew;
                    if (temp_test > LT_MAX_TEMP_FEEDBACK)
                    {
#ifdef DEBUG
                        VERBOSE_ALL(2,
                                    "task=%d: Particle id=%llu,m=%e Correcting too large temperature after cooling in "
                                    "cooling %e>%e !",
                                    ThisTask, (unsigned long long)P[i].ID, (double)P[i].Mass, (double)temp_test,
                                    (double)LT_MAX_TEMP_FEEDBACK);
#endif // DEBUG
                        unew = LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC;
                        temp_test = LT_MAX_TEMP_FEEDBACK;
#ifdef LT_STELLAREVOLUTION
                        SphP[i].EgyRes = 0;
                        SNEgy = 0;
#endif // LT_STELLAREVOLUTION
                    }
#ifdef DEBUG
                    if (!(temp_test >= 0 && temp_test <= LT_MAX_TEMP_FEEDBACK))
                    {
                        VERBOSE_ALL(
                            2, "task=%d: Particle id=%llu,m=%e strange temperature after cooling in cooling %e>%e !",
                            ThisTask, (unsigned long long)P[i].ID, (double)P[i].Mass, (double)temp_test,
                            (double)LT_MAX_TEMP_FEEDBACK);
                    }
                    if (temp_test == 0)
                    {
                        VERBOSE_ALL(2,
                                    "task=%d: WARNING Particle id=%llu,m=%e,uold=%e,rho=%e got zero tempersture "
                                    "strange after cooling in cooling (unew=%e) !",
                                    ThisTask, (unsigned long long)P[i].ID, (double)P[i].Mass, (double)uold,
                                    (double)SphP[i].Density, (double)unew);
                    }

#endif
                }
#endif // LT_MAX_TEMP_FEEDBACK

                unew += SNEgy;
#ifdef LT_MAX_TEMP_FEEDBACK
                // As we want to imply a maximum temperature cut, we can savely assume full ionization
                // and ignore all effects of additional physics
                {
                    MyAtLeastDouble temp_test = U_TO_TEMP_FAC * unew;
                    if (temp_test > LT_MAX_TEMP_FEEDBACK)
                    {
#ifdef DEBUG
                        VERBOSE_ALL(2,
                                    "task=%d: Particle id=%llu,m=%e Correcting too large temperature after receiving "
                                    "SN feedback in cooling %e>%e !",
                                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, (double)temp_test,
                                    (double)LT_MAX_TEMP_FEEDBACK);
#endif // DEBUG
                        unew = LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC;
#ifdef LT_STELLAREVOLUTION
                        SphP[i].EgyRes = 0;
                        SNEgy = 0;
#endif // LT_STELLAREVOLUTION
                    }
                }
#endif // LT_MAX_TEMP_FEEDBACK

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
#ifdef LT_MAX_TEMP_FEEDBACK
                            // As we want to imply a maximum temperature cut, we can savely assume full ionization
                            // and ignore all effects of additional physics
                            {
                                MyAtLeastDouble temp_test = U_TO_TEMP_FAC * unew;
                                if (temp_test > LT_MAX_TEMP_FEEDBACK)
                                {
#ifdef DEBUG
                                    VERBOSE_ALL(2,
                                                "task=%d: Particle id=%llu,m=%e Correcting too large temperature after "
                                                "receiving BH feedback in cooling %e>%e !",
                                                ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass,
                                                (double)temp_test, (double)LT_MAX_TEMP_FEEDBACK);
#endif
                                    unew = LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC;
#ifdef LT_STELLAREVOLUTION
                                    SphP[i].EgyRes = 0;
                                    SphP[i].Injected_BH_Energy = 0;
#endif // LT_STELLAREVOLUTION
                                }
                            }
#endif
                            SphP[i].Injected_BH_Energy = 0;
                        }
                    }
#endif

                SphP[i].elec = ne;
                SphP[i].Entropy = unew / fac_entr_to_u;
#if GADGET_HYDRO == HYDRO_MFM
                SphP[i].InternalEnergy = unew; // need variable gamma?!
#endif // GADGET_HYDRO == HYDRO_MFM

                SphP[i].EntropyPred = SphP[i].Entropy;
#if GADGET_HYDRO == HYDRO_MFM
                SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
                mfm_synchronize_entropy(i);
#endif // GADGET_HYDRO == HYDRO_MFM
                SphP[i].Pressure = get_pressure(i);
                /* Commented out due to seg-fault - to be fixed
                #if !defined(LT_METAL_COOLING_WAL) // paranoid: recompute the temperature before assigning it
                                Temperature = convert_u_to_tempSTD(
                                    uold * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                                    SphP[i].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam *
                a3inv, &ne_guess); #else Temperature = convert_u_to_tempMET(uold * All.UnitPressure_in_cgs /
                All.UnitDensity_in_cgs, Redshift, DZ, &Metallicities[0]); #endif SphP[i].Temperature = Temperature;
                */
            }

            if (flag == 0) /* active star formation */
            {
#if !defined(QUICK_LYALPHA)
                tsfr = sqrt(myPhysDensThresh / rho_phys) * myMaxSfrTimescale;

                factorEVP = pow(rho_phys / myPhysDensThresh, -0.8) * myFactorEVP;

                egyhot = myEgySpecSN / (1 + factorEVP) + All.EgySpecCold;

                ne = SphP[i].elec;

#ifndef LT_METAL_COOLING_WAL
#ifndef LT_METAL_COOLING
                tcool = GetCoolingTimeSTD(egyhot, rho_phys, &ne);
#else
#ifndef UM_CHEMISTRY
                /*Z = get_metallicity(i); already done */
                if ((tcool = GetCoolingTimeZ(egyhot, rho_phys, &ne, Zcool, &Temperature)) == 0)
                    tcool = 1e13;
#else
                if ((tcool = Um_GetCoolingTime(egyhot, rho_phys, &ne, Zcool, i)) == 0)
                    tcool = 1e13;
#endif
#endif
#else
#ifndef UM_CHEMISTRY

#ifdef GL_DUST_COOLING
                tcool = GetCoolingTimeMET(egyhot, rho_phys, Redshift, DZ, DL, DS, &Metallicities[0], &Temperature);
#else // GL_DUST_COOLING
                tcool = GetCoolingTimeMET(egyhot, rho_phys, Redshift, DZ, &Metallicities[0], &Temperature);
#endif // GL_DUST_COOLING
#else
                if ((tcool = Um_GetCoolingTime(egyhot, rho_phys, &ne, &Metallicities[0], Redshift, DZ, i)) == 0)
                    tcool = 1e13;
#endif
#endif

                SphP[i].elec = ne;

                y = tsfr / tcool * egyhot / (myFactorSN * myEgySpecSN - (1 - myFactorSN) * All.EgySpecCold);

                x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

                egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

                cloudmass = x * P[i].Mass;

                if (tsfr < dtime)
                {
                    tsfr = dtime;
                }

                sm = dtime / tsfr * cloudmass;

                p = sm / P[i].Mass;

                SphP[i].Sfr = (1 - myFactorSN) * cloudmass / tsfr * (All.UnitMass_in_g / SOLAR_MASS) /
                              (All.UnitTime_in_s / SEC_PER_YEAR);

#ifdef _OPENMP
#pragma omp critical(_sfr_)
#endif
                {
                    sum_sm[mySFi] += P[i].Mass * (1 - exp(-p));
                    TimeBinSfr[P[i].TimeBin] += SphP[i].Sfr;
                    sfrrates[mySFi] += SphP[i].Sfr;
                }
                /*
                   if(SphP[i].Sfr > 0)
                   {
                   int byrho_idx, byZ_idx;
                   byrho_idx = (int)(log10(rho_phys / rhomean) - SFR_by_DENSITY_LOGMIN);
                   if(byrho_idx < 0 )
                   byrho_idx = 0;
                   if(byrho_idx > NBINS_SFR_by_DENSITY)
                   byrho_idx = NBINS_SFR_by_DENSITY;
                   sfrrates_bydensity[byrho_idx] += SphP[i].Sfr;

                   byZ_idx = (int)(Zcool - (NO_METAL));
                   if(byZ_idx < 0)
                   byZ_idx = 0;
                   if(byZ_idx > NBINS_SFR_by_Z)
                   byZ_idx = NBINS_SFR_by_Z;
                   sfrrates_byZ[byZ_idx] += SphP[i].Sfr;
                   }
                 */
                double exp_minus_p = exp(-p);
#ifndef LT_STOCHASTIC_IRA
                mstar = sm * exp_minus_p;
#else
                mstar = P[i].Mass * (1 - exp_minus_p);
#endif

                if (dt > 0)
                {
                    if (P[i].TimeBin) /* upon start-up, we need to protect against dt==0 */
                    {
                        trelax = tsfr * (1 - x) / x / (myFactorSN * (1 + factorEVP));

                        egycurrent = SphP[i].Entropy * pow(rho_phys, gm1) / gm1;

#ifdef LT_MAX_TEMP_FEEDBACK
                        // As we want to imply a maximum temperature cut, we can savely assume full ionization
                        // and ignore all effects of additional physics
                        {
                            MyAtLeastDouble temp_test = U_TO_TEMP_FAC * egycurrent;
                            if (temp_test > LT_MAX_TEMP_FEEDBACK)
                            {
#ifdef DEBUG
                                VERBOSE_ALL(2,
                                            "task=%d: Particle id=%llu,m=%e Detected too large temperature for sfr "
                                            "particle starting in cooling/sfr %e>%e !",
                                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, (double)temp_test,
                                            (double)LT_MAX_TEMP_FEEDBACK);
#endif // DEBUG
                            }
                        }
#endif // LT_MAX_TEMP_FEEDBACK

                        egycurrent += SNEgy;
#ifdef LT_MAX_TEMP_FEEDBACK
                        // As we want to imply a maximum temperature cut, we can savely assume full ionization
                        // and ignore all effects of additional physics
                        {
                            MyAtLeastDouble temp_test = U_TO_TEMP_FAC * egycurrent;
                            if (temp_test > LT_MAX_TEMP_FEEDBACK)
                            {
#ifdef DEBUG
                                VERBOSE_ALL(2,
                                            "task=%d: Particle id=%llu,m=%e Correcting too large temperature after "
                                            "receiving SN feedback in cooling/sfr %e>%e !",
                                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, (double)temp_test,
                                            (double)LT_MAX_TEMP_FEEDBACK);
#endif

                                egycurrent = LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC;
#ifdef LT_STELLAREVOLUTION
                                SphP[i].EgyRes = 0;
                                SNEgy = 0;
#ifdef BLACK_HOLES
                                SphP[i].Injected_BH_Energy = 0;
#endif
#endif // LT_STELLAREVOLUTION
                            }
                        }
#endif

#ifdef BLACK_HOLES
                        if (All.BlackHoleThermalFeedbackOn == 1 || All.BlackHoleKineticFeedbackOn == 1)
                            if (SphP[i].Injected_BH_Energy > 0)
                            {
                                egycurrent += SphP[i].Injected_BH_Energy / P[i].Mass;
#ifdef LT_MAX_TEMP_FEEDBACK
                                // As we want to imply a maximum temperature cut, we can savely assume full ionization
                                // and ignore all effects of additional physics
                                {
                                    MyAtLeastDouble temp_test = U_TO_TEMP_FAC * egycurrent;
                                    if (temp_test > LT_MAX_TEMP_FEEDBACK)
                                    {
#ifdef DEBUG
                                        VERBOSE_ALL(2,
                                                    "task=%d: Particle id=%llu,m=%e Correcting too large temperature "
                                                    "after receiving BH feedback in cooling/sfr %e>%e !",
                                                    ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass,
                                                    (double)temp_test, (double)LT_MAX_TEMP_FEEDBACK);
#endif

                                        egycurrent = LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC;
#ifdef LT_STELLAREVOLUTION
                                        SphP[i].EgyRes = 0;
                                        SNEgy = 0;
#ifdef BLACK_HOLES
                                        SphP[i].Injected_BH_Energy = 0;
#endif
#endif // LT_STELLAREVOLUTION
                                    }
                                }
#endif

                                if (egycurrent > egyeff)
                                {

#if defined(LT_METAL_COOLING) /* metal cooling S & D */
                                    ne = 1.0;
#ifndef UM_CHEMISTRY /* low-T metal & molecular cooling */
                                    if ((tcool = GetCoolingTimeZ(egycurrent, rho_phys, &ne, Zcool, &Temperature)) == 0)
                                        tcool = 1e13;
#else
                                    ne = SphP[i].elec;
                                    if ((tcool = Um_GetCoolingTime(egycurrent, rho_phys, &ne, Zcool, i)) == 0)
                                        tcool = 1e13;
#endif
#endif

#if defined(LT_METAL_COOLING_WAL) /* WAL metal cooling */
#ifndef UM_CHEMISTRY /* low-T metal & molecular cooling */
#ifdef GL_DUST_COOLING
                                    tcool = GetCoolingTimeMET(egycurrent, rho_phys, Redshift, DZ, DL, DS,
                                                              &Metallicities[0], &Temperature);
#else // GL_DUST_COOLING
                                    tcool = GetCoolingTimeMET(egycurrent, rho_phys, Redshift, DZ, &Metallicities[0],
                                                              &Temperature);
#endif // GL_DUST_COOLING
#else
                                    if ((tcool = Um_GetCoolingTime(egyhot, rho_phys, &ne, &Metallicities[0], Redshift,
                                                                   DZ, i)) == 0)
                                        tcool = 1e13;
#endif
#endif

                                    if (tcool < trelax && tcool > 0)
                                        trelax = tcool;
                                }

                                SphP[i].Injected_BH_Energy = 0;
                            }
#endif

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

#if !defined(NOISMPRESSURE)

#ifdef NO_SLOW_RELAX_TO_EOS // default (here, and in precedence)
                        // We need to set the entropy to the new value, as we need it for the pressure calculation
                        // in the next
                        SphP[i].Entropy = egyeff * gm1 / pow(rho_phys, gm1);
#else // ! NO_SLOW_RELAX_TO_EOS
                        SphP[i].Entropy =
                            (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax)) * gm1 / pow(rho_phys, gm1);
#endif

#ifdef GM_SFEFF_TTHRESH
                        // if we are using the temperature threshold, in case this particle WILL NOT
                        // be eligible for SF in the next timestep, we allow heating from SnIa & AGN
                        // without the exponential decay
#if !defined(LT_METAL_COOLING_WAL)
                        Temperature = convert_u_to_tempSTD(
                            egycurrent * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                            SphP[i].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam * a3inv,
                            &ne_guess);
#else
                        Temperature =
                            convert_u_to_tempMET(egycurrent * All.UnitPressure_in_cgs / All.UnitDensity_in_cgs,
                                                 Redshift, DZ, &Metallicities[0]);
#endif

                        if (Temperature > GM_SFEFF_TTHRESH * Temperature_eff)
                        {
                            SphP[i].Entropy = egycurrent * gm1 / pow(rho_phys, gm1);
                        }
#endif

#ifdef LT_MAX_TEMP_FEEDBACK
                        // As we want to imply a maximum temperature cut, we can savely assume full ionization
                        // and ignore all effects of additional physics
                        {
                            MyAtLeastDouble u_final = SphP[i].Entropy * pow(rho_phys, gm1) / gm1;
                            MyAtLeastDouble temp_test = U_TO_TEMP_FAC * u_final;
                            if (temp_test > LT_MAX_TEMP_FEEDBACK)
                            {
#ifdef DEBUG
                                VERBOSE_ALL(2,
                                            "task=%d: Particle id=%llu,m=%e Detected too large temperature for sfr "
                                            "particle on EQS in cooling/sfr %e>%e !",
                                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, (double)temp_test,
                                            (double)LT_MAX_TEMP_FEEDBACK);
#endif // DEBUG
                            }
                        }
#endif // LT_MAX_TEMP_FEEDBACK

                        SphP[i].DtEntropy = 0;
#if GADGET_HYDRO == HYDRO_MFM
                        SphP[i].InternalEnergy = (egyeff + (egycurrent - egyeff) * exp(-dtime / trelax));

                        MyFloat div_vel = limited_divvel(i);
                        SphP[i].DtInternalEnergy =
                            eos->GetDtUFromDtEntropy((MyFloat)0.0, SphP[i].Entropy, SphP[i].InternalEnergy, div_vel,
                                                     All.cf_hubble_a, All.ComovingIntegrationOn);
                        SphP[i].dQdt[NUMDIMS + 1] = eos->GetDtQFromDtEntropy(
                            (MyFloat)0.0, SphP[i].Entropy, SphP[i].InternalEnergy, P[i].Mass, div_vel, All.cf_hubble_a);
#endif // GADGET_HYDRO == HYDRO_MFM
#endif

                        SphP[i].EntropyPred = SphP[i].Entropy;
#if GADGET_HYDRO == HYDRO_MFM
                        SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
                        mfm_synchronize_entropy(i);
#endif // GADGET_HYDRO == HYDRO_MFM
                        SphP[i].Pressure = get_pressure(i);
                    }
                }
#ifndef LT_LOCAL_IRA
                SphP[i].mstar = 0;
#endif

#endif /* belongs to ifndef(QUICK_LYALPHA) */

                /* the upper bits of the gas particle ID store how man stars this gas
                   particle gas already generated                                     */
                if (All.StarBits > 0)
                {
                    number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType) * 8 - All.StarBits));
                }
                else
                {
                    number_of_stars_generated = 0;
                }
                mass_of_star = P[i].Mass / (All.Generations - number_of_stars_generated) * GENERATIONS_BUNCH;
#ifdef DEBUG
                VERBOSE_ALL(2,
                            "RUNAWAY_TREATMENT: ID %llu, number_of_stars_generated %d, mass_of_star %e, "
                            "sizeof(MyIDType)*8 %lu, All.StarBits %d",
                            P[i].ID, number_of_stars_generated, mass_of_star, sizeof(MyIDType) * 8, All.StarBits);
#endif
#ifndef QUICK_LYALPHA
                prob = P[i].Mass / mass_of_star * (1 - exp(-p));
#else
                prob = 2.0; /* this will always cause a star creation event */
#endif /* ends to QUICK_LYALPHA */

#ifdef LS_SF_RUNAWAY_TREATMENT
                /*
                We assume that beyond the runaway threshold predicted by SH03
                gas particles are forced to become star particles
                (although still in generations according to the code below).
                Albeit it is a rather crude treatment, it fits the
                original framework of the SH03 model. As per their paragraph 4.2
                beyond this threshold "self-regulation should break down", and
                the model's cold clouds "may all collapse and form stars
                on the time-scale of their lifetime". In their study they do not
                model this process explicitly and of course a better treatment
                is necessary. However, it contemplates the possibility
                of runaway star formation, since for politropic index <4/3 the
                eEOS become soft and the effective pressure is not able to prevent
                gravitational collapse. Since we expect a runaway conversion of gas
                into stars, it is possible to activate this treatment
                to prevent gas particles to undergo runaway collapse
                that has been observed in some cases at gas mass resolution
                <~ 10^6 Msun, that halts the simulation.
                */
                if ((rho_phys >= RunawayDensityThreshold)
#ifdef WINDS
                    && (SphP[i].DelayTime <= 0)
#endif
                )
                {
                    prob = 2;

                    MyAtLeastDouble sfr_predicted_by_eff_model = SphP[i].Sfr;
                    SphP[i].Sfr =
                        mass_of_star / dtime * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);
#ifdef _OPENMP
#pragma omp critical(_sfr_)
#endif
                    {
                        sum_sm[mySFi] -= P[i].Mass * (1 - exp(-p));
                        TimeBinSfr[P[i].TimeBin] -= sfr_predicted_by_eff_model;
                        sfrrates[mySFi] -= sfr_predicted_by_eff_model;
                        sum_sm[mySFi] += mass_of_star;
                        TimeBinSfr[P[i].TimeBin] += SphP[i].Sfr;
                        sfrrates[mySFi] += SphP[i].Sfr;
                    }
#ifdef DEBUG
                    VERBOSE_ALL(2,
                                "RUNAWAY_TREATMENT: ID %llu, rho %e, eff_pred_sfr %e, runaway_sfr %e, mass_of_star %e, "
                                "All.Generations %d, number_of_stars_generated %d, GENERATIONS_BUNCH %d",
                                P[i].ID, rho_phys, sfr_predicted_by_eff_model, SphP[i].Sfr, mass_of_star,
                                All.Generations, number_of_stars_generated, GENERATIONS_BUNCH);
#endif
                }
#endif // LS_SF_RUNAWAY_TREATMENT

                if (get_random_number(P[i].ID + 1) < prob) /* ok, make a star */
                {
                    if ((number_of_stars_generated + GENERATIONS_BUNCH) >= All.Generations)
                        P[i].Type |= KD_I_WILL_BE_A_STAR;
                    else
                        P[i].Type |= KD_I_WILL_MAKE_A_STAR;
                }

#ifdef WINDS
                /* Here comes the wind model */

                // In the original scheme a particle spawning a star was allowed to get a wind particle, but with the
                // split of the look this is not longer possible, as including a || P[i].Type == KD_I_WILL_MAKE_A_STAR
                // in the if can lead to stars which have wind velocity
                if (P[i].Type == 0 && /* to protect using a particle that has been turned into a star */
                    SFs[mySFi].WindEfficiency > 0)
                {
                    p = SFs[mySFi].WindEfficiency * sm / P[i].Mass;

                    prob = 1 - exp(-p);

                    if (get_random_number(P[i].ID + 2) < prob) /* ok, make the particle go into the wind */
                    {
                        // SphP[i].XColdCloud = 0; /* no cold cloud anymore */

#if WINDS + 0 > 0
                        v = WINDS;
#else
                        v = sqrt(2 * SFs[mySFi].WindEnergyFraction / SFs[mySFi].WindEfficiency *
                                 (SFs[mySFi].totFactorSN / (1 - SFs[mySFi].totFactorSN) * myEgySpecSN + SNEgy));
#endif

                        dir[0] = P[i].GravAccel[1] * P[i].Vel[2] - P[i].GravAccel[2] * P[i].Vel[1];
                        dir[1] = P[i].GravAccel[2] * P[i].Vel[0] - P[i].GravAccel[0] * P[i].Vel[2];
                        dir[2] = P[i].GravAccel[0] * P[i].Vel[1] - P[i].GravAccel[1] * P[i].Vel[0];

                        for (j = 0, norm = 0; j < 3; j++)
                            norm += dir[j] * dir[j];

                        norm = sqrt(norm);
                        if (get_random_number(P[i].ID + 5) < 0.5)
                            norm = -norm;

                        if (norm != 0)
                        {
                            for (j = 0; j < 3; j++)
                                dir[j] /= norm;

                            for (j = 0; j < 3; j++)
                            {
                                P[i].Vel[j] += v * units->velocity_to_comoving * dir[j];
                                SphP[i].VelPred[j] += v * units->velocity_to_comoving * dir[j];
                            }

#ifdef UM_WIND_DELAYTIME
                            SphP[i].DelayTime = max(2. * P[i].Hsml, All.WindFreeTravelLength) / v;
#else
                            SphP[i].DelayTime = All.WindFreeTravelMaxTimeFactor / hubble_function(All.Time);
#endif
                        }
                    }
                }
#endif
            }

            ne = SphP[i].elec;
#ifdef UM_CHEMISTRY

            Um_Compute_MeanMolecularWeight(i);
            unew = max(All.MinEgySpec, (double)(SphP[i].Entropy / gm1 * pow(SphP[i].Density * a3inv, gm1)));
            Temperature = gm1 / BOLTZMANN * PROTONMASS * SphP[i].Um_MeanMolecularWeight * unew *
                          All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;

            if (Temperature >= T_SUP_INTERPOL_LIMIT)
#endif

#ifndef LT_METAL_COOLING_WAL
                Temperature = convert_u_to_tempSTD(
                    max(All.MinEgySpec, (double)((SphP[i].Entropy + SphP[i].DtEntropy * dt) / gm1 *
                                                 pow(SphP[i].Density * a3inv, gm1))) *
                        All.UnitEnergy_in_cgs / All.UnitMass_in_g,
                    SphP[i].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam * a3inv, &ne);
#else
            Temperature = convert_u_to_tempMET(max(All.MinEgySpec, (double)((SphP[i].Entropy + SphP[i].DtEntropy * dt) /
                                                                            gm1 * pow(SphP[i].Density * a3inv, gm1))) *
                                                   All.UnitEnergy_in_cgs / All.UnitMass_in_g,
                                               Redshift, DZ, &Metallicities[0]);
#endif

            //            SphP[i].EgyRes = 0; //GM: considered at line 394

            SphP[i].XColdCloud = (float)x;
            SphP[i].Temperature = (float)Temperature;
#ifdef LT_MAX_TEMP_FEEDBACK
#ifdef DEBUG
            if (SphP[i].Temperature == 0)
            {
                MyAtLeastDouble u_final = SphP[i].Entropy * pow(rho_phys, gm1) / gm1;
                MyAtLeastDouble temp_test = U_TO_TEMP_FAC * u_final;
                VERBOSE_ALL(2,
                            "task=%d: WARNING Particle id=%llu,m=%e,entr=%e,rho=%e got zero tempersture after "
                            "cooling/sfr (T=%e/%e) !",
                            ThisTask, (unsigned long long)P[i].ID, (double)P[i].Mass, (double)SphP[i].Entropy,
                            (double)SphP[i].Density, (double)SphP[i].Temperature, temp_test);
            }
#endif // DEBUG

            if (!(SphP[i].Temperature < LT_MAX_TEMP_FEEDBACK))
            {
                // As we want to imply a maximum temperature cut, we can savely assume full ionization
                // and ignore all effects of additional physics
#ifdef DEBUG
                VERBOSE_ALL(2, "task=%d: Particle id=%llu,m=%e too large temperature after cooling/sfr %e>%e !",
                            ThisTask, (unsigned long long)P[i].ID, (float)P[i].Mass, (double)Temperature,
                            (double)LT_MAX_TEMP_FEEDBACK);
#endif // DEBUG
                SphP[i].Temperature = 0.95 * LT_MAX_TEMP_FEEDBACK;
                SphP[i].EntropyPred = SphP[i].Entropy = SphP[i].Temperature / U_TO_TEMP_FAC * gm1 / pow(rho_phys, gm1);
                if (SphP[i].DtEntropy > 0)
                {
                    SphP[i].DtEntropy = 0;
                }
#ifdef LT_STELLAREVOLUTION
                SphP[i].EgyRes = 0;
#endif // LT_STELLAREVOLUTION
#ifdef BLACK_HOLES
                SphP[i].Injected_BH_Energy = 0;
#endif // BLACK_HOLES
                SphP[i].Pressure = get_pressure(i);
            }

            assert(SphP[i].Temperature < 1.05 * LT_MAX_TEMP_FEEDBACK);
#endif // LT_MAX_TEMP_FEEDBACK

        } // if(P[i].Type == 0)
    } /* end of main loop over active particles */

    VERBOSE(2, "LT_SFR: Checking for stars to be formed");

// Ensure that the list of particles is walked the same way as befoe
#ifdef _OPENMP
    for (int il = 0; il < NActivePart; il++)
    {
        i = ActiveParticleList[il];
#else /* _OPENMP */
    for (i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif
        if (P[i].Type == KD_I_WILL_MAKE_A_STAR || P[i].Type == KD_I_WILL_BE_A_STAR)
        {
            ENDRUN_IF(N_stars + 1 >= All.MaxPartMet,
                      "On Task=%d with NumPart=%d we try to spawn %d particles. Sorry, no space "
                      "left...(All.MaxPartMet=%d)\n",
                      ThisTask, NumPart, stars_spawned, All.MaxPartMet);

            int myIMFi, mySFi;
            get_SF_index(i, &mySFi, &myIMFi);

            GENERATIONS_BUNCH = All.Generations / SFs[mySFi].Generations;

            if (All.StarBits > 0)
            {
                number_of_stars_generated = (P[i].ID >> (sizeof(MyIDType) * 8 - All.StarBits));
            }
            else
            {
                number_of_stars_generated = 0;
            }
            mass_of_star = P[i].Mass / (All.Generations - number_of_stars_generated) * GENERATIONS_BUNCH;

            if (P[i].Type == KD_I_WILL_MAKE_A_STAR)
            {
                /* here we spawn a new star particle */

                ENDRUN_IF(NumPart + stars_spawned >= All.MaxPart,
                          "On Task=%d with NumPart=%d we try to spawn %d particles. Sorry, no space "
                          "left...(All.MaxPart=%d)\n",
                          ThisTask, NumPart, stars_spawned, All.MaxPart);

                P[i].Type = 0;
                P[NumPart + stars_spawned] = P[i];
                P[NumPart + stars_spawned].Type = 4;

                if (P[NumPart + stars_spawned].Hsml > All.MaxChemSpreadL)
                {
                    P[NumPart + stars_spawned].Hsml = 0.9 * All.MaxChemSpreadL;
                }

                P[i].ID += (GENERATIONS_BUNCH << (sizeof(MyIDType) * 8 - All.StarBits));

                NextActiveParticle[NumPart + stars_spawned] = FirstActiveParticle;
                FirstActiveParticle = NumPart + stars_spawned;
                NumForceUpdate++;

                assert(P[NumPart + stars_spawned].TimeBin >= 0);
                assert(P[NumPart + stars_spawned].TimeBin < TIMEBINS);
                TimeBinCount[P[NumPart + stars_spawned].TimeBin]++;

                PrevInTimeBin[NumPart + stars_spawned] = i;
                NextInTimeBin[NumPart + stars_spawned] = NextInTimeBin[i];
                if (NextInTimeBin[i] >= 0)
                    PrevInTimeBin[NextInTimeBin[i]] = NumPart + stars_spawned;
                NextInTimeBin[i] = NumPart + stars_spawned;
                if (LastInTimeBin[P[i].TimeBin] == i)
                    LastInTimeBin[P[i].TimeBin] = NumPart + stars_spawned;

                P[NumPart + stars_spawned].Mass = mass_of_star;
                MetP[N_stars].iMass = P[NumPart + stars_spawned].Mass;

                factor = P[NumPart + stars_spawned].Mass / P[i].Mass;

#if GADGET_HYDRO == HYDRO_MFM
                mfm_add_mass(i, -P[NumPart + stars_spawned].Mass);
#endif // GADGET_HYDRO == HYDRO_MFM
                P[i].Mass -= P[NumPart + stars_spawned].Mass;
                sum_mass_stars[mySFi] += P[NumPart + stars_spawned].Mass;
                P[NumPart + stars_spawned].Mass *= (1 - SFs[mySFi].metFactorSN);

                P[NumPart + stars_spawned].pt.MetID = N_stars;
                MetP[N_stars].PID = NumPart + stars_spawned;

                MetP[N_stars].LastChemTime = SNtimesteps[mySFi][0][1];
                MPP(NumPart + stars_spawned).StellarAge = All.Time;
                if (MPP(NumPart + stars_spawned).StellarAge < 0.0)
                    MPP(NumPart + stars_spawned).StellarAge = 0.0; /* fix negative stardate in isolation */

                index_of_star = NumPart + stars_spawned;

                for (j = 0; j < LT_NMetP; j++)
                {
                    MetP[N_stars].Metals[j] = SphP[i].Metals[j] * factor;
                    SphP[i].Metals[j] *= (1 - factor);
                }
#ifdef LT_USE_KERNEL_IN_WEIGHT
                MetP[N_stars].weight = SphP[i].Density;
#else
                MetP[N_stars].weight = MetP[N_stars].iMass * P[i].TrueNGB;
#endif
#ifdef GL_CR_DUST
                MetP[N_stars].Metals[j] += factor * (SphP[i].DustL[j] + SphP[i].DustS[j]);
                SphP[i].DustL[j] *= (1. - factor);
                SphP[i].DustS[j] *= (1. - factor);
#endif

#ifdef LT_TRACK_CONTRIBUTES
                MetP[N_stars].contrib = SphP[i].contrib;
#endif
                N_stars++;

                force_add_star_to_tree(i, NumPart + stars_spawned);

                stars_spawned++;
            }
            else
            {
                /* here we turn the gas particle itself into a star */
                Particles_converted++;
                stars_converted++;

                sum_mass_stars[mySFi] += P[i].Mass;

                assert(P[i].TimeBin >= 0);
                assert(P[i].TimeBin < TIMEBINS);

                P[i].Type = 4;
                TimeBinCountSph[P[i].TimeBin]--;
                TimeBinSfr[P[i].TimeBin] -= SphP[i].Sfr;

                if (P[i].Hsml > All.MaxChemSpreadL)
                    P[i].Hsml = 0.9 * All.MaxChemSpreadL;

                P[i].pt.MetID = N_stars;
                MetP[N_stars].PID = i;

                MetP[N_stars].iMass = P[i].Mass;

                for (j = 0; j < LT_NMetP; j++)
                    MetP[N_stars].Metals[j] = SphP[i].Metals[j];
#ifdef LT_USE_KERNEL_IN_WEIGHT
                MetP[N_stars].weight = SphP[i].Density;
#else
                MetP[N_stars].weight = MetP[N_stars].iMass * P[i].TrueNGB;
#endif
#ifdef GL_CR_DUST
                for (j = 0; j < LT_NMetP; j++)
                    MetP[N_stars].Metals[j] += SphP[i].DustL[j] + SphP[i].DustS[j];
#endif

                MetP[N_stars].LastChemTime = SNtimesteps[mySFi][0][1];
                MPP(i).StellarAge = All.Time;
                if (MPP(i).StellarAge < 0.0)
                    MPP(i).StellarAge = 0.0; /* fix negative stardate in isolation */
#ifdef LT_TRACK_CONTRIBUTES
                MetP[N_stars].contrib = SphP[i].contrib;
#endif
                index_of_star = i;

                P[i].Mass *= (1 - SFs[mySFi].metFactorSN);

                N_stars++;
            }

            NextChemTime = get_NextChemTime(0, mySFi, 0x0); /* !! note: this is look-back time */

            bin = get_chemstep_bin(All.Time, All.Time_Age - NextChemTime, &chem_step, index_of_star);

            MetP[P[index_of_star].pt.MetID].ChemTimeBin = bin;

            assert(bin >= 0);
            assert(bin < TIMEBINS);

            TimeBinCountStars[bin]++;
        }
    }

    MPI_Allreduce(&stars_spawned, &tot_spawned, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&stars_converted, &tot_converted, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    if (tot_spawned > 0 || tot_converted > 0)
    {
        VERBOSE(2, "LT_SFR: spawned %d stars, converted %d gas particles into stars\n", tot_spawned, tot_converted);

        All.TotNumPart += tot_spawned;
        All.TotN_gas -= tot_converted;
        NumPart += stars_spawned;
        All.TotN_stars += tot_spawned + tot_converted;

        /* Note: N_gas is only reduced once rearrange_particle_sequence is called */

        /* Note: New tree construction can be avoided because of  `force_add_star_to_tree()' */
    }
    else
    {
        VERBOSE(2, "LT_SFR: no new stars formed");
    }

#ifdef _OPENMP
    if (stars_spawned > 0)
        build_active_particle_list();
#endif

    for (bin = 0, sfrrate = 0; bin < TIMEBINS; bin++)
        if (TimeBinCount[bin])
            sfrrate += TimeBinSfr[bin];

    MPI_Allreduce(&sfrrate, &totsfrrate, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(sfrrates, totsfrrates, SFs_dim, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(sfrrates_bydensity, totsfrrates_bydensity, NBINS_SFR_by_DENSITY, MPI_DOUBLE, MPI_SUM,
                  MYMPI_COMM_WORLD);
    MPI_Allreduce(sfrrates_byZ, totsfrrates_byZ, NBINS_SFR_by_DENSITY, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    for (i = 0; i < SFs_dim; i++)
    {
        Sum_sm += sum_sm[i];
        Sum_mass_stars += sum_mass_stars[i];
    }
    MPI_Reduce(&Sum_sm, &Total_sm, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&Sum_mass_stars, &Total_sum_mass_stars, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sum_sm[0], &total_sm[0], SFs_dim, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&sum_mass_stars[0], &total_sum_mass_stars[0], SFs_dim, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);

    if (ThisTask == 0)
    {
        if (All.TimeStep > 0)
        {
            rate = Total_sm / (All.TimeStep * units->timestep_to_physical);

            /* convert to solar masses per yr */

            double rate_in_msunperyear = rate * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

            fprintf(FdSfr, "%10.8e %10.8e %10.8e %10.8e %10.8e ", All.Time, Total_sm, totsfrrate, rate_in_msunperyear,
                    Total_sum_mass_stars);
            for (i = 0; i < SFs_dim; i++)
                fprintf(FdSfr, "%10.8e %10.8e ", totsfrrates[i],
                        total_sm[i] / (All.TimeStep * units->timestep_to_physical) * (All.UnitMass_in_g / SOLAR_MASS) /
                            (All.UnitTime_in_s / SEC_PER_YEAR));

            for (i = 0; i < SFs_dim; i++)
                fprintf(FdSfr, "%10.8e ", total_sum_mass_stars[i]);

            fprintf(FdSfrDens, "%10.8e ", All.Time);
            for (i = 0; i < NBINS_SFR_by_DENSITY; i++)
                fprintf(FdSfrDens, "%10.8e ", totsfrrates_bydensity[i]);

            fprintf(FdSfrZ, "%10.8e ", All.Time);
            for (i = 0; i < NBINS_SFR_by_Z; i++)
                fprintf(FdSfrZ, "%10.8e ", totsfrrates_byZ[i]);
        }
        else
        {
            fprintf(FdSfr, "%10.8e %10.8e 0 0 %10.8e ", All.Time, Total_sm, Total_sum_mass_stars);
            for (i = 0; i < SFs_dim; i++)
                fprintf(FdSfr, "0 ");
            for (i = 0; i < SFs_dim; i++)
                fprintf(FdSfr, "%10.8e ", total_sum_mass_stars[i]);

            fprintf(FdSfrDens, "%10.8e ", All.Time);
            for (i = 0; i < NBINS_SFR_by_DENSITY; i++)
                fprintf(FdSfrDens, "0 ");
            for (i = 0; i < NBINS_SFR_by_Z; i++)
                fprintf(FdSfrZ, "0 ");
        }
        fprintf(FdSfr, "\n");
        fprintf(FdSfrDens, "\n");
        fprintf(FdSfrZ, "\n");

        fflush(FdSfr);
    }
    VERBOSE(2, "LT_SFR: global statistics done");

#ifdef WIND_DEBUG
    unsigned long long trace_id_merk = trace_id;
    MPI_Bcast(&trace_id, 1, MPI_LONG_LONG, 0, MYMPI_COMM_WORLD);
    if (ThisTask != 0)
    {
        if (trace_id_merk == 0 && trace_id != 0)
            VERBOSE_ALL(2, "  COOLING: %d received trace_id=%llu", ThisTask, trace_id);
    }
#endif // WIND_DEBUG
}

double get_starformation_rate(int i, float *Temperature, float *xclouds)
{
    double rateOfSF;
    double a3inv, dt;
    int mySFi, myIMFi, flag;
    double tsfr;
    double factorEVP, egyhot, tcool, y, x, cloudmass;
    double Z;
    double myFactorEVP, myPhysDensThresh, myEgySpecSN, myFactorSN, myMaxSfrTimescale;
#if !defined(LT_METAL_COOLING_WAL) || defined(UM_CHEMISTRY)
    double ne;
#endif // !LT_METAL_COOLING_WAL || UM_CHEMISTRY
#if !defined(UM_CHEMISTRY) && (defined(LT_METAL_COOLING_WAL) || defined(LT_METAL_COOLING))
    double temp;
#endif // !UM_CHEMISTRY && (LT_METAL_COOLING_WAL || LT_METAL_COOLING)

#ifdef UM_CHEMISTRY
    double u;
#endif

#ifdef LT_METAL_COOLING_WAL
    double Metallicities[LT_NMet], myDZ, myRedshift;
#endif

#ifdef WINDS
    if (SphP[i].DelayTime > 0)
        return 0;
#endif

    *xclouds = 0;

    mySFi = get_SF_index(i, &mySFi, &myIMFi);
    myFactorSN = SFs[mySFi].FactorSN;
    myEgySpecSN = SFs[mySFi].EgySpecSN;
    myMaxSfrTimescale = SFs[mySFi].MaxSfrTimescale;

    if (All.ComovingIntegrationOn)
        a3inv = 1 / (All.Time * All.Time * All.Time);
    else
        a3inv = 1;

    dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;

#ifdef LT_METAL_COOLING_WAL
    ENDRUN_IF(LT_NMet < get_cool_n_el(), "2009");
    set_metallicities(i, &Metallicities[0], a3inv);
    if (All.ComovingIntegrationOn)
    {
        myRedshift = 1.0 / All.Time - 1;
        get_cool_redshift(myRedshift, &myDZ);
    }
    else
    {
        myRedshift = 0;
        myDZ = 0;
    }
#else // LT_METAL_COOLING_WAL
    Z = get_metallicity_solarunits(get_metallicity(i, Iron));
#endif // LT_METAL_COOLING_WAL

#ifdef LT_METAL_COOLING_WAL
    if (SFs[mySFi].SFTh_Zdep)
    {
#ifdef UM_METAL_COOLING
        um_ZsPoint = SphP[i].Metals;
        um_mass = P[i].Mass;
#endif // UM_METAL_COOLING

        getindex(&CoolZvalue[0], 0, ZBins - 1, &Z, &flag);

        // ToDo: account for EB_SFR_MAGNETIC
        if (flag == 0 || flag == ZBins - 1)
        {
            myFactorEVP = SFs[mySFi].FEVP[flag];
            myPhysDensThresh = SFs[mySFi].PhysDensThresh[flag];
        }
        else
        {
            x = (Z - CoolZvalue[flag + 1]) / (CoolZvalue[flag + 1] - CoolZvalue[flag]);
            myFactorEVP = SFs[mySFi].FEVP[flag] * (1 - x) + SFs[mySFi].FEVP[flag + 1] * x;
            myPhysDensThresh = SFs[mySFi].PhysDensThresh[flag] * (1 - x) + SFs[mySFi].PhysDensThresh[flag + 1] * x;
        }
    }
    else
#endif
    {
        myFactorEVP = SFs[mySFi].FEVP[0];
        myPhysDensThresh = SFs[mySFi].PhysDensThresh[0];

#ifdef UM_MYTH
        myPhysDensThresh = UM_MYTH; // 1.e20,  0.480065; //<-new th
#endif
    }

    flag = 1; /* default is normal cooling */

    if (SphP[i].Density * a3inv >= myPhysDensThresh)
        flag = 0;

    if (All.ComovingIntegrationOn)
        if (SphP[i].Density < All.OverDensThresh)
            flag = 1;

    if (flag == 1)
    {

#if !defined(LT_METAL_COOLING_WAL) || defined(UM_CHEMISTRY)
        ne = SphP[i].elec;
#endif // !LT_METAL_COOLING_WAL || UM_CHEMISTRY
        const double gm1 = get_gamma_minus1(i);
#ifdef UM_CHEMISTRY
        Um_Compute_MeanMolecularWeight(i);
        u = max(All.MinEgySpec, (double)(SphP[i].Entropy / gm1 * pow(SphP[i].Density * a3inv, gm1)));
        *Temperature = gm1 / BOLTZMANN * PROTONMASS * SphP[i].Um_MeanMolecularWeight * u * All.UnitPressure_in_cgs /
                       All.UnitDensity_in_cgs;
        if (*Temperature >= T_SUP_INTERPOL_LIMIT)
#endif

#ifndef LT_METAL_COOLING_WAL
            *Temperature = convert_u_to_tempSTD(
                max(All.MinEgySpec,
                    (double)((SphP[i].Entropy + SphP[i].DtEntropy * dt) / gm1 * pow(SphP[i].Density * a3inv, gm1))) *
                    All.UnitEnergy_in_cgs / All.UnitMass_in_g,
                SphP[i].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam * a3inv, &ne);
#else
        *Temperature = convert_u_to_tempMET(max(All.MinEgySpec, (double)((SphP[i].Entropy + SphP[i].DtEntropy * dt) /
                                                                         gm1 * pow(SphP[i].Density * a3inv, gm1))) *
                                                All.UnitEnergy_in_cgs / All.UnitMass_in_g,
                                            myRedshift, myDZ, &Metallicities[0]);
#endif

        return 0;
    }

    tsfr = sqrt(myPhysDensThresh / (SphP[i].Density * a3inv)) * myMaxSfrTimescale;

    factorEVP = pow(SphP[i].Density * a3inv / myPhysDensThresh, -0.8) * myFactorEVP;

    egyhot = myEgySpecSN / (1 + factorEVP) + All.EgySpecCold;

#if !defined(LT_METAL_COOLING_WAL) || defined(UM_CHEMISTRY)
    ne = SphP[i].elec;
#endif // !LT_METAL_COOLING_WAL || UM_CHEMISTRY

#ifndef LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
#ifndef UM_CHEMISTRY
    if ((tcool = GetCoolingTimeZ(egyhot, SphP[i].Density * a3inv, &ne, Z, &temp)) == 0)
        tcool = 1e13;
#else
    if ((tcool = Um_GetCoolingTime(egyhot, SphP[i].Density * a3inv, &ne, Z, i)) == 0)
        tcool = 1e13;
#endif
#else
    tcool = GetCoolingTimeSTD(egyhot, SphP[i].Density * a3inv, &ne);
#endif
#else /* LT_METAL_COOLING_WAL */
#ifndef UM_CHEMISTRY
#ifdef GL_DUST_COOLING
    double DL, DS; // particle's dust to gas ration for small and big grains
    if (P[i].Mass < 0)
    {
        DL = get_metalmass(SphP[i].DustL) / P[i].Mass;
        DS = get_metalmass(SphP[i].DustS) / P[i].Mass;
    }
    else
    {
        DL = 0;
        DS = 0;
    }
    tcool = GetCoolingTimeMET(egyhot, SphP[i].Density * a3inv, myRedshift, myDZ, DL, DS, &Metallicities[0], &temp);
#else // GL_DUST_COOLING
    tcool = GetCoolingTimeMET(egyhot, SphP[i].Density * a3inv, myRedshift, myDZ, &Metallicities[0], &temp);
#endif // GL_DUST_COOLING
#else
    if ((tcool = Um_GetCoolingTime(egyhot, SphP[i].Density * a3inv, &ne, &Metallicities[0], myRedshift, myDZ, i)) == 0)
        tcool = 1e13;
#endif
#endif


    y = tsfr / tcool * egyhot / (myFactorSN * myEgySpecSN - (1 - myFactorSN) * All.EgySpecCold);

    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));


    cloudmass = x * P[i].Mass;

    *xclouds = x;

    rateOfSF = (1 - myFactorSN) * cloudmass / tsfr;

    /* convert to solar masses per yr */

    rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

    return rateOfSF;
}

#ifdef SUBFIND

double get_starformation_rate_subfind(int i, float *Temperature, float *xclouds)
{
    double rateOfSF;
    double a3inv, dt;
    int mySFi, myIMFi, flag;
    double tsfr;
    double factorEVP, egyhot, tcool, y, x, cloudmass;
    double Z;
    double myFactorEVP, myPhysDensThresh, myEgySpecSN, myFactorSN, myMaxSfrTimescale;
#if !defined(LT_METAL_COOLING_WAL) || defined(UM_CHEMISTRY)
    double ne;
#endif // !LT_METAL_COOLING_WAL || UM_CHEMISTRY
#if !defined(UM_CHEMISTRY) && (defined(LT_METAL_COOLING_WAL) || defined(LT_METAL_COOLING))
    double temp;
#endif // !UM_CHEMISTRY && (LT_METAL_COOLING_WAL || LT_METAL_COOLING)

#ifdef UM_CHEMISTRY
    double u;
#endif

#ifdef LT_METAL_COOLING_WAL
    double Metallicities[LT_NMet], myDZ, myRedshift;
#endif

#ifdef WINDS
    if (SphP[P[i].origindex].DelayTime > 0)
        return 0;
#endif

    *xclouds = 0;

    mySFi = get_SF_index_subfind(i, &mySFi, &myIMFi);

    myFactorSN = SFs[mySFi].FactorSN;
    myEgySpecSN = SFs[mySFi].EgySpecSN;
    myMaxSfrTimescale = SFs[mySFi].MaxSfrTimescale;

    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1;
    }

    dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval;

#ifdef LT_METAL_COOLING_WAL
    ENDRUN_IF(LT_NMet < get_cool_n_el(), "3009");
    set_metallicities_subfind(i, &Metallicities[0], a3inv);
    if (All.ComovingIntegrationOn)
    {
        myRedshift = 1.0 / All.Time - 1;
        get_cool_redshift(myRedshift, &myDZ);
    }
    else
    {
        myRedshift = 0;
        myDZ = 0;
    }
#else // LT_METAL_COOLING_WAL
#ifndef LT_METAL_COOLING_on_SMOOTH_Z
    Z = get_metallicity_solarunits(get_metallicity_subfind(i));
#else // LT_METAL_COOLING_on_SMOOTH_Z
    double metalmass = get_metalmass(SphP[P[i].origindex].Metals);
    if (metalmass > 0)
    {
        Z = get_metallicity_solarunits(SphP[P[i].origindex].Zsmooth * SphP[P[i].origindex].Metals[Iron] / metalmass);
    }
    else
    {
        Z = NO_METAL;
    }
#endif // LT_METAL_COOLING_on_SMOOTH_Z
#endif // LT_METAL_COOLING_WAL

#ifdef LT_METAL_COOLING_WAL
    if (SFs[mySFi].SFTh_Zdep)
    {
#if defined(UM_METAL_COOLING)
        um_ZsPoint = SphP[P[i].origindex].Metals;
        um_mass = P[i].Mass;
#endif

        getindex(&CoolZvalue[0], 0, ZBins - 1, &Z, &flag);
        if (flag == 0 || flag == ZBins - 1)
        {
            myFactorEVP = SFs[mySFi].FEVP[flag];
            myPhysDensThresh = SFs[mySFi].PhysDensThresh[flag];
        }
        else
        {
            x = (Z - CoolZvalue[flag + 1]) / (CoolZvalue[flag + 1] - CoolZvalue[flag]);
            myFactorEVP = SFs[mySFi].FEVP[flag] * (1 - x) + SFs[mySFi].FEVP[flag + 1] * x;
            myPhysDensThresh = SFs[mySFi].PhysDensThresh[flag] * (1 - x) + SFs[mySFi].PhysDensThresh[flag + 1] * x;
        }
    }
    else
#endif
    {
        myFactorEVP = SFs[mySFi].FEVP[0];
        myPhysDensThresh = SFs[mySFi].PhysDensThresh[0];

#ifdef UM_MYTH
        myPhysDensThresh = UM_MYTH; // 1.e20,  0.480065; //<-new th
#endif
    }

    flag = 1; /* default is normal cooling */

    if (SphP[P[i].origindex].Density * a3inv >= myPhysDensThresh)
        flag = 0;

    if (All.ComovingIntegrationOn)
        if (SphP[P[i].origindex].Density < All.OverDensThresh)
            flag = 1;

    if (flag == 1)
    {

#if !defined(LT_METAL_COOLING_WAL) || defined(UM_CHEMISTRY)
        ne = SphP[P[i].origindex].elec;
#endif // !LT_METAL_COOLING_WAL || UM_CHEMISTRY
        const double gm1 = get_gamma_minus1(i);
#ifdef UM_CHEMISTRY
        Um_Compute_MeanMolecularWeight(i);
        u = max(All.MinEgySpec,
                (double)(SphP[P[i].origindex].Entropy / gm1 * pow(SphP[P[i].origindex].Density * a3inv, gm1)));
        *Temperature = gm1 / BOLTZMANN * PROTONMASS * SphP[P[i].origindex].Um_MeanMolecularWeight * u *
                       All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
        if (*Temperature >= T_SUP_INTERPOL_LIMIT)
#endif

#ifndef LT_METAL_COOLING_WAL
            *Temperature = convert_u_to_tempSTD(
                max(All.MinEgySpec, (double)((SphP[P[i].origindex].Entropy + SphP[P[i].origindex].DtEntropy * dt) /
                                             gm1 * pow(SphP[P[i].origindex].Density * a3inv, gm1))) *
                    All.UnitEnergy_in_cgs / All.UnitMass_in_g,
                SphP[P[i].origindex].Density * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam * a3inv, &ne);
#else
        *Temperature = convert_u_to_tempMET(
            max(All.MinEgySpec, (double)((SphP[P[i].origindex].Entropy + SphP[P[i].origindex].DtEntropy * dt) / gm1 *
                                         pow(SphP[P[i].origindex].Density * a3inv, gm1))) *
                All.UnitEnergy_in_cgs / All.UnitMass_in_g,
            myRedshift, myDZ, &Metallicities[0]);
#endif

        return 0;
    }

    tsfr = sqrt(myPhysDensThresh / (SphP[P[i].origindex].Density * a3inv)) * myMaxSfrTimescale;

    factorEVP = pow(SphP[P[i].origindex].Density * a3inv / myPhysDensThresh, -0.8) * myFactorEVP;

    egyhot = myEgySpecSN / (1 + factorEVP) + All.EgySpecCold;

#if !defined(LT_METAL_COOLING_WAL) || defined(UM_CHEMISTRY)
    ne = SphP[P[i].origindex].elec;
#endif // !LT_METAL_COOLING_WAL || UM_CHEMISTRY

#ifndef LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
#ifndef UM_CHEMISTRY
    if ((tcool = GetCoolingTimeZ(egyhot, SphP[P[i].origindex].Density * a3inv, &ne, Z, &temp)) == 0)
        tcool = 1e13;
#else
    if ((tcool = Um_GetCoolingTime(egyhot, SphP[P[i].origindex].Density * a3inv, &ne, Z, i)) == 0)
        tcool = 1e13;
#endif
#else
    tcool = GetCoolingTimeSTD(egyhot, SphP[P[i].origindex].Density * a3inv, &ne);
#endif
#else /* LT_METAL_COOLING_WAL */
#ifndef UM_CHEMISTRY

#ifdef GL_DUST_COOLING
    double DL, DS; // particle's dust to gas ration for small and big grains
    if (P[i].Mass < 0)
    {
        DL = get_metalmass(SphP[i].DustL) / P[i].Mass;
        DS = get_metalmass(SphP[i].DustS) / P[i].Mass;
    }
    else
    {
        DL = 0;
        DS = 0;
    }

    tcool = GetCoolingTimeMET(egyhot, SphP[P[i].origindex].Density * a3inv, myRedshift, myDZ, DL, DS, &Metallicities[0],
                              &temp);
#else // GL_DUST_COOLING
    tcool = GetCoolingTimeMET(egyhot, SphP[P[i].origindex].Density * a3inv, myRedshift, myDZ, &Metallicities[0], &temp);
#endif // GL_DUST_COOLING

#else
    if ((tcool = Um_GetCoolingTime(egyhot, SphP[i].Density * a3inv, &ne, &Metallicities[0], myRedshift, myDZ, i)) == 0)
        tcool = 1e13;
#endif
#endif

    y = tsfr / tcool * egyhot / (myFactorSN * myEgySpecSN - (1 - myFactorSN) * All.EgySpecCold);

    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

    cloudmass = x * P[i].Mass;

    *xclouds = x;

    rateOfSF = (1 - myFactorSN) * cloudmass / tsfr;

    /* convert to solar masses per yr */

    rateOfSF *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

    return rateOfSF;
}

#endif


int write_eff_model(int num, int mySFi)
{
    char string[200];

    FILE *file;

    int i;

    if (ThisTask == 0)
    {
        if (num < 0)
            sprintf(string, "%s.%02d", "eff_model.dat", mySFi);
        else
            sprintf(string, "eff_model.dat.%02d.%03d", mySFi, num);

        /* write data on a file to save time in the next run */
        if ((file = fopen(string, "w")) == NULL)
        {
            printf("it's impossible to write in <%s> \n", string);
            fclose(file);
            return -1;
        }
        else
        {
            fprintf(file, "%d\n", ZBins);
            for (i = 0; i < ZBins; i++)
                fprintf(file, "%lg %lg %lg\n", CoolZvalue[i], SFs[mySFi].PhysDensThresh[i], SFs[mySFi].FEVP[i]);
            fclose(file);
        }
    }
    return 0;
}

int read_eff_model(int num, int mySFi)
{
    int i, j, k, Zbins;

    char string[200];

    FILE *file;

    k = 0;

    if (num < 0)
        sprintf(string, "eff_model.dat.%02d", mySFi);
    else
        sprintf(string, "eff_model.dat.%02d.%03d", mySFi, num);

    if ((file = fopen(string, "r")) != 0x0)
    {
        if (fscanf(file, "%d", &Zbins) < 1)
        {
            return k;
        }

        ENDRUN_IF(Zbins != ZBins, "# of bins in %s is different than that found in cooling tables!\n", string);

        for (i = j = 0; i < Zbins; i++)
        {
            j += fscanf(file, "%*g %lg %lg", &SFs[mySFi].PhysDensThresh[i], &SFs[mySFi].FEVP[i]);
            printf("[Fe/H]: %-4.3lg - RhoTh: %-9.7lg  - fEVP: %-9.7lg\n", CoolZvalue[i], SFs[mySFi].PhysDensThresh[i],
                   SFs[mySFi].FEVP[i]);
        }
        if (j != Zbins * 2)
        {
            k = 0;
        }
        else
        {
            k = 1;
        }
    }
    return k;
}

void init_clouds_cm(int mode, double *PhDTh, double *fEVP, double EgySN, double FSN, double SFt, int Zbins,
                    double *ZArray)
{
    int i;

    int *offset, *counts;

    if (mode > 0)
    {
        if (ThisTask == 0)
            printf("\n\ninitialize effective model.. \n"
                   "it is needed to recalculate metallicity dependence for effective model.. it will take a while\n");
        fflush(stdout);
        for (i = 0; i < Zbins; i++)
            PhDTh[i] = 0;
    }

    /* distribute work over processor */
    offset = (int *)mymalloc("offset", sizeof(int) * NTask);
    counts = (int *)mymalloc("counts", sizeof(int) * NTask);

    offset[0] = 0;
    counts[0] = Zbins / NTask + (0 < (int)fmod(Zbins, NTask));
    for (i = 1; i < NTask; i++)
    {
        if ((counts[i] = Zbins / NTask + (i < (int)fmod(Zbins, NTask))))
            offset[i] = offset[i - 1] + counts[i - 1];
        else
            offset[i] = 0;
    }

    /* ThisTask will calculate params for its range of Z */
    for (i = 0; i < counts[ThisTask]; i++)
        init_clouds(mode, EgySN, FSN, SFt, ZArray[i + offset[ThisTask]], &PhDTh[i + offset[ThisTask]],
                    &fEVP[i + offset[ThisTask]]);

    MPI_Barrier(MYMPI_COMM_WORLD);
    if (mode > 0)
    {
        /* collect informations */
        MPI_Allgatherv((void *)&PhDTh[offset[ThisTask]], counts[ThisTask], MPI_DOUBLE, (void *)&PhDTh[0], counts,
                       offset, MPI_DOUBLE, MYMPI_COMM_WORLD);

        MPI_Allgatherv((void *)&fEVP[offset[ThisTask]], counts[ThisTask], MPI_DOUBLE, (void *)&fEVP[0], counts, offset,
                       MPI_DOUBLE, MYMPI_COMM_WORLD);

        MPI_Barrier(MYMPI_COMM_WORLD);
    }

    myfree(counts);
    myfree(offset);

    return;
}

void init_clouds(int mode, double egySN, double fSN, double SFt, double Z, double *PhDTh, double *fEVP)
{
#ifdef LT_METAL_COOLING
    int Zbin;
#endif
    double A0, dens, tcool, coolrate, egyhot, x, u4, meanweight, temp;
#ifndef LT_METAL_COOLING_WAL
    double ne;
#endif // !LT_METAL_COOLING_WAL
    double tsfr, y, peff, fac, neff, egyeff, factorEVP, sigma, thresholdStarburst;
    const double gm1 = GAMMA_MINUS1;
    double u_to_temp_fact = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * All.UnitEnergy_in_cgs /
                            All.UnitMass_in_g * gm1;

#ifdef UM_MET_IN_LT_COOLING
    float dummy_Metals[LT_NMet];

    memset(dummy_Metals, 0, sizeof(float) * LT_NMet);
    um_ZsPoint = dummy_Metals;
#endif

#ifdef LT_METAL_COOLING_WAL
    double Metallicities[LT_NMet];

#ifdef LS_SF_RUNAWAY_TREATMENT
    /*
    Note that as far as I can see, without this compiler switch,
    init_clouds cannot work because temperature, redshift, and most
    importantly the Metallicity array are declared but
    not assigned values values.
    For the moment I encapsulated all my modifications in idefs,
    but I think it might be worth discussing whether to extend
    the modifications to become default.
    */
    dens = *PhDTh;
    memset(Metallicities, 0, sizeof(double) * LT_NMet);
    /*
    Note: we assume zero metallicity to compute the runaway threshold
    and print the eEOS in eos.txt. At the moment, since this operation
    is done only once at the beginning and the threshold will be used
    for all particles indiscriminately, it is in principle
    inconsistent to use this for enriched gas. Besides, it follows the
    same principle as the rest of the code,
    to use the redshift zero table for initialisation.
    Of course all these aspects could be refined further.
    */
    Metallicities[HydPos] = HYDROGEN_MASSFRAC * dens * All.UnitDensity_in_cgs / PROTONMASS;
    Metallicities[HelPos] = (1 - HYDROGEN_MASSFRAC);

    Redshift = 0;
    DZ = 0;
#endif
#endif

    /*
     * calculating effective model parameters you have different choices
     * in the case you have settled on LT_METAL_COOLING:
     *   (a) make A0 parameter (onset of thermal instability) dependent on
     *       metal cooling
     *   (b) leave A0 parameter independent of metal cooling
     * the following code trigger this choice as stated by compiler directives
     */

#if defined(LT_METAL_COOLING)
    for (Zbin = ZBins - 1; Z < CoolZvalue[Zbin] && Zbin > 0; Zbin--)
        ;
#endif

    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)); /* note: assuming FULL ionization */
    u4 = 1 / meanweight * (1.0 / gm1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
    u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

    A0 = *fEVP;

#ifndef LT_METAL_COOLING_WAL
    if (All.ComovingIntegrationOn)
    {
        All.Time = 1.0; /* to be guaranteed to get z=0 rate */
        set_cosmo_factors_for_current_time();
        IonizeParams();
    }
#endif

    if (mode)
    {

        egyhot = egySN / A0;

        // ToDo: there is no reason why there should be an if/else here
        if (All.ComovingIntegrationOn)
            dens = 1.0e6 * 3 * All.Hubble * All.Hubble /
                   (8 * M_PI * All.GravityConstantInternal); // these lines are identical!
        else
            dens = 1.0e6 * 3 * All.Hubble * All.Hubble /
                   (8 * M_PI * All.GravityConstantInternal); // these lines are identical!


#ifndef LT_METAL_COOLING_WAL

        ne = 1.0;
        SetZeroIonization();

#ifdef LT_METAL_COOLING
        /* #ifndef UM_CHEMISTRY */
        /*       if((tcool = GetCoolingTimeZ(egyhot, dens, &ne, Z)) == 0) */
        /* 	tcool = 1e13; */
        /* #else */
        /*       /\* what about Um_GetCoolingTime() here ???? I don't know!!! *\/ */
        /* #endif */
        if ((tcool = GetCoolingTimeZ(egyhot, dens, &ne, Z, &temp)) == 0)
            tcool = 1e13;
#else
        tcool = GetCoolingTimeSTD(egyhot, dens, &ne);
#endif
#else
#ifdef GL_DUST_COOLING
        if ((tcool = GetCoolingTimeMET(egyhot, dens, Redshift, DZ, DL, DS, &Metallicities[0], &temp)) == 0)
            tcool = 1e13;
#else // GL_DUST_COOLING
        if ((tcool = GetCoolingTimeMET(egyhot, dens, Redshift, DZ, &Metallicities[0], &temp)) == 0)
            tcool = 1e13;
#endif // GL_DUST_COOLING
#endif

        coolrate = egyhot / tcool / dens;

        x = (egyhot - u4) / (egyhot - All.EgySpecCold);

        *PhDTh = x / pow(1 - x, 2) * (fSN * egySN - (1 - fSN) * All.EgySpecCold) / (SFt * coolrate);
        if (mode == 2)
            return;
    }

    dens = *PhDTh * 10;

    do
    {

        tsfr = sqrt(*PhDTh / (dens)) * SFt;
        factorEVP = pow(dens / *PhDTh, -0.8) * *fEVP;
        egyhot = egySN / (1 + factorEVP) + All.EgySpecCold;

#ifndef LT_METAL_COOLING_WAL
        ne = 0.5;
#endif // !LT_METAL_COOLING_WAL

#ifdef LS_SF_RUNAWAY_TREATMENT
        Metallicities[HydPos] = HYDROGEN_MASSFRAC * dens * All.UnitDensity_in_cgs / PROTONMASS;
        temp = egyhot * u_to_temp_fact;
#endif
#ifndef LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
        if ((tcool = GetCoolingTimeZ(egyhot, dens, &ne, Z, &temp)) == 0)
            tcool = 1e13;
#else
        tcool = GetCoolingTimeSTD(egyhot, dens, &ne);
#endif
#else
#ifdef GL_DUST_COOLING
        tcool = GetCoolingTimeMET(egyhot, dens, Redshift, DZ, DL, DS, &Metallicities[0], &temp);
#else // GL_DUST_COOLING
        tcool = GetCoolingTimeMET(egyhot, dens, Redshift, DZ, &Metallicities[0], &temp);
#endif // GL_DUST_COOLING
#endif

        y = tsfr / tcool * egyhot / (fSN * egySN - (1 - fSN) * All.EgySpecCold);
        if (y < 1e-4)
            x = y;
        else
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));


        egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

        peff = gm1 * dens * egyeff;
        fac = 1 / (log(dens * 1.025) - log(dens));
        dens *= 1.025;
        neff = -log(peff) * fac;

        tsfr = sqrt(*PhDTh / (dens)) * SFt;
        factorEVP = pow(dens / *PhDTh, -0.8) * *fEVP;
        egyhot = egySN / (1 + factorEVP) + All.EgySpecCold;

#ifndef LT_METAL_COOLING_WAL
        ne = 0.5;
#endif // !LT_METAL_COOLING_WAL

#ifdef LS_SF_RUNAWAY_TREATMENT
        Metallicities[HydPos] = HYDROGEN_MASSFRAC * dens * All.UnitDensity_in_cgs / PROTONMASS;
        temp = egyhot * u_to_temp_fact;
#endif

#ifndef LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
        if ((tcool = GetCoolingTimeZ(egyhot, dens, &ne, Z, &temp)) == 0)
            tcool = 1e13;
#else
        tcool = GetCoolingTimeSTD(egyhot, dens, &ne);
#endif
#else
#ifdef GL_DUST_COOLING
        tcool = GetCoolingTimeMET(egyhot, dens, Redshift, DZ, DL, DS, &Metallicities[0], &temp);
#else // GL_DUST_COOLING
        tcool = GetCoolingTimeMET(egyhot, dens, Redshift, DZ, &Metallicities[0], &temp);
#endif // GL_DUST_COOLING

#endif

        y = tsfr / tcool * egyhot / (fSN * egySN - (1 - fSN) * All.EgySpecCold);
        if (y < 1e-4)
            x = y;
        else
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

        egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

        peff = gm1 * dens * egyeff;

        neff += log(peff) * fac;
#ifdef LS_SF_RUNAWAY_TREATMENT
#ifdef DEBUG
        VERBOSE(0, "dens %g | Metallicities[HydPos] %g | tcool %g | y %g | x %g | egyeff %g | egyhot %g\n", dens,
                Metallicities[HydPos], tcool, y, x, egyeff, egyhot);
#endif
#endif
    } while (neff > 4.0 / 3);

    thresholdStarburst = dens;
#ifdef LS_SF_RUNAWAY_TREATMENT
    RunawayDensityThreshold = thresholdStarburst;
#endif
    integrate_sfr(*PhDTh, *fEVP, egySN, fSN, SFt, Z);

    sigma = 10.0 / All.Hubble * 1.0e-10 / pow(1.0e-3, 2);

    printf("\n[%i] .: [Fe] = %.3g :. \n"
           "\nA0= %g  \n"
           "Computed: PhysDensThresh= %g  (int units)         %g h^2 cm^-3\n"
           "EXPECTED FRACTION OF COLD GAS AT THRESHOLD = %g\n\n"
           "tcool=%g dens=%g egyhot=%g\n"
           "Run-away sets in for dens=%g\n"
           "Dynamic range for quiescent star formation= %g\n\n"
           "Isotherm sheet central density: %g   z0=%g\n",
           ThisTask,
#ifdef LT_METAL_COOLING
           CoolZvalue[Zbin],
#else
           (double)0,
#endif
           A0, *PhDTh, *PhDTh / (PROTONMASS / HYDROGEN_MASSFRAC / All.UnitDensity_in_cgs), x, tcool, dens, egyhot,
           thresholdStarburst, thresholdStarburst / *PhDTh,
           M_PI * All.GravityConstantInternal * sigma * sigma / (2 * gm1) / u4,
           gm1 * u4 / (2 * M_PI * All.GravityConstantInternal * sigma));
    fflush(stdout);

#ifndef LT_METAL_COOLING_WAL
    if (All.ComovingIntegrationOn)
    {
        All.Time = All.TimeBegin;
        set_cosmo_factors_for_current_time();
        IonizeParams();
    }
#endif

    return;
}

void integrate_sfr(double PhDTh, double fEVP, double egySN, double fSN, double SFt, double Z)
{
    double rho0, rho, rho2, q, dz, gam, sigma = 0, sigma_u4, sigmasfr = 0, P1;
#ifndef LT_METAL_COOLING_WAL
    double ne;
#endif // !LT_METAL_COOLING_WAL
    double x = 0, y, P, P2, x2, y2, tsfr2, factorEVP2, egyhot2, tcool2, drho, dq;
    double meanweight, u4, tsfr, tcool, egyhot, factorEVP, egyeff, egyeff2, temp;
    char buff[30];
    FILE *fd;

#ifdef LT_METAL_COOLING_WAL
    double Metallicities[LT_NMet];
#ifdef LS_SF_RUNAWAY_TREATMENT
    memset(Metallicities, 0, sizeof(double) * LT_NMet);
    Metallicities[HydPos] = HYDROGEN_MASSFRAC * PhDTh * All.UnitDensity_in_cgs / PROTONMASS;
    Metallicities[HelPos] = (1 - HYDROGEN_MASSFRAC);
#endif
#endif
    const double gm1 = GAMMA_MINUS1;
    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)); /* note: assuming FULL ionization */
    u4 = 1 / meanweight * (1.0 / gm1) * (BOLTZMANN / PROTONMASS) * 1.0e4;
    u4 *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;
#ifdef LS_SF_RUNAWAY_TREATMENT
    double u_to_temp_fact = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * All.UnitEnergy_in_cgs /
                            All.UnitMass_in_g * gm1;
#endif

#ifndef LT_METAL_COOLING_WAL
    if (All.ComovingIntegrationOn)
    {
        All.Time = 1.0; /* to be guaranteed to get z=0 rate */
        set_cosmo_factors_for_current_time();
        IonizeParams();
    }
#endif

    if (ThisTask == 0)
        fd = fopen("eos.txt", "w");
    else
        fd = 0;

    for (rho = PhDTh; rho <= 1000 * PhDTh; rho *= 1.1)
    {
        tsfr = sqrt(PhDTh / rho) * SFt;

        factorEVP = pow(rho / PhDTh, -0.8) * fEVP;

        egyhot = egySN / (1 + factorEVP) + All.EgySpecCold;

#ifndef LT_METAL_COOLING_WAL
        ne = 1.0;
#endif // !LT_METAL_COOLING_WAL

#ifdef LS_SF_RUNAWAY_TREATMENT
        Metallicities[HydPos] = HYDROGEN_MASSFRAC * rho * All.UnitDensity_in_cgs / PROTONMASS;
        temp = egyhot * u_to_temp_fact;
#endif

#ifndef LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
        if ((tcool = GetCoolingTimeZ(egyhot, rho, &ne, Z, &temp)) == 0)
            tcool = 1e13;
#else
        tcool = GetCoolingTimeSTD(egyhot, rho, &ne);
#endif
#else
#ifdef GL_DUST_COOLING
        tcool = GetCoolingTimeMET(egyhot, rho, Redshift, DZ, DL, DS, &Metallicities[0], &temp);
#else // GL_DUST_COOLING
        tcool = GetCoolingTimeMET(egyhot, rho, Redshift, DZ, &Metallicities[0], &temp);
#endif // GL_DUST_COOLING
#endif

        y = tsfr / tcool * egyhot / (fSN * egySN - (1 - fSN) * All.EgySpecCold);
        if (y < 1e-4)
            x = y;
        else
            x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));
        egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

        P = gm1 * rho * egyeff;

        if (ThisTask == 0)
        {
            fprintf(fd, "%g %g\n", rho, P);
        }
    }

    if (ThisTask == 0)
        fclose(fd);

    sprintf(buff, "sfrrate.%03d.txt", sfrrate_filenum);
    if (ThisTask == 0)
        fd = fopen(buff, "w");
    else
        fd = 0;

    for (rho0 = PhDTh; rho0 <= 10000 * PhDTh; rho0 *= 1.02)
    {
        rho = rho0;
        q = 0;
        dz = 0.001;

        sigma = sigmasfr = sigma_u4 = 0;

        while (rho > 0.0001 * rho0)
        {
            if (rho > PhDTh)
            {
                tsfr = sqrt(PhDTh / rho) * SFt;

                factorEVP = pow(rho / PhDTh, -0.8) * fEVP;

                egyhot = egySN / (1 + factorEVP) + All.EgySpecCold;

#ifndef LT_METAL_COOLING_WAL
                ne = 1.0;
#endif // !LT_METAL_COOLING_WAL

#ifdef LS_SF_RUNAWAY_TREATMENT
                Metallicities[HydPos] = HYDROGEN_MASSFRAC * rho * All.UnitDensity_in_cgs / PROTONMASS;
                temp = egyhot * u_to_temp_fact;
#endif

#ifndef LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
                if ((tcool = GetCoolingTimeZ(egyhot, rho, &ne, Z, &temp)) == 0)
                    tcool = 1e13;
#else
                tcool = GetCoolingTimeSTD(egyhot, rho, &ne);
#endif
#else
#ifdef GL_DUST_COOLING
                tcool = GetCoolingTimeMET(egyhot, rho, Redshift, DZ, DL, DS, &Metallicities[0], &temp);
#else // GL_DUST_COOLING
                tcool = GetCoolingTimeMET(egyhot, rho, Redshift, DZ, &Metallicities[0], &temp);
#endif // GL_DUST_COOLING

#endif

                y = tsfr / tcool * egyhot / (fSN * egySN - (1 - fSN) * All.EgySpecCold);
                if (y < 1e-4)
                    x = y;
                else
                    x = 1 + 1 / (2 * y) - sqrt(1 / y + 1 / (4 * y * y));

                egyeff = egyhot * (1 - x) + All.EgySpecCold * x;

                P = P1 = gm1 * rho * egyeff;

                rho2 = 1.1 * rho;
                tsfr2 = sqrt(PhDTh / rho2) * SFt;
                factorEVP2 = pow(rho2 / PhDTh, -0.8) * fEVP;
                egyhot2 = egySN / (1 + factorEVP2) + All.EgySpecCold;

#ifdef LS_SF_RUNAWAY_TREATMENT
                Metallicities[HydPos] = HYDROGEN_MASSFRAC * rho2 * All.UnitDensity_in_cgs / PROTONMASS;
                temp = egyhot2 * u_to_temp_fact;
#endif

#ifndef LT_METAL_COOLING_WAL
#ifdef LT_METAL_COOLING
                if ((tcool2 = GetCoolingTimeZ(egyhot2, rho2, &ne, Z, &temp)) == 0)
                    tcool2 = 1e13;
#else
                tcool2 = GetCoolingTimeSTD(egyhot2, rho2, &ne);
#endif
#else
#ifdef GL_DUST_COOLING
                tcool2 = GetCoolingTimeMET(egyhot2, rho2, Redshift, DZ, DL, DS, &Metallicities[0], &temp);
#else // GL_DUST_COOLING
                tcool2 = GetCoolingTimeMET(egyhot2, rho2, Redshift, DZ, &Metallicities[0], &temp);
#endif // GL_DUST_COOLING
#endif

                y2 = tsfr2 / tcool2 * egyhot2 / (fSN * egySN - (1 - fSN) * All.EgySpecCold);
                if (y2 < 1e-4)
                    x2 = y2;
                else
                    x2 = 1 + 1 / (2 * y2) - sqrt(1 / y2 + 1 / (4 * y2 * y2));

                egyeff2 = egyhot2 * (1 - x2) + All.EgySpecCold * x2;
                P2 = gm1 * rho2 * egyeff2;

                gam = log(P2 / P1) / log(rho2 / rho);
            }
            else
            {
                tsfr = 0;

                P = gm1 * rho * u4;
                gam = 1.0;

                sigma_u4 += rho * dz;
            }

            drho = q;
            dq = -(gam - 2) / rho * q * q - 4 * M_PI * All.GravityConstantInternal / (gam * P) * rho * rho * rho;

            sigma += rho * dz;
            if (tsfr > 0)
            {
                sigmasfr += (1 - SFs[sfrrate_filenum].FactorSN) * rho * x / tsfr * dz;
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

#ifndef LT_METAL_COOLING_WAL
    if (All.ComovingIntegrationOn)
    {
        All.Time = All.TimeBegin;
        set_cosmo_factors_for_current_time();
        IonizeParams();
    }
#endif

    if (ThisTask == 0)
        fclose(fd);
}

#endif

#ifdef UM_CHECK
void Um_cooling_check(void)
{
    int i, ifunc;
    double a_start, a_end, um_u;

    for (i = 0; i < N_gas; i++)
    {
        a_start = All.TimeBegin * exp(P[i].Ti_begstep * All.Timebase_interval);
        a_end = All.TimeBegin * exp(P[i].Ti_endstep * All.Timebase_interval);

        if (ThisTask == 0 && i == 1)
        {
            printf("--- in  Um_cooling_check: start = %g, end = %g\n", a_start, a_end);
            printf("--- From  Um_cooling_check: Step %d, Time: %g, Systemstep: %g, Dloga=log(Time-Systemstep): %g\n",
                   All.NumCurrentTiStep, All.Time, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
        }
        if (P[i].Type == 0)
            ifunc = compute_abundances(1, i, a_start, a_end, &um_u);
    }
}
#endif

#endif
