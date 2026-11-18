/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the main black hole routines.
 *
 * @author OpenGadget3-development team, in particular Alice Damiano, Klaus Dolag, Luca Sala and Volker Springel.
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
#include "../Hydro/kernel.hpp"
#include "blackhole.hpp"
#include "../System/communication.hpp"

#ifdef BLACK_HOLES

MyAtLeastDouble hubble_a, ascale;

extern void blackhole_feedback_loop();
extern void blackhole_swallow_loop();

/**
 * @brief Perform black hole accretion and spin calculation.
 *
 * This function computes the mass accretion rate for black holes based on their
 * surrounding environment and updates their properties such as mass and spin. It
 * takes into account different accretion modes (hot or cold) and applies necessary
 * corrections for efficiency and constraints. It also updates the black hole's spin
 * according to the accretion dynamics.
 *
 * The function operates in a loop over active particles, checking if they are black
 * holes and then computing the accretion rate (`mdot`) based on their surrounding
 * gas properties. The accretion is adjusted by considering the Eddington limit,
 * physical limits, and various efficiency factors. If the subgrid angular momentum
 * model is enabled, it further adjusts the black hole's spin and mass based on the
 * angular momentum of the surrounding gas.
 */
void blackhole(void)
{

    VERBOSE(1, "BH: Beginning black-hole calculation");

    TIMER_PUSH(1, "BLACKHOLES");

    if (All.ComovingIntegrationOn)
    {
        ascale = All.Time;
        hubble_a = hubble_function(All.Time);
    }
    else
    {
        hubble_a = ascale = 1;
    }

    /***********************************************************************************/
    /************************** Local computation of values ****************************/
    /***********************************************************************************/

    TIMER_PUSH(1, "COMPUTE_PROPERTIES");

/* The following neighbour loop, for each BH:
- computes the Bondi rate (mdot)
- limits mdot to Eddington, using All.BlackHoleEddingtonFactor
- carries out spin evolution, if the module is on
- updates BH_Mass (physical mass) and BH_Mdot (accretion rate)
- sets BH_TotalFeedbackEfficiency (feedback efficiency)
*/
#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < NActivePart; i++)
    {
        int n = ActiveParticleList[i];
#else // _OPENMP
    for (int n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    {
#endif // _OPENMP
        if (P[n].Type == 5 && All.BlackHoleHotColdAccretionOn >= 0)
        {
            MyAtLeastDouble dt = (P[n].TimeBin ? (1 << P[n].TimeBin) : 0) * All.Timebase_interval / hubble_a;
            MyAtLeastDouble mdot = 0;
            MyAtLeastDouble meddington = 0;
            MyAtLeastDouble meddington_auxiliary = 0;
            std::ostringstream bh_log; // buffer for storing BH detail string

            /* Compute Bondi rate */
            if (All.BlackHoleDetails >= 1)
            {
                bh_log << std::fixed << std::setprecision(7); // set precicsion of output in bh_log
                bh_log << "BHGROWTH: idBH=" << (unsigned long long)P[n].ID << " (" << (double)P[n].Pos[0] << " "
                       << (double)P[n].Pos[1] << " " << (double)P[n].Pos[2] << ") time=" << All.Time
                       << " mbh=" << (double)BPP(n).BH_Mass << " (";
            }
            assert(BHP[P[n].pt.BHID].PID == n);
            if (All.BlackHoleHotColdAccretionOn == 0)
            {
                Get_Bondi_Parameter(n, BPP(n).BH_Density, BPP(n).BH_SurroundingGasVel, BPP(n).BH_Entropy,
                                    All.BlackHoleAccretionFactor, dt, &mdot, bh_log);
            }
            else
            {
                if (All.BlackHoleDetails >= 1)
                {
                    bh_log << " rho=" << (double)BPP(n).BH_Density << " (cold: ";
                }
                Get_Bondi_Parameter(n, BPP(n).BH_ColdDensity, BPP(n).BH_SurroundingColdGasVel, BPP(n).BH_ColdEntropy,
                                    All.BlackHoleColdAccretionFactor, dt, &mdot, bh_log);

                MyAtLeastDouble hotmdot;
                if (All.BlackHoleDetails >= 1)
                {
                    bh_log << " hot:";
                }
                Get_Bondi_Parameter(n, BPP(n).BH_HotDensity, BPP(n).BH_SurroundingHotGasVel, BPP(n).BH_HotEntropy,
                                    All.BlackHoleAccretionFactor, dt, &hotmdot, bh_log);
                mdot += hotmdot;
            }

            /* Compute Eddington rate */
            meddington = Mdot_Eddington(BPP(n).BH_Mass);
            meddington_auxiliary = meddington;
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
            if (All.BlackHoleSubgridADOn == 1)
            {
                meddington *=
                    BlackHoleMaximumRadiativeEfficiency(BPP(n).BH_Mass) /
                    BPP(n).BH_EtaDisk; /* If spin is active, we want to use EtaDisk to compute Mdot_Edd.
                                        Hence, we multiply by BlackHoleMaximumRadiativeEfficiency(BPP(n).BH_Mass)
                                        that cancels out, and divide by EtaDisk. */
            }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION

            /* Limit mdot to Eddington rate */
            if (All.BlackHoleEddingtonFactor > 0 && mdot > All.BlackHoleEddingtonFactor * meddington)
            {
                mdot = All.BlackHoleEddingtonFactor * meddington;
            }

            /* Limit to the accretion rate BlackHolePhysMdotUpperLimit, if >0 */
            MyAtLeastDouble mdot_upper_limit_code_units =
                All.BlackHolePhysMdotUpperLimit / (All.UnitMass_in_g / All.UnitTime_in_s / SOLAR_MASS * SEC_PER_YEAR);
            if (All.BlackHolePhysMdotUpperLimit > 0 && mdot >= mdot_upper_limit_code_units)
            {
                mdot = mdot_upper_limit_code_units;
            }

            MyAtLeastDouble epsTot = BlackHoleTotalEfficiency(mdot / meddington_auxiliary, BPP(n).BH_Mass);

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
            if (All.BlackHoleSubgridADOn > 0)
            {
                /* printf("BH: Subgrid AD model computations... "); */
                if (dt > 0)
                {
                    BPP(n).BH_AccrCounter +=
                        1; /* BH_AccrCounter is reset to 0 at the completion of an accretion episode. */
                    /* Inizializations */
                    MyAtLeastDouble a = 0.; /* Dimensionless BH spin parameter */
                    MyAtLeastDouble mratio = 0.; /* Ratio final/initial bh mass */
                    MyAtLeastDouble J_BH = 0.; /* BH spin magnitude */
                    MyAtLeastDouble j_BH[3] = {0, 0, 0}; /* BH spin direction unit vector */
                    MyAtLeastDouble J_d = 0.; /* AD AM magnitude */
                    MyAtLeastDouble j_d[3] = {0, 0, 0}; /* AD AM direction unit vector */
                    MyAtLeastDouble md = 0.; /* AD mass */
                    MyAtLeastDouble rd = 0.; /* AD mass */
                    MyAtLeastDouble j_g[3] = {0, 0, 0}; /* AM direction unit vector of the gas in the BH kernel */
                    MyAtLeastDouble J_g[3] = {0, 0, 0}; /* AM direction unit vector of the gas in the BH kernel */
                    MyAtLeastDouble J_g_mod = 0; /* Magnitude of AM direction unit vector of the gas in the BH kernel */
                    MyAtLeastDouble J_tot[3] = {0, 0, 0}; /* total AM direction */
                    MyAtLeastDouble J_tot_mod = 0; /* total AM direction magnitude */
                    MyAtLeastDouble j_tot[3] = {0, 0, 0}; /* total AM direction unit vector */
                    MyAtLeastDouble Jd_Jbh_ratio = 0;
                    MyAtLeastDouble jbh_scalar_jd = 0;
                    MyAtLeastDouble risco = 0;
                    MyAtLeastDouble rwarp = 0;
                    MyAtLeastDouble rsg = 0;
                    MyAtLeastDouble taccr = 0;
                    MyAtLeastDouble msg = 0;
                    MyAtLeastDouble average_mdot = 0;
                    MyAtLeastDouble mbh = 0;
                    MyAtLeastDouble average_dt = 0;
                    int num = 0;

                    /* Parameters */
                    MyAtLeastDouble alpha01 =
                        All.BlackHoleAccretionDiskAlpha / 0.1; /* Viscosity parameter normalized to 0.1 */
                    MyAtLeastDouble nu2_over_nu1 =
                        2 * (1 + 7 * All.BlackHoleAccretionDiskAlpha) / (4 + pow(All.BlackHoleAccretionDiskAlpha, 2)) /
                        pow(All.BlackHoleAccretionDiskAlpha, 2); /* Prescription for viscosity ratios (Ogilvie99) */
                    /* Computations */
                    for (int k = 0; k < 3; k++)
                    {
                        BPP(n).Jgas_in_Kernel_Sum[k] += BPP(n).Vphi[k];
                        J_g[k] = BPP(n).Jgas_in_Kernel_Sum[k] / BPP(n).BH_AccrCounter;
                        J_g_mod += J_g[k] * J_g[k];
                    }
                    J_g_mod = sqrt(J_g_mod);
                    for (int k = 0; k < 3; k++)
                    {
                        J_BH += BPP(n).BH_Spin[k] * BPP(n).BH_Spin[k];
                    }
                    J_BH = sqrt(J_BH);
                    a = J_BH / All.GravityConstantInternal / (BPP(n).BH_Mass * BPP(n).BH_Mass) *
                        (C_LIGHT / All.UnitVelocity_in_cm_per_s); /* This is the very definition of a from J_BH */
                    for (int k = 0; k < 3; k++)
                    {
                        j_BH[k] = BPP(n).BH_Spin[k] / J_BH;
                    }

                    if (J_g_mod > 0) /* If no gas is found close to the BH such that Jgas in kernel=0, then we skip the
                                        spin evolution. */
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            j_g[k] = J_g[k] / J_g_mod;
                        }

                        MyAtLeastDouble eta01 =
                            BPP(n).BH_EtaDisk / 0.1; /* Disk radiative efficiency normalized to 0.1 */

                        BPP(n).BH_Mass_Sum += BPP(n).BH_Mass;
                        BPP(n).BH_accr_mass_Sum += mdot * dt;
                        BPP(n).BH_dt_Sum += dt;
                        average_mdot = BPP(n).BH_accr_mass_Sum /
                                       BPP(n).BH_dt_Sum; /* This reduces to mdot if the counter is 1. Otherwise computes
                                                            the timestep-weighted accretion rate */
                        MyAtLeastDouble fedd = average_mdot / meddington;
                        /* NB. In the average over multiple timesteps M_BH factors out of the sum
                           because it is updated only at the completion of the whole accretion episode,
                           so the average of mdots is the same as the average of the fedds  */
                        mbh = BPP(n).BH_Mass_Sum / BPP(n).BH_AccrCounter;
                        MyAtLeastDouble mbh8 = BPP(n).BH_Mass_Sum * (All.UnitMass_in_g / All.HubbleParam) /
                                               (1.e8 * SOLAR_MASS) / BPP(n).BH_AccrCounter;
                        MyAtLeastDouble average_accr_mass = BPP(n).BH_accr_mass_Sum / BPP(n).BH_AccrCounter;
                        average_dt = BPP(n).BH_dt_Sum / BPP(n).BH_AccrCounter;

                        // printf("Task = %d, BHID= %d, a = %g, mbh8 = %g, eta01 = %g, fedd= %g ", ThisTask, P[n].ID, a,
                        // mbh8, eta01, fedd);

                        rwarp = 4.e2 * pow(a, 5. / 8.) * pow(mbh8, 1. / 8.) * pow(eta01 / fedd, 1. / 4.) *
                                pow(nu2_over_nu1 / 85., -5. / 8.) * pow(alpha01, -1. / 2.); /* This is R_warp/R_BH */
                        rsg = 5.e2 * pow(alpha01, 28. / 45.) * pow(mbh8, -52. / 45.) *
                              pow(eta01 / fedd, 22. / 45.); /* This is R_sg/R_BH */
                        taccr = 3.4e5 * pow(a, 7. / 8.) * pow(mbh8, 11. / 8.) * pow(eta01 / fedd, 3. / 4.) *
                                pow(nu2_over_nu1 / 85., -7. / 8.) * pow(alpha01, -3. / 2.) * SEC_PER_YEAR /
                                All.UnitTime_in_s; /* This is t_accr in internal units */
                        msg = 6.e5 * pow(alpha01, -1. / 45.) * pow(mbh8, 34. / 45.) * pow(fedd / eta01, 4. / 45.) *
                              SOLAR_MASS / (All.UnitMass_in_g / All.HubbleParam); /* This is M_sg in internal units */
                        if (a > 0)
                        {
                            if (BPP(n).BH_AccrCounter <= 1)
                            {
                                if (rsg > rwarp)
                                {
                                    md = average_mdot * taccr; /* average_mdot here is just mdot */
                                    rd = rwarp;
                                    BPP(n).Mode = 0;
                                }
                                else
                                {
                                    md = msg;
                                    rd = rsg;
                                    BPP(n).Mode = 1;
                                }
                                if (md <= average_accr_mass) /* Case of multiple accretion episodes in one BH timestep.
                                                                Here average_accr_mass = mdot*dt. Effectively we are
                                                                asking that t_accr < dt. */
                                {
                                    int stop = 0;
                                    for (num = 1; (num * md <= average_dt * average_mdot || stop == 1);
                                         num++) /* Here average_dt = dt */
                                    /* num * taccr * average_mdot <= average_dt * average_mdot <-> num * md <= dM
                                       formulated in this way it takes care of both mode 0 and 1.
                                       If mode == 0, this means that the BH will accrete N * mdot * t_accr = mdot * dt.
                                       If mode == 1, this means that in the BH will accrete msg * N = mdot*dt.
                                       In mode 1, t_accr is actually not the relevant timescale, because the region of
                                       the disc that gets accreted extend up to rsg < rwarp. In both cases, the amount
                                       of accreted mass is correct for the mdot, and the BH accretes enough.
                                     */
                                    {
                                        /* Need to set again all this variables because while subcycling
                                        the BH mass, spin and efficiency change at the end of each loop
                                        iteration. */
                                        mbh = BPP(n).BH_Mass;
                                        mbh8 = BPP(n).BH_Mass * (All.UnitMass_in_g / All.HubbleParam) /
                                               (1.e8 * SOLAR_MASS);
                                        meddington = Mdot_Eddington(BPP(n).BH_Mass) *
                                                     BlackHoleMaximumRadiativeEfficiency(BPP(n).BH_Mass) /
                                                     BPP(n).BH_EtaDisk;
                                        fedd = average_mdot / meddington;
                                        /* Here we do a pedantic check if by chance accretion has become
                                        super-Eddington because of change in mass or efficiency. */
                                        if (All.BlackHoleEddingtonFactor > 0 &&
                                            average_mdot > All.BlackHoleEddingtonFactor * meddington)
                                        {
                                            average_mdot = All.BlackHoleEddingtonFactor * meddington;
                                        }
                                        eta01 = BPP(n).BH_EtaDisk / 0.1;
                                        taccr = 3.4e5 * pow(a, 7. / 8.) * pow(mbh8, 11. / 8.) *
                                                pow(eta01 / fedd, 3. / 4.) * pow(nu2_over_nu1 / 85., -7. / 8.) *
                                                pow(alpha01, -3. / 2.) * SEC_PER_YEAR /
                                                All.UnitTime_in_s; /* This is t_accr in internal units */
                                        msg =
                                            6.e5 * pow(alpha01, -1. / 45.) * pow(mbh8, 34. / 45.) *
                                            pow(fedd / eta01, 4. / 45.) * SOLAR_MASS /
                                            (All.UnitMass_in_g / All.HubbleParam); /* This is M_sg in internal units */
                                        rsg = 5.e2 * pow(alpha01, 28. / 45.) * pow(mbh8, -52. / 45.) *
                                              pow(eta01 / fedd, 22. / 45.); /* This is R_sg/R_BH */
                                        rwarp = 4.e2 * pow(a, 5. / 8.) * pow(mbh8, 1. / 8.) *
                                                pow(eta01 / fedd, 1. / 4.) * pow(nu2_over_nu1 / 85., -5. / 8.) *
                                                pow(alpha01, -1. / 2.); /* This is R_warp/R_BH */

                                        if (BPP(n).Mode == 0)
                                        {
                                            md = mdot * taccr;
                                            rd = rwarp;
                                        }
                                        else if (BPP(n).Mode == 1)
                                        {
                                            md = msg;
                                            rd = rsg;
                                        }

                                        Jd_Jbh_ratio = sqrt(2) * md / a / mbh * sqrt(rd);
                                        J_d = Jd_Jbh_ratio * J_BH;

                                        jbh_scalar_jd = 0;
                                        J_tot_mod = 0;
                                        for (int k = 0; k < 3; k++)
                                        {
                                            j_d[k] = j_g[k];
                                            J_tot[k] = BPP(n).BH_Spin[k] + J_d * j_d[k];
                                            jbh_scalar_jd += j_d[k] * j_BH[k];
                                            J_tot_mod += J_tot[k] * J_tot[k];
                                        }
                                        J_tot_mod = sqrt(J_tot_mod);

                                        for (int k = 0; k < 3; k++)
                                        {
                                            j_tot[k] = J_tot[k] / J_tot_mod;
                                            j_BH[k] = j_tot[k];
                                        }

                                        /* Alignment */
                                        if (jbh_scalar_jd >= -0.5 * Jd_Jbh_ratio) /* Case a/c/d, Figure 1, King+05 */
                                        {
                                            risco = calc_risco(a); /* NB. Here we are using the **initial** BH spin
                                                                      because **first** the direction align, **then**
                                                                      the AM is accreted at ISCO and a is updated */
                                            BPP(n).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));
                                        }
                                        /* Counter-alignment */
                                        else /* Case b, Figure 1, King+05 */
                                        {
                                            risco = calc_risco(-a);
                                            BPP(n).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));
                                        }

                                        BPP(n).BH_Mass += (1. - BPP(n).BH_EtaDisk) * md;

                                        mratio =
                                            BPP(n).BH_Mass / mbh; /* BH_Mass has been updated, mbh is still the value
                                                                     set at the beginning of every iteration. */
                                        if (3. * risco / (mratio * mratio) - 2. > 0)
                                        {
                                            a = 1. / 3. * sqrt(risco) / mratio *
                                                (4. -
                                                 sqrt(3. * risco / (mratio * mratio) - 2.)); /* Update final BH spin */
                                        }
                                        else
                                        {
                                            a = 0.998;
                                        }
                                        if (a > 0.998)
                                        {
                                            a = 0.998;
                                        }
                                        if (a < 0)
                                        {
                                            a = -a;
                                        }
                                        if (a < MinimumAllowedDimensionlessSpinValue)
                                        {
                                            a = MinimumAllowedDimensionlessSpinValue;
                                        }

                                        for (int k = 0; k < 3; k++)
                                        {
                                            BPP(n).BH_Spin[k] = a * All.GravityConstantInternal *
                                                                (BPP(n).BH_Mass * BPP(n).BH_Mass) /
                                                                (C_LIGHT / All.UnitVelocity_in_cm_per_s) * j_BH[k];
                                        }

                                        J_BH = 0;
                                        for (int k = 0; k < 3; k++)
                                        {
                                            J_BH += BPP(n).BH_Spin[k] * BPP(n).BH_Spin[k];
                                        }
                                        J_BH = sqrt(J_BH);

                                        if (All.BlackHoleDetails >= 4)
                                        {
#ifdef _OPENMP
#pragma omp critical
#endif // _OPENMP
                                            {
                                                fprintf(
                                                    FdBlackHolesSpinDetails,
                                                    "%g\t%g\t%llu\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%"
                                                    "g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t\n",
                                                    (double)(All.Time + num * taccr), (double)average_dt,
                                                    (unsigned long long)P[n].ID, (double)mbh, (double)BPP(n).BH_Mass,
                                                    (double)a, (double)BPP(n).BH_EtaDisk, (double)BPP(n).BH_Spin[0],
                                                    (double)BPP(n).BH_Spin[1], (double)BPP(n).BH_Spin[2], (double)J_BH,
                                                    (double)J_g[0], (double)J_g[1], (double)J_g[2], (double)J_g_mod,
                                                    (double)J_d, (double)Jd_Jbh_ratio, (double)jbh_scalar_jd,
                                                    (double)J_tot[0], (double)J_tot[1], (double)J_tot[2],
                                                    (double)J_tot_mod, (double)num, (double)rwarp, (double)rsg,
                                                    (double)msg, (double)taccr, (double)BPP(n).Mode,
                                                    (double)average_mdot, (double)md);
                                                fflush(FdBlackHolesSpinDetails);
                                            }
                                        }
                                    }
                                    if (num > 1000000)
                                    {
                                        VERBOSE_ALL(1,
                                                    "BH: emergency break of sub-cycling, probably vers small "
                                                    "accreation (%g,%g,%g)!",
                                                    md, average_dt, average_mdot);
                                        stop = 1;
                                    }
                                    BPP(n).BH_AccrCounter = 0;
                                    BPP(n).BH_Mass_Sum = 0;
                                    BPP(n).BH_accr_mass_Sum = 0;
                                    BPP(n).BH_dt_Sum = 0;
                                    for (int k = 0; k < 3; k++)
                                    {
                                        BPP(n).Jgas_in_Kernel_Sum[k] = 0;
                                    }
                                }
                            }
                            else /* Case in which multiple timesteps are needed to complete a single accretion episode
                                  */
                            {
                                if (BPP(n).Mode == 0)
                                {
                                    md = average_mdot * taccr;
                                    rd = rwarp;
                                }
                                else if (BPP(n).Mode == 1)
                                {
                                    md = msg;
                                    rd = rsg;
                                }
                                /* If mode == 0, then the following condition will be equivalent to
                                   BPP(n).BH_AccrCounter * average_dt * average_mdot >= taccr * average_mdot.
                                   If mode == 1, then it will be equivalent to
                                   BPP(n).BH_AccrCounter * average_dt * average_mdot >= msg.
                                   In both cases, this formulation of the condition takes into account correctly
                                   of the amount of mass the BH should accrete in a given amount of time.
                                 */
                                if (BPP(n).BH_AccrCounter * average_dt * average_mdot >= md)
                                {

                                    Jd_Jbh_ratio = sqrt(2) * md / a / mbh * sqrt(rd);
                                    J_d = Jd_Jbh_ratio * J_BH;

                                    jbh_scalar_jd = 0;
                                    J_tot_mod = 0;
                                    for (int k = 0; k < 3; k++)
                                    {
                                        j_d[k] = j_g[k];
                                        J_tot[k] = BPP(n).BH_Spin[k] + J_d * j_d[k];
                                        jbh_scalar_jd += j_d[k] * j_BH[k];
                                        J_tot_mod += J_tot[k] * J_tot[k];
                                    }
                                    J_tot_mod = sqrt(J_tot_mod);

                                    for (int k = 0; k < 3; k++)
                                    {
                                        j_tot[k] = J_tot[k] / J_tot_mod;
                                        j_BH[k] = j_tot[k];
                                    }

                                    /* Alignment */
                                    if (jbh_scalar_jd >= -0.5 * Jd_Jbh_ratio) /* Case a/c/d, Figure 1, King+05 */
                                    {
                                        risco = calc_risco(a); /* NB. Here we are using the **initial** BH spin because
                                                                  **first** the direction align, **then** the AM is
                                                                  accreted at ISCO and a is updated */
                                        BPP(n).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));
                                    }
                                    /* Counter-alignment */
                                    else /* Case b, Figure 1, King+05 */
                                    {
                                        risco = calc_risco(-a);
                                        BPP(n).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));
                                    }

                                    BPP(n).BH_Mass += (1. - BPP(n).BH_EtaDisk) * md;

                                    /* Update final BH spin and BH mass */
                                    mratio = BPP(n).BH_Mass / mbh;
                                    if (3. * risco / (mratio * mratio) - 2. > 0)
                                    {
                                        a = 1. / 3. * sqrt(risco) / mratio *
                                            (4. - sqrt(3. * risco / (mratio * mratio) - 2.)); /* Update final BH spin */
                                    }
                                    else
                                    {
                                        a = 0.998;
                                    }
                                    if (a > 0.998)
                                    {
                                        a = 0.998;
                                    }
                                    if (a < 0)
                                    {
                                        a = -a;
                                    }
                                    if (a < MinimumAllowedDimensionlessSpinValue)
                                    {
                                        a = MinimumAllowedDimensionlessSpinValue;
                                    }

                                    for (int k = 0; k < 3; k++)
                                    {
                                        BPP(n).BH_Spin[k] = a * All.GravityConstantInternal *
                                                            (BPP(n).BH_Mass * BPP(n).BH_Mass) /
                                                            (C_LIGHT / All.UnitVelocity_in_cm_per_s) * j_BH[k];
                                    }
                                    J_BH = 0;
                                    for (int k = 0; k < 3; k++)
                                    {
                                        J_BH += BPP(n).BH_Spin[k] * BPP(n).BH_Spin[k];
                                    }
                                    J_BH = sqrt(J_BH);

                                    if (All.BlackHoleDetails >= 4)
                                    {
#ifdef _OPENMP
#pragma omp critical
#endif // _OPENMP
                                        {
                                            fprintf(
                                                FdBlackHolesSpinDetails,
                                                "%g\t%g\t%llu\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%"
                                                "g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t\n",
                                                (double)All.Time, (double)average_dt, (unsigned long long)P[n].ID,
                                                (double)mbh, (double)BPP(n).BH_Mass, (double)a,
                                                (double)BPP(n).BH_EtaDisk, (double)BPP(n).BH_Spin[0],
                                                (double)BPP(n).BH_Spin[1], (double)BPP(n).BH_Spin[2], (double)J_BH,
                                                (double)J_g[0], (double)J_g[1], (double)J_g[2], (double)J_g_mod,
                                                (double)J_d, (double)Jd_Jbh_ratio, (double)jbh_scalar_jd,
                                                (double)J_tot[0], (double)J_tot[1], (double)J_tot[2], (double)J_tot_mod,
                                                (double)BPP(n).BH_AccrCounter, (double)rwarp, (double)rsg, (double)msg,
                                                (double)taccr, (double)BPP(n).Mode, (double)average_mdot, (double)md);
                                            fflush(FdBlackHolesSpinDetails);
                                        }
                                    }

                                    BPP(n).BH_AccrCounter = 0;
                                    BPP(n).BH_Mass_Sum = 0;
                                    BPP(n).BH_accr_mass_Sum = 0;
                                    BPP(n).BH_dt_Sum = 0;
                                    for (int k = 0; k < 3; k++)
                                    {
                                        BPP(n).Jgas_in_Kernel_Sum[k] = 0;
                                    }
                                }
                            }
                        }
                        else
                        {
                            /* Here we essentially consider a single accretion episode
                               that will increase the spin of the correct amount. */
                            md = msg;
                            rd = rsg;
                            /* Update final BH spin and BH mass */
                            risco = calc_risco(a);
                            BPP(n).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));

                            BPP(n).BH_Mass += (1. - BPP(n).BH_EtaDisk) * md;

                            mratio = BPP(n).BH_Mass / mbh;
                            if (3. * risco / (mratio * mratio) - 2. > 0)
                            {
                                a = 1. / 3. * sqrt(risco) / mratio *
                                    (4. - sqrt(3. * risco / (mratio * mratio) - 2.)); /* Update final BH spin */
                            }
                            else
                            {
                                a = 0.998;
                            }
                            if (a > 0.998)
                            {
                                a = 0.998;
                            }
                            if (a < 0)
                            {
                                a = -a;
                            }
                            if (a < MinimumAllowedDimensionlessSpinValue)
                            {
                                a = MinimumAllowedDimensionlessSpinValue;
                            }

                            for (int k = 0; k < 3; k++)
                            {
                                BPP(n).BH_Spin[k] = a * All.GravityConstantInternal * BPP(n).BH_Mass * BPP(n).BH_Mass /
                                                    (C_LIGHT / All.UnitVelocity_in_cm_per_s) * j_g[k];
                            }
                            J_BH = 0;
                            for (int k = 0; k < 3; k++)
                            {
                                J_BH += BPP(n).BH_Spin[k] * BPP(n).BH_Spin[k];
                            }
                            J_BH = sqrt(J_BH);
                        }
                        BPP(n).BH_Mdot = mdot;
                    } /* if(J_g_mod>0) */
                    else
                    {
                        BPP(n).BH_AccrCounter = 0;
                        BPP(n).BH_Mass_Sum = 0;
                        BPP(n).BH_accr_mass_Sum = 0;
                        BPP(n).BH_dt_Sum = 0;
                        for (int k = 0; k < 3; k++)
                        {
                            BPP(n).Jgas_in_Kernel_Sum[k] = 0;
                        }
                        BPP(n).BH_Mdot = mdot;
                    }

                    if (All.BlackHoleRadiativeEfficiencyTreatment == 0)
                    {
                        BPP(n).BH_TotalFeedbackEfficiency =
                            BPP(n).BH_EtaDisk * BlackHoleCouplingEfficiency(mdot / meddington);
                    }
                    else if (All.BlackHoleRadiativeEfficiencyTreatment > 0)
                    {
                        BPP(n).BH_TotalFeedbackEfficiency =
                            BlackHoleTotalFeedbackEfficiency(mdot / meddington_auxiliary, BPP(n).BH_Mass);
                    }
                    int counter = 0;
                    num != 0 ? counter = num : counter = BPP(n).BH_AccrCounter;

                    if (All.BlackHoleDetails >= 2 && All.BlackHoleDetails < 4)
                    {
#ifdef _OPENMP
#pragma omp critical
#endif // _OPENMP
                        {
                            fprintf(
                                FdBlackHolesSpinDetails,
                                "%g\t%g\t%llu\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%"
                                "g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t\n",
                                (double)All.Time, (double)average_dt, (unsigned long long)P[n].ID, (double)mbh,
                                (double)BPP(n).BH_Mass, (double)a, (double)BPP(n).BH_EtaDisk, (double)BPP(n).BH_Spin[0],
                                (double)BPP(n).BH_Spin[1], (double)BPP(n).BH_Spin[2], (double)J_BH, (double)J_g[0],
                                (double)J_g[1], (double)J_g[2], (double)J_g_mod, (double)J_d, (double)Jd_Jbh_ratio,
                                (double)jbh_scalar_jd, (double)J_tot[0], (double)J_tot[1], (double)J_tot[2],
                                (double)J_tot_mod, (double)counter, (double)rwarp, (double)rsg, (double)msg,
                                (double)taccr, (double)BPP(n).Mode, (double)average_mdot, (double)md);
                            fflush(FdBlackHolesSpinDetails);
                        }
                    }
                } /* if(dt>0) */
            } /* if(All.BlackHoleSubgridADOn==1) */
            else
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
            {
                BPP(n).BH_Mass += (1. - epsTot) * mdot * dt;
                BPP(n).BH_TotalFeedbackEfficiency = BlackHoleTotalFeedbackEfficiency(
                    mdot / meddington, BPP(n).BH_Mass); // Here the quantities are passed in internal units.
                BPP(n).BH_Mdot = mdot;
            }

            if (All.BlackHoleDetails >= 1)
            {
                bh_log << ") MdotEdd=" << (double)meddington
                       << " EpsTotFeed=" << (double)BPP(n).BH_TotalFeedbackEfficiency << " EpsTot=" << (double)epsTot
                       << " dt=" << (double)dt << " mdot=" << (double)BPP(n).BH_Mdot
                       << " Mnow=" << (double)BPP(n).BH_Mass << "\n";

#ifdef _OPENMP
#pragma omp critical
#endif // _OPENMP
                {
                    fprintf(FdBlackHolesDetails, "%s", bh_log.str().c_str());
                }
            }
        }
    }

    TIMER_POP(1, "COMPUTE_PROPERTIES");

    /***********************************************************************************/
    /************************** FIRST loop (accreation) ********************************/
    /***********************************************************************************/

    blackhole_feedback_loop();

    /***********************************************************************************/
    /************************** SECOND loop (swallow) **********************************/
    /***********************************************************************************/

    blackhole_swallow_loop();

    /***********************************************************************************/
    /************************** Final computation of values ****************************/
    /***********************************************************************************/

    if (All.BlackHoleRepositioningOn)
    {
        TIMER_PUSH(1, "REPOSITIONING");
#ifdef _OPENMP
#pragma omp parallel for
        for (int i = 0; i < NActivePart; i++)
        {
            int n = ActiveParticleList[i];
#else // _OPENMP
        for (int n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
        {
#endif // _OPENMP
            if (P[n].Type == 5)
#ifdef GM_REPOSITION_ON_STARDENSITY_MAX
            {
                blackhole_stardens_reposition(n);
            }
#else // GM_REPOSITION_ON_STARDENSITY_MAX
            {
                if (BPP(n).BH_MinPot < 0.5 * BHPOTVALUEINIT)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        P[n].Pos[k] = BPP(n).BH_MinPotPos[k];
                    }
                }
            }
#endif // GM_REPOSITION_ON_STARDENSITY_MAX
        }
        TIMER_POP(1, "REPOSITIONING");
    } // RepositioningOn

    TIMER_PUSH(1, "FINAL_OPERATIONS");

    for (int n = 0; n < TIMEBINS; n++)
    {
        if (TimeBinActive[n])
        {
            TimeBin_BH_mass[n] = 0;
            TimeBin_BH_dynamicalmass[n] = 0;
            TimeBin_BH_Mdot[n] = 0;
            TimeBin_BH_Medd[n] = 0;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for
    for (int i = 0; i < NActivePart; i++)
    {
        int n = ActiveParticleList[i];
#else // _OPENMP
    for (int n = FirstActiveParticle; n >= 0; n = NextActiveParticle[n])
    {
#endif // _OPENMP
        if (P[n].Type == 5)
        {
            assert(BHP[P[n].pt.BHID].PID == n);
            if (BPP(n).BH_accreted_Mass > 0 && P[n].Mass > 0)
            {
                double accreted_Mass = BPP(n).BH_accreted_Mass;

                if (All.BlackHoleIgnoreMomentum == 0)
                {
                    double Mass = P[n].Mass;

                    for (int k = 0; k < 3; k++)
                    {
                        P[n].Vel[k] = (P[n].Vel[k] * Mass + BPP(n).BH_accreted_momentum[k]) / (Mass + accreted_Mass);
                    }
                }

                P[n].Mass += accreted_Mass;
                BPP(n).BH_Mass += BPP(n).BH_accreted_BHMass;
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                if (BPP(n).BH_flag_merged == 1)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        BPP(n).BH_Spin[k] = BPP(n).BH_final_spin[k];
                    }
                }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
            }
            BPP(n).BH_accreted_Mass = 0;
            if (All.BlackHoleIgnoreMomentum == 1)
            {
                BPP(n).BH_accreted_BHMass = 0;
            }
            for (int k = 0; k < 3; k++)
            {
                BPP(n).BH_final_spin[k] = 0;
            }
            BPP(n).BH_flag_merged = 0;

            int bin = P[n].TimeBin;
            TimeBin_BH_mass[bin] += BPP(n).BH_Mass;
            TimeBin_BH_dynamicalmass[bin] += P[n].Mass;
            TimeBin_BH_Mdot[bin] += BPP(n).BH_Mdot;
            if (BPP(n).BH_Mass > 0)
            {
                double temp_meddington = Mdot_Eddington(BPP(n).BH_Mass);
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                if (All.BlackHoleSubgridADOn == 1)
                {
                    temp_meddington *= BlackHoleMaximumRadiativeEfficiency(BPP(n).BH_Mass) / BPP(n).BH_EtaDisk;
                }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
                TimeBin_BH_Medd[bin] +=
                    BPP(n).BH_Mdot /
                    temp_meddington; // Here we are summing f_Edd = Mdot/Mdot_Edd, to compute the mean afterwards.
            }
        }
    }

    VERBOSE(2, "BH: Final operation done");

    double bh_sums[4] = {0.0, 0.0, 0.0, 0.0}; // {mass_holes, mass_real, mdot, mdotedd}
    for (int bin = 0; bin < TIMEBINS; bin++)
    {
        if (TimeBinCount[bin])
        {
            bh_sums[0] += TimeBin_BH_mass[bin];
            bh_sums[1] += TimeBin_BH_dynamicalmass[bin];
            bh_sums[2] += TimeBin_BH_Mdot[bin];
            bh_sums[3] += TimeBin_BH_Medd[bin];
        }
    }

    double total_bh_sums[4]; // {total_mass_holes, total_mass_real, total_mdot, total_mdotedd}
    MPI_Reduce(&bh_sums[0], &total_bh_sums[0], 4, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    if (ThisTask == 0)
    {
        // Map back
        double total_mass_holes = total_bh_sums[0];
        double total_mass_real = total_bh_sums[1];
        double total_mdot = total_bh_sums[2];
        double total_mdotedd = total_bh_sums[3];
        /* convert to solar masses per yr */
        double mdot_in_msun_per_year =
            total_mdot * (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR);

        /* Here we divide Sum(f_Edd) by the number of BHs to get the average Eddington ratio.*/
        total_mdotedd /= All.TotBHs;

        fprintf(FdBlackHoles, "%g %d %g %g %g %g %g\n", All.Time, All.TotBHs, total_mass_holes, total_mdot,
                mdot_in_msun_per_year, total_mass_real, total_mdotedd);
        fflush(FdBlackHoles);
    }

    if (All.BlackHoleDetails >= 1)
    {
        fflush(FdBlackHolesDetails);
    }

    VERBOSE(2, "BH: Final stats for %d black holes done", All.TotBHs);
    TIMER_POP(1, "FINAL_OPERATIONS");
    TIMER_POP(1, "BLACKHOLES");
}
#endif // BLACK_HOLES
