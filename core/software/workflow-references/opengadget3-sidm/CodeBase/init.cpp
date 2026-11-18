/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the functions needed for the initialisation of a simulation from initial conditions.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "allvars.hpp"
#include "proto.hpp"
#include "../IO/io.hpp"

#ifdef CS_DECOUPLING
#include "../CoolingSfr/cs_model/cs_model.hpp"
#endif // CS_DECOUPLING

/**
 * @brief Initialise the simulation
 *
 * This function reads the initial conditions, and allocates storage for the
 * tree(s). Various variables of the particle data are initialised and An
 * intial domain decomposition is performed. If SPH particles are present,
 * the inial SPH smoothing lengths are determined.
 */
void init(void)
{
    double a3;

#ifdef MAGNETIC
    double a2_fac, gauss2gadget = 1.0;
#endif // MAGNETIC


#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    double min_t_cool, max_t_cool;
    double min_t_elec, max_t_elec;
    double a_start, a_end;
#endif // defined (CHEMISTRY) || defined (UM_CHEMISTRY)

#ifdef BLACK_HOLES
    int count_holes = 0;
#endif // BLACK_HOLES


#ifdef LT_STELLAREVOLUTION
    double this_age, NextChemTime;
    int IMFi, SFi;
    unsigned int I;
    int chem_step, ti_min;
#endif // LT_STELLAREVOLUTION

    // double t0_ics, t1_ics;

#if defined(LT_ADD_GAL_TO_SUB) && defined(OBSERVER_FRAME) && defined(INTERP_OBSERVER_FRAME)
    flag_readCB07obs_redshifts = 0;
#endif // defined(LT_ADD_GAL_TO_SUB) && defined(OBSERVER_FRAME) && defined(INTERP_OBSERVER_FRAME)

    All.Time = All.TimeBegin;
    set_cosmo_factors_for_current_time();

    if (RestartFlag == -1)
    {
        allocate_memory();
#ifdef BLACK_HOLES
        if (ThisTask == 0)
        {
            VERBOSE(2, "BH: Evaluating Settings of the parameters");
            report_internal_model_for_blackhole();
        }
#endif // BLACK_HOLES


#ifdef LMB_CR_OUTPUT_SYNCHROTRON
        if (ThisTask == 0)
        {
            report_synchrotron_kernel();
        }
#endif // LMB_CR_OUTPUT_SYNCHROTRON

        MPI_Barrier(MYMPI_COMM_WORLD);

        VERBOSE(0, "\nEnding OpenGadget after explaining all physics modules.\n");
        // end code succesfully
        MPI_Comm_free(&MYMPI_COMM_WORLD);
        MPI_Finalize();
        exit(0);
    }

#if defined(SFR) || defined(BLACK_HOLES)
    for (All.bits = 0; All.Generations > (1 << All.bits); All.bits++)
        ;
#endif // defined(SFR) || defined(BLACK_HOLES)

    ENDRUN_IF((RestartFlag == 3 && RestartSnapNum < 0),
              "Need to give the snapshot number if FOF/SUBFIND is selected for output");

    ENDRUN_IF((RestartFlag == 4 && RestartSnapNum < 0),
              "Need to give the snapshot number if snapshot should be converted");

    ENDRUN_IF(
        (RestartFlag == 5 && RestartSnapNum < 0),
        "Need to give the snapshot number if power spectrum and two-point correlation function should be calculated");

    ENDRUN_IF((RestartFlag == 6 && RestartSnapNum < 0),
              "Need to give the snapshot number if velocity power spectrum for the gas cells should be calculated");

    TIMER_PUSH(1, "reading_ICs");

    if (RestartFlag == 3 || RestartFlag == 2)
    {
        All.ICFormat = All.SnapFormat;
    }

    switch (All.ICFormat)
    {
    case 1:
    case 2:
    case 3:
    case 4:
        if (RestartFlag >= 2 && RestartSnapNum >= 0)
        {
            char fname[1000];

            if (All.NumFilesPerSnapshot > 1)
            {
                sprintf(fname, "%s/snapdir_%03d/%s_%03d", All.OutputDir, RestartSnapNum, All.SnapshotFileBase,
                        RestartSnapNum);
            }
            else
            {
                sprintf(fname, "%s%s_%03d", All.OutputDir, All.SnapshotFileBase, RestartSnapNum);
            }
            read_ic(fname);
        }
        else
        {
            if (All.ICFormat == 4)
            {
                read_ic_cluster(All.InitCondFile);
            }
            else
            {
                read_ic(All.InitCondFile);
            }
        }
        break;

    default:
        ENDRUN("ICFormat=%d not supported.", All.ICFormat);
    }
    TIMER_POP(1, "reading_ICs");

    All.Time = All.TimeBegin;
    set_cosmo_factors_for_current_time();

#if defined(COOLING) && !defined(LT_METAL_COOLING_WAL)
    IonizeParams();
#endif // defined(COOLING) && !defined(LT_METAL_COOLING_WAL)

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    InitChem();
#endif // defined (CHEMISTRY) || defined (UM_CHEMISTRY)


    if (All.ComovingIntegrationOn)
    {
        All.Timebase_interval = (log(All.TimeMax) - log(All.TimeBegin)) / TIMEBASE;
        All.Previous_Ti_Current = All.Ti_Current = 0;
        a3 = All.Time * All.Time * All.Time;
#ifdef MAGNETIC
#ifndef MU0_UNITY
        gauss2gadget *= sqrt(All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g /
                             (All.HubbleParam * All.HubbleParam));
#endif // MU0_UNITY
        a2_fac = (All.Time * All.Time);
#endif // MAGNETIC
    }
    else
    {
        All.Timebase_interval = (All.TimeMax - All.TimeBegin) / TIMEBASE;
        All.Previous_Ti_Current = All.Ti_Current = 0;
        a3 = 1;
#ifdef MAGNETIC
#ifndef MU0_UNITY
        gauss2gadget *= sqrt(All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g);
#endif // MU0_UNITY
        a2_fac = 1;
#endif // MAGNETIC
    }
    set_softenings();

#ifdef OUTPUT_LONGRANGE_POTENTIAL
    All.PotentialFileCount = 0;
#endif // OUTPUT_LONGRANGE_POTENTIAL

    All.NumCurrentTiStep = 0; /* setup some counters */
    All.SnapshotFileCount = 0;
    if (RestartFlag == 2)
    {
        if (RestartSnapNum < 0)
        {
            char *underscore = strrchr(All.InitCondFile, '_');
            if (!underscore)
            {
                ENDRUN("Your input file '%s' lacks an underscore. Cannot infer next snapshot number.",
                       All.InitCondFile);
            }
            else
            {
                All.SnapshotFileCount = atoi(underscore + 1) + 1;
#ifdef OUTPUT_LONGRANGE_POTENTIAL
                All.PotentialFileCount = atoi(underscore + 1) + 1;
#endif // OUTPUT_LONGRANGE_POTENTIAL
            }
        }
        else
        {
            All.SnapshotFileCount = RestartSnapNum + 1;
            All.SnapshotFileCount--;
#ifdef OUTPUT_LONGRANGE_POTENTIAL
            All.PotentialFileCount = RestartSnapNum + 1;
#endif // OUTPUT_LONGRANGE_POTENTIAL
        }
    }

    All.TotNumOfForces = 0;
    All.TopNodeAllocFactor = 0.008;
    All.TreeAllocFactor = 0.35;

    if (All.ComovingIntegrationOn)
    {
        if (All.PeriodicBoundariesOn == 1)
        {
            check_omega();
        }
    }

    All.TimeLastStatistics = All.TimeBegin - All.TimeBetStatistics;
#ifdef BLACK_HOLES
    All.TimeNextOnTheFlyFoF = All.TimeBegin;
#endif // BLACK_HOLES

    for (int i = 0; i < Gravcostlevels; i++)
    {
        All.LevelToTimeBin[i] = 0;
    }

    for (int i = 0; i < NumPart; i++)
    {
        for (int j = 0; j < Gravcostlevels; j++)
        {
            P[i].GravCost[j] = 0;
        }
    }

    if (All.ComovingIntegrationOn) /*  change to new velocity variable */
    {
        for (int i = 0; i < NumPart; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                P[i].Vel[j] *= sqrt(All.Time) * All.Time;
            }
        }
    }

    for (int i = 0; i < NumPart; i++) /*  start-up initialization */
    {
        for (int j = 0; j < 3; j++)
        {
            P[i].GravAccel[j] = 0;
        }

#ifdef KEEP_DM_HSML_AS_GUESS
        if (RestartFlag != 1)
        {
            P[i].DM_Hsml = -1;
        }
#endif // KEEP_DM_HSML_AS_GUESS

#ifdef PMGRID
        for (int j = 0; j < 3; j++)
        {
            P[i].GravPM[j] = 0;
        }
#endif // PMGRID
        P[i].Ti_begstep = 0;
        P[i].Ti_current = 0;
        P[i].TimeBin = 0;

        if (header.flag_ic_info != FLAG_SECOND_ORDER_ICS)
        {
            P[i].OldAcc = 0; /* Do not zero in 2lpt case as masses are stored here */
        }
#if defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY)
        P[i].Potential = 0;
#endif // defined(EVALPOTENTIAL) || defined(COMPUTE_POTENTIAL_ENERGY)

        if (RestartFlag == 0)
        {
#ifdef SFR
            if (P[i].Type == type_star)
            {
                MPP(i).StellarAge = 0;
            }
#endif // SFR

#ifdef BLACK_HOLES
            if (P[i].Type == type_bh)
            {
                BPP(i).StellarAge = 0;
            }
#endif // BLACK_HOLES
        }

#if defined(KD_MAIN_HALO) && defined(SUBFIND)
        if (RestartFlag == 0)
        {
            P[i].SubLevel = 0;
        }
#endif // defined(KD_MAIN_HALO) && defined(SUBFIND)

#if defined(SFR) && !defined(LT_STELLAREVOLUTION)
        if (RestartFlag == 0)
        {
            MPP(i).Metallicity = 0;
        }
#endif // SFR && !LT_STELLAREVOLUTION

#ifdef LT_STELLAREVOLUTION
        if (RestartFlag != 5)
        {
            All.Time_Age = get_age(All.Time);

            if (P[i].Type == type_star)
            {
                this_age = get_age(MPP(i).StellarAge) - All.Time_Age;
                SFi = get_SF_index(i, &SFi, &IMFi);

                NextChemTime = get_NextChemTime(this_age, SFi, 0x0);
                int j = get_chemstep_bin(All.Time, All.Time_Age - NextChemTime, &chem_step, i);

                I = P[i].pt.MetID;

                if (All.Ti_Current >= TIMEBASE)
                {
                    chem_step = j = 0;
                }

                if ((TIMEBASE - All.Ti_Current) < chem_step)
                {
                    chem_step = TIMEBASE - All.Ti_Current;
                    ti_min = TIMEBASE;
                    while (ti_min > chem_step)
                    {
                        ti_min >>= 1;
                    }
                    chem_step = ti_min;
                    j = get_timestep_bin(chem_step);
                }

                if (j != MetP[I].ChemTimeBin)
                {
                    TimeBinCountStars[MetP[I].ChemTimeBin]--;
                    TimeBinCountStars[j]++;
                    MetP[I].ChemTimeBin = j;
                }
                MetP[I].LastChemTime = this_age;

                if (this_age < SFs[SFi].ShortLiv_TimeTh)
                {
                    MetP[I].LastChemTime = SFs[SFi].ShortLiv_TimeTh;
                }
            }
        }
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
        if (P[i].Type == type_bh)
        {
            count_holes++;

            if (RestartFlag == 0)
            {
#ifndef BHMASS_IN_ICS
                BPP(i).BH_Mass = All.SeedBlackHoleMass;
#endif // BHMASS_IN_ICS
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                BPP(i).BH_AccrCounter = 0;
                BPP(i).BH_Mass_Sum = 0;
                BPP(i).BH_accr_mass_Sum = 0;
                BPP(i).BH_dt_Sum = 0;
                for (int k = 0; k < 3; k++)
                    BPP(i).Jgas_in_Kernel_Sum[k] = 0;
                BPP(i).BH_flag_merged = 0;
                MyAtLeastDouble a = 0;
                MyAtLeastDouble risco = 0;
#ifndef BHSPIN_IN_ICS // Avoid initializing the spin if the ICS already have it
                a = All.SeedBlackHoleSpin;
                MyAtLeastDouble BH_Spin_mod = a * All.GravityConstantInternal * (pow(BPP(i).BH_Mass, 2)) /
                                              (C_LIGHT / All.UnitVelocity_in_cm_per_s);
                BPP(i).BH_Spin[0] = BH_Spin_mod * 0; // sin(M_PI/6.)*cos(M_PI/4.);
                BPP(i).BH_Spin[1] = BH_Spin_mod * 0; // sin(M_PI/6.)*sin(M_PI/4.);
                BPP(i).BH_Spin[2] = BH_Spin_mod * 1; //-cos(M_PI/6.);
#else // BHSPIN_IN_ICS
                MyAtLeastDouble J_BH = 0;
                for (int k = 0; k < 3; k++)
                    J_BH += pow(BPP(i).BH_Spin[k], 2);
                J_BH = sqrt(J_BH);
                a = J_BH / All.GravityConstantInternal / (pow(BPP(i).BH_Mass, 2)) *
                    (C_LIGHT / All.UnitVelocity_in_cm_per_s);
#endif // BHSPIN_IN_ICS
                MyAtLeastDouble z1 =
                    1. + pow((1. - pow(a, 2.)), 1. / 3.) * (pow((1. + a), 1. / 3.) + pow((1. - a), 1. / 3.));
                MyAtLeastDouble z2 = pow((3 * pow(a, 2.) + pow(z1, 2.)), 1. / 2.);
                if (a <= 0.)
                {
                    risco = 3. + z2 + pow(((3. - z1) * (3. + z1 + 2 * z2)), 1. / 2.);
                }
                else
                {
                    risco = 3. + z2 - pow(((3. - z1) * (3. + z1 + 2 * z2)), 1. / 2.);
                }
                BPP(i).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
            }
            else if (RestartFlag == 2)
            {
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                BPP(i).BH_AccrCounter = 0;
                BPP(i).BH_Mass_Sum = 0;
                BPP(i).BH_accr_mass_Sum = 0;
                BPP(i).BH_dt_Sum = 0;
                for (int k = 0; k < 3; k++)
                {
                    BPP(i).Jgas_in_Kernel_Sum[k] = 0;
                }
                BPP(i).BH_flag_merged = 0;
                MyAtLeastDouble a = 0;
                MyAtLeastDouble risco = 0;
                MyAtLeastDouble J_BH = 0;
                for (int k = 0; k < 3; k++)
                {
                    J_BH += pow(BPP(i).BH_Spin[k], 2);
                }
                J_BH = sqrt(J_BH);
                a = J_BH / All.GravityConstantInternal / (pow(BPP(i).BH_Mass, 2)) *
                    (C_LIGHT / All.UnitVelocity_in_cm_per_s);
                MyAtLeastDouble z1 =
                    1. + pow((1. - pow(a, 2.)), 1. / 3.) * (pow((1. + a), 1. / 3.) + pow((1. - a), 1. / 3.));
                MyAtLeastDouble z2 = pow((3 * pow(a, 2.) + pow(z1, 2.)), 1. / 2.);
                if (a <= 0.)
                {
                    risco = 3. + z2 + pow(((3. - z1) * (3. + z1 + 2 * z2)), 1. / 2.);
                }
                else
                {
                    risco = 3. + z2 - pow(((3. - z1) * (3. + z1 + 2 * z2)), 1. / 2.);
                }
                BPP(i).BH_EtaDisk = 1. - sqrt(1 - 2 / (3 * risco));
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
            }
        }
#endif // BLACK_HOLES
    }

#ifdef BLACK_HOLES
    MPI_Allreduce(&count_holes, &All.TotBHs, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
#endif // BLACK_HOLES

    for (int i = 0; i < TIMEBINS; i++)
    {
        TimeBinActive[i] = 1;
    }

    reconstruct_timebins();

#ifdef PMGRID
    All.PM_Ti_endstep = All.PM_Ti_begstep = 0;
#endif // PMGRID

#ifdef CONDUCTION
    All.Conduction_Ti_endstep = All.Conduction_Ti_begstep = 0;
#endif // CONDUCTION



    for (int i = 0; i < N_gas; i++) /* initialize sph_properties */
    {
        SphP[i].EntropyPred = SphP[i].Entropy;

        for (int j = 0; j < 3; j++)
        {
            SphP[i].VelPred[j] = P[i].Vel[j];
            SphP[i].HydroAccel[j] = 0;
        }
        SphP[i].DtEntropy = 0;
        SPHP(i, wakeup) = TIMEBINS;

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        SphP[i].Gamma = GAMMA; /* set universal value */
        SphP[i].t_cool = 0;
        SphP[i].t_elec = 0;
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

        if (RestartFlag == 0)
        {
#ifndef READ_HSML
            P[i].Hsml = 0;
#endif // READ_HSML
            SphP[i].Density = -1;
#if defined(COOLING) && !defined(UM_CHEMISTRY)
            SphP[i].elec = 1.0;
#endif //  defined(COOLING) && !defined(UM_CHEMISTRY)
            SphP[i].DivVel = 0;
        }

#ifdef WINDS
        if (RestartFlag == 0)
        {
            SphP[i].DelayTime = 0;
        }
#endif // WINDS

#ifdef PB_BH_KIN_FEED_DECOUPLE_WIND
        if (RestartFlag == 0)
        {
            SphP[i].DecoupleTime_AGN = 0;
        }
#endif // PB_BH_KIN_FEED_DECOUPLE_WIND

#ifdef PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT
        if (RestartFlag == 0)
        {
            SphP[i].lastKick_Time_AGN = 0;
        }
#endif // PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT

#ifdef SFR
        SphP[i].Sfr = 0;
#endif // SFR

#ifdef CS_DECOUPLING
        SphP[i].CS_DensityOld = -1;
#endif // CS_DECOUPLING

#ifdef STATICBRANDT_INIT
        P[i].Vel[0] = 0.0;
        P[i].Vel[1] = 0.0;
        P[i].Vel[2] = 0.0;

        P[i].Vel[0] = -OmegaR(i, BRANDT_MODE) * (P[i].Pos[1] - LONG_Y / 2.);
        P[i].Vel[1] = OmegaR(i, BRANDT_MODE) * (P[i].Pos[0] - LONG_X / 2.);

        if (P[i].Type == type_gas)
        {
            for (int j = 0; j < 3; j++)
            {
                SphP[i].VelPred[j] = P[i].Vel[j];
            }
        }
#endif // STATICBRANDT_INIT

#if defined(UM_CHEMISTRY) && defined(UM_CHEMISTRY_INISET)
        SphP[i].elec = All.START_elec;

        SphP[i].HI = All.START_HI;
        SphP[i].HII = All.START_HII;
        SphP[i].HM = All.START_HM;

        SphP[i].HeI = All.START_HeI;
        SphP[i].HeII = All.START_HeII;
        SphP[i].HeIII = All.START_HeIII;

        SphP[i].H2I = All.START_H2I;
        SphP[i].H2II = All.START_H2II;

        SphP[i].HD = All.START_HD;
        SphP[i].DI = All.START_DI;
        SphP[i].DII = All.START_DII;

        SphP[i].HeHII = All.START_HeHII;
#endif //  defined (UM_CHEMISTRY) && defined (UM_CHEMISTRY_INISET)

#ifdef MAGNETIC
#if defined(BINISET) && !defined(MAGNETICZERO)
        if (RestartFlag == 0)
        { /* Set only when starting from ICs */
            SphP[i].BPred[0] = All.BiniX;
            SphP[i].BPred[1] = All.BiniY;
            SphP[i].BPred[2] = All.BiniZ;
        }
#endif // defined(BINISET) && !defined(MAGNETICZERO)
        for (int j = 0; j < 3; j++)
        {
            SphP[i].BPred[j] *= a2_fac * gauss2gadget;
            SphP[i].B[j] = SphP[i].BPred[j];
        }

#ifdef TIME_DEP_MAGN_DISP
        SphP[i].Balpha =
            All.ArtificialMagneticDissipationMin +
            High_Magn_Disp_Start * (All.ArtificialMagneticDissipationConstant - All.ArtificialMagneticDissipationMin);
#if (TIME_DEP_MAGN_DISP == 0) || (TIME_DEP_MAGN_DISP == 1) || (TIME_DEP_MAGN_DISP == 3)
        SphP[i].DtBalpha = 0.0;
#endif // (TIME_DEP_MAGN_DISP == 0) || (TIME_DEP_MAGN_DISP == 1) || (TIME_DEP_MAGN_DISP == 3)
#endif // TIME_DEP_MAGN_DISP

#ifdef DIVBCLEANING_TRICCO
        SphP[i].Psi = SphP[i].PsiPred = SphP[i].DtPsi = 0.0;
#endif // DIVBCLEANING_TRICCO

#ifdef DIVBCLEANING_DEDNER
        SphP[i].Phi = SphP[i].PhiPred = SphP[i].DtPhi = 0.0;
#endif // DIVBCLEANING_DEDNER

#endif // MAGNETIC

#ifdef TIME_DEP_ART_COND
        SphP[i].Calpha = TIME_DEP_ART_COND * All.ArtCondMin;
#endif // TIME_DEP_ART_COND

#ifdef TIME_DEP_ART_VISC
        SphP[i].alpha = All.ViscosityAlphaMin + High_Art_Visc_Start * (All.ArtBulkViscConst - All.ViscosityAlphaMin);
#if (TIME_DEP_ART_VISC == 0) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
        SphP[i].Dtalpha = 0.0;
#endif // (TIME_DEP_ART_VISC == 0) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#if (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
        SphP[i].visc_R = 0.0;
        SphP[i].visc_OldDvel = 0.0;
        SphP[i].visc_Oldtime = 0;
#endif // (TIME_DEP_ART_VISC == 1)|| (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#endif // TIME_DEP_ART_VISC

#ifdef OUTPUT_ENERGY_VISC
        SphP[i].visc_E = 0.0;
#endif // OUTPUT_ENERGY_VISC

#ifdef BLACK_HOLES
        SphP[i].Injected_BH_Energy = 0;
#endif // BLACK_HOLES


#ifdef LT_STELLAREVOLUTION
        SphP[i].EgyRes = 0.; // needed to avoid energy problem at the VERY FIRST timestep
        SphP[i].EgyStep = 0.;
#endif // LT_STELLAREVOLUTION
    }

#ifdef TWODIMS
    for (int i = 0; i < NumPart; i++)
    {
        P[i].Pos[2] = 0;
#ifndef MAGNETIC
        P[i].Vel[2] = 0;
#endif // MAGNETIC

        P[i].GravAccel[2] = 0;

        if (P[i].Type == type_gas)
        {
#ifndef MAGNETIC
            SphP[i].VelPred[2] = 0;
            SphP[i].HydroAccel[2] = 0;
#endif // MAGNETIC
        }
    }
#endif // TWODIMS

#ifdef ONEDIM
    for (int i = 0; i < NumPart; i++)
    {
        P[i].Pos[1] = P[i].Pos[2] = 0;
#ifndef MAGNETIC
        P[i].Vel[1] = P[i].Vel[2] = 0;
#endif // MAGNETIC

        P[i].GravAccel[1] = P[i].GravAccel[2] = 0;

        if (P[i].Type == type_gas)
        {
#ifndef MAGNETIC
            SphP[i].VelPred[1] = SphP[i].VelPred[2] = 0;
            SphP[i].HydroAccel[1] = SphP[i].HydroAccel[2] = 0;
#endif // MAGNETIC
        }
    }
#endif // ONEDIM

#ifdef ASSIGN_NEW_IDS
    assign_unique_ids();
#endif // ASSIGN_NEW_IDS

#ifndef NOTEST_FOR_IDUNIQUENESS
    test_id_uniqueness();
#endif // NOTEST_FOR_IDUNIQUENESS

    Flag_FullStep = 1; /* to ensure that Peano-Hilber order is done */
    TreeReconstructFlag = 1;

#ifdef SHIFT_BY_HALF_BOX
    for (int i = 0; i < NumPart; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P[i].Pos[j] += 0.5 * All.BoxSize;
        }
    }
#endif // SHIFT_BY_HALF_BOX
#if GADGET_HYDRO == HYDRO_MFM
#ifdef GT_DUST
    QFluidVector<mfm_numdims> QFluidVector_zero{};
#endif // GT_DUST
    QFluidVector<mfm_numdims, MyAtLeastDouble> dQdt_zero{};
    WFluidVector<mfm_numdims> WFluidVector_zero{};
    SymmetricMatrix<NUMDIMS> SymmetricMatrix_zero{};
    Vector<NUMDIMS> Vector_numdims_zero{};
    Vector<mfm_numdims> Vector_mfmnumdims_zero{};
    for (int i = 0; i < N_gas; i++) /* initialize sph_properties */
    {
        // properly initialize all C++ class type variables
        memcpy(&SphP[i].dQdt, &dQdt_zero, sizeof(QFluidVector<mfm_numdims, MyAtLeastDouble>));

        memcpy(&SphP[i].Wprim, &WFluidVector_zero, sizeof(WFluidVector<mfm_numdims>));
        memcpy(&SphP[i].Wmin, &WFluidVector_zero, sizeof(WFluidVector<mfm_numdims>));
        memcpy(&SphP[i].Wmax, &WFluidVector_zero, sizeof(WFluidVector<mfm_numdims>));

        memcpy(&SphP[i].Bgrad, &SymmetricMatrix_zero, sizeof(SymmetricMatrix<NUMDIMS>));

        memcpy(&SphP[i].Vel_smooth, &Vector_mfmnumdims_zero, sizeof(Vector<mfm_numdims>));
        for (int k = 0; k < numvar; ++k)
        {
            memcpy(&SphP[i].gradW[k], &Vector_numdims_zero, sizeof(Vector<NUMDIMS>));
        }

#ifdef GT_DUST
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            memcpy(&SphP[i].dQdt_dust[idust], &QFluidVector_zero, sizeof(QFluidVector<mfm_numdims>));

            memcpy(&SphP[i].Wprim_dust[idust], &WFluidVector_zero, sizeof(WFluidVector<mfm_numdims>));
            memcpy(&SphP[i].Wmin_dust[idust], &WFluidVector_zero, sizeof(WFluidVector<mfm_numdims>));
            memcpy(&SphP[i].Wmax_dust[idust], &WFluidVector_zero, sizeof(WFluidVector<mfm_numdims>));
            for (int k = 0; k < numvar; ++k)
            {
                memcpy(&SphP[i].gradW_dust[idust][k], &Vector_numdims_zero, sizeof(Vector<NUMDIMS>));
            }
        }
#endif // GT_DUST
    }
#endif // GADGET_HYDRO == HYDRO_MFM

    domain_Decomposition(0, 0); /* do initial domain decomposition (gives equal numbers of particles) */
    set_softenings();
    /* will build tree */
    ngb_treebuild();
    All.Ti_Current = 0;

#ifdef ACC_RUN
    on_beginning_of_timestep();
#endif // ACC_RUN

    if (RestartFlag != 3 && RestartFlag != 5)
    {
        setup_smoothinglengths();
    }

#ifdef ACC_RUN
    on_end_of_timestep();
#endif // ACC_RUN

#ifdef ADAPTGRAVSOFT
    if (RestartFlag != 3 && RestartFlag != 5)
    {
        ags_setup_smoothinglengths();
    }
#endif // ADAPTGRAVSOFT

#if GADGET_HYDRO == HYDRO_MFM
    for (int i = 0; i < N_gas; i++) /* initialize sph_properties */
    {
        if (header.flag_entropy_instead_u == 0)
        {
            // TODO : Also add restart option later
            SphP[i].InternalEnergy = SphP[i].Entropy;
            SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
        }
        else
        {
            SphP[i].InternalEnergy = SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].Density / a3, GAMMA_MINUS1);
            SphP[i].InternalEnergyPred = SphP[i].InternalEnergy;
        }
        SphP[i].DtInternalEnergy = (MyFloat)0.0;

        for (int k = 0; k < 3; k++)
        {
            SphP[i].HydroAccelExtra[k] = (MyAtLeastDouble)0.0;
#ifdef MFM_FACE_CLOSURE_RESIDUAL
            // find_hsml() clears this before every flux walk, but the DtB and internal-energy
            // limiters read it in the first drift and kick, which come before the first walk.
            SphP[i].FaceAreaSum[k] = (MyFloat)0.0;
#endif // MFM_FACE_CLOSURE_RESIDUAL
        }
        SphP[i].DtEntropyExtra = (MyFloat)0.0;

#ifdef GT_DUST_DIFFUSION
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            for (int k = 0; k < NUMDIMS; k++)
            {
                SphP[i].DtDeltaVelExtra[idust][k] = (MyFloat)0.0;
            }
        }
#endif // GT_DUST_DIFFUSION
    }
#endif // GADGET_HYDRO == HYDRO_MFM

    /* at this point, the entropy variable actually contains the
     * internal energy, read in from the initial conditions file.
     * Once the density has been computed, we can convert to entropy.
     */
    for (int i = 0; i < N_gas; i++) /* initialize sph_properties */
    {
        if (header.flag_entropy_instead_u == 0)
        {
#ifdef GT_DUST
            SphP[i].InternalEnergySPH = SphP[i].Entropy;
#endif // GT_DUST
            if (ThisTask == 0 && i == 0)
            {
                VERBOSE(2, "Converting u -> entropy !");
            }

            SphP[i].Entropy = GAMMA_MINUS1 * SphP[i].Entropy / pow(SphP[i].Density / a3, GAMMA_MINUS1);

#ifdef GT_DUST
            SphP[i].Entropy /= pow((1. - SphP[i].TotalDustFracPred) / a3, GAMMA_MINUS1);
#endif // GT_DUST

            SphP[i].EntropyPred = SphP[i].Entropy;
        }

        SphP[i].DtEntropy = 0;
        SphP[i].DivVel = 0;



#ifdef GT_DUST
        SphP[i].InternalEnergySPHPred = SphP[i].InternalEnergySPH;

        SphP[i].TotalDustFrac = 0.0;
        for (int j = 0; j < 3; j++)
        {
            SphP[i].TotalDeltaVel[j] = 0.0;
        }

        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            SphP[i].DustFracPred[idust] = SphP[i].DustFrac[idust];
            SphP[i].TotalDustFrac += dustfrac(SphP[i].DustFrac[idust]);
            for (int j = 0; j < 3; j++)
            {
                SphP[i].DeltaVelPred[idust][j] = SphP[i].DeltaVel[idust][j];
                SphP[i].TotalDeltaVel[j] += dustfrac(SphP[i].DustFrac[idust]) * SphP[i].DeltaVel[idust][j];
            }
        }

        SphP[i].TotalDustFracPred = SphP[i].TotalDustFrac;
        for (int j = 0; j < 3; j++)
        {
            if (SphP[i].TotalDustFrac > 0.0)
            {
                SphP[i].TotalDeltaVel[j] /= SphP[i].TotalDustFrac;
            }
            else
            {
                SphP[i].TotalDeltaVel[j] = 0.0;
            }
            SphP[i].TotalDeltaVelPred[j] = SphP[i].TotalDeltaVel[j];
        }

#endif // GT_DUST
    }

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    VERBOSE(2, "Initial abundances (for P[1].ID=%llu):", (unsigned long long)P[1].ID);
    VERBOSE(2, "HI=%g, HII=%g, HeI=%g, HeII=%g, HeIII=%g ", SphP[1].HI, SphP[1].HII, SphP[1].HeI, SphP[1].HeII,
            SphP[1].HeIII);
    VERBOSE(2, "HM=%g, H2I=%g, H2II=%g, elec=%g", SphP[1].HM, SphP[1].H2I, SphP[1].H2II, SphP[1].elec);

#if defined(UM_CHEMISTRY)
    VERBOSE(2, "HD=%g,  DI=%g, DII=%g", SphP[1].HD, SphP[1].DI, SphP[1].DII);
    VERBOSE(2, "HeHII=%g", SphP[1].HeHII);
#endif // defined(UM_CHEMISTRY)

    VERBOSE(2, "x=%g, y=%g, z=%g, vx=%g, vy=%g, vz=%g,\n density=%g, entropy=%g", P[N_gas - 1].Pos[0],
            P[N_gas - 1].Pos[1], P[N_gas - 1].Pos[2], P[N_gas - 1].Vel[0], P[N_gas - 1].Vel[1], P[N_gas - 1].Vel[2],
            SphP[N_gas - 1].Density, SphP[N_gas - 1].Entropy);

#if defined(UM_METAL_COOLING) && defined(LT_METAL_COOLING)
    VERBOSE(2, "::  mass of C=%g, O=%g, Si=%g, Fe=%g, N=%g", SphP[N_gas - 1].Metals[Carbon],
            SphP[N_gas - 1].Metals[Oxygen], SphP[N_gas - 1].Metals[Silicon], SphP[N_gas - 1].Metals[Iron],
            SphP[N_gas - 1].Metals[Nitrogen]);
#endif // defined(UM_METAL_COOLING) && defined (LT_METAL_COOLING)

    /* need predict the cooling time and elec_dot here */
    min_t_cool = min_t_elec = 1.0e30;
    max_t_cool = max_t_elec = -1.0e30;

    for (int i = 0; i < N_gas; i++) /* Init Chemistry: */
    {
#ifdef CHEMISTRY
        a_start = All.Time;
        a_end = All.Time + 0.001; /*  0.001 as an arbitrary value */
        (void)compute_abundances(0, i, a_start, a_end);
#endif // CHEMISTRY

#ifdef UM_CHEMISTRY /* Init um_Chemistry/Z: */
#if defined(UM_METAL_COOLING) && defined(LT_METAL_COOLING)
        um_ZsPoint = SphP[i].Metals;
        um_mass = P[i].Mass;
#endif // defined(UM_METAL_COOLING) && defined(LT_METAL_COOLING)
        double um_u;

        a_start = All.Time;
        a_end = All.Time + 0.001; /* 0.001 as an arbitrary value */
        (void)compute_abundances(0, i, a_start, a_end, &um_u);
#endif // UM_CHEMISTRY

        if (abs(SphP[i].t_cool) < min_t_cool)
        {
            min_t_cool = abs(SphP[i].t_cool);
        }
        if (abs(SphP[i].t_cool) > max_t_cool)
        {
            max_t_cool = abs(SphP[i].t_cool);
        }

        if (abs(SphP[i].t_elec) < min_t_elec)
        {
            min_t_elec = abs(SphP[i].t_elec);
        }
        if (abs(SphP[i].t_elec) > max_t_elec)
        {
            max_t_elec = abs(SphP[i].t_elec);
        }
    }

    fprintf(stdout, "PE %d t_cool min= %g, max= %g in yrs", ThisTask, min_t_cool, max_t_cool);
    fflush(stdout);
    fprintf(stdout, "PE %d t_elec min= %g, max= %g in yrs", ThisTask, min_t_elec, max_t_elec);
    fflush(stdout);
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

    if (RestartFlag == 3)
    {
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
        VERBOSE(3, "SUBFIND_RESHUFFLE_AND_POTENTIAL: Calculating potential energy before reshuffling...");
#ifdef PMGRID
        long_range_init_regionsize();
#endif // PMGRID
        compute_potential();
        VERBOSE(3, "potential energy done.");
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL

#ifdef ADAPTGRAVSOFT
#ifdef PMGRID
        /* If PMGRID was on during the run, then at least a Maximum softening must have also existed:
         * we then need to take that into account during the computation of the potential energy of the particles */
        long_range_init_regionsize();
        All.AGS_MaxSoft[0] = AGS_MAXSOFT * All.Asmth[0];
        All.AGS_MinSoft[0] = AGS_MINSOFT * All.Asmth[0];
#ifdef PLACEHIGHRESREGION
        All.AGS_MaxSoft[1] = AGS_MAXSOFT * All.Asmth[1];
        All.AGS_MinSoft[1] = AGS_MINSOFT * All.Asmth[1];
#endif // PLACEHIGHRESREGION
#endif // PMGRID
        VERBOSE(3, "*ADAPTGRAVSOFT* Computation of softening lengths...");
        ags_setup_smoothinglengths();
        VERBOSE(3, "*ADAPTGRAVSOFT* Computation of softening lengths done.");
#endif // ADAPTGRAVSOFT

#ifdef FOF
        fof_fof(RestartSnapNum);
#endif // FOF

#ifdef LMB_SPECTRAL_CRs_POST_PROCESS
        TIMER_PUSH(1, "CR_injection");
        // CR injection
        inject_crs_postprocess();
        TIMER_POPPUSH(1, "CR_injection", "CR_evolution"); // CR_injection
        // spectral evolution
        evolve_spectral_crs();
        TIMER_POP(1, "CR_evolution"); // CR_evolution
                                      // write the snapshot
        savepositions(RestartSnapNum, -1);
#endif // LMB_SPECTRAL_CRs_POST_PROCESS
        ENDRUN_MAIN("End of restart run.");
    }

    if (RestartFlag == 5)
    {
        /* calculating powerspec and twopoint function */
#ifdef PMGRID
        long_range_init_regionsize();
#ifdef PERIODIC
        int n_type[6];
        long long ntot_type_all[6];
        /* determine global and local particle numbers */
        for (int n = 0; n < 6; n++)
        {
            n_type[n] = 0;
        }
        for (int n = 0; n < NumPart; n++)
        {
            n_type[P[n].Type]++;
        }
        sumup_large_ints(6, n_type, ntot_type_all);
        calculate_power_spectra(RestartSnapNum, ntot_type_all);
#endif // PERIODIC
#endif // PMGRID
        force_treebuild(NumPart, NULL);
        ENDRUN_MAIN("End of restart run.");
    }

    ENDRUN_IF(RestartFlag == 6, "End of restart run.");

#ifdef CHEMISTRY
    VERBOSE(2, "Initial abundances:");
    VERBOSE(2, "HI=%g, HII=%g, HeI=%g, HeII=%g, HeIII=%g", SphP[1].HI, SphP[1].HII, SphP[1].HeI, SphP[1].HeII,
            SphP[1].HeIII);
    VERBOSE(2, "HM=%g, H2I=%g, H2II=%g, elec=%g, %d", SphP[1].HM, SphP[1].H2I, SphP[1].H2II, SphP[1].elec, P[1].ID);
    VERBOSE(2, "x=%g, y=%g, z=%g, vx=%g, vy=%g, vz=%g, density=%g, entropy=%g", P[N_gas - 1].Pos[0],
            P[N_gas - 1].Pos[1], P[N_gas - 1].Pos[2], P[N_gas - 1].Vel[0], P[N_gas - 1].Vel[1], P[N_gas - 1].Vel[2],
            SphP[N_gas - 1].Density, SphP[N_gas - 1].Entropy);

    /* need predict the cooling time and elec_dot here */
    min_t_cool = min_t_elec = 1.0e30;
    max_t_cool = max_t_elec = -1.0e30;

    for (int i = 0; i < N_gas; i++)
    {
        a_start = All.Time;
        a_end = All.Time + 0.001; /* 0.001 as an arbitrary value */
        (void)compute_abundances(0, i, a_start, a_end);

        if (abs(SphP[i].t_cool) < min_t_cool)
        {
            min_t_cool = abs(SphP[i].t_cool);
        }
        if (abs(SphP[i].t_cool) > max_t_cool)
        {
            max_t_cool = abs(SphP[i].t_cool);
        }
        if (abs(SphP[i].t_elec) < min_t_elec)
        {
            min_t_elec = abs(SphP[i].t_elec);
        }
        if (abs(SphP[i].t_elec) > max_t_elec)
        {
            max_t_elec = abs(SphP[i].t_elec);
        }
    }

    fprintf(stdout, "PE %d t_cool min= %g, max= %g in yrs", ThisTask, min_t_cool, max_t_cool);
    fflush(stdout);
    fprintf(stdout, "PE %d t_elec min= %g, max= %g in yrs", ThisTask, min_t_elec, max_t_elec);
    fflush(stdout);
#endif // CHEMISTRY

    if (RestartFlag == 4)
    {
        All.Time = All.TimeBegin = header.time;
        // Temporary buffer to avoid undefined behavior
        char tempBuffer[sizeof(All.SnapshotFileBase)];
        snprintf(tempBuffer, sizeof(tempBuffer), "%s_converted", All.SnapshotFileBase);
        strncpy(All.SnapshotFileBase, tempBuffer, sizeof(All.SnapshotFileBase) - 1);
        VERBOSE(2, "Start writing file %s", All.SnapshotFileBase);
        VERBOSE_ALL(2, "RestartSnapNum %d", RestartSnapNum);

        All.TopNodeAllocFactor = 0.008;
        savepositions(RestartSnapNum, -1);
        ENDRUN_MAIN("End of restart run.");
    }
}

/**
 * @brief Checks the mass content of the box and compares it to the specified value of Omega-matter.
 *
 * This routine iterates over all particles in the simulation box to compute the total mass content.
 * The total mass is then normalized by the volume of the simulation box and the critical density to obtain
 * the value of Omega-matter. If the absolute difference between the computed Omega-matter and the specified
 * Omega0 exceeds a certain threshold, the routine prints a message indicating the discrepancy and terminates
 * the simulation run.
 */
void check_omega(void)
{
    double mass = 0, masstot, omega;
    // Compute total mass content of the box
    for (int i = 0; i < NumPart; i++)
    {
        mass += P[i].Mass;
    }
    // Sum total mass across all MPI processes
    MPI_Allreduce(&mass, &masstot, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    // Compute Omega-matter
    omega = masstot / (All.BoxSize * All.BoxSize * All.BoxSize) /
            (3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal));
#ifdef TIMEDEPGRAV
    omega *= All.Gini / All.GravityConstantInternal;
#endif // TIMEDEPGRAV

    // Check if computed Omega-matter significantly differs from specified Omega0
    if (abs(omega - All.Omega0) > 1.0e-2)
    {
        VERBOSE(0, "\n\nI've found something odd!");
        VERBOSE(0, "The mass content accounts only for Omega=%g,\nbut you specified Omega=%g in the parameterfile.",
                omega, All.Omega0);
#ifndef LMB_SPECTRAL_CRs_POST_PROCESS
        VERBOSE(0, "\nI better stop.");
#else // LMB_SPECTRAL_CRs_POST_PROCESS
        VERBOSE(0, "\nBut since we're only running the CR component in post-processing we don't have to care!\nYay!");
#endif // LMB_SPECTRAL_CRs_POST_PROCESS

#ifndef LMB_SPECTRAL_CRs_POST_PROCESS
        ENDRUN_MAIN("Stopping CR post processing due to odd Omega-matter value.");
#endif // LMB_SPECTRAL_CRs_POST_PROCESS
    }
}

/**
 * @brief Setup smooting legnths for SPH particles.
 *
 * This function is used to find an initial smoothing length for each SPH
 * particle. It guarantees that the number of neighbours will be between
 * desired_ngb-MAXDEV and desired_ngb+MAXDEV. For simplicity, a first guess
 * of the smoothing length is provided to the function density(), which will
 * then iterate if needed to find the right smoothing length.
 */
void setup_smoothinglengths(void)
{
    int no;
    // Check if restarting simulation
    if (RestartFlag == 0)
    {
        // Loop over gas particles
        for (int i = 0; i < N_gas; i++)
        {
            no = Father[i];
            // Iterate to find the correct node
            while (10 * All.DesNumNgb * P[i].Mass > Nodes[no].u.d.mass)
            {
                int p = Nodes[no].u.d.father;

                if (p < 0)
                {
                    break;
                }

                no = p;
            }

#ifndef READ_HSML
// Calculate smoothing length in one/two or three dimensions
#ifndef TWODIMS
#ifndef ONEDIM
            P[i].Hsml = pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
#else // ONEDIM
            P[i].Hsml = All.DesNumNgb * (P[i].Mass / Nodes[no].u.d.mass) * Nodes[no].len;
#endif // ONEDIM
#else // TWODIMS
            P[i].Hsml = pow(1.0 / (M_PI)*All.DesNumNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 2) * Nodes[no].len;
#endif // TWODIMS
            if (All.SofteningTable[0] != 0 && P[i].Hsml > 200.0 * All.SofteningTable[0])
            {
                P[i].Hsml = All.SofteningTable[0];
            }
#ifdef HSML_DEBUG
            if (P[i].ID == MyIDType(HSML_DEBUG))
            {
                VERBOSE_ALL(2, "HSML_DEBUG Init: i=%d task=%d ID=%llu Hsml=%g", i, ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Hsml);
                fflush(stdout);
            }
#endif
#endif // READ_HSML
        }
    }

#if (defined(mSIDM) || defined(mIDM)) && !defined(ADAPTGRAVSOFT)
    // Check if restarting simulation or setting up for the first time
    if (RestartFlag == 0 || RestartFlag == 2)
    {
        for (int i = 0; i < NumPart; ++i)
        {
            // Apply special treatment for dark matter particles
            if (P[i].Type == type_dm)
            {
                no = Father[i];

                while (10 * All.DesNumNgb_pt1 * P[i].Mass > Nodes[no].u.d.mass)
                {
                    int p = Nodes[no].u.d.father;

                    if (p < 0)
                    {
                        break;
                    }

                    no = p;
                }

                P[i].Hsml =
                    pow(3.0 / (4 * M_PI) * All.DesNumNgb_pt1 * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
            }
        }
    }
    // Calculate density for dark matter particles
    scatter::pt1_find_hsml();
#endif // (defined(mSIDM) || defined(mIDM)) && !defined(ADAPTGRAVSOFT)

#ifdef BLACK_HOLES
    // Check if restarting simulation
    if (RestartFlag == 0 || RestartFlag == 2)
    {
        // Loop over all particles
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].Type == type_bh)
            {
                P[i].Hsml = All.SofteningTable[5];
            }
        }
    }
#endif // BLACK_HOLES

#ifdef LT_STELLAREVOLUTION
    // Check if restarting simulation
    if (RestartFlag == 0 || RestartFlag == 2)
    {
        // Loop over all particles
        for (int i = 0; i < NumPart; i++)
        {
            if (P[i].Type == type_star)
            {
                // Set smoothing length for star particles
                P[i].Hsml = All.SofteningTable[4];
            }
        }
    }
#endif // LT_STELLAREVOLUTION
#ifdef CS_DECOUPLING
    TIMER_PUSH(1, "CS_DECOUPLE");
    cs_copy_densities();
    cs_find_low_density_tail();
    TIMER_POP(1, "CS_DECOUPLE");
#endif // CS_DECOUPLING
    find_hsml(); /* computes hsml, density, and pressure */
    compute_unified_gradients(); /* computes gradients and other additional values */
}

/**
 * @brief Assign unique IDs to particles.
 *
 * This function assigns unique IDs to particles based on the number of particles
 * on each MPI task. It ensures that each particle has a unique identifier.
 */
void assign_unique_ids(void)
{
    int *numpartlist;
    MyIDType idfirst;
    // Allocate memory for storing the number of particles on each MPI task
    numpartlist = (int *)mymalloc("numpartlist", NTask * sizeof(int));
    // Gather the number of particles from all MPI tasks
    MPI_Allgather(&NumPart, 1, MPI_INT, numpartlist, 1, MPI_INT, MYMPI_COMM_WORLD);
    idfirst = 1;
    // Calculate the starting ID for this MPI task based on the number of particles on previous tasks
    for (int i = 0; i < ThisTask; i++)
    {
        idfirst += numpartlist[i];
    }
    // Assign IDs to particles sequentially
    for (int i = 0; i < NumPart; i++)
    {
        P[i].ID = idfirst;
        idfirst++;
    }
    // Free memory allocated for storing the number of particles on each MPI task
    myfree(numpartlist);
}

#ifdef ADAPTGRAVSOFT
/**
 * @brief Sets up the smoothing lengths for the Adaptive Gravitational Softening (AGS) scheme.
 *
 * This function calculates and assigns smoothing lengths to particles based on the AGS scheme.
 * If gravitational softening lengths are stored in output files, it reads them; otherwise, it
 * computes them from scratch.
 */
void ags_setup_smoothinglengths(void)
{
    int no;

#ifdef AGS_OUTPUTGRAVSOFT
    /* If the gravitational softening lengths are stored in the outputfiles, then we'd rather read those than
     * recompute them from scratch when the code is restarted from a snapshot.*/
    // well, actually...
    //  Check if restarting simulation
    if (RestartFlag == 0 || RestartFlag == 2)
    //  if(RestartFlag == 0)
    {
        // Loop over all particles to assign smoothing lengths
        for (int i = 0; i < NumPart; i++)
        {
            no = Father[i];
            // Adjust smoothing length until it satisfies the AGS criterion
            while (10 * All.AGS_DesNumNgb * P[i].Mass > Nodes[no].u.d.mass)
            {
                int p = Nodes[no].u.d.father;

                if (p < 0)
                {
                    break;
                }

                no = p;
            }
            // Compute and assign the smoothing length based on the AGS scheme
            P[i].AGS_Hsml =
                pow(3.0 / (4 * M_PI) * All.AGS_DesNumNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
        }
    }
    // Compute density using the AGS scheme
    ags_density();

#else // AGS_OUTPUTGRAVSOFT

    // Check if restarting simulation
    if (RestartFlag == 0 || RestartFlag == 2)
    {
        // Loop over all particles to assign smoothing lengths
        for (int i = 0; i < NumPart; i++)
        {
            no = Father[i];
            // Adjust smoothing length until it satisfies the AGS criterion
            while (10 * All.AGS_DesNumNgb * P[i].Mass > Nodes[no].u.d.mass)
            {
                int p = Nodes[no].u.d.father;

                if (p < 0)
                {
                    break;
                }

                no = p;
            }
            // Compute and assign the smoothing length based on the AGS scheme
            P[i].AGS_Hsml =
                pow(3.0 / (4 * M_PI) * All.AGS_DesNumNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
        }
    }
    // Compute density using the AGS scheme
    ags_density();

#endif // AGS_OUTPUTGRAVSOFT
}
#endif // ADAPTGRAVSOFT

/**
 * @brief Tests if particle IDs are truly unique.
 *
 * This function checks whether particle IDs are unique across all particles in the simulation.
 * If any duplicate IDs are found, it terminates the simulation.
 */
void test_id_uniqueness(void)
{
    MyIDType *ids, *ids_first;

    VERBOSE(2, "Testing ID uniqueness...");
    // Check if there are particles present
    ENDRUN_IF(NumPart == 0, "We need at least one particle per CPU");
    TIMER_PUSH(1, "test_id_uniqueness");

#ifndef SPH_BND_PARTICLES
    // Allocate memory for storing particle IDs
    ids = (MyIDType *)mymalloc("ids", NumPart * sizeof(MyIDType));
    ids_first = (MyIDType *)mymalloc("ids_first", NTask * sizeof(MyIDType));
    // Copy particle IDs to the array
    for (int i = 0; i < NumPart; i++)
    {
        ids[i] = P[i].ID;
    }

#ifdef ALTERNATIVE_PSORT
    // Sort the IDs using an alternative sorting method
    init_sort_ID(ids, NumPart);
#else // ALTERNATIVE_PSORT
    // Sort the IDs in parallel using the default sorting method
    parallel_sort(ids, NumPart, sizeof(MyIDType), compare_IDs);
#endif // ALTERNATIVE_PSORT
       // Check for duplicate IDs
    for (int i = 1; i < NumPart; i++)
    {
        if (ids[i] == ids[i - 1])
        {
#ifdef LONGIDS
            ENDRUN("Non-unique ID=%d%09d found on task=%d (i=%d NumPart=%d)", (int)(ids[i] / 1000000000),
                   (int)(ids[i] % 1000000000), ThisTask, i, NumPart);
#else // LONGIDS
            ENDRUN("Non-unique ID=%d found on task=%d   (i=%d NumPart=%d)", (int)ids[i], ThisTask, i, NumPart);
#endif // LONGIDS
        }
    }
    // Gather the first ID from each task to check for continuity
    MPI_Allgather(&ids[0], sizeof(MyIDType), MPI_BYTE, ids_first, sizeof(MyIDType), MPI_BYTE, MYMPI_COMM_WORLD);
    // Check for continuity in IDs across tasks
    if (ThisTask < NTask - 1)
    {
        ENDRUN_IF(ids[NumPart - 1] == ids_first[ThisTask + 1], "Non-unique ID=%d found on task=%d",
                  (int)ids[NumPart - 1], ThisTask);
    }
    // Free memory allocated for storing first IDs from each task and ids in general
    myfree(ids_first);
    myfree(ids);
#endif // SPH_BND_PARTICLES

    TIMER_POP(1, "test_id_uniqueness");
}

/**
 * @brief Compares two IDs to check if they are identical or one is greater/less than the other.
 *
 * This function is used as a comparison function for sorting IDs. It returns -1 if the first ID is less than
 * the second, +1 if the first ID is greater than the second, and 0 if they are identical.
 *
 * @param a Pointer to the first ID.
 * @param b Pointer to the second ID.
 * @return -1 if the first ID is less than the second, +1 if the first ID is greater than the second, and 0 if they are
 * identical.
 */
int compare_IDs(const void *a, const void *b)
{
    if (*((MyIDType *)a) < *((MyIDType *)b))
    {
        return -1;
    }

    if (*((MyIDType *)a) > *((MyIDType *)b))
    {
        return +1;
    }

    return 0;
}
