/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the main loop, which iterates over timesteps.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <ctype.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "allvars.hpp"
#include "greentree.hpp"
#include "proto.hpp"
#ifdef ACC_RUN
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // ACC_RUN

void domain_decomposition_intensity_decision();
void domain_decomposition_intensity_execute();

/**
 * @brief Main simulation loop for iterating over timesteps.
 *
 * The loop terminates when the CPU-time limit is reached, a 'stop' file is found,
 * or when the simulation ends because TimeMax is reached. It includes initialization,
 * domain decomposition, physics updates, and other necessary computations.
 */
void run(void)
{
    TIMER_PUSH(0, "Initialisation");
#ifdef FPE_TRAPPING
    feenableexcept(FE_INVALID);
#endif // FPE_TRAPPING

    if (RestartFlag != 1)
    {
        // Compute forces at initial synchronization time, unless we restarted from restart files
        output_log_messages();
        domain_Decomposition(0, 0);

#ifdef RECOMPOSE_DOMAIN_FACTOR
        LastDomainUpdate = 0;
#endif // RECOMPOSE_DOMAIN_FACTOR

        set_non_standard_physics_for_current_time();
        // Compute gravitational accelerations for synchronous particles
        compute_grav_accelerations();
        // Compute densities for synchronous particles
        compute_densities();

#ifdef GM_STARDENSITY
        VERBOSE(1, "Start stellar density computation...");
        // Claculate stellar densities for star particles
        density_stars();
#endif // GM_STARDENSITY

#if GADGET_HYDRO == HYDRO_SPH || GADGET_HYDRO == HYDRO_PESPH
        // Compute hydro-accels for synchronous particles
        compute_hydro_accelerations();
#elif GADGET_HYDRO == HYDRO_MFM
        // Compute limiters once all gradients are known
        compute_limiters();
       // Compute Godunov fluxes
        compute_fluxes();
#endif // GADGET_HYDRO
       // Source terms are treated in a strang-split fashion
        calculate_non_standard_physics();

#ifdef mSIDM
        msidm.init_calc();
#endif // mSIDM

    }

    TIMER_POP(0, "Initialisation");
// Measure time for first timestep. This is deactivated in case of using the nvtx/roctx timers as we dont want
// cummulative times in this case, but for each timestep individually.
#if !defined(TIMER_NVTX) || !defined(TIMER_ROCTX)
    TIMER_PUSH(0, "Timestep");
#endif // TIMER_NVTX || TIMER_ROCTX

    /* main timestep iteration loop  */
    // Iterate over timesteps until the simulation ends
    while (1)
    {

#ifdef DEBUG
        check_particles();
#endif // DEBUG

#ifdef KD_COOLING_TEST
        savepositions(++All.SnapshotFileCount,
                      -1); /* this will be overwritten if All.TimeMax is increased and the run is continued */
        break;
#endif // KD_COOLING_TEST

        compute_statistics(); /* regular statistics outputs (like total energy) */
        create_snapshot_if_desired();
        write_cpu_log(); /* output some CPU usage log-info (accounts for everything needed up to the current sync-point)
                          */
#ifdef WRITE_ANOTHER_SNAPSHOT
        savepositions(All.SnapshotFileCount++,
                      0); /* this will be overwritten if All.TimeMax is increased and the run is continued */
        break;
#endif // WRITE_ANOTHER_SNAPSHOT

        // Check stop conditions
        int stopflag = check_stop_condition();

#ifndef SWITCH_TIMER_OFF
        // Write timer information to files
        TIMER_WRITE_TIMES_TO_CPUFILES();
        TIMER_WRITE_SYMBOLS_TO_BALANCEFILE();
        // Reset the timestep timer for a clean stack for the next timestep
        TIMER_RESET(TimestepTimer);
#endif // SWITCH_TIMER_OFF

        // End run if necessary
        if (stopflag > 0)
        {
            return;
        }

        find_timesteps(); /* find-timesteps */

        do_first_halfstep_kick(); /* half-step kick at beginning of timestep for synchronous particles */

        find_next_sync_point_and_drift(); /* find next synchronization point and drift particles to this time.
                                           * If needed, this function will also write an output file
                                           * at the desired time. */

        domain_decomposition_intensity_decision();

        output_log_messages(); /* write some info to log-files */

        set_non_standard_physics_for_current_time(); /* update auxiliary physics for current time */

        domain_decomposition_intensity_execute();

        compute_grav_accelerations(); /* compute gravitational accelerations for synchronous particles */

#ifdef AR_TREE_STATS
        ar_tree_stats(); // provides information on number of particles per tree-level
#endif // AR_TREE_STATS
        compute_densities(); /* densities for synchronous particles */
        /* GM WARNING : Supernovae evaluation is now here! */

#ifdef GM_STARDENSITY /*...and stellar densities for star particles */
        VERBOSE(1, "Start stellar density computation...");
        density_stars();
#endif // GM_STARDENSITY

#if GADGET_HYDRO == HYDRO_SPH || GADGET_HYDRO == HYDRO_PESPH
        compute_hydro_accelerations(); /* hydro-accels for synchronous particles */
#elif GADGET_HYDRO == HYDRO_MFM
        compute_limiters(); // Compute limiters once all gradients are known
        compute_fluxes(); // Compute Godunov fluxes
#endif // GADGET_HYDRO

        do_second_halfstep_kick(); /* this does the half-step kick at the end of the timestep */

        calculate_non_standard_physics(); /* source terms are here treated in a strang-split fashion */

#ifdef mSIDM
        msidm.do_full_kick(); /* computes new velocities due to self-interactions */
#endif // mSIDM

    }
}

/**
 * @brief Decides the domain decomposition intensity based on timebin distribution.
 *
 * This function analyzes the distribution of active particles across timebins
 * and determines the appropriate intensity for domain decomposition. It adjusts
 * flags for full, diffuse, and fix domain decomposition accordingly.
 */
void domain_decomposition_intensity_decision()
{
    long long tot_count[TIMEBINS];
    // long long tot_cumulative[TIMEBINS];
    int max_occupied_bin = 0, min_occupied_bin = TIMEBINS;

    // Calculate the total count and cumulative count of active particles in each timebin
    sumup_large_ints(TIMEBINS, TimeBinCount, tot_count);

    // tot_cumulative[0] = tot_count[0];
    // Determine the maximum and minimum occupied timebins
    for (int i = 1; i < TIMEBINS; i++)
    {
        // tot_cumulative[i] = tot_count[i] + tot_cumulative[i - 1];
        if (tot_count[i] > 0)
        {
            if (i > max_occupied_bin)
            {
                max_occupied_bin = i;
            }
            if (i < min_occupied_bin)
            {
                min_occupied_bin = i;
            }
        }
    }

    // Initialize intensity flags for domain decomposition
    AllDerived.DoDomainDecompositionTBDiffuse = 0;
    int DecompositionTBFull_merk = AllDerived.DoDomainDecompositionTBFull;
    // int DecompositionTBFix_merk = AllDerived.DoDomainDecompositionTBFix;

    // Place domain decomposition at 2/3 of the timebin distribution
    AllDerived.DoDomainDecompositionTBFull =
        (int)(min_occupied_bin + (max_occupied_bin - min_occupied_bin) * 2 / 3.0 + 0.5);
#ifdef AR_DD_TBFULL_SHIFT
    AllDerived.DoDomainDecompositionTBFull += AR_DD_TBFULL_SHIFT;
#endif // AR_DD_TBFULL_SHIFT

    // Paranoic check, should not happen
    if (AllDerived.DoDomainDecompositionTBFull > max_occupied_bin)
    {
        AllDerived.DoDomainDecompositionTBFull = max_occupied_bin;
    }
    // Never do a domain decomposition on the lowest timebin level
    if (AllDerived.DoDomainDecompositionTBFull == min_occupied_bin)
    {
        AllDerived.DoDomainDecompositionTBFull = max_occupied_bin;
    }
    // Never shift it more than 7 bins back from the lowest timebin
    if (AllDerived.DoDomainDecompositionTBFull > min_occupied_bin + 7)
    {
        AllDerived.DoDomainDecompositionTBFull = min_occupied_bin + 7;
    }

    // If domain decomposition is on the largest timebin, once can do it only every second time
    if (AllDerived.DoDomainDecompositionTBFull == max_occupied_bin && DecompositionTBFull_merk != 0)
    {
        AllDerived.DoDomainDecompositionTBFull = 0;
        AllDerived.DoDomainDecompositionTBDiffuse = 0;
        AllDerived.DoDomainDecompositionTBFix = 0;
    }

    // If domain decomposition is on the largest timebin, but timebin is not active
    if (AllDerived.DoDomainDecompositionTBFull == max_occupied_bin &&
        TimeBinActive[AllDerived.DoDomainDecompositionTBFull] == 0)
    {
        AllDerived.DoDomainDecompositionTBFull = 0;
        AllDerived.DoDomainDecompositionTBDiffuse = 0;
        AllDerived.DoDomainDecompositionTBFix = 0;
    }

    // So lets see if it makes sense to schedule repair steps
    if (AllDerived.DoDomainDecompositionTBFull > 0)
    {
        // diffusion makes only sense if there are at least 3 timebins occupied below
        if (AllDerived.DoDomainDecompositionTBFull - 1 > min_occupied_bin + 2)
        {
            AllDerived.DoDomainDecompositionTBDiffuse = AllDerived.DoDomainDecompositionTBFull - 1;
        }
        else
        {
            AllDerived.DoDomainDecompositionTBDiffuse = 0;
        }
        // repair makes only sense if done 2 steps below full and there are at least 2 more timebins occupied below
        if (AllDerived.DoDomainDecompositionTBFull - 2 > min_occupied_bin + 2)
        {
            AllDerived.DoDomainDecompositionTBFix = AllDerived.DoDomainDecompositionTBFull - 2;
        }
        else
        {
            AllDerived.DoDomainDecompositionTBFix = 0;
        }
    }

    VERBOSE(1, "RUN: Domain decomposition intensity (%d,%d,%d) decision", AllDerived.DoDomainDecompositionTBFull,
            AllDerived.DoDomainDecompositionTBDiffuse, AllDerived.DoDomainDecompositionTBFix);
}

/**
 * @brief Executes the domain decomposition intensity algorithm.
 *
 * This function performs domain decomposition based on the specified intensity flags.
 * It updates the tree dynamically with kicks from the last step and sets a new list
 * of active particles. Depending on the intensity flags, it may also perform additional
 * operations such as diffuse domain decomposition and chain list updates.
 */
void domain_decomposition_intensity_execute()
{
#ifdef NEUTRINOS
    // Enforce full domain decomposition when Time_tree_on_nu is reached or a clean setup
    static int neutrino_tree_rebuild_forced = 0;
    if (!neutrino_tree_rebuild_forced && All.Time > All.Time_tree_on_nu)
    {
        neutrino_tree_rebuild_forced = 1;
        VERBOSE(1,
                "DOMAIN: forcing full domain decomposition, neutrinos just became tree-active "
                "(Time=%g > Time_tree_on_nu=%g)\n",
                All.Time, All.Time_tree_on_nu);
        domain_Decomposition(0, 0);
        return;
    }
#endif // NEUTRINOS

    // Check if full domain decomposition is required
    if (AllDerived.DoDomainDecompositionTBFull != 0 && TimeBinActive[AllDerived.DoDomainDecompositionTBFull])
    {
#ifdef TIMER_DOMAIN
        double timer_frac = 1;
        if (ThisTask == 0)
        {
            double domain_timer = DomainTimer.get_region_time("DOMAIN");
            double total_timer = DomainTimer.get_total_time();
            timer_frac = domain_timer / total_timer;
        }

        // communicate fraction
        MPI_Bcast(&timer_frac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (timer_frac < DomainCritFractionFix)
        {
            if (ThisTask == 0)
            {
                TIMER_RESET(DomainTimer);
                TIMER_RESET(DomainRecompositionTimer);
            }
            VERBOSE(1, "DOMAIN: domain decomposition Intensity: Full");
            // Do domain decomposition if step is big enough, and set new list of active particles
            domain_Decomposition(0, 0);
        }
        else
        {
            VERBOSE(1, "Skipping domain decomposition Intensity: Full since time needed for last decomposition > 30 "
                       "per cent of time since last decomposition. ");
        }
#else // TIMER_DEBUG

        VERBOSE(1, "DOMAIN: domain decomposition Intensity: Full");
        // Do domain decomposition if step is big enough, and set new list of active particles
        domain_Decomposition(0, 0);
#endif // TIMER_DEBUG
        return;
    }
    // Update tree dynamically with kicks of last step so that it can be reused
    force_update_tree();
    // Check and perform diffuse domain decomposition if flagged
    if (AllDerived.DoDomainDecompositionTBDiffuse != 0 && TimeBinActive[AllDerived.DoDomainDecompositionTBDiffuse])
    {
        VERBOSE(1, "DOMAIN: domain decomposition Intensity: Diffuse");
#ifdef RECOMPOSE_DOMAIN_DIFFUSE
        domain_diffuse();
        domain_recomposition();
#else // RECOMPOSE_DOMAIN_DIFFUSE
        VERBOSE(2, "domain decomposition: No RECOMPOSE_DOMAIN_DIFFUSE flag");
#endif // RECOMPOSE_DOMAIN_DIFFUSE
    }
    // Check and perform fix domain decomposition if flagged
    else if (AllDerived.DoDomainDecompositionTBFix != 0 && TimeBinActive[AllDerived.DoDomainDecompositionTBFix])
    {
#ifdef TIMER_DOMAIN
        double timer_frac = 1;
        if (ThisTask == 0)
        {
            double domain_timer = DomainRecompositionTimer.get_region_time("DOMAIN_RECOMPOSITION");
            double total_timer = DomainRecompositionTimer.get_total_time();
            timer_frac = domain_timer / total_timer;
        }
        // communicate fraction
        MPI_Bcast(&timer_frac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (timer_frac < DomainCritFractionFix)
        {
            TIMER_RESET(DomainRecompositionTimer);
            VERBOSE(1, "DOMAIN: domain decomposition Intensity: Fix");
            domain_recomposition();
        }
        else
        {
            VERBOSE(1, "Skipping domain decomposition Intensity: Fix since time needed for last decomposition fix > 30 "
                       "per cent of time since last decomposition. ");
        }
#else // TIMER_DOMAIN
        VERBOSE(1, "DOMAIN: domain decomposition Intensity: Fix");
        domain_recomposition();
#endif // TIMER_DOMAIN
    }
    make_list_of_active_particles(); /* now we can set the new chain list of active particles */
}

/**
 * @brief Checks the stop condition for interrupting the run.
 *
 * This function checks various conditions, such as the presence of a stop-file,
 * running out of CPU-time, or reaching a specific time limit, to determine if
 * the simulation should be interrupted.
 *
 * @return An integer representing the stop condition:
 *         - 0: No interruption needed.
 *         - 1: Stop-file present or reached a specified time limit.
 *         - 2: Running out of CPU-time.
 *         - 3: Write a regularly spaced restart file.
 */
int check_stop_condition(void)
{
    TIMER_PUSH(3, "check_stop_condition");

    int stopflag = 0;
    // Check if we need to interupt run only on master task
    if (ThisTask == 0)
    {
        FILE *fd;
        char stopfname[1000];
        sprintf(stopfname, "%sstop", All.OutputDir);

        if (All.NumCurrentTiStep - All.StartTiStep ==
            All.StopAfterNSteps) // Check if the simulation has reached the specified number of steps and then
                                 // interrupt the run
        {
            VERBOSE(0,
                    "\n### Stopping after N-Steps active ###\nStarted at step %d \nCurrent step number: %d \n%d steps "
                    "done, stopping now.",
                    All.StartTiStep, All.NumCurrentTiStep, All.StopAfterNSteps);
            stopflag = 1;
        }
        else if ((CPUThisRun - All.TimeLastRestartFile) >=
                 All.CpuTimeBetRestartFile) // Check if it's time to write a regularly spaced restart file
        {
            All.TimeLastRestartFile = CPUThisRun;
            stopflag = 3;
        }
        else if (All.TimeLimitCPU > 0 &&
                 CPUThisRun >
                     0.85 * All.TimeLimitCPU) // Check if running out of CPU-time, interrupt the run if necessary
        {
            VERBOSE(0, "### Reaching time-limit. stopping.");
            stopflag = 2;
        }
        else if (All.Time >= All.StopAtA) // Interrupt the run if a specified time is reached
        {
            VERBOSE(0, "### Stopping at scale factor a is active with user-specified value %g.\n### Current time %g",
                    All.StopAtA, All.Time);
            stopflag = 1;
        }
        else if ((fd = fopen(stopfname, "r"))) // Check for the presence of the stop-file and interrupt the run if found
        {
            fclose(fd);
            stopflag = 1;
            unlink(stopfname);
        }
    }

    MPI_Bcast(&stopflag, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    if (stopflag == 1 || stopflag == 2)
    {
#ifndef DISABLE_OUTPUT_AT_FINAL_TIME
        restart(0); /* write restart file */
        MPI_Barrier(MYMPI_COMM_WORLD);
#endif // DISABLE_OUTPUT_AT_FINAL_TIME
        if (stopflag == 2 && ThisTask == 0)
        {
            char contfname[1000];
            sprintf(contfname, "%scont", All.OutputDir);
            FILE *fd = fopen(contfname, "w");
            if (fd)
            {
                fclose(fd);
            }
        }
        return stopflag;
    }
    else if (stopflag == 3) // write regularily spaced restart file
    {
#ifndef DISABLE_OUTPUT_AT_FINAL_TIME
        // write an occasional restart file
        restart(0);
        MPI_Barrier(MYMPI_COMM_WORLD);
#endif // DISABLE_OUTPUT_AT_FINAL_TIME
        stopflag = 0;
        // All.TimeLastRestartFile += report_time();
    }
    // draw a new list of random numbers
    set_random_numbers();

    report_memory_usage(&HighMark_run, "RUN");

    if (All.Ti_Current >= TIMEBASE) /* check whether we reached the final time */
    {
        VERBOSE(0, "\nFinal time=%g reached. Simulation ends.", All.TimeMax);

#ifndef DISABLE_OUTPUT_AT_FINAL_TIME
        restart(0); /* write a restart file to allow continuation of the run for a larger value of TimeMax */

        if (All.Ti_lastoutput !=
            All.Ti_Current) /* make a snapshot at the final time in case none has produced at this time */
        {
            savepositions(All.SnapshotFileCount++,
                          0); /* this will be overwritten if All.TimeMax is increased and the run is continued */
        }
#endif // DISABLE_OUTPUT_AT_FINAL_TIME

        stopflag = 1;
    }

    TIMER_POP(3, "check_stop_condition");
    return stopflag;
}

/**
 * @brief Sets up non-standard physics parameters for the current simulation time.
 */
void set_non_standard_physics_for_current_time(void)
{
    TIMER_PUSH(3, "set_non_standard_physics_for_current_time");

#if defined(UM_CHEMISTRY) && !defined(UM_DISABLE_RADIATION)
    // Initialize radiative rates and radiation for the current time
    init_rad(All.Time);
#endif // UM_DISABLE_RADIATION

#ifdef COOLING
    // Set up the UV background for the current time
    IonizeParams();
#endif // COOLING
    TIMER_POP(3, "set_non_standard_physics_for_current_time");
}

/**
 * @brief Performs computations for non-standard physics processes during the simulation.
 *
 * This function handles various non-standard physics processes based on compilation flags.
 * It includes the following functionalities:
 * - Conduction calculations (if CONDUCTION is defined)
 * - Black hole accretion and feedback (if BLACK_HOLES is defined)
 * - Spectral cosmic rays evolution (if LMB_SPECTRAL_CRs is defined)
 * - Spectral cosmic rays diffusion (if LMB_SPECTRAL_CRs_DIFFUSION is defined)
 * - On-the-fly calculation of black hole seeding and variable wind model (if BLACK_HOLES is defined)
 * - Radiative cooling and star formation processes (if COOLING is defined)
 */
void calculate_non_standard_physics(void)
{

#ifdef CONDUCTION
    // Thermal conduction calculations if enabled
    if (All.Conduction_Ti_endstep == All.Ti_Current)
    {
        conduction();
    }
#endif // CONDUCTION

#ifdef BLACK_HOLES

#ifdef PB_INITIALIZE_BH_PART
    // Do at start (0'th timestep) of Isolated System (Galaxy or Cluster) Simulation.
    if (All.NumCurrentTiStep == 0)
    {
        Initialize_Black_Hole();
    }
#endif // PB_INITIALIZE_BH_PART

    // Black hole accretion and feedback if enabled
    if (All.BlackHolesOn == 1)
    {
        blackhole();
    }
#endif // BLACK_HOLES


#ifdef ACC_RUN
    // Execute specific actions at the end of each timestep if ACC_RUN is defined
    on_end_of_timestep();
#endif // ACC_RUN

#ifdef BLACK_HOLES
#ifdef FOF
    // Find new black hole seed halos and/or assign host halo masses for the variable wind model
    if (All.Time >= All.TimeNextOnTheFlyFoF)
    {
        TIMER_PUSH(1, "BH_SEEDING");
        // Determine the appropriate fof_fof mode based on the black hole seed star mass fraction
        if (All.BlackHoleSeedStarMassFraction > 0)
        {
#ifdef SFR
            if (All.TotN_stars > 1000)
            {
#endif // SFR
                fof_fof(-2);
#ifdef SFR
            }
#endif // SFR
        }
        else
        {
            fof_fof(-1);
        }
        // Update the next on-the-fly FoF time
        if (All.ComovingIntegrationOn)
        {
            All.TimeNextOnTheFlyFoF *= All.TimeBetOnTheFlyFoF;
        }
        else
        {
            All.TimeNextOnTheFlyFoF += All.TimeBetOnTheFlyFoF;
        }
        TIMER_POP(1, "BH_SEEDING");
    }
#endif // FOF
#endif // BLACK_HOLES

#ifdef COOLING
    // Perform radiative cooling and star formation processes if COOLING is defined
    TIMER_PUSH(1, "COOLING_SFR");
#if GADGET_SFR == SFR_NONE

    if (All.CoolingOn == 1)
    {
        VERBOSE(1, "Start Cooling ..");
        TIMER_PUSH(1, "COOLING");
        // Execute cooling calculations only
        cl_CoolingOnly();
        TIMER_POP(1, "COOLING");
        VERBOSE(1, ".. Cooling done.");
    }
#endif // GADGET_SFR == SFR_NONE


#ifdef SFR
    if (All.StarformationOn == 1)
    {
        VERBOSE(1, "Start Cooling and Starformation..");
        TIMER_PUSH(1, "SFR");
        // Execute cooling and star formation calculations
        cooling_and_starformation();
        TIMER_POP(1, "SFR");
        VERBOSE(1, ".. Cooling and Starformation done.");
    }


#else // SFR
    if (All.CoolingOn == 1)
    {
        VERBOSE(1, "Start Cooling ..");
        TIMER_PUSH(1, "COOLING");
        // Execute cooling calculations only
        cl_CoolingOnly();
        TIMER_POP(1, "COOLING");
        VERBOSE(1, ".. Cooling done.");
    }
#endif // SFR

    TIMER_POP(1, "COOLING_SFR");
#endif // COOLING
}

/**
 * @brief Computes and outputs various statistics if the specified time interval has elapsed.
 *
 * This function checks if the time elapsed since the last statistics computation exceeds the
 * specified time interval (All.TimeBetStatistics). If so, it performs various statistical
 * computations, including potential energy calculation (if enabled) and energy statistics output.
 * The computed statistics are then updated.
 */
void compute_statistics(void)
{
    TIMER_PUSH(3, "compute_statistics");
    // Check if the time interval for statistics computation has elapsed
    if ((All.Time - All.TimeLastStatistics) >= All.TimeBetStatistics)
    {
#ifdef COMPUTE_POTENTIAL_ENERGY
        compute_potential();
#endif // COMPUTE_POTENTIAL_ENERGY
        energy_statistics(); /* compute and output energy statistics */

        // Update the time of the last statistics computation
        All.TimeLastStatistics += All.TimeBetStatistics;
    }

    TIMER_POP(3, "compute_statistics");
}

/**
 * @brief Create and write a simulation snapshot.
 *
 * This function generates an output snapshot of the current simulation state.
 * Depending on compile-time flags, it may also recompute physical quantities
 * such as the gravitational potential before writing the snapshot.
 */
void create_snapshot(void)
{
#ifdef OUTPUTPOTENTIAL
#if !defined(EVALPOTENTIAL) || (defined(EVALPOTENTIAL) && defined(RECOMPUTE_POTENTIAL_ON_OUTPUT))
    // Perform domain decomposition and compute potential energy if necessary
    domain_Decomposition(0, 0);
    compute_potential();
#endif // !defined(EVALPOTENTIAL) || (defined(EVALPOTENTIAL) && defined(RECOMPUTE_POTENTIAL_ON_OUTPUT))
#endif // OUTPUTPOTENTIAL

#ifdef LT_SEvDbg
    get_metals_sumcheck(9);
#endif // LT_SEvDbg

    savepositions(All.SnapshotFileCount++, 0);
    // Find the next scheduled output time
    All.Ti_nextoutput = find_next_outputtime(All.Ti_Current + 1);
}

/**
 * @brief Creates a snapshot if the current simulation time matches the next output time.
 *
 * In case WRITE_SNAP_AT_EACH_STEP is set, it will create a snapshot file. If not this function checks if the highest
 * active and occupied time bins are synchronized and if the current simulation time surpasses the next scheduled output
 * time. If both conditions are met, the function triggers the creation of a snapshot file.
 */
void create_snapshot_if_desired(void)
{
#ifdef SYNCRONIZ_OUTPUT
    // Evaluate snapshot condition once
    const bool should_create_snapshot = (All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) &&
                                        (All.Ti_Current >= All.Ti_nextoutput) && (All.Ti_nextoutput >= 0);

#ifdef WRITE_SNAP_AT_EACH_STEP
    // dp snapshot (potentiall with domain), otherwise just savepositions
    if (should_create_snapshot)
    {
        create_snapshot();
    }
    else
    {
        savepositions(All.SnapshotFileCount++, 0);
    }

#else // WRITE_SNAP_AT_EACH_STEP
    // Only create snapshot when condition holds
    if (should_create_snapshot)
    {
        create_snapshot();
    }

#endif // WRITE_SNAP_AT_EACH_STEP
#endif // SYNCRONIZ_OUTPUT
}

/**
 * @brief Finds the next synchronization point of the system and drifts the system to that point.
 *
 * This function determines the earliest point in time at which any particle requires a force
 * computation (synchronization point) and drifts the system to that moment. If the system drifts
 * over the desired time of a snapshot file, the function generates an output and resumes the drift.
 */
void find_next_sync_point_and_drift(void)
{
    All.NumCurrentTiStep++; // Move to the next sync point -> keep before PUSH for MERIC instrumentation
    TIMER_PUSH(1, "DRIFT");

    int n;
    integertime ti_next_kick, ti_next_kick_global;
    int highest_active_bin, highest_occupied_bin;
    double timeold = All.Time;

    // Find the next kick time
    for (n = 0, ti_next_kick = TIMEBASE, highest_occupied_bin = 0; n < TIMEBINS; n++)
    {
        if (TimeBinCount[n])
        {
            integertime dt_bin, ti_next_for_bin;
            if (n > 0)
            {
                highest_occupied_bin = n;
                dt_bin = (((integertime)1) << n); // = 2^(n+1)
                ti_next_for_bin = (All.Ti_Current / dt_bin) * dt_bin + dt_bin; // Next kick time for this timebin
            }
            else
            {
                // dt_bin = 0;
                ti_next_for_bin = All.Ti_Current;
            }

            if (ti_next_for_bin < ti_next_kick)
            {
                ti_next_kick = ti_next_for_bin;
            }
        }
    }

#ifdef ENLARGE_DYNAMIC_RANGE_IN_TIME
    minimum_large_ints(1, &ti_next_kick, &ti_next_kick_global);
#else // ENLARGE_DYNAMIC_RANGE_IN_TIME
    MPI_Allreduce(&ti_next_kick, &ti_next_kick_global, 1, MPI_INT, MPI_MIN, MYMPI_COMM_WORLD);
#endif // ENLARGE_DYNAMIC_RANGE_IN_TIME

#ifndef SYNCRONIZ_OUTPUT
#ifdef WRITE_SNAP_AT_EACH_STEP
    bool snapshot_created = false;
#endif // WRITE_SNAP_AT_EACH_STEP
    VERBOSE(2, "ti_next_kick_global = %i, All.Ti_nextoutput = %i", ti_next_kick_global, All.Ti_nextoutput);
    // Generate outputs and resume drift until synchronized
    while (ti_next_kick_global >= All.Ti_nextoutput && All.Ti_nextoutput >= 0)
    {
        All.Ti_Current = All.Ti_nextoutput;
        // Update the simulation time
        if (All.ComovingIntegrationOn)
        {
            All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);
        }
        else
        {
            All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;
        }
        // Update cosmological factors
        set_cosmo_factors_for_current_time();

#ifdef TIMEDEPGRAV
        All.GravityConstantInternal = All.Gini * dGfak(All.Time);
#endif // TIMEDEPGRAV
       // Move particles to the next output time
        move_particles(All.Ti_nextoutput);

        // create snapshot
        create_snapshot();
#ifdef WRITE_SNAP_AT_EACH_STEP
        snapshot_created = true;
#endif // WRITE_SNAP_AT_EACH_STEP
    }

#ifdef WRITE_SNAP_AT_EACH_STEP
    if (!snapshot_created)
    {
        savepositions(All.SnapshotFileCount++, 0);
    }
#endif // WRITE_SNAP_AT_EACH_STEP

#endif // SYNCRONIZ_OUTPUT

    // Update time-related variables
    All.Previous_Ti_Current = All.Ti_Current;
    All.Ti_Current = ti_next_kick_global;

    if (All.ComovingIntegrationOn)
    {
        All.Time = All.TimeBegin * exp(All.Ti_Current * All.Timebase_interval);
    }
    else
    {
        All.Time = All.TimeBegin + All.Ti_Current * All.Timebase_interval;
    }
    // Update cosmological factors
    set_cosmo_factors_for_current_time();

#ifdef TIMEDEPGRAV
    All.GravityConstantInternal = All.Gini * dGfak(All.Time);
#endif // TIMEDEPGRAV
    // Update the time step
    All.TimeStep = All.Time - timeold;

#ifdef LT_STELLAREVOLUTION
    All.Time_Age = get_age(All.Time);
#endif // LT_STELLAREVOLUTION

    // Mark the bins that will be active
    for (n = 1, TimeBinActive[0] = 1, NumForceUpdate = TimeBinCount[0], highest_active_bin = 0; n < TIMEBINS; n++)
    {
        integertime dt_bin = (((integertime)1) << n);
        if ((ti_next_kick_global % dt_bin) == 0)
        {
            TimeBinActive[n] = 1;
            NumForceUpdate += TimeBinCount[n];
            if (TimeBinCount[n])
            {
                highest_active_bin = n;
            }
        }
        else
        {
            TimeBinActive[n] = 0;
        }
    }
    // Update global variables
    sumup_large_ints(1, &NumForceUpdate, &GlobNumForceUpdate);
    MPI_Allreduce(&highest_active_bin, &All.HighestActiveTimeBin, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
    MPI_Allreduce(&highest_occupied_bin, &All.HighestOccupiedTimeBin, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
    // Check if the system is fully synchronized
    if (GlobNumForceUpdate == All.TotNumPart)
    {
        Flag_FullStep = 1;
        if (All.HighestActiveTimeBin != All.HighestOccupiedTimeBin)
        {
            ENDRUN_MAIN("Something is wrong with the time bins. ");
        }
    }
    else
    {
        Flag_FullStep = 0;
    }

    /* Move the new set of active/synchronized particles
     Note: We do not yet call make_list_of_active_particles(), since we
     may still need the old list in the dynamic tree update */
#ifndef KD_DRIFT_ALL_PARTICLES
    // Build flat index list
    std::vector<int> active_particles;
    active_particles.reserve(NumPart); // avoid reallocs

    for (int ind = 0; ind < TIMEBINS; ind++)
    {
        if (TimeBinActive[ind])
        {
            for (int i = FirstInTimeBin[ind]; i >= 0; i = NextInTimeBin[i])
            {
                active_particles.push_back(i);
            }
        }
    }

#pragma omp parallel for schedule(dynamic,                                                                             \
                                  64) // dynamic sheduling due to non uniform work per particle (e.g. is SPH or not)
    for (int k = 0; k < (int)active_particles.size(); k++)
    {
        drift_particle(active_particles[k], All.Ti_Current);
    }

#else // KD_DRIFT_ALL_PARTICLES
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        drift_particle(i, All.Ti_Current);
    }
#pragma omp parallel for
    for (int no = All.MaxPart; no < All.MaxPart + Numnodestree; no++)
    {
        force_drift_node(no, All.Ti_Current);
    }
#endif // KD_DRIFT_ALL_PARTICLES

    TIMER_POP(1, "DRIFT");
}

/**
 * @brief Makes a linked list of active particles based on their time bins.
 *
 * This function iterates through the time bins, identifies active particles,
 * and creates a linked list of these particles. The linked list is used to
 * efficiently access and process active particles.
 */
void make_list_of_active_particles(void)
{
    TIMER_PUSH(3, "make_list_of_active_particles");

    // Make a link list with the particles in the active time bins
    FirstActiveParticle = -1;
    int prev = -1;
    // Iterate through the time bins
    for (int n = 0; n < TIMEBINS; n++)
    {
        // Check if the time bin is active
        if (TimeBinActive[n])
        {
            // Iterate through particles in the active time bin
            for (int i = FirstInTimeBin[n]; i >= 0; i = NextInTimeBin[i])
            {
#ifdef BLACK_HOLES
                // Skip particles with zero mass (for black holes)
                if (P[i].Mass == 0)
                {
                    continue;
                }
#endif // BLACK_HOLES

                // Update linked list pointers
                if (prev == -1)
                {
                    FirstActiveParticle = i;
                }

                if (prev >= 0)
                {
                    NextActiveParticle[prev] = i;
                }

                prev = i;
            }
        }
    }

    // Complete the linked list
    if (prev >= 0)
    {
        NextActiveParticle[prev] = -1;
    }

#ifdef _OPENMP
    // Build an array of active particles for OpenMP parallelization
    build_active_particle_list();
#endif // _OPENMP

    TIMER_POP(3, "make_list_of_active_particles");
}

#ifdef _OPENMP
/**
 * @brief  This function builds a list of active particles by iterating through the linked list of active particles.
    It updates the ActiveParticleList array and maintains the count of active particles in NActivePart.
 */
void build_active_particle_list()
{
    NActivePart = 0;
    // Iterate through the linked list of active particles
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        // Add the current particle index to the ActiveParticleList array
        ActiveParticleList[NActivePart] = i;
        // Increment the count of active particles
        NActivePart++;
    }
}
#endif // _OPENMP

/**
 * @brief Find the next output time that is equal or larger than the given current time.
 *
 * This function searches for the next valid output time based on the simulation parameters, including the
 * user-defined output list (if enabled) and the default output time intervals. It returns the time in terms
 * of the simulation's internal time units and sets the DumpFlag to indicate the type of output (snapshot, etc.).
 *
 * @param ti_curr The current simulation time.
 *
 * @return The next output time in simulation time units.
 */
integertime find_next_outputtime(integertime ti_curr)
{
    integertime ti, ti_next;
    double next, time;

    DumpFlag = 1;
    ti_next = -1;

    // Check if user-defined output list is enabled
    if (All.OutputListOn)
    {
        for (int i = 0; i < All.OutputListLength; i++)
        {
            time = All.OutputListTimes[i];

            if (time >= All.TimeBegin && time <= All.TimeMax)
            {
                // Calculate time in internal units
                if (All.ComovingIntegrationOn)
                {
                    ti = (integertime)(log(time / All.TimeBegin) / All.Timebase_interval);
                }
                else
                {
                    ti = (integertime)((time - All.TimeBegin) / All.Timebase_interval);
                }

#ifdef PROCESS_TIMES_OF_OUTPUTLIST
                // Determine maximum output interval based on All.MaxSizeTimestep
                integertime timax = (integertime)(All.MaxSizeTimestep / All.Timebase_interval);

                // Make a power 2 subdivision
                integertime ti_min = TIMEBASE;
                while (ti_min > timax)
                {
                    ti_min >>= 1;
                }
                timax = ti_min;

                double multiplier = ti / ((double)timax);

                // Round to the nearest multiple of timax
                ti = ((integertime)(multiplier + 0.5)) * timax;
#endif // PROCESS_TIMES_OF_OUTPUTLIST
       // Check if the calculated time is equal or larger than the current time
                if (ti >= ti_curr)
                {
                    if (ti_next == -1)
                    {
                        // Update ti_next and DumpFlag based on the conditions
                        ti_next = ti;
                        DumpFlag = All.OutputListFlag[i];
                        if (i > All.SnapshotFileCount)
                        {
                            All.SnapshotFileCount = i;
                        }
                    }

                    if (ti_next > ti)
                    {
                        ti_next = ti;
                        DumpFlag = All.OutputListFlag[i];
                        if (i > All.SnapshotFileCount)
                        {
                            All.SnapshotFileCount = i;
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Check if valid TimeBetSnapshot is provided for the simulation
        if (All.ComovingIntegrationOn)
        {
            ENDRUN_IF(All.TimeBetSnapshot <= 1.0, "TimeBetSnapshot > 1.0 required for your simulation.");
        }
        else
        {
            ENDRUN_IF(All.TimeBetSnapshot <= 0.0, "TimeBetSnapshot > 0.0 required for your simulation.");
        }

        // Initialize time based on the first snapshot time
        time = All.TimeOfFirstSnapshot;
        int iter = 0;
        // Iterate until a valid next output time is found
        while (time < All.TimeBegin)
        {
            // Calculate the next time based on the simulation conditions
            if (All.ComovingIntegrationOn)
            {
                time *= All.TimeBetSnapshot;
            }
            else
            {
                time += All.TimeBetSnapshot;
            }

            iter++;
            // Check for iteration limit
            ENDRUN_IF(iter > 1000000, "Can't determine next output time.");
        }
        // Continue the search for the next output time
        while (time <= All.TimeMax)
        {
            // Calculate time in internal units
            if (All.ComovingIntegrationOn)
            {
                ti = (integertime)(log(time / All.TimeBegin) / All.Timebase_interval);
            }
            else
            {
                ti = (integertime)((time - All.TimeBegin) / All.Timebase_interval);
            }
            // Check if the calculated time is equal or larger than the current time
            if (ti >= ti_curr)
            {
                ti_next = ti;
                break;
            }
            // Calculate the next time based on the simulation conditions
            if (All.ComovingIntegrationOn)
            {
                time *= All.TimeBetSnapshot;
            }
            else
            {
                time += All.TimeBetSnapshot;
            }

            iter++;
            // Check for iteration limit
            ENDRUN_IF(iter > 1000000, "Can't determine next output time.");
        }
    }

    // Check if a valid next output time is found
    if (ti_next == -1)
    {
        ti_next = 2 * TIMEBASE; /* this will prevent any further output */

        VERBOSE(0, "\nThere is no valid time for a further snapshot file.");
    }
    else
    {
        // Convert the next time to physical units for logging
        if (All.ComovingIntegrationOn)
        {
            next = All.TimeBegin * exp(ti_next * All.Timebase_interval);
        }
        else
        {
            next = All.TimeBegin + ti_next * All.Timebase_interval;
        }
        VERBOSE(2, "\nSetting next time for snapshot file to Time_next= %g  (DumpFlag=%d) ", next, DumpFlag);
    }

    return ti_next;
}

/**
 * @brief Output log messages at synchronization points in the timeline.
 *
 * This routine writes information to two log files, FdInfo and FdTimebin, for every synchronization point
 * in the simulation timeline. In FdInfo, it lists the timesteps that have been completed, and in FdTimebin,
 * it provides information about the distribution of particles over the time bins and which time bins are active
 * during the current step.
 */
void output_log_messages(void)
{
    TIMER_PUSH(3, "output_log_messages");
    double z;
    long long tot, tot_sph;
    long long tot_count[TIMEBINS];
    long long tot_count_sph[TIMEBINS];
    long long tot_cumulative[TIMEBINS];
    int corr_weight;
    double avg_CPU_TimeBin[TIMEBINS], frac_CPU_TimeBin[TIMEBINS];


    // Calculate total counts for time bins
    sumup_large_ints(TIMEBINS, TimeBinCount, tot_count);
    sumup_large_ints(TIMEBINS, TimeBinCountSph, tot_count_sph);

    // Check if this is master the master task
    if (ThisTask == 0)
    {
        // Output information about the synchronization point
        if (All.ComovingIntegrationOn)
        {
            z = 1.0 / (All.Time) - 1;
            fprintf(FdInfo, "\nSync-Point %d, Time: %g, Redshift: %g, Nf = %d%09d, Systemstep: %g, Dloga: %g\n",
                    All.NumCurrentTiStep, All.Time, z, (int)(GlobNumForceUpdate / 1000000000),
                    (int)(GlobNumForceUpdate % 1000000000), All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
            VERBOSE(0, "\nSync-Point %d, Time: %g, Redshift: %g, Systemstep: %g, Dloga: %g\n", All.NumCurrentTiStep,
                    All.Time, z, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
            fprintf(FdTimebin, "\nSync-Point %d, Time: %g, Redshift: %g, Systemstep: %g, Dloga: %g\n",
                    All.NumCurrentTiStep, All.Time, z, All.TimeStep, log(All.Time) - log(All.Time - All.TimeStep));
            fflush(FdInfo);
        }
        else
        {
            fprintf(FdInfo, "\nSync-Point %d, Time: %g, Nf = %d%09d, Systemstep: %g\n", All.NumCurrentTiStep, All.Time,
                    (int)(GlobNumForceUpdate / 1000000000), (int)(GlobNumForceUpdate % 1000000000), All.TimeStep);
            VERBOSE(0, "\nSync-Point %d, Time: %g, Systemstep: %g\n", All.NumCurrentTiStep, All.Time, All.TimeStep);
            fprintf(FdTimebin, "\nSync-Point %d, Time: %g, Systemstep: %g\n", All.NumCurrentTiStep, All.Time,
                    All.TimeStep);
            fflush(FdInfo);
        }

        // Calculate cumulative counts and average CPU time per time bin
        tot_cumulative[0] = tot_count[0];
        for (int i = 1; i < TIMEBINS; i++)
        {
            tot_cumulative[i] = tot_count[i] + tot_cumulative[i - 1];
        }

        for (int i = 0; i < TIMEBINS; i++)
        {
            double sum = 0;
            for (int j = 0; j < All.CPU_TimeBinCountMeasurements[i]; j++)
            {
                sum += All.CPU_TimeBinMeasurements[i][j];
            }

            if (All.CPU_TimeBinCountMeasurements[i])
            {
                avg_CPU_TimeBin[i] = sum / All.CPU_TimeBinCountMeasurements[i];
            }
            else
            {
                avg_CPU_TimeBin[i] = 0;
            }
        }

        // Calculate the fraction of CPU time for each time bin
        int weight = 1;
        double sum = 0;
        for (int i = All.HighestOccupiedTimeBin; i >= 0; i--, weight *= 2)
        {
            if (weight > 1)
            {
                corr_weight = weight / 2;
            }
            else
            {
                corr_weight = weight;
            }

            if (tot_count_sph[i] > 0 || tot_count[i] > 0)
            {
                frac_CPU_TimeBin[i] = corr_weight * avg_CPU_TimeBin[i];
                sum += frac_CPU_TimeBin[i];
            }
        }

        // Normalize the fractions
        for (int i = All.HighestOccupiedTimeBin; i >= 0; i--)
        {
            if (sum)
            {
                frac_CPU_TimeBin[i] /= sum;
            }
        }

        // Output log information for each time bin
        VERBOSE(2,
                "Occupied timebins: non-cells     cells       dt                 cumulative A D    avg-time  cpu-frac");
        fprintf(
            FdTimebin,
            "Occupied timebins: non-cells     cells       dt                 cumulative A D    avg-time  cpu-frac\n");
        tot = 0;
        tot_sph = 0;
        for (int i = TIMEBINS - 1; i >= 0; i--)
        {
            if (tot_count_sph[i] > 0 || tot_count[i] > 0)
            {
                VERBOSE(2, " %c  bin=%2d      %10llu  %10lld   %16.12f       %10lld %c %c  %10.2f    %5.1f%% %c%c%c",
                        TimeBinActive[i] ? 'X' : ' ', i, tot_count[i] - tot_count_sph[i], tot_count_sph[i],
                        i > 0 ? (((integertime)1) << i) * All.Timebase_interval : 0.0, tot_cumulative[i],
                        (i == All.HighestActiveTimeBin) ? '<' : ' ', ' ', avg_CPU_TimeBin[i],
                        100.0 * frac_CPU_TimeBin[i], AllDerived.DoDomainDecompositionTBFull == i ? 'D' : ' ',
                        AllDerived.DoDomainDecompositionTBDiffuse == i ? 'd' : ' ',
                        AllDerived.DoDomainDecompositionTBFix == i ? 'f' : ' ');
                fprintf(FdTimebin,
                        " %c  bin=%2d      %10llu  %10lld   %16.12f       %10lld %c %c  %10.2f    %5.1f%% %c%c%c\n",
                        TimeBinActive[i] ? 'X' : ' ', i, tot_count[i] - tot_count_sph[i], tot_count_sph[i],
                        i > 0 ? (((integertime)1) << i) * All.Timebase_interval : 0.0, tot_cumulative[i],
                        (i == All.HighestActiveTimeBin) ? '<' : ' ', ' ', avg_CPU_TimeBin[i],
                        100.0 * frac_CPU_TimeBin[i], AllDerived.DoDomainDecompositionTBFull == i ? 'D' : ' ',
                        AllDerived.DoDomainDecompositionTBDiffuse == i ? 'd' : ' ',
                        AllDerived.DoDomainDecompositionTBFix == i ? 'f' : ' ');

                if (TimeBinActive[i])
                {
                    tot += tot_count[i];
                    tot_sph += tot_count_sph[i];
                }
            }
        }
        VERBOSE(2, "               ------------------------");
        fprintf(FdTimebin, "               ------------------------\n");

#ifdef PMGRID
        // Output PM-Step information
        if (All.PM_Ti_endstep == All.Ti_Current)
        {
            VERBOSE_ALL(2, "PM-Step. Total: %10lld  %10lld    Sum: %10llu", tot - tot_sph, tot_sph, tot);
            fprintf(FdTimebin, "PM-Step. Total: %10lld  %10lld    Sum: %10llu\n", tot - tot_sph, tot_sph, tot);
        }
        else
#endif // PMGRID
        {
            // Output total active particles information
            VERBOSE_ALL(2, "Total active:   %10lld  %10lld    Sum: %10lld", tot - tot_sph, tot_sph, tot);
            fprintf(FdTimebin, "Total active:   %10lld  %10lld    Sum: %10lld\n", tot - tot_sph, tot_sph, tot);
        }
        fprintf(FdTimebin, "\n");
        fflush(FdTimebin);
    }


    // Output additional log messages
    output_extra_log_messages();

    TIMER_POP(3, "output_log_messages");
}

/**
 * @brief Write CPU usage information to log files.
 *
 * This function writes CPU time measurements to log files, specifically 'cpu.txt' and 'balance.txt'.
 * It includes the time taken for the current timestep and updates cumulative CPU usage for the entire simulation run.
 *
 * Note: The writing of symbols to 'balance.txt' and times to 'cpu.txt' is controlled by timer-related macros.
 * Additionally, the function resets the timestep timer to prepare for new measurements in the next timestep.
 */
void write_cpu_log(void)
{
    // Check if the current task is the master task (ThisTask == 0)
    if (ThisTask == 0)
    {
        // Adjust measurements buffer for circular recording
        if (All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin] == NUMBER_OF_MEASUREMENTS_TO_RECORD)
        {
            All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin]--;
            memmove(&All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][0],
                    &All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin][1],
                    (NUMBER_OF_MEASUREMENTS_TO_RECORD - 1) * sizeof(double));
        }
        double Current_timestep_time = TimestepTimer.get_total_time();
        VERBOSE(1, "TIMERS: Step=%d sec=%g NF=%lld", All.NumCurrentTiStep, Current_timestep_time, GlobNumForceUpdate);
        // Record the time of the current timestep in the circular buffer
        All.CPU_TimeBinMeasurements[All.HighestActiveTimeBin]
                                   [All.CPU_TimeBinCountMeasurements[All.HighestActiveTimeBin]++] =
            Current_timestep_time;

        // update total cumulative CPU usage for the this run
        CPUThisRun += TimestepTimer.get_total_time();
    }
}

/**
 * @brief Compute global quantities of the particle distribution.
 *
 * This routine first calls a computation of various global
 * quantities of the particle distribution, and then writes some
 * statistics about the energies in the various particle components to
 * the file FdEnergy.
 */
void energy_statistics(void)
{
    // compute energy statistics
    compute_global_quantities_of_system();
    // Check if the current task is the master task (ThisTask == 0)
    if (ThisTask == 0)
    {
        // Print output to energy file
        fprintf(FdEnergy, "%g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g",
                All.Time, SysState.EnergyInt, SysState.EnergyPot, SysState.EnergyKin, SysState.EnergyIntComp[0],
                SysState.EnergyPotComp[0], SysState.EnergyKinComp[0], SysState.EnergyIntComp[1],
                SysState.EnergyPotComp[1], SysState.EnergyKinComp[1], SysState.EnergyIntComp[2],
                SysState.EnergyPotComp[2], SysState.EnergyKinComp[2], SysState.EnergyIntComp[3],
                SysState.EnergyPotComp[3], SysState.EnergyKinComp[3], SysState.EnergyIntComp[4],
                SysState.EnergyPotComp[4], SysState.EnergyKinComp[4], SysState.EnergyIntComp[5],
                SysState.EnergyPotComp[5], SysState.EnergyKinComp[5], SysState.MassComp[0], SysState.MassComp[1],
                SysState.MassComp[2], SysState.MassComp[3], SysState.MassComp[4], SysState.MassComp[5]);

#ifdef MAGNETIC_STATISTICS
        fprintf(FdEnergy, " %g", SysState.EnergyMag);
        fprintf(FdEnergy, " %g", SysState.DivBerr);
#endif // MAGNETIC_STATISTICS

        fprintf(FdEnergy, " \n");
        fflush(FdEnergy);
    }
}

/**
 * @brief Output extra log messages for diagnostic purposes.
 *
 * This function outputs various log messages for diagnostic purposes, providing additional information
 * based on specific conditions.
 */
void output_extra_log_messages(void)
{
    // Check if the current task is the master task (ThisTask == 0)
    if (ThisTask == 0)
    {
#ifdef LT_STELLAREVOLUTION_NOT_YET_FIXED
        long long tot, tot_sph, tot_stars;
        long long tot_count[TIMEBINS], tot_count_sph[TIMEBINS], tot_count_stars[TIMEBINS];
        int i;
        // Sum up counts for stars in each time bin
        sumup_large_ints(TIMEBINS, TimeBinCountStars, tot_count_stars);

        VERBOSE_ALL(2, "Occupied timebins: non-sph         sph       stars    dt");
        for (i = TIMEBINS - 1, tot = tot_sph = tot_stars = 0; i >= 0; i--)
        {
            if (tot_count_sph[i] > 0 || tot_count[i] > 0 || tot_count_stars[i] > 0)
            {
                VERBOSE_ALL(2, " %c  bin=%2d     %2d%09d %2d%09d %2d%09d   %6g", TimeBinActive[i] ? 'X' : ' ', i,
                            (int)((tot_count[i] - tot_count_sph[i]) / 1000000000),
                            (int)((tot_count[i] - tot_count_sph[i]) % 1000000000), (int)(tot_count_sph[i] / 1000000000),
                            (int)(tot_count_sph[i] % 1000000000), (int)(tot_count_stars[i] / 1000000000),
                            (int)(tot_count_stars[i] % 1000000000),
                            i > 0 ? (((integertime)1) << i) * All.Timebase_interval : 0.0);
                if (TimeBinActive[i])
                {
                    tot += tot_count[i];
                    tot_sph += tot_count_sph[i];
                    tot_stars += tot_count_stars[i];
                }
            }
        }
#endif // LT_STELLAREVOLUTION_NOT_YET_FIXED

#ifdef CHEMISTRY
        VERBOSE_ALL("Abundances elec: %g, HM: %g, H2I: %g, H2II: %g", SphP[1].elec, SphP[1].HM, SphP[1].H2I,
                    SphP[1].H2II);
#endif // CHEMISTRY

#ifdef XXLINFO
        if (Flag_FullStep == 1)
        {
            fprintf(FdXXL, "%d %g ", All.NumCurrentTiStep, All.Time);
#ifdef MAGNETIC
            fprintf(FdXXL, "%e ", MeanB);
            fprintf(FdXXL, "%e ", MaxDivB);
#endif // MAGNETIC
#ifdef TIME_DEP_ART_VISC
            fprintf(FdXXL, "%f ", MeanAlpha);
#endif // TIME_DEP_ART_VISC
            fprintf(FdXXL, "\n");
            fflush(FdXXL);
        }
#endif // XXLINFO

#ifdef DARKENERGY
        if (All.ComovingIntegrationOn == 1)
        {
            double hubble_a;

            hubble_a = hubble_function(All.Time);
            fprintf(FdDE, "%d %g %e ", All.NumCurrentTiStep, All.Time, hubble_a);
#ifndef TIMEDEPDE
            fprintf(FdDE, "%e ", All.DarkEnergyParam);
#else // TIMEDEPDE
            fprintf(FdDE, "%e %e ", get_wa(All.Time), DarkEnergy_a(All.Time));
#endif // TIMEDEPDE
#ifdef TIMEDEPGRAV
            fprintf(FdDE, "%e %e", dHfak(All.Time), dGfak(All.Time));
#endif // TIMEDEPGRAV
            fprintf(FdDE, "\n");
            fflush(FdDE);
        }
#endif // DARKENERGY
    }
}

/**
 * @brief Check node data for consistency and validity.
 *
 * This function performs a check on node data in the tree structure,
 * ensuring that tree nodes are correctly linked and that top-level tree nodes are reached.
 * It prints verbose messages for potential issues and terminates the program if critical errors are detected.
 */
void check_node_info(const char *func, const char *file, int linenr)
{
    int maxpart = All.MaxPart;
    // Synchronize all MPI processes before performing checks.
    MPI_Barrier(MYMPI_COMM_WORLD);

    VERBOSE(2, "Checking node data (function %s in file %s at line %d) ...", func, file, linenr);
    // Loop through active particles
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        // Get the father node of the current particle
        int no = Father[i];
        // Check if the father node is within valid range
        ENDRUN_IF(no < All.MaxPart, "error at check 1: i=%d, no=%d < All.MaxPart=%d ", i, no, All.MaxPart);
        ENDRUN_IF(no >= maxpart + MaxNodes + 1, "error: at check 2: i=%d, no=%d >= maxpart=%d ", i, no,
                  maxpart + MaxNodes + 1);

        // Traverse up the tree until the top-level node is reached
        while (no >= 0)
        {
            // Check if the current node is a top-level tree node
            if (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* top-level tree-node reached */
            {
                break; // Exit the loop if the top-level tree node is reached
            }
            // Check for fixed point error
            ENDRUN_IF(no == Nodes[no].u.d.father,
                      "ettot at check 3: Fixed point  no == Nodes[no].u.d.father  detected, %d/%d ", no,
                      Nodes[no].u.d.father);
            // Move up to the father node
            no = Nodes[no].u.d.father;
            // Check if the current node is within valid range
            ENDRUN_IF(no < All.MaxPart, "error: at check 4: i=%d, no=%d < All.MaxPart=%d ", i, no, All.MaxPart);
            ENDRUN_IF(no >= maxpart + MaxNodes + 1, "error: at check 5: i=%d, no=%d >= maxpart=%d ", i, no,
                      maxpart + MaxNodes + 1);
        }
    }

    VERBOSE(2, "Node check done ...");
    // Synchronize all MPI processes after completing checks
    MPI_Barrier(MYMPI_COMM_WORLD);
}

/**
 * @brief Check particle data for consistency and validity.
 *
 * This function performs a check on particle data, including
 * velocities, positions, masses, and additional properties for gas, star, and
 * black hole particles. It prints verbose messages for potential issues and
 * terminates the program if critical errors are detected.
 */
void check_particles_info(const char *func, const char *file, int linenr)
{
    int vok = 0, pok = 0, vsph = 0, mok = 0, vstar = 0;
#ifdef LT_MAX_TEMP_FEEDBACK
    double a3inv = 1;
#endif // LT_MAX_TEMP_FEEDBACK
    // double Redshift = 0;
    int *double_check;

    if (All.ComovingIntegrationOn)
    {
#ifdef LT_MAX_TEMP_FEEDBACK
        a3inv = 1 / (All.Time * All.Time * All.Time);
#endif // LT_MAX_TEMP_FEEDBACK
       // Redshift = 1.0 / All.Time - 1;
    }
    // Synchronize all MPI processes before performing checks.
    MPI_Barrier(MYMPI_COMM_WORLD);

    VERBOSE(2, "Checking particle data (function %s in file %s at line %d) ...", func, file, linenr);

    // double_check = (int *) mymalloc("DoubleCount", NumPart * sizeof(int));
    double_check = static_cast<int *>(mymalloc("DoubleCount", NumPart * sizeof(int)));

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : vok, pok, vsph, vstar)
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        double_check[i] = 0;
        // Check velocity components
        for (int k = 0; k < 3; k++)
        {
            if (P[i].Vel[k] > -1e8 && P[i].Vel[k] < 1e8)
            {
                // Calculate velocity magnitude
                MyAtLeastDouble vv = std::sqrt((MyAtLeastDouble)(P[i].Vel[0] * P[i].Vel[0] + P[i].Vel[1] * P[i].Vel[1] +
                                                                 P[i].Vel[2] * P[i].Vel[2]));
                // If velocity magnitude is too large, renormalize it
                if (vv > 15000)
                {
                    VERBOSE_ALL(
                        1, "task=%d: WARNING: Large velocity for particle %d ID %llu v[%d]=%g, renormalizing it !!",
                        ThisTask, i, (unsigned long long)P[i].ID, k, vv);
                    // Renormalize velocity components
                    P[i].Vel[0] = P[i].Vel[0] / vv * 10000;
                    P[i].Vel[1] = P[i].Vel[1] / vv * 10000;
                    P[i].Vel[2] = P[i].Vel[2] / vv * 10000;
                }
                vok++;
            }
            else
            {
                // Handle strange velocity values
                VERBOSE_ALL(
                    0, "task=%d:  strange value in velocity in for particle %d ID %llu , type=%d, mass=%g, v[%d]=%g",
                    ThisTask, i, (unsigned long long)P[i].ID, P[i].Type, (float)P[i].Mass, k, (float)P[i].Vel[k]);
                // Check additional information and terminate the program
                if (P[i].Type == type_gas)
                {
                    VERBOSE_ALL(0, "        vred[%d]=%g  a_hydro[%d]=%g", k, (float)SphP[i].VelPred[k], k,
                                (double)SphP[i].HydroAccel[k]);
                }
                ENDRUN("Strange values in velocity. Stopping here.");
            }

            // Check particle position
            if (P[i].Pos[k] > -10000 && P[i].Pos[k] < All.BoxSize + 10000)
            {
                pok++;
            }
            else
            {
                // Handle strange position values and end the program
                ENDRUN("Task=%d:  strange value in position in for particle %d ID %llu x[%d]=%g", ThisTask, i,
                       (unsigned long long)P[i].ID, k, (double)P[i].Pos[k]);
            }
        }

        // Check mass of dark matter particles
        double massDMpart;
        if (All.MassTable[1] > 0)
        {
            massDMpart = All.MassTable[1];
        }
        else
#ifdef BLACK_HOLES
            massDMpart = All.massDMpart;
#else // BLACK_HOLES
            if (All.MassTable[0] > 0)
            {
                massDMpart = All.MassTable[0] * 10;
            }
            else
            {
                massDMpart = P[0].Mass * 10;
            }
#endif // BLACK_HOLES

        if ((P[i].Mass > massDMpart / 5000 && P[i].Mass < massDMpart * 10) || P[i].Type == 2 || P[i].Type == 3 ||
            P[i].Type == 5)
        {
            mok++;
        }
        else
        {
            // Handle strange mass values
            VERBOSE_ALL(1, "Task=%d:  strange value in Mass in for particle %d ID %llu Type %d mass=%g", ThisTask, i,
                        (unsigned long long)P[i].ID, P[i].Type, (float)P[i].Mass);
            ENDRUN_IF(P[i].Mass > 0,
                      "Task=%d:  strange value in Mass larger than 0 for particle %d ID %llu Type %d mass=%g", ThisTask,
                      i, (unsigned long long)P[i].ID, P[i].Type, (float)P[i].Mass);
        }

        // Check additional properties for gas particles
        if (P[i].Type == type_gas && P[i].Mass > 0)
        {
            // Check density of gas particles
            if (SphP[i].Density >= 0 && SphP[i].Density < 1e5)
            {
                vsph++;
            }
            else
            {
                // Handle strange density values for gas particles
                VERBOSE_ALL(1, "task=%d: Gas Particle id=%llu,m=%e,divV=%e strange Density value: %e", ThisTask,
                            (unsigned long long)P[i].ID, (double)P[i].Mass, (double)SphP[i].DivVel,
                            (double)SphP[i].Density);
            }

            // Check predicted entropy of gas particles
            if ((SphP[i].EntropyPred > -1e20 && SphP[i].EntropyPred < 1e20))
            {
                vsph++;
            }
            else
            {
                // Handle strange predicted entropy values for gas particles
                VERBOSE_ALL(1, "task=%d: Particle id=%llu, m=%e strange predicted Entropy value in hydro: %e", ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Mass, (double)SphP[i].EntropyPred);
            }

            // Check entropy of gas particles
            if ((SphP[i].Entropy > -1e20 && SphP[i].Entropy < 1e20))
            {
                vsph++;
            }
            else
            {
                // Handle strange entropy values for gas particles
                VERBOSE_ALL(1, "task=%d: Particle id=%llu, m=%e strange Entropy value in hydro: %e", ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Mass, (double)SphP[i].Entropy);
            }

            // Check change in entropy of gas particles
            if ((SphP[i].DtEntropy > -1e20 && SphP[i].DtEntropy < 1e20))
            {
                vsph++;
            }
            else
            {
                // Handle strange change in entropy values for gas particles
                VERBOSE_ALL(1, "task=%d: Particle id=%llu, m=%e strange DtEntropy value in hydro: %e", ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Mass, (double)SphP[i].DtEntropy);
            }

            // Check pressure of gas particles
            if ((SphP[i].Pressure > -1e20 && SphP[i].Pressure < 1e20))
            {
                vsph++;
            }
            else
            {
                // Handle strange values for gas pressure
                VERBOSE_ALL(1, "task=%d: Particle id=%llu,m=%e strange Pressure value in hydro: %e,%e", ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Mass, (double)SphP[i].EntropyPred,
                            (float)SphP[i].Pressure);
            }

            // Check hydrodynamical acceleration of gas particles
            if (SphP[i].HydroAccel[0] > -1e20 && SphP[i].HydroAccel[0] < 1e20 && SphP[i].HydroAccel[1] > -1e20 &&
                SphP[i].HydroAccel[1] < 1e20 && SphP[i].HydroAccel[2] > -1e20 && SphP[i].HydroAccel[2] < 1e20)
            {
                vsph++;
            }
            else
            {
                // Handle case of strange acceleration values for gas particles
                VERBOSE_ALL(1, "task=%d: Particle id=%llu,m=%e strange acceleration value in hydro: %e,%e,%e", ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Mass, (double)SphP[i].HydroAccel[0],
                            (double)SphP[i].HydroAccel[1], (double)SphP[i].HydroAccel[2]);
            }
            // Check Temperature
#ifdef LT_MAX_TEMP_FEEDBACK
            // As we want to imply a maximum temperature cut, we can savely assume full ionizateiona
            // and ignore all effects of additional physics

            double Temperature =
                U_TO_TEMP_FAC / GAMMA_MINUS1 * SphP[i].Entropy * pow(SphP[i].Density * a3inv, GAMMA_MINUS1);
            if (Temperature > LT_MAX_TEMP_FEEDBACK)
            {
                VERBOSE_ALL(1, "task=%d: Particle id=%llu,m=%e too large temperature %e>%e !", ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Mass, (double)Temperature,
                            (double)LT_MAX_TEMP_FEEDBACK);
                SphP[i].Entropy = SphP[i].EntropyPred = 0.95 * LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC * GAMMA_MINUS1 /
                                                        pow(SphP[i].Density * a3inv, GAMMA_MINUS1);
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
            }
#endif // LT_MAX_TEMP_FEEDBACK
#ifdef KD_RESTRICT_MAXVEL
            restrict_maxvel(SphP[i].VelPred, i);
#endif // KD_RESTRICT_MAXVEL
        }

        // Check particle types to ensure it is within valid range
        ENDRUN_IF((P[i].Type > 5 || P[i].Type < 0), "task=%d:  P[i=%d].Type=%d", ThisTask, i, P[i].Type);

#ifdef LT_STELLAREVOLUTION
        // Check additional properties for star particles
        if (P[i].Type == type_star)
        {
            // Check cross-indexes for star particles
            ENDRUN_IF(MetP[P[i].pt.MetID].PID != i, "Task=%d:  error in cross-indexes for star-particle %d ID %llu",
                      ThisTask, i, (unsigned long long)P[i].ID);
            // Check weight of star particles
            if (MetP[P[i].pt.MetID].weight >= 0 && MetP[P[i].pt.MetID].weight < 1e3)
            {
                vstar++;
            }
            else
            {
                // Handle strange weight values for star particles
                VERBOSE_ALL(1, "task=%d: Star Particle id=%llu,m=%e strange weight for star: %e", ThisTask,
                            (unsigned long long)P[i].ID, (float)P[i].Mass, (float)MetP[P[i].pt.MetID].weight);
            }
        }
#endif // LT_STELLAREVOLUTION

#ifdef BLACK_HOLES
        // Check additional properties for black hole particles
        if (P[i].Type == type_bh)
        {
            // Check cross-indexes for black hole particles
            ENDRUN_IF(BHP[P[i].pt.BHID].PID != i, "Task=%d:  error in cross-indexes for bh-particle %d ID %llu",
                      ThisTask, i, (unsigned long long)P[i].ID);
        }
#endif // BLACK_HOLES
    }

#ifdef _OPENMP
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP

        double_check[i]++;
    }

    int dcount = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dcount)
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        dcount += (double_check[i] > 1);
    }

    myfree(double_check);

    VERBOSE(2,
            "Positions, Velocities, Masses and  SphVelocities fine for (%d,%d,%d,%d) of %d/%d/%d cases on task 0 (and "
            "%d stars)... ",
            pok, vok, mok, vsph / 6, NumPart * 3, NumPart, N_gas, vstar);

    ENDRUN_IF(dcount > 0, "Task %d: Number of double counts in active particle list %d > 0 !", ThisTask, dcount);

    // Synchronize all MPI processes after completing checks
    MPI_Barrier(MYMPI_COMM_WORLD);
}

/**
 * @brief Allocate memory before each timestep for OpenACC.
 *
 * This function allocates memory structures required for gravitational calculations
 * when using OpenACC. It allocates memory for the WalkLiteP structure and, if stellar evolution is enabled,
 * for the Acc_SFs_PhysDensThresh_table array. Additionally, it copies relevant values from the SFs array
 * for later use.
 */
void on_beginning_of_timestep()
{
// Uncomment the following line if you want to dump memory table if ThisTask == 0.
//  if(ThisTask==0)  dump_memory_table();
#ifdef ACC_RUN

    VERBOSE(2, "Alloc WalkLieP %d", NumPart);
    WalkLiteP = (struct gravity_walkdata *)mymalloc("WalkLiteP", NumPart * sizeof(struct gravity_walkdata));

#if defined(LT_STELLAREVOLUTION) && defined(ACC_GRADIENTS)

    // Allocate memory for the Acc_SFs_PhysDensThresh_table array.
    Acc_SFs_PhysDensThresh_table = (double *)mymalloc("Acc_SFs_PhysDensThresh_table", SFs_dim * ZBins * sizeof(double));

    // Copy the values from SFs array to Acc_SFs_PhysDensThresh_table for later use.
    for (int i = 0; i < SFs_dim; i++)
    {
        for (int j = 0; j < ZBins; j++)
        {
            Acc_SFs_PhysDensThresh_table[i * ZBins + j] = SFs[i].PhysDensThresh[j];
        }
    }
#endif // LT_STELLAREVOLUTION
#endif // ACC_RUN
}

/**
 * @brief Clean and update memory after each timestep, when OpenACC is used.
 *
 * This function is responsible for cleaning up and updating memory structures after each timestep
 * in a simulation using OpenACC. It ensures proper handling of memory allocated during the simulation
 * and releases resources. The function checks if it is inside the data region for the accelerator run,
 * performs the necessary updates, and frees allocated memory. Additionally, specific memory related to
 * stellar evolution (if enabled) is freed. This function is conditionally compiled based on the ACC_RUN flag.
 */
void on_end_of_timestep()
{
#ifdef ACC_RUN
    // Check if inside the data region for accelerator run
    if (acc_all.inside_data_region)
    {
        update_openacc_structures_exit();
    }

// Free memory allocated
#if defined(LT_STELLAREVOLUTION) && defined(ACC_GRADIENTS)
    myfree(Acc_SFs_PhysDensThresh_table);
#endif // LT_STELLAREVOLUTION
    VERBOSE(2, "Free WalkLieP %d", NumPart);

    // Nitya - don't delete it here if running OMP offload. We will check in the next timestep if the size has increased
    // to then delete the variable or not
    myfree(WalkLiteP);
#endif // ACC_RUN
}
