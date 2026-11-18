/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the initial set-up of a simulation run.
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
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <libgen.h> /* libraries to easily build path of included files */
#include <string.h>

#include "allvars.hpp"
#include "proto.hpp"

#include "../IO/io_matrix.hpp"

#ifdef BLACK_HOLES
#include "../Blackholes/blackhole_begrun.hpp"
#endif // BLACK_HOLES

#ifdef ACC
#ifdef ACC_OPENACC
#include <openacc.h>
#elif defined(ACC_OPENMP)
#include <omp.h>
#endif // ACC_OPENACC
#endif // ACC
#define MAX_PATH_LENGTH 300 // maximum path length of a included parameter file
#define MEMORY_UNDERUSE_WARN_FRACTION 0.9 // below this fraction of node memory used, hint at more efficient settings

// Defined below, after the Param_Type/Param_Restart enums; forward-declared here
void change_simple_parameters(struct global_data_all_processes &all,
                              std::multimap<std::string, ParamInfo> &parameter_pointers_map);

/**
 * @brief Copy selected parameters from saved 'all' to new 'All' struct.
 *
 * This function copies the saved parameters from the parameter file back to the main struct.
 *
 * @param all Reference to the struct containing saved parameters.
 */
void change_array_like_params_to_All(struct global_data_all_processes &all)
{
    // Fields below aren't representable as a single setup_parameter()-registered scalar
    // (they're arrays, or derived rather than read directly from the parameter file), so they
    // can't go through the generic reconcile_restart_parameters() pass and stay hand-copied here.
    All.OutputListLength = all.OutputListLength;
    memcpy(All.OutputListTimes, all.OutputListTimes, sizeof(double) * All.OutputListLength);
    memcpy(All.OutputListFlag, all.OutputListFlag, sizeof(char) * All.OutputListLength);

#ifdef LT_STELLAREVOLUTION
    memcpy(All.MetalsCheckSum, all.MetalsCheckSum, sizeof(long double) * LT_NMetP);
#endif // LT_STELLAREVOLUTION

}

void change_restart_parameters(struct global_data_all_processes &all,
                               std::multimap<std::string, ParamInfo> &parameter_pointers_map)
{
    // Re-apply every parameter file value that's allowed to change on restart (and error out
    // if one that isn't was edited), then copy the remaining hand-managed state (arrays and
    // BH seed fractions) that can't go through the generic mechanism.
    change_simple_parameters(all, parameter_pointers_map);
    change_array_like_params_to_All(all);
}

/**
 * @brief Initialize different modules.
 *
 * This function is called in begrun and initializes various modules depending on the compilation
 * options. It initializes modules related to dark energy, cooling, chemistry, star formation,
 * periodic boundary conditions, artificial viscosity, random number generation, and other
 * components as required by the simulation.
 */
void init_modules()
{
#if defined(DARKENERGY) && defined(TIMEDEPDE)
    fwa_init();
#endif // defined(DARKENERGY) && defined(TIMEDEPDE)

#ifdef GROWING_DISK_POTENTIAL
    growing_disk_init();
#endif // GROWING_DISK_POTENTIAL

#ifdef COOLING
    All.Time = All.TimeBegin;
    set_cosmo_factors_for_current_time();
#ifndef LT_STELLAREVOLUTION
    InitCool();
#endif // LT_STELLAREVOLUTION
#endif // COOLING

#ifdef UM_CHEMISTRY
    VERBOSE_ALL(2, "Initialize UVB rates to be interpolated, Task %d... ", ThisTask);
    gJH0uvb = gJHe0uvb = gJHepuvb = 0;
    epsH0uvb = epsHe0uvb = epsHepuvb = 0;
    HIshielding = 1;
#endif // UM_CHEMISTRY

#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    VERBOSE_ALL(0, "Initialize chemistry.. ");
    InitChem();
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)

#if defined(SFR) && !defined(LT_STELLAREVOLUTION) && !defined(GM_MUPPI) && !defined(EB_SFR_MAGNETIC)
    init_clouds();
#endif // defined(SFR) && !defined(LT_STELLAREVOLUTION) && !defined(GM_MUPPI) && !defined(EB_SFR_MAGNETIC)

#ifdef PERIODIC
    ewald_init();
#endif // PERIODIC

#ifdef PERIODIC
    boxSize = All.BoxSize;
    boxHalf = 0.5 * All.BoxSize;
    inverse_boxSize = 1. / boxSize;
#ifdef LONG_X
    boxHalf_X = boxHalf * LONG_X;
    boxSize_X = boxSize * LONG_X;
    inverse_boxSize_X = 1. / boxSize_X;
#endif // LONG_X
#ifdef LONG_Y
    boxHalf_Y = boxHalf * LONG_Y;
    boxSize_Y = boxSize * LONG_Y;
    inverse_boxSize_Y = 1. / boxSize_Y;
#endif // LONG_Y
#ifdef LONG_Z
    boxHalf_Z = boxHalf * LONG_Z;
    boxSize_Z = boxSize * LONG_Z;
    inverse_boxSize_Z = 1. / boxSize_Z;
#endif // LONG_Z
#endif // PERIODIC

    random_generator.seed(42);
    set_random_numbers();

#ifdef PMGRID
#if !defined(ADAPTGRAVSOFT) && !defined(SUBFIND_RESHUFFLE_AND_POTENTIAL)
    if (RestartFlag != 3 && RestartFlag != 4)
#endif // !ADAPTGRAVSOFT  && !SUBFIND_RESHUFFLE_AND_POTENTIAL
        long_range_init();
#endif // PMGRID

#ifdef SUBFIND
    GrNr = -1;
#endif // SUBFIND
}

#ifdef ACC
#ifdef ACC_OPENACC
/**
 * @brief Initialize OpenACC for GPU acceleration.
 *
 * This function initializes OpenACC for GPU acceleration. It checks the number of available GPUs
 * and selects a GPU based on the MPI rank. Each MPI rank is assigned one GPU, assuming that the
 * number of GPUs per node is greater than or equal to the number of MPI ranks per node.
 *
 * If GPUs are available, the function sets the device number, initializes OpenACC, and prints
 * diagnostic messages indicating the GPU selection. If no GPUs are found, it prints a message
 * indicating that no GPU was found.
 */
void init_ACC()
{
    // basic GPU selection - assumes (number GPUs / node >= MPI ranks / node )
    const int num_devices = acc_get_num_devices(acc_device_nvidia);

    if (num_devices > 0)
    {
        int mydevice = ThisTask % num_devices; // 1 device per MPI rank, less MPI_ranks then GPUs / node
        acc_set_device_num(mydevice, acc_device_nvidia);
        acc_init(acc_device_nvidia);
        VERBOSE_ALL(2, "OpenACC init: MPI Rank %d/%d using GPU %d/%d ", ThisTask, NTask, mydevice, num_devices);
#pragma omp parallel
        {
            acc_set_device_num(mydevice, acc_device_nvidia);
            acc_init(acc_device_nvidia);
        }
    }
    else
    {
        VERBOSE_ALL(2, "OpenACC init: MPI Rank %d/%d No GPU found. ", ThisTask, NTask);
    }
    VERBOSE(0, "\nUsing %d GPU(s) per MPI rank via OpenACC", num_devices);
}
#endif // ACC_OPENACC
// Nitya - check for OMP offload
#ifdef ACC_OPENMP
/**
 * @brief Initialize OpenMP for GPU acceleration.
 *
 * This function initializes OpenMP for GPU acceleration. It checks the number of available GPUs
 * and selects a GPU based on the MPI rank. Each MPI rank is assigned one GPU, assuming that the
 * number of GPUs per node is greater than or equal to the number of MPI ranks per node.
 *
 * If GPUs are available, the function sets the device number, initializes OpenMP, and prints
 * diagnostic messages indicating the GPU selection. If no GPUs are found, it prints a message
 * indicating that no GPU was found.
 */
void init_ACC()
{
    // basic GPU selection - assumes (number GPUs / node >= MPI ranks / node )

    const int num_devices = omp_get_num_devices();
    if (num_devices > 0)
    {
        int mydevice = ThisTask % num_devices; // 1 device per MPI rank, less MPI_ranks then GPUs / node

        // Nitya - adding a dummy target call to get the compilation done
#pragma omp target device(omp_get_default_device())

        printf("OpenMP init: MPI Rank %d/%d using GPU %d/%d\n", ThisTask, NTask, mydevice, num_devices);
    }
    else
    {
        VERBOSE_ALL(2, "OpenMP init:  MPI Rank %d/%d No GPU found. ", ThisTask, NTask);
    }
    VERBOSE(0, "\nUsing %d GPU(s) per MPI rank via OpenMP", num_devices);
}
#endif // ACC_OPENMP

#ifdef ACC_NONE
void init_ACC()
{
    VERBOSE(0, "\nEmulating the usage of one GPU per MPI rank on CPU");
}
#endif // ACC_NONE
#endif // ACC

/**
 * @brief Print initialization information.
 *
 * This function prints information about the initialisation of the simulation.
 * It prints the current version of the code, the number of MPI tasks, and the number of OpenMP threads
 * as well as information about the data structures.
 */
void print_initialisation_info()
{
    /* #ifdef _OPENMP
    int tid;
    #endif // _OPENMP */

    int flag = 0;
    int *tag_ub_p = NULL;
    MPI_Comm_get_attr(MYMPI_COMM_WORLD, MPI_TAG_UB, &tag_ub_p, &flag);
    MYMPI_MAX_TAG_UB = *tag_ub_p;

    if (ThisTask == 0)
    {
        VERBOSE(0, "\nThis is OpenGadget3, Version %s, SHA %s.", GADGETVERSION, GIT_SHA);
        VERBOSE(0, "\nRunning on %d MPI tasks. ", NTask);
        VERBOSE(2, "   (Max number for TAGS is %d !)\n", MYMPI_MAX_TAG_UB);
#ifdef _OPENMP
#pragma omp parallel // private(tid)
        {
#pragma omp master
            {
                VERBOSE(0, "Using %d OpenMP threads", omp_get_num_threads());
                VERBOSE(0, "Using %d omp threads for fftw", omp_get_num_threads());
            }
        }
#else // _OPENMP
        VERBOSE(0, "OpenMP is switched off");
#endif // _OPENMP

        VERBOSE(0, "\nCode was compiled with settings:");
        output_compile_time_options();

        VERBOSE(0, "\nCode was compiled with additional (default) settings:");
        VERBOSE(0, "   EXPERT_LEVEL=%d", EXPERT_LEVEL);
        VERBOSE(0, "   VERBOSE_LEVEL=%d", VERBOSE_LEVEL);
        VERBOSE(0, "   TIMER_LEVEL=%d", Timer_Level);
        VERBOSE(0, "   WAKEUP=%d", WAKEUP);
        VERBOSE(0, "   AVERAGE_ART_VISC=%g (0:max, 1:min, 2:arit.mean, 3:geom.mean, 4:harm.mean)", Average_Art_Visc);
#ifdef TIME_DEP_ART_VISC
        VERBOSE(0, "   HIGH_ART_VISC_START=%g", High_Art_Visc_Start);
#ifdef DIVB_TO_ALPHA
        VERBOSE(0, "   DIVB_TO_ALPHA=%g", (double)DIVB_TO_ALPHA);
#endif // DIVB_TO_ALPHA
#endif // TIME_DEP_ART_VISC
#ifdef TIME_DEP_MAGN_DISP
        VERBOSE(0, "   HIGH_MAGN_DISP_START=%g", High_Magn_Disp_Start);
#endif // TIME_DEP_MAGN_DISP
#ifdef PMGRID
        VERBOSE(0, "   ASMTH=%g", ASMTH);
        VERBOSE(0, "   RCUT=%g", RCUT);
#endif // PMGRID
        VERBOSE(0, "   OMEGAR=%g", OMEGAR);

#ifndef DOUBLEPRECISION
        VERBOSE(0, "\nCode was compile with no extra precission settings");
#else // DOUBLEPRECISION
        VERBOSE(0, "\nCode was compiled with DOUBLEPRECISION set to %d", DOUBLEPRECISION);
#endif // DOUBLEPRECISION
        VERBOSE(0, "   Size MyFloat is                  %d  [bytes] ", (int)sizeof(MyFloat));
        VERBOSE(0, "   Size MyDouble is                 %d  [bytes] ", (int)sizeof(MyDouble));
        VERBOSE(0, "   Size MyIDType is                 %d  [bytes] ", (int)sizeof(MyIDType));
        VERBOSE(0, "   Size MyLongDouble is             %d  [bytes] ", (int)sizeof(MyLongDouble));
        VERBOSE(0, "   Size MyAtLeastDouble is          %d  [bytes] ", (int)sizeof(MyAtLeastDouble));
        VERBOSE(0, "   Size of All structure            %d  [bytes] ", (int)sizeof(global_data_all_processes));
        VERBOSE(0, "   Size of particle structure       %d  [bytes] ", (int)sizeof(struct particle_data));
        VERBOSE(0, "   Size of sph particle structure   %d  [bytes] ", (int)sizeof(struct sph_particle_data));
#ifdef LT_STELLAREVOLUTION
        VERBOSE(0, "   Size of metal particle structure %d  [bytes]  ", (int)sizeof(struct met_particle_data));
#endif // LT_STELLAREVOLUTION
#ifdef BLACK_HOLES
        VERBOSE(0, "   Size of BH particle structure    %d  [bytes]  ", (int)sizeof(struct bh_particle_data));
#endif // BLACK_HOLES
    }
}

/**
 * @brief Check the configured per-task memory against the memory physically available on the node.
 *
 * This function estimates how many MPI tasks share the node that Task 0 runs on
 * and compares their combined `MaxMemSize` memory pools against the
 * node's physical RAM. It prints a warning if the configuration would request more memory than the node
 * provides, and a hint if it uses substantially less than what is available, so that the number of MPI
 * tasks and OpenMP threads per node can be tuned. As only Task 0's node is inspected, the result is only
 * representative for jobs that use the same task/thread layout on every node.
 */
void check_memory_configuration()
{
    int ranks_per_node = 1;

    MPI_Comm shmcomm;
    MPI_Comm_split_type(MYMPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ThisTask, MPI_INFO_NULL, &shmcomm);
    MPI_Comm_size(shmcomm, &ranks_per_node);
    MPI_Comm_free(&shmcomm);

    if (ThisTask != 0)
    {
        return;
    }

#ifdef _OPENMP
    int threads_per_rank = omp_get_max_threads();
#else // _OPENMP
    int threads_per_rank = 1;
#endif // _OPENMP

    // get pages and number of pages of node for memory
    long phys_pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    // get number of cores
    long cores_per_node = sysconf(_SC_NPROCESSORS_ONLN);

    if (phys_pages < 0 || page_size < 0)
    {
        VERBOSE(1, "\nCould not determine the physical memory of the node: skipping memory configuration check.");
        return;
    }

    double node_mem_MB = (double)phys_pages * (double)page_size / (1024.0 * 1024.0);
    double used_mem_MB = (double)ranks_per_node * (double)All.MaxMemSize;
    long cores_used = (long)ranks_per_node * (long)threads_per_rank;

    VERBOSE(0,
            "\n=== Memory Information ==\nMemory configuration (based on Task 0's node): %d MPI task(s) x %d OpenMP "
            "thread(s) = %ld core(s) "
            "used (node has %ld core(s)).",
            ranks_per_node, threads_per_rank, cores_used, cores_per_node);
    VERBOSE(0, "MaxMemSize=%d MB per task -> %.0f MB used on this node out of %.0f MB physical memory (%.1f%%).",
            All.MaxMemSize, used_mem_MB, node_mem_MB, 100.0 * used_mem_MB / node_mem_MB);

    if (used_mem_MB > node_mem_MB)
    {
        VERBOSE(0,
                "WARNING: The requested memory (MaxMemSize=%d MB x %d MPI task(s) per node = %.0f MB) exceeds "
                "the %.0f MB of physical memory available on this node.\nReduce MaxMemSize, or the number of MPI "
                "tasks per node, to avoid running out of memory.",
                All.MaxMemSize, ranks_per_node, used_mem_MB, node_mem_MB);
    }
    else if (used_mem_MB < MEMORY_UNDERUSE_WARN_FRACTION * node_mem_MB)
    {
        VERBOSE(0,
                "NOTE: Only %.1f%% of the physical memory on this node is being used (MaxMemSize=%d MB x %d MPI "
                "task(s) per node).\nDepending on the node architecture (part of the memory may be reserved for "
                "the system or shared with other jobs!), it might be possible to use the node more efficiently,\n"
                "e.g. by increasing MaxMemSize, or the number of MPI tasks / OpenMP threads per node (%ld "
                "core(s) are avaliable, %ld are currently used).",
                100.0 * used_mem_MB / node_mem_MB, All.MaxMemSize, ranks_per_node, cores_per_node, cores_used);
    }
}

/**
 * @brief Set up the simulation.
 *
 * This function performs the initial set-up of the simulation. First, the
 * parameterfile is set, then routines for setting units, reading
 * ICs/restart-files are called, auxialiary memory is allocated, etc.
 */
void begrun(void)
{
    std::multimap<std::string, ParamInfo> parameter_pointers_map;

#ifdef ACC
    init_ACC();
#endif // ACC

    TIMER_PUSH(0, "Initialisation");

    // print initialisation information
    print_initialisation_info();

#if defined(X86FIX) && defined(SOFTDOUBLEDOUBLE)
    x86_fix(); /* disable 80bit treatment of internal FPU registers in favour of proper IEEE 64bit double precision
                  arithmetic */
#endif // defined(X86FIX) && defined(SOFTDOUBLEDOUBLE)

    // read in parameters for this run, parameter 1 implies to set default values.
    read_parameter_file(ParameterFile, NULL, NULL, NULL, 0, NULL, parameter_pointers_map);

    // compare the requested MaxMemSize/MPI/OpenMP configuration against the node's physical memory
    check_memory_configuration();

    mymalloc_init();

#ifdef DISABLE_MEMORY_MANAGER
    FreeBytes = All.MaxMemSize * ((size_t)1024 * 1024) / 3;
#endif // DISABLE_MEMORY_MANAGER

#ifdef DEBUG
    write_pid_file();
    enable_core_dumps_and_fpu_exceptions();
#endif // DEBUG

    // Computes conversion between internal code units and the cgs-system
    set_units();

    // intialize different modules
    init_modules();

    All.TimeLastRestartFile = CPUThisRun;

    VERBOSE(2, "Opening log files ... ");
    open_outputfiles(); // Opening output files


    // check if simulation starts new or from restart file
    if (RestartFlag == -1 || RestartFlag == 0 || RestartFlag == 2 || RestartFlag == 3 || RestartFlag == 4 ||
        RestartFlag == 5 || RestartFlag == 6)
    {
        init(); /* ... read in initial model */
    }
    else
    {
        // get initial TimeMax from restart file
        double savedTimeMax = All.TimeMax;

        // scope of all is limited to restart
        struct global_data_all_processes all;
        all = All; /* save global variables. (will be read from restart file) */

        // initialise io fields table
        io_matrix::init_io_fields(
        );

        restart(RestartFlag); /* ... read restart file. Note: This also resets
                     all variables in the struct `All'.
                     However, during the run, some variables in the parameter
                     file are allowed to be changed, if desired. These need to
                     copied in the way below.
                     Note:  All.PartAllocFactor is treated in restart() separately.
                   */
        TIMER_PUSH(0, "RESTART");
        set_random_numbers();

        // change/update parameters based on parameter file
        change_restart_parameters(all, parameter_pointers_map);

        // initialize Units struct after setting the time, as it is not contained in the All struct.
        set_cosmo_factors_for_current_time();

        // When restarting and the new max time is different to the old, we need to reset the timebins
        if (All.TimeMax != savedTimeMax)
        {
            readjust_timebase(All.TimeMax, savedTimeMax);
        }
    }

    // Write parameters and values to console and log file
    print_current_parameters(parameter_pointers_map);

#ifdef LT_STELLAREVOLUTION
    if (All.TestSuite && (ThisTask == 0))
    {
        VERBOSE_ALL(2, ">>>>>>>>>>>>>>>>>>>>>\n"
                       "  testing SE\n"
                       ">>>>>>>>>>>>>>>>>>>>>");
        TestStellarEvolution();
    }

    // if All.MetalsCheckSum was not read from restart file
    // we initialize it here
    if (RestartFlag != 1)
    {
        get_metals_checksum(1, All.MetalsCheckSum);
    }
#endif // LT_STELLAREVOLUTION

    char contfname[1000];
    sprintf(contfname, "%scont", All.OutputDir);
    unlink(contfname);

#ifdef PMGRID
    long_range_init_regionsize();

#ifdef ADAPTGRAVSOFT
    /* Setting the maximum\minimum allowed gravitational softening length */
    All.AGS_MaxSoft[0] = AGS_MAXSOFT * All.Asmth[0];
    All.AGS_MinSoft[0] = AGS_MINSOFT * All.Asmth[0];
    VERBOSE(2, "*ADAPTGRAVSOFT* MinSoft, MaxSoft= %g %g ", All.AGS_MinSoft[0], All.AGS_MaxSoft[0]);
#ifdef PLACEHIGHRESREGION
    All.AGS_MaxSoft[1] = AGS_MAXSOFT * All.Asmth[1];
    All.AGS_MinSoft[1] = AGS_MINSOFT * All.Asmth[1];
    VERBOSE(2, "*ADAPTGRAVSOFT* for Highresolution Region: MinSoft, MaxSoft= %g %g ", All.AGS_MinSoft[1],
            All.AGS_MaxSoft[1]);
    VERBOSE(2, "ASmth[0] = %.2f ASmth[1] = %.2f RCut[0] = %.2f RCut[1] = %.2f ", All.Asmth[0], All.Asmth[1],
            All.Rcut[0], All.Rcut[1]);
#endif // PLACEHIGHRESREGION
#endif // ADAPTGRAVSOFT
#endif // PMGRID

    // Reconstruct time bins and update particle information
    reconstruct_timebins();

#ifdef TWODIMS
    for (int i = 0; i < NumPart; i++)
    {
        P[i].Pos[2] = 0;
#ifndef MAGNETIC
        P[i].Vel[2] = 0;
#endif // MAGNETIC

        P[i].GravAccel[2] = 0;

#ifndef MAGNETIC
        if (P[i].Type == 0)
        {
            SphP[i].VelPred[2] = 0;
            SphP[i].HydroAccel[2] = 0;
        }
#endif // MAGNETIC
    }
#endif // TWODIMS

    if (All.ComovingIntegrationOn)
    {
        init_drift_table();
    }

    if (RestartFlag == 2)
    {
        All.Ti_nextoutput = find_next_outputtime(All.Ti_Current + 100);
    }
    else if (RestartFlag == 1)
    {
        All.Ti_nextoutput = find_next_outputtime(All.Ti_Current + 1);
    }
    else
    {
        All.Ti_nextoutput = find_next_outputtime(All.Ti_Current);
    }

#ifdef OUTPUT_LONGRANGE_POTENTIAL
    All.Ti_nextpotdump = All.Ti_nextoutput;
    All.dumped_after_snap = 0; /* need to do first potential dump */
#endif // OUTPUT_LONGRANGE_POTENTIAL

    All.TimeLastRestartFile = CPUThisRun;
    All.StartTiStep = All.NumCurrentTiStep;

#ifdef mSIDM
    msidm.init();
#endif // mSIDM


#ifdef mSIDM_EVAPORATION
    analytic_potentials.init();
#endif // mSIDM_EVAPORATION

    // Close correct timer, depending if restart or not
    if (RestartFlag == -1 || RestartFlag == 0 || RestartFlag == 2 || RestartFlag == 3 || RestartFlag == 4 ||
        RestartFlag == 5 || RestartFlag == 6)
    {
        TIMER_POP(0, "Initialisation");
    }
    else
    {
        TIMER_POP(0, "RESTART");
    }
}

/**
 * @brief Computes conversion factors between internal code units and the cgs-system.
 *
 * This function calculates various conversion factors between internal code units used in the simulation
 * and the cgs (centimeter-gram-second) system, which is commonly used in physics.
 * It also performs unit conversions for physical input parameters to internal units.
 *
 * This function initializes several global variables related to unit conversions and physical parameters.
 */
void set_units(void)
{

    if (All.ComovingIntegrationOn)
    {
        units = new UnitsComoving{};
    }
    else
    {
        units = new UnitsNoncomoving{};
    }
    // future default behaviour with Units struct
    units->init_cgs_factors(All.UnitLength_in_cm, All.UnitVelocity_in_cm_per_s, All.UnitMass_in_g);
    units->init_h0_factors(All.HubbleParam);

#ifdef CONDUCTION
#ifndef CONDUCTION_CONSTANT
    double coulomb_log;
#endif // CONDUCTION_CONSTANT
#endif // CONDUCTION

#ifdef STATICNFW
    double Mtot;
#endif // STATICNFW
    // Calculate unit of time in seconds and Megayears
    All.UnitTime_in_s = All.UnitLength_in_cm / All.UnitVelocity_in_cm_per_s;
    All.UnitTime_in_Megayears = All.UnitTime_in_s / SEC_PER_MEGAYEAR;
    // Calculate gravitational constant G
    if (All.GravityConstantInternal == 0)
    {
        All.GravityConstantInternal =
            GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);
    }

#ifdef TIMEDEPGRAV
    All.Gini = All.GravityConstantInternal;
    All.GravityConstantInternal = All.Gini * dGfak(All.TimeBegin);
#endif // TIMEDEPGRAV

    // Calculate unit of density, pressure, cooling rate, and energy in cgs units
    All.UnitDensity_in_cgs = All.UnitMass_in_g / pow(All.UnitLength_in_cm, 3);
    All.UnitPressure_in_cgs = All.UnitMass_in_g / All.UnitLength_in_cm / pow(All.UnitTime_in_s, 2);
    All.UnitCoolingRate_in_cgs = All.UnitPressure_in_cgs / All.UnitTime_in_s;
    All.UnitEnergy_in_cgs = All.UnitMass_in_g * pow(All.UnitLength_in_cm, 2) / pow(All.UnitTime_in_s, 2);
#ifdef MAGNETIC
#ifndef MU0_UNITY
    All.UnitMagneticField_in_gauss =
        1 / sqrt(All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g);
#else // MU0_UNITY
    All.UnitMagneticField_in_gauss = 1;
#endif // MU0_UNITY
#endif // MAGNETIC
    /* convert some physical input parameters to internal units */
    // Calculate Hubble parameter
    All.Hubble = HUBBLE * All.UnitTime_in_s;

    VERBOSE(0, "\nCode Units:");
    VERBOSE(0, "Hubble (internal units) = %g ", All.Hubble);
    VERBOSE(0, "G (internal units) = %g ", All.GravityConstantInternal);
    VERBOSE(0, "UnitMass_in_g = %g  ", All.UnitMass_in_g);
    VERBOSE(0, "UnitTime_in_s = %g  ", All.UnitTime_in_s);
    VERBOSE(0, "UnitLength_in_cm = %g ", All.UnitLength_in_cm);
    VERBOSE(0, "UnitVelocity_in_cm_per_s = %g  ", All.UnitVelocity_in_cm_per_s);
    VERBOSE(0, "UnitDensity_in_cgs = %g  ", All.UnitDensity_in_cgs);
    VERBOSE(0, "UnitEnergy_in_cgs = %g  \n", All.UnitEnergy_in_cgs);
#ifdef MAGNETIC
    VERBOSE(0, "UnitMagneticField_in_gauss = %g  ", All.UnitMagneticField_in_gauss);
#endif
#ifdef INCLUDE_RADIATION
    VERBOSE(0, "Omega_R = %g ", OMEGAR);
#endif // INCLUDE_RADIATION
    // Calculate mean molecular weight for neutral gas
    double meanweight = 4.0 / (1 + 3 * HYDROGEN_MASSFRAC); /* note: assuming NEUTRAL GAS */

    All.MinEgySpec = 1 / meanweight * (1.0 / GAMMA_MINUS1) * (BOLTZMANN / PROTONMASS) * All.MinGasTemp;
    All.MinEgySpec *= All.UnitMass_in_g / All.UnitEnergy_in_cgs;

#ifdef LT_STELLAREVOLUTION
    // Convert chemical time step to internal units and initialize supernova module
    All.MinChemTimeStep /= 86400 * 365 * 1e9;
    /*
       Add here Spreading Length in the case of fixed distance
     */
    All.SnIaEnergy /= All.UnitEnergy_in_cgs;
    All.SnIIEnergy /= All.UnitEnergy_in_cgs;

#ifdef LT_HOT_EJECTA
    /* All.EgySpecEjecta is supposed to be given in paramfile as Km/sec        */
    /* then we have to transform in erg / g through the 1e10 conversion factor */
    All.EgySpecEjecta = pow(All.EgySpecEjecta, 2) * 1e10 * All.UnitMass_in_g / All.UnitEnergy_in_cgs;
#endif // LT_HOT_EJECTA

    Hyd = LT_NMet - 1;
    All.Time = All.TimeBegin;
    set_cosmo_factors_for_current_time();
    init_SN();
#endif // LT_STELLAREVOLUTION

#if defined(SFR) && ((!defined(LT_STELLAREVOLUTION)) || defined(GM_MUPPI))
    set_units_sfr();
#endif //  defined(SFR) && ( (!defined(LT_STELLAREVOLUTION)) ||  defined(GM_MUPPI) )
// Define conversion factors for commonly used units in the code
#define cm (All.HubbleParam / All.UnitLength_in_cm)
#define g (All.HubbleParam / All.UnitMass_in_g)
#define s (All.HubbleParam / All.UnitTime_in_s)
#define erg (g * cm * cm / (s * s))
#define keV (1.602e-9 * erg)
#define deg 1.0
#define m_p (PROTONMASS * g)
#define k_B (BOLTZMANN * erg / deg)

    U_TO_TEMP_FAC_WITHOUT_GAMMA =
        (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) * PROTONMASS / BOLTZMANN * All.UnitEnergy_in_cgs / All.UnitMass_in_g;
    U_TO_TEMP_FAC = U_TO_TEMP_FAC_WITHOUT_GAMMA * GAMMA_MINUS1;

#ifdef NAVIERSTOKES
    /* Braginskii-Spitzer shear viscosity parametrization */
    /* mu = 0.406 * m_p^0.5 * (k_b* T)^(5/2) / e^4 / logLambda  [g/cm/s] */
    /* eta = frac * mu */

    meanweight = 4.0 / (8 - 5 * (1 - HYDROGEN_MASSFRAC)); /* assuming full ionization */

#ifdef NAVIERSTOKES_CONSTANT
    All.NavierStokes_ShearViscosity = All.FractionSpitzerViscosity * 0.406 * pow(PROTONMASS, 0.5) *
                                      pow(BOLTZMANN * All.ShearViscosityTemperature, 5. / 2.) / pow(ELECTRONCHARGE, 4) /
                                      LOG_LAMBDA; /*in cgs units */

    VERBOSE(2, "Constant shear viscosity in cgs units: eta = %g ", All.NavierStokes_ShearViscosity);
    All.NavierStokes_ShearViscosity *=
        All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g / All.HubbleParam; /* in internal code units */

    VERBOSE(2, "Constant shear viscosity in internal code units: eta = %g ", All.NavierStokes_ShearViscosity);

#else // NAVIERSTOKES_CONSTANT
    All.NavierStokes_ShearViscosity =
        0.406 * pow(PROTONMASS, 0.5) * pow(BOLTZMANN, 2.5) / pow(ELECTRONCHARGE, 4.) / LOG_LAMBDA; /* in cgs units */

    /* T = S*(rho/a³)^(gamma-1) * mu*m_p/k_b * UnitEnergy/UnitMass */
    /* We multiply by the temperature in compute_gradients.cpp */

    All.NavierStokes_ShearViscosity *= All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g / All.HubbleParam;
    /* in internal code units */

    All.NavierStokes_ShearViscosity *= All.FractionSpitzerViscosity;

#endif // NAVIERSTOKES_CONSTANT


#ifdef NAVIERSTOKES_VISCOSITY_SATURATION
    /* calculate ion mean free path assuming complete ionization:
       ion mean free path for hydrogen is similar to that of helium,
       thus we calculate only for hydrogen */
    /* l_i = 3^(3/2)*(k*T)^2 / (4*\pi^(1/2)*ni*(Z*e)^4*lnL) */

    All.IonMeanFreePath =
        pow(3.0, 1.5) * pow(BOLTZMANN, 2.0) / (4.0 * sqrt(M_PI) * pow(ELECTRONCHARGE, 4) * LOG_LAMBDA);

    /* T = S*(rho/a³)^(gamma-1) * mu*m_p/k_b * UnitEnergy/UnitMass */
    /* We multiply by the temperature in hydra.c */

    All.IonMeanFreePath /=
        (HYDROGEN_MASSFRAC / PROTONMASS) * (All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam);

    All.IonMeanFreePath *= All.HubbleParam / All.UnitLength_in_cm;
    /* in internal code units */
#endif // NAVIERSTOKES_VISCOSITY_SATURATION

#endif // NAVIERSTOKES

#ifdef CONDUCTION
#ifndef CONDUCTION_CONSTANT
    meanweight = m_p * 4.0 / (8 - 5 * (1 - HYDROGEN_MASSFRAC));
    /* assuming full ionization */
    coulomb_log = 37.8;
    /* according to Sarazin's book */
    All.ConductionEfficiency *=
        (1.84e-5 / coulomb_log * pow(meanweight / k_B * GAMMA_MINUS1, 2.5) * erg / (s * deg * cm));
    /* Kappa_Spitzer definition taken from Zakamska & Narayan 2003
     * ( ApJ 582:162-169, Eq. (5) )
     */

    /* Note: Because we replace \nabla(T) in the conduction equation with
     * \nable(u), our conduction coefficient is not the usual kappa, but
     * rather kappa*(gamma-1)*mu/kB. We therefore need to multiply with
     * another factor of (meanweight / k_B * GAMMA_MINUS1).
     */
    All.ConductionEfficiency *= meanweight / k_B * GAMMA_MINUS1;

    /* The conversion of ConductionEfficiency between internal units and cgs
     * units involves one factor of 'h'. We take care of this here.
     */
    All.ConductionEfficiency /= All.HubbleParam;

#ifdef CONDUCTION_SATURATION
    All.ElectronFreePathFactor = 8 * pow(3.0, 1.5) * pow(GAMMA_MINUS1, 2) / pow(3 + 5 * HYDROGEN_MASSFRAC, 2) /
                                 (1 + HYDROGEN_MASSFRAC) / sqrt(M_PI) / coulomb_log * pow(PROTONMASS, 3) /
                                 pow(ELECTRONCHARGE, 4) / (All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam) *
                                 pow(All.UnitPressure_in_cgs / All.UnitDensity_in_cgs, 2);

    /* If the above value is multiplied with u^2/rho in code units (with rho being the physical density), then
     * one gets the electrong mean free path in centimeter. Since we want to compare this with another length
     * scale in code units, we now add an additional factor to convert back to code units.
     */
    All.ElectronFreePathFactor *= All.HubbleParam / All.UnitLength_in_cm;
#endif // CONDUCTION_SATURATION
#endif // CONDUCTION_CONSTANT

#ifdef CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR
    // constants from omega_g * tau
    All.ConductionPerpendicularConstants =
        pow(meanweight * GAMMA_MINUS1, 1.5) * pow(ELECTRONMASS * All.HubbleParam / All.UnitMass_in_g, 0.5);
    All.ConductionPerpendicularConstants /=
        16.0 * M_PI * M_PI *
        pow(ELECTRONCHARGE * All.HubbleParam / pow(All.UnitEnergy_in_cgs * All.UnitLength_in_cm, 0.5), 3.0) *
        2.9979e10 / All.UnitVelocity_in_cm_per_s; // c
#endif // CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR
#endif // CONDUCTION


#ifdef STATICNFW
    R200 = pow(Nfw_M200 * All.GravityConstantInternal / (100 * All.Hubble * All.Hubble), 1.0 / 3);
    Rs = R200 / Nfw_C;
    Dc = 200.0 / 3 * Nfw_C * Nfw_C * Nfw_C / (log(1 + Nfw_C) - Nfw_C / (1 + Nfw_C));
    RhoCrit = 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal);
    V200 = 10 * All.Hubble * R200;
    VERBOSE(2, "V200= %g ", V200);

    fac = 1.0;
    Mtot = enclosed_mass(R200);
    VERBOSE(2, "M200= %g ", Mtot);

    /* fac = M200 / Mtot */
    fac = V200 * V200 * V200 / (10 * All.GravityConstantInternal * All.Hubble) / Mtot;
    Mtot = enclosed_mass(R200);
    VERBOSE(2, "M200= %g ", Mtot);
#endif // STATICNFW
}

#ifdef STATICNFW
/**
 * @brief Auxiliary function for static NFW potential.
 *
 * This function calculates the enclosed mass for a given radius in the context of a static NFW potential.
 *
 * @param R The radius at which to calculate the enclosed mass.
 * @return The enclosed mass at the given radius.
 */
double enclosed_mass(double R)
{
    /* Eps is in units of Rs !!!! */
    /* use unsoftened NFW if Nfw_Eps=0 */

    // Check if Nfw_Eps is greater than 0 to determine if unsoftened NFW should be used
    if (Nfw_Eps > 0.0)
    {
        if (R > Rs * Nfw_C)
        {
            R = Rs * Nfw_C;
        }
    }
    // Calculate the enclosed mass based on whether Nfw_Eps is greater than 0
    if (Nfw_Eps > 0.0)
    {
        // Calculate enclosed mass using softened NFW potential formula
        return fac * 4 * M_PI * RhoCrit * Dc *
               (-(Rs * Rs * Rs *
                  (1 - Nfw_Eps + log(Rs) - 2 * Nfw_Eps * log(Rs) + Nfw_Eps * Nfw_Eps * log(Nfw_Eps * Rs))) /
                    ((Nfw_Eps - 1) * (Nfw_Eps - 1)) +
                (Rs * Rs * Rs *
                 (Rs - Nfw_Eps * Rs - (2 * Nfw_Eps - 1) * (R + Rs) * log(R + Rs) +
                  Nfw_Eps * Nfw_Eps * (R + Rs) * log(R + Nfw_Eps * Rs))) /
                    ((Nfw_Eps - 1) * (Nfw_Eps - 1) * (R + Rs)));
    }
    else
    {
        // Calculate enclosed mass using analytic NFW potential formula
        return fac * 4 * M_PI * RhoCrit * Dc *
               (-(Rs * Rs * Rs * (1 + log(Rs))) + Rs * Rs * Rs * (Rs + (R + Rs) * log(R + Rs)) / (R + Rs));
    }
}
#endif // STATICNFW

/**
 * @brief Function that generates an output file with a specific name and writes two lines.
 *
 * This function generates an output file with a given name and writes two lines (header) of text to it.
 * The output file name is constructed based on the output directory, output name, and task number.
 * If the file opening fails, an error message is displayed, and the program ends.
 *
 * @param output_name The name of the output file.
 * @param output_line1 The first line to be written to the output file.
 * @param output_line2 The second line to be written to the output file.
 * @param mode The mode in which the file should be opened (e.g., "w" for write).
 * @return A pointer to the opened output file.
 */
FILE *write_header_to_outputfile(const char *output_name, const char *output_line1, const char *output_line2,
                                 const char *mode)
{
    char buf[200];
    FILE *output;
    // Generate the output file name based on the output directory, output name, and task number
    sprintf(buf, "%s%s_%03d.txt", All.OutputDir, output_name, ThisTask);
    // Try to open the output file
    if (!(output = fopen(buf, mode)))
    {
        // If the file opening fails, display error message and end program
        ENDRUN("error in opening file '%s' ", buf);
    }
    else
    {
        // If the file is opened successfully, write lines to the file and flush the output
        fprintf(output, "%s", output_line1);
        fprintf(output, "%s", output_line2);
        fflush(output);
    }
    return output;
}

/**
 * @brief This function opens various log-files that report on the status and performance of the simulation.
 *
 * On restart from restart-files (start-option 1), the code will append to these files.
 */
void open_outputfiles(void)
{
    char mode[2], buf[200];
    std::ios_base::openmode modeCpp;

    // Set the file opening mode based on the RestartFlag
    if (RestartFlag == 0)
    {
        strcpy(mode, "w"); // Write mode
        modeCpp = std::ios::out;
    }
    else
    {
        strcpy(mode, "a"); // Append mode
        modeCpp = std::ios::app;
    }

    if (ThisTask == 0)
    {
        mkdir(All.OutputDir, 02755); // Create the output directory
    }
    MPI_Barrier(MYMPI_COMM_WORLD);

#if defined(MV_GM_AGNMUPPI_OUTPUT) || defined(MV_GM_STELLAR_KIN_FB2_OUTPUT) ||                                         \
    defined(MV_KRUMHOLZ_MOLECULAR_FRACTION_OUTPUT) || defined(MV_EARLY_FB_HIGH_SN_ENERGY_FOR_LOW_Z_OUTPUT) ||          \
    defined(MV_GM_AGNMUPPI) || defined(MV_GM_COVER_FACT_MCLS) || defined(MV_AGNMUPPI_COOLING_OFF_OUTPUT) ||            \
    defined(MV_AGNMUPPI_COOLING_OFF) || defined(GM_COUNT_PARTICLES_IN_CONE)

    if (ThisTask == 0)
    {
        char buf[200];

        sprintf(buf, "%s/MuppiDetails", All.OutputDir); // create folder to save timer detail information
        mkdir(buf, 02755);
    }
#endif

#ifdef TIMER_DETAILS
    if (ThisTask == 0)
    {
        sprintf(buf, "%s/timer_details", All.OutputDir); // create folder to save timer detail information
        mkdir(buf, 02755);
    }
    MPI_Barrier(MYMPI_COMM_WORLD);
#endif // defined(TIMER_DETAILS)

#ifdef BLACK_HOLES
    /* Note: This is done by everyone */
    if ((ThisTask == 0) && (All.BlackHoleDetails >= 1))
    {
        sprintf(buf, "%sblackhole_details", All.OutputDir);
        mkdir(buf, 02755); // Create a subdirectory for black hole details
    }
    MPI_Barrier(MYMPI_COMM_WORLD);
    if (All.BlackHoleDetails >= 1)
    {
        sprintf(buf, "%sblackhole_details/blackhole_details_%d.txt", All.OutputDir, ThisTask);
        if (!(FdBlackHolesDetails = fopen(buf, mode)))
        {
            ENDRUN("Error in opening file '%s' ", buf);
        }
    }

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    if (All.BlackHoleDetails >= 2)
    {
        sprintf(buf, "%sblackhole_details/blackhole_spin_details_%d.txt", All.OutputDir, ThisTask);
        if (!(FdBlackHolesSpinDetails = fopen(buf, mode)))
        {
            ENDRUN("Error in opening file '%s' ", buf);
        }
        else
        {
            fprintf(FdBlackHolesSpinDetails,
                    "# "
                    "Time\tdt\tID\tMbh_average\tBH_Mass\ta\tEtadisk\tJ_BH_x\tJ_BH_y\tJ_BH_z\tJ_BH\tJ_g_x\tJ_g_y\tJ_g_"
                    "z\tJ_g\tJ_d\tJd_Jbh_ratio\tjbh_scalar_jd\tJ_tot_x\tJ_tot_y\tJ_tot_z\tJ_tot\tN_"
                    "accretion\tRwarp\tRsg\tMsg\tt_accr\tMode\tMdot\tM_d\n");
            fflush(FdBlackHolesSpinDetails);
        }
    }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
#endif // BLACK_HOLES

#ifdef MV_GM_AGNMUPPI_OUTPUT
    // Write header to output file for the total AGN energy output
    sprintf(buf, "%sMuppiDetails/EgyAGN_global.txt", All.OutputDir);
    if (!(FdEgyAGNtot = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
    else
    {
        fprintf(FdEgyAGNtot, "# AGN energy output: global information\n");
        fprintf(FdEgyAGNtot, "   # 1.time       2.TotE_AGNM_tot       3.TotE_AGNM_cool       4.TotE_AGNM_sfr_c    "
                             "5.TotE_AGNM_sfr_h        6.TotE_AGNM_sfr_E      7.TotE_AGNM_sfr_E_aft        "
                             "8.TotE_AGNM_sfr_BHEcoldD       9.Tot_n_AGNM           10.Tot_n_AGNcool\n");
    }
#endif // MV_GM_AGNMUPPI_OUTPUT

#ifdef LT_TRACK_WINDS
    sprintf(buf, "%s%s%d%s", All.OutputDir, "trckwinds.", ThisTask, ".txt");
    if (!(FdTrackW = fopen(buf, mode)))
    {
        ENDRUN("[%d] error in opening file '%s' ", ThisTask, buf);
    }
#endif // LT_TRACK_WINDS

#ifdef MV_GM_STELLAR_KIN_FB2_OUTPUT
    // Write header to output file for Winds output
    FdWind = write_header_to_outputfile("MuppiDetails/winds", "# Statistics of wind gas particles\n",
                                        " #    time  x y z vx vy vz mass rho pres temp id vkick/All.Time SphP[i].E_kin "
                                        "SphP[i].xkin SphP[i].ykin SphP[i].zkin DelayTime\n",
                                        mode);
#endif // MV_GM_STELLAR_KIN_FB2_OUTPUT

#ifdef MV_KRUMHOLZ_MOLECULAR_FRACTION_OUTPUT
    // Write header to output file for SF Molecular fraction output
    FdMolFrac =
        write_header_to_outputfile("MuppiDetails/SF_Molecular_fraction", "# SF molecular fractions - output\n",
                                   " # 1.case   2.time    3.particleID    4.H2_frac_Krumholz    5.H2_frac_Blitz&R    "
                                   "6.metallicity    7.number_dens hot(1)/cold(2)   8.temperature hot(1)/cold(2)\n",
                                   mode);
#endif // MV_KRUMHOLZ_MOLECULAR_FRACTION_OUTPUT

#ifdef MV_EARLY_FB_HIGH_SN_ENERGY_FOR_LOW_Z_OUTPUT
    // Write header to output file for Hypernovae output
    FdHypernovae = write_header_to_outputfile(
        "MuppiDetails/Hypernovae",
        "# Info about MP star-forming particles with low Z injecting higher E_out from early SNe\n",
        "# 1.time    2.ID     3.x     4.y     5.z      6.rho     7.temp     8.mass     9.metallicity     10.E_out\n",
        mode);
#endif // MV_EARLY_FB_HIGH_SN_ENERGY_FOR_LOW_Z_OUTPUT

#ifdef MV_GM_AGNMUPPI_OUTPUT
    // Write header to output file for AGN energy output
    FdEgyAGN =
        write_header_to_outputfile("MuppiDetails/EgyAGN", "# AGN energy output\n",
                                   " # 1.time    2.x    3.y    4.z    5.vx    6.vy    7.vz    8.mass    9.rho    "
                                   "10.pres    11.temp    12.id    13.flag    14.Injected_BH_Energy    15.MultiPhase   "
                                   " 16.DelayTime      17.CouplingBHEnHot    18.CouplingBHEnCold\n",
                                   mode);
#endif // MV_GM_AGNMUPPI_OUTPUT

#if defined(MV_GM_AGNMUPPI) && defined(MV_GM_COVER_FACT_MCLS) && defined(MV_GM_AGNMUPPI_OUTPUT)
    // Write header to output file for AGN covering factors output
    FdCovFacAGN = write_header_to_outputfile(
        "MuppiDetails/CovFacAGN", "# AGN covering factors-coupling factors output for MP particles\n",
        " # 1.time    2.ID    3.filling_fac_hot    4.hsml_Muppi      5.Injected_BH_Energy      "
        "6.SphP[i].CouplingBHEnHot     7.SphP[i].CouplingBHEnCold    8.SphP[i].M_h    9.SphP[i].M_c    10.SphP[i].T_h  "
        "  11.n_h    12.n_c    13.SphP[i].Sfr     14.hsml \n",
        mode);
#endif // defined(MV_GM_AGNMUPPI) && defined(MV_GM_COVER_FACT_MCLS)

#if defined(MV_GM_AGNMUPPI) && !defined(MV_AGNMUPPI_COOLING_OFF_OUTPUT)
    // Write header to output file for properties of MP particles when AGNMuppi is on
    FdAGNmuppi = write_header_to_outputfile(
        "MuppiDetails/AGNmuppi", "# MV_GM_AGNMUPPI_OUTPUT - Properties of MP particles when AGNmuppi is on\n",
        " # 1.time    2.timestep   3.Egy_to_Hot_phase_from_AGNfb    4.Egy_to_Cold_phase_from_AGNfb    "
        "5.M_c_in_timestep    6.M_h_in_timestep   7.T_h_in_timestep    8.M_c_fin_timestep    9.M_h_fin_timestep   "
        "10.T_h_fin_timestep   11.InitialBHEcold   12.y[4]    13.tcool    14.Sph_density     "
        "15.Pressione_sph_preAGN_(cgs/K_B)      16.Pressione_postAGN\n",
        mode);
#endif // defined (MV_GM_AGNMUPPI) && !defined(MV_AGNMUPPI_COOLING_OFF_OUTPUT)

#if defined(MV_AGNMUPPI_COOLING_OFF) && defined(MV_AGNMUPPI_COOLING_OFF_OUTPUT)
    // Write header to output file for particles with cooling
    FdCoolOffAGNmuppi = write_header_to_outputfile(
        "MuppiDetails/CoolOff",
        "# MV_AGNMUPPI_COOLING_OFF - Properties of MP particles with cooling (eventually) switched off\n",
        "# 1.time    2.timestep    3.timeCoolOff   4.Egy_to_Hot_phase_from_AGNfb    5.Egy_to_Cold_phase_from_AGNfb    "
        "6.M_c_in_timestep   7.M_h_in_timestep    8.T_h_in_timestep    9.M_c_fin_timestep    10.M_h_fin_timestep   "
        "11.T_h_fin_timestep   12.InitialBHEcold   13.y[4]    14.tcool    15.Sph_density     "
        "16.Pressione_sph_preAGN_(cgs/K_B)      17.Pressione_postAGN\n",
        mode);
#endif // defined(MV_AGNMUPPI_COOLING_OFF) && defined(MV_AGNMUPPI_COOLING_OFF_OUTPUT)

#if !defined(SWITCH_TIMER_OFF) && defined(TIMER_DETAILS)
    // open timestep file for each task
    sprintf(buf, "%s%s%d.txt", All.OutputDir, "/timer_details/cpu_timestep_", ThisTask);
    FdCPUTimestep.open(buf, modeCpp);
    if (!(FdCPUTimestep.is_open()))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
    else
    {
        FdCPUTimestep << "\nThis is Task: " << ThisTask << std::endl;
        FdCPUTimestep << "This file provides timing information for each timestep:" << std::endl;
        FdCPUTimestep.flush();
    }
#endif // !defined (SWITCH_TIMER_OFF) && defined(TIMER_DETAILS)

    /* only the root processors writes to the log files */
    if (ThisTask != 0)
    {
        return;
    }

    sprintf(buf, "%s%s", All.OutputDir, All.InfoFile);
    if (!(FdInfo = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

    sprintf(buf, "%s%s", All.OutputDir, All.EnergyFile);
    if (!(FdEnergy = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

    sprintf(buf, "%s%s", All.OutputDir, All.TimingsFile);
    if (!(FdTimings = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

    sprintf(buf, "%s%s", All.OutputDir, All.TimebinFile);
    if (!(FdTimebin = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

    sprintf(buf, "%s%s", All.OutputDir, "paramchanges.txt");
    if (!(FdParamChangeLog = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
    sprintf(buf, "%s%s", All.OutputDir, "pm.txt");
    if (!(FdPM = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#endif // PMGRID

#ifdef TIMER_TIMESTAMP
    sprintf(buf, "%s%s", All.OutputDir, "Timers_timestamp.txt");
    FdTimersTimestamp.open(buf, modeCpp);
    if (!(FdTimersTimestamp.is_open()))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
    else
    {
        FdTimersTimestamp << "PUSH/POP, level, Region_name, Timestamp" << std::endl;
    }
#endif // TIMER_TIMESTAMP

#ifndef SWITCH_TIMER_OFF
    sprintf(buf, "%s%s", All.OutputDir, All.CpuFile);
    FdCPU.open(buf, modeCpp);
    if (!(FdCPU.is_open()))
    {
        ENDRUN("Error in opening file '%s' ", All.CpuFile);
    }
    else
    {
        if (RestartFlag == 0)
        {
            FdCPU << "This file provides accumulated timing information for the global timer:" << std::endl;
        }
        else if (RestartFlag == 1)
        {
            FdCPU << "\nRestarting:" << std::endl;
        }
        FdCPU.flush();
    }

    sprintf(buf, "%s%s", All.OutputDir, All.BalanceFile);
    FdBalance.open(buf, modeCpp);
    if (!(FdBalance.is_open()))
    {
        ENDRUN("Error in opening file '%s' ", All.BalanceFile);
    }
    else
    {
        if (RestartFlag == 0)
        {
            FdBalance << "This file provides timing balance information for each timestep: " << std::endl;
        }
        else if (RestartFlag == 1)
        {
            FdBalance << "Restarting: " << std::endl;
        }
        FdBalance.flush();
    }
#endif // SWITCH_TIMER_OFF

#ifdef SFR
    sprintf(buf, "%s%s", All.OutputDir, "sfr.txt");
    if (!(FdSfr = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#endif // SFR

#ifdef BLACK_HOLES
    sprintf(buf, "%s%s", All.OutputDir, "blackholes.txt");
    if (!(FdBlackHoles = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#if (defined(PB_BH_HSML_PREVENT_HOLE) || defined(PB_OUTPUT_SINGLE_BH))
    sprintf(buf, "%s%s", All.OutputDir, "BlackHole_GasNgbs.txt");
    if (!(Fd_BlackHole_GasNgbs = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#endif // PB_BH_HSML_PREVENT_HOLE || PB_OUTPUT_SINGLE_BH
#endif // BLACK_HOLES

#ifdef FORCETEST
    if (RestartFlag == 0)
    {
        sprintf(buf, "%s%s", All.OutputDir, "forcetest.txt");
        if (!(FdForceTest = fopen(buf, "w")))
        {
            ENDRUN("Error in opening file '%s' ", buf);
        }
        fclose(FdForceTest);
    }
#endif // FORCETEST

#ifdef XXLINFO
    sprintf(buf, "%s%s", All.OutputDir, "xxl.txt");
    if (!(FdXXL = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
    else
    {
        if (RestartFlag == 0)
        {
            fprintf(FdXXL, "nstep time ");
#ifdef MAGNETIC
            fprintf(FdXXL, "<|B|> ");
            fprintf(FdXXL, "max(divB) ");
#endif // MAGNETIC
#ifdef TIME_DEP_ART_VISC
            fprintf(FdXXL, "<alpha> ");
#endif // TIME_DEP_ART_VISC
            fprintf(FdXXL, "\n");
            fflush(FdXXL);
        }
    }
#endif // XXLINFO

#ifdef DARKENERGY
    sprintf(buf, "%s%s", All.OutputDir, "darkenergy.txt");
    if (!(FdDE = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
    else // DARKENERGY
    {
        if (RestartFlag == 0)
        {
            fprintf(FdDE, "nstep time H(a) ");
#ifndef TIMEDEPDE
            fprintf(FdDE, "w0 Omega_L ");
#else // TIMEDEPDE
            fprintf(FdDE, "w(a) Omega_L ");
#endif // TIMEDEPDE
#ifdef TIMEDEPGRAV
            fprintf(FdDE, "dH dG ");
#endif // TIMEDEPGRAV
            fprintf(FdDE, "\n");
            fflush(FdDE);
        }
    }
#endif // DARKENERGY

#ifdef LT_EXTEGY_INFO
    sprintf(buf, "%s%s", All.OutputDir, "extegy.txt");
    if (!(FdExtEgy = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#endif // LT_EXTEGY_INFO

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
    sprintf(buf, "%s%s", All.OutputDir, "spinfo.txt");
    if (!(FdSPinfo = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#endif // LT_SEv_INFO_DETAILS_onSPREAD

#ifdef LT_STELLAREVOLUTION
#ifdef LT_SEv_INFO
#ifdef LT_SEvDbg
    sprintf(buf, "%s%s", All.OutputDir, "met_sumcheck.txt");
    if (!(FdMetSumCheck = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#endif // LT_SEvDbg

    sprintf(buf, "%s%s", All.OutputDir, "metals.txt");
    if (!(FdMetals = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

    sprintf(buf, "%s%s", All.OutputDir, "sn.txt");
    if (!(FdSn = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

    sprintf(buf, "%s%s", All.OutputDir, "sn_lost.txt");
    if (!(FdSnLost = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }

#ifdef WINDS
    sprintf(buf, "%s%s", All.OutputDir, "winds.txt");
    if (!(FdWinds = fopen(buf, mode)))
    {
        ENDRUN("Error in opening file '%s' ", buf);
    }
#endif // WINDS
#endif // LT_SEv_INFO
#endif // LT_STELLAREVOLUTION
}

/**
 * @brief This function closes the global log-files.
 *
 * This function is responsible for closing the global log-files that were opened during program execution.
 * Make sure to close all files opened in void open_outputfiles(void) here.
 */
void close_outputfiles(void)
{
#ifdef BLACK_HOLES
    if (All.BlackHoleDetails >= 1)
    {
        fclose(FdBlackHolesDetails); /* needs to be done by everyone */
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
        fclose(FdBlackHolesSpinDetails);
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
    }
#endif // BLACK_HOLES

#ifdef LT_TRACK_WINDS
    fclose(FdTrackW);
#endif // LT_TRACK_WINDS

#ifdef MV_GM_STELLAR_KIN_FB2_OUTPUT
    fclose(FdWind);
#endif // MV_GM_STELLAR_KIN_FB2_OUTPUT

#if defined(MV_GM_AGNMUPPI) && defined(MV_GM_AGNMUPPI_OUTPUT)
    fclose(FdAGNmuppi);
    fclose(FdEgyAGN);
    fclose(FdEgyAGNtot);
#if defined(MV_GM_COVER_FACT_MCLS)
    fclose(FdCovFacAGN);
#endif // MV_GM_COVER_FACT_MCLS
#endif // MV_GM_AGNMUPPI && MV_GM_AGNMUPPI_OUTPUT

#if defined(MV_AGNMUPPI_COOLING_OFF) && defined(MV_AGNMUPPI_COOLING_OFF_OUTPUT)
    fclose(FdCoolOffAGNmuppi);
#endif // defined(MV_AGNMUPPI_COOLING_OFF) && defined(MV_AGNMUPPI_COOLING_OFF_OUTPUT)
#ifdef MV_KRUMHOLZ_MOLECULAR_FRACTION_OUTPUT
    fclose(FdMolFrac);
#endif // MV_KRUMHOLZ_MOLECULAR_FRACTION_OUTPUT
#ifdef MV_EARLY_FB_HIGH_SN_ENERGY_FOR_LOW_Z_OUTPUT
    fclose(FdHypernovae);
#endif // MV_EARLY_FB_HIGH_SN_ENERGY_FOR_LOW_Z_OUTPUT

    if (ThisTask != 0) /* only the root processors writes to the log files */
    {
        return;
    }

#ifndef SWITCH_TIMER_OFF
    FdCPU.close();
    FdBalance.close();
#ifdef TIMER_TIMESTAMP
    FdTimersTimestamp.close();
#endif // TIMER_TIMESTAMP;
#ifdef TIMER_DETAILS
    FdCPUTimestep.close();
#endif // TIMER_DETAILS
#endif // SWITCH_TIMER_OFF

    fclose(FdInfo);
    fclose(FdEnergy);
    fclose(FdTimings);
    fclose(FdTimebin);
    fclose(FdParamChangeLog);

#if defined(PMGRID) && defined(PLACEHIGHRESREGION)
    fclose(FdPM);
#endif // PMGRID

#ifdef SFR
    fclose(FdSfr);
#endif // SFR

#ifdef BLACK_HOLES
    fclose(FdBlackHoles);
#if (defined(PB_BH_HSML_PREVENT_HOLE) || defined(PB_OUTPUT_SINGLE_BH))
    fclose(Fd_BlackHole_GasNgbs);
#endif // PB_BH_HSML_PREVENT_HOLE || PB_OUTPUT_SINGLE_BH
#endif // BLACK_HOLES

#ifdef XXLINFO
    fclose(FdXXL);
#endif // XXLINFO

#ifdef DARKENERGY
    fclose(FdDE);
#endif // DARKENERGY

#ifdef LT_SEv_INFO_DETAILS_onSPREAD
    fclose(FdSPinfo);
#endif // LT_SEv_INFO_DETAILS_onSPREAD

#ifdef LT_EXTEGY_INFO
    fclose(FdExtEgy);
#endif // LT_EXTEGY_INFO

#ifdef LT_STELLAREVOLUTION
#ifdef LT_SEv_INFO
#ifdef LT_SEvDbg
    fclose(FdMetSumCheck);
#endif // LT_SEvDbg
    fclose(FdSn);
    fclose(FdSnLost);
    fclose(FdMetals);
#ifdef WINDS
    fclose(FdWinds);
#endif // WINDS
#endif // LT_SEv_INFO
#endif // LT_STELLAREVOLUTION
}

/**
 * @brief Enum representing different levels of parameters.
 */
enum class Param_Level
{
    PARAMETER_NECESSARY = -1,
    PARAMETER_PUBLIC = 0,
    PARAMETER_ADVANCED = 1,
    PARAMETER_EXPERIENCED = 2,
    PARAMETER_DEVELOPER = 42
};

/**
 * @brief Enum representing different types of parameters.
 */
enum class Param_Type
{
    REAL = 1,
    STRING = 2,
    INT = 3
};

/**
 * @brief Enum representing whether/how a parameter may change across a restart (RestartFlag==1).
 *
 * RESTART_NEVER is the default: a parameter not explicitly classified otherwise is assumed to
 * be frozen at whatever was stored in the restart file, and changing it in the parameter file
 * before a restart is treated as an error rather than being silently ignored.
 */
enum class Param_Restart
{
    RESTART_NEVER = 0, // default: differing from the restart file is an error
    RESTART_ALWAYS = 1, // freely re-applied from the parameter file on every restart
    RESTART_DANGEROUS = 2, // re-applied only if EXPERT_LEVEL allows it; always logged
};

/**
 * @brief Function to setup a parameter and define its default levels etc.
 *
 * This function sets up a parameter by assigning its name, address, default indicator,
 * and type to corresponding arrays. It is used to manage parameters within the code.
 *
 * @param paramName The name of the parameter.
 * @param paramAddr Pointer to the parameter's address.
 * @param defaultIndicator The default indicator for the parameter from the Param_Level class.
 * @param parameter_type The type of the parameter from the Param_Type class.
 * @param tag Pointer to an array of character pointers representing parameter tags.
 * @param addr Pointer to an array of void pointers representing parameter addresses.
 * @param default_indicator Pointer to an array of integers representing default indicators.
 * @param id Pointer to an array of integers representing parameter IDs.
 * @param nt Reference to an integer representing the number of parameters.
 * @param parameter_pointers_map A multimap that maps parameter names to their addresses and types.
 * @param restartClass Whether/how this parameter may change across a restart, see Param_Restart.
 */
void setup_parameter(const char *paramName, void *paramAddr, Param_Level defaultIndicator, Param_Type parameter_type,
                     char **tag, void **addr, int *default_indicator, int *id, int &nt,
                     std::multimap<std::string, ParamInfo> &parameter_pointers_map,
                     Param_Restart restartClass = Param_Restart::RESTART_NEVER, int minExpertLevel = 0)
{
    strcpy(tag[nt], paramName); // parameter name within the parameter file
    addr[nt] = paramAddr; // parameter name within the code
    default_indicator[nt] = static_cast<int>(
        defaultIndicator); // PARAMETER_NECESSARY, PARAMETER_PUBLIC, PARAMETER_ADVANCED or PARAMETER_DEVELOPER
    id[nt++] = static_cast<int>(parameter_type); // REAL, STRING or INT
    parameter_pointers_map.insert({paramName,
                                   {paramAddr, static_cast<int>(parameter_type), static_cast<int>(restartClass),
                                    minExpertLevel}}); // add variable and type to multimap
}

/**
 * @brief Reconcile every setup_parameter()-registered parameter across a restart.
 *
 * Called after restart() has read `All` from the restart file. For every registered
 * parameter whose freshly-parsed value (in `all`) differs from the restart-file value (already in
 * `All`), this either applies the change (Param_Restart::RESTART_ALWAYS, optionally gated by
 * EXPERT_LEVEL for RESTART_DANGEROUS) or aborts the run (RESTART_NEVER) instead of silently
 * keeping the stale restart-file value.
 *
 * @param all Freshly parsed parameter-file values, saved before restart() overwrote `All`.
 * @param parameter_pointers_map Registry of every setup_parameter()-declared parameter.
 */
void change_simple_parameters(struct global_data_all_processes &all,
                              std::multimap<std::string, ParamInfo> &parameter_pointers_map)
{
    for (auto &entry : parameter_pointers_map)
    {
        const std::string &name = entry.first;
        ParamInfo &info = entry.second;

        void *curAddr = info.addr; // already points at All.<field>, i.e. the restart-file value
        ptrdiff_t offset = (char *)info.addr - (char *)&All;
        void *newAddr = (char *)&all + offset; // freshly parsed parameter-file value

        bool changed;
        // check if value has changed
        if (info.type == static_cast<int>(Param_Type::STRING))
        {
            changed = strcmp((char *)curAddr, (char *)newAddr) != 0;
        }
        else if (info.type == static_cast<int>(Param_Type::INT))
        {
            changed = *(int *)curAddr != *(int *)newAddr;
        }
        else // REAL
        {
            changed = *(double *)curAddr != *(double *)newAddr;
        }

        if (!changed)
        {
            continue;
        }

        // small helper function/lambda for output
        auto format_value = [&](void *addr) -> std::string {
            std::ostringstream oss;

            // dereference depending on type
            if (info.type == static_cast<int>(Param_Type::STRING))
            {
                oss << "\"" << (char *)addr << "\"";
            }
            else if (info.type == static_cast<int>(Param_Type::INT))
            {
                oss << *(int *)addr;
            }
            else // REAL (double)
            {
                oss << *(double *)addr;
            }

            return oss.str();
        };
        // save values
        const std::string old_val = format_value(curAddr);
        const std::string new_val = format_value(newAddr);

        if (info.restart_class == static_cast<int>(Param_Restart::RESTART_NEVER))
        {
            ENDRUN("Parameter '%s' was changed in the parameter file but cannot be changed on restart. "
                   "Changed from %s to %s",
                   name.c_str(), old_val.c_str(), new_val.c_str());
        }
        else if (info.restart_class == static_cast<int>(Param_Restart::RESTART_DANGEROUS) &&
                 EXPERT_LEVEL < info.min_expert_level)
        {
            ENDRUN("Parameter '%s' was changed in the parameter file but changing it on restart needs "
                   "EXPERT_LEVEL >= %d. Changed from %s to %s",
                   name.c_str(), info.min_expert_level, old_val.c_str(), new_val.c_str());
        }

        if (ThisTask == 0)
        {
            fprintf(FdParamChangeLog, "Time %f: %s changed from %s to %s\n", All.Time, name.c_str(), old_val.c_str(),
                    new_val.c_str());

            VERBOSE(0, "%s changed from %s to %s\n", name.c_str(), old_val.c_str(), new_val.c_str());

            fflush(FdParamChangeLog);
        }

        // copy new value
        if (info.type == static_cast<int>(Param_Type::STRING))
        {
            strcpy((char *)curAddr, (char *)newAddr);
        }
        else if (info.type == static_cast<int>(Param_Type::INT))
        {
            *(int *)curAddr = *(int *)newAddr;
        }
        else // REAL
        {
            *(double *)curAddr = *(double *)newAddr;
        }
    }
}

/**
 * @brief This function sets up all parameters.
 *
 * In this function all parameters are set up by calling the setup_parameter function. Add new parameters  with
 * corresponding directives here.
 *
 * @param tag Pointer to an array of character pointers representing parameter tags.
 * @param addr Pointer to an array of void pointers representing parameter addresses.
 * @param default_indicator Pointer to an array of integers representing default indicators.
 * @param id Pointer to an array of integers representing parameter IDs.
 * @param nt Reference to an integer representing the number of parameters.
 * @param parameter_pointers_map A multimap that maps parameter names to their addresses and types.
 */
void setup_all_parameters(char **tag, void **addr, int *default_indicator, int *id, int &nt,
                          std::multimap<std::string, ParamInfo> &parameter_pointers_map)
{
    setup_parameter("InitCondFile", All.InitCondFile, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map);
    setup_parameter("OutputDir", All.OutputDir, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("SnapshotFileBase", All.SnapshotFileBase, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("EnergyFile", All.EnergyFile, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("CpuFile", All.CpuFile, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("BalanceFile", All.BalanceFile, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map);
    setup_parameter("BalanceFileHead", All.BalanceFileHead, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map);
    setup_parameter("InfoFile", All.InfoFile, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimingsFile", All.TimingsFile, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimebinFile", All.TimebinFile, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("RestartFile", All.RestartFile, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("OutputListFilename", All.OutputListFilename, Param_Level::PARAMETER_PUBLIC, Param_Type::STRING,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("OutputListOn", &All.OutputListOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("Omega0", &All.Omega0, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("OmegaBaryon", &All.OmegaBaryon, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("OmegaLambda", &All.OmegaLambda, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("HubbleParam", &All.HubbleParam, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BoxSize", &All.BoxSize, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("PeriodicBoundariesOn", &All.PeriodicBoundariesOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MaxMemSize", &All.MaxMemSize, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimeOfFirstSnapshot", &All.TimeOfFirstSnapshot, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("CpuTimeBetRestartFile", &All.CpuTimeBetRestartFile, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimeBetStatistics", &All.TimeBetStatistics, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimeBegin", &All.TimeBegin, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("TimeMax", &All.TimeMax, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimeBetSnapshot", &All.TimeBetSnapshot, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("UnitVelocity_in_cm_per_s", &All.UnitVelocity_in_cm_per_s, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("UnitLength_in_cm", &All.UnitLength_in_cm, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("UnitMass_in_g", &All.UnitMass_in_g, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("ErrTolIntAccuracy", &All.ErrTolIntAccuracy, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("ErrTolTheta", &All.ErrTolTheta, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("StopAfterNSteps", &All.StopAfterNSteps, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("StopAtA", &All.StopAtA, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);

    setup_parameter("TimerReportLevel", &All.TimerReportLevel, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimerReportBalance", &All.TimerReportBalance, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);

#ifdef fSIDM
#ifdef fSIDM_SIMPLE
    setup_parameter("FSIDM_CrossSectionPerMass_in_cgs", &All.FSIDM_CrossSectionPerMass_in_cgs,
                    Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("FSIDM_CrossSectionDef", &All.FSIDM_CrossSectionDef, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::STRING, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 3);
#endif // fSIDM_SIMPLE
#ifdef mSIDM_TIMESTEP
    setup_parameter("FSIDM_tfact", &All.FSIDM_tfact, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // mSIDM_TIMESTEP
#ifdef fSIDM_DISSIPATION
    setup_parameter("FSIDM_const_r_diss", &All.FSIDM_const_r_diss, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // fSIDM_DISSIPATION
#endif // fSIDM

#ifdef rSIDM
#ifdef rSIDM_ISOTROPIC
    setup_parameter("RSIDM_CrossSectionPerMass_in_cgs", &All.RSIDM_CrossSectionPerMass_in_cgs,
                    Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("RSIDM_CrossSectionDef", &All.RSIDM_CrossSectionDef, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::STRING, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 3);
#endif // rSIDM_ISOTROPIC
#ifdef mSIDM_TIMESTEP
    setup_parameter("RSIDM_tfact", &All.RSIDM_tfact, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // mSIDM_TIMESTEP
#ifdef rSIDM_DISSIPATION
    setup_parameter("RSIDM_f_diss", &All.RSIDM_f_diss, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // rSIDM_DISSIPATION
#endif // rSIDM


#ifdef mSIDM
    setup_parameter("MSIDM_SEED", &All.MSIDM_SEED, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#if defined(mSIDM_VDEP0) || defined(mSIDM_VDEP1) || defined(mSIDM_VDEP_MV) || defined(mSIDM_VDEP_RV)
    setup_parameter("MSIDM_vdep_w", &All.MSIDM_vdep_w, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // mSIDM_VDEP0 || mSIDM_VDEP1 || mSIDM_VDEP_MV || mSIDM_VDEP_RV

#ifdef mSIDM_VDEP0
    setup_parameter("MSIDM_vdep_alpha", &All.MSIDM_vdep_alpha, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_vdep_beta", &All.MSIDM_vdep_beta, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // mSIDM_VDEP0

#ifdef mSIDM_VDEP1
    setup_parameter("MSIDM_vdep_u", &All.MSIDM_vdep_u, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // mSIDM_VDEP1

#ifdef mSIDM_DIFFERENTIAL_CROSS_SECTION
    setup_parameter("MSIDM_CrossSection", &All.MSIDM_CrossSection, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_DCS_params", &All.MSIDM_DCS_params, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);

    setup_parameter("MSIDM_thetaC_fwd", &All.MSIDM_thetaC_fwd, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_thetaC_bwd", &All.MSIDM_thetaC_bwd, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_v_min", &All.MSIDM_v_min, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_v_max", &All.MSIDM_v_max, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_len_xh", &All.MSIDM_len_xh, Param_Level::PARAMETER_ADVANCED, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_len_v", &All.MSIDM_len_v, Param_Level::PARAMETER_ADVANCED, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MSIDM_reserve_fact", &All.MSIDM_reserve_fact, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // mSIDM_DIFFERENTIAL_CROSS_SECTION

#ifdef ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
    setup_parameter("MSIDM_first_target_id", &All.MSIDM_first_target_id, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 3);
#endif // ANGULAR_DEFLECTION_TEST_ON_FIXED_TARGET
#endif // mSIDM



#if defined(mSIDM) || defined(mIDM)
#ifndef ADAPTGRAVSOFT
    setup_parameter("DesNumNgb_pt1", &All.DesNumNgb_pt1, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("MaxNumNgbDeviation_pt1", &All.MaxNumNgbDeviation_pt1, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 3);
#endif // ADAPTGRAVSOFT
#endif // mSIDM || mIDM


#ifdef SUBFIND
    setup_parameter("ErrTolThetaSubfind", &All.ErrTolThetaSubfind, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("DesLinkNgb", &All.DesLinkNgb, Param_Level::PARAMETER_DEVELOPER, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // SUBFIND

    setup_parameter("ErrTolForceAcc", &All.ErrTolForceAcc, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("MinGasHsmlFractional", &All.MinGasHsmlFractional, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);

#ifdef MAXHSML
    setup_parameter("MaxHsml", &All.MaxHsml, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // MAXHSML

    setup_parameter("MaxSizeTimestep", &All.MaxSizeTimestep, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("MinSizeTimestep", &All.MinSizeTimestep, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("MaxRMSDisplacementFac", &All.MaxRMSDisplacementFac, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("ArtBulkViscConst", &All.ArtBulkViscConst, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("ArtBulkViscBeta", &All.ArtBulkViscBeta, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);


#ifdef ARTIFICIAL_CONDUCTIVITY
    setup_parameter("ArtCondConstant", &All.ArtCondConstant, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#ifdef TIME_DEP_ART_COND
    setup_parameter("ArtCondMin", &All.ArtCondMin, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // TIME_DEP_ART_COND
#endif // ARTIFICIAL_CONDUCTIVITY

    setup_parameter("CourantFac", &All.CourantFac, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("DesNumNgb", &All.DesNumNgb, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);

#ifdef NEUTRINOS
    setup_parameter("Time_tree_on_nu", &All.Time_tree_on_nu, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
#endif // NEUTRINOS



    setup_parameter("MaxNumNgbDeviationRel", &All.MaxNumNgbDeviationRel, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);

    setup_parameter("ComovingIntegrationOn", &All.ComovingIntegrationOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("ICFormat", &All.ICFormat, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("SnapFormat", &All.SnapFormat, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("NumFilesPerSnapshot", &All.NumFilesPerSnapshot, Param_Level::PARAMETER_PUBLIC, Param_Type::INT,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("NumFilesWrittenInParallel", &All.NumFilesWrittenInParallel, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("CoolingOn", &All.CoolingOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("StarformationOn", &All.StarformationOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("TypeOfTimestepCriterion", &All.TypeOfTimestepCriterion, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("TimeLimitCPU", &All.TimeLimitCPU, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("SofteningHalo", &All.SofteningHalo, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningDisk", &All.SofteningDisk, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningBulge", &All.SofteningBulge, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningGas", &All.SofteningGas, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningStars", &All.SofteningStars, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningBndry", &All.SofteningBndry, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningHaloMaxPhys", &All.SofteningHaloMaxPhys, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningDiskMaxPhys", &All.SofteningDiskMaxPhys, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningBulgeMaxPhys", &All.SofteningBulgeMaxPhys, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningGasMaxPhys", &All.SofteningGasMaxPhys, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningStarsMaxPhys", &All.SofteningStarsMaxPhys, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("SofteningBndryMaxPhys", &All.SofteningBndryMaxPhys, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BufferSize", &All.BufferSize, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("PartAllocFactor", &All.PartAllocFactor, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("MaxTreeAllocFactor", &All.MaxTreeAllocFactor, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("GravityConstantInternal", &All.GravityConstantInternal, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("InitGasTemp", &All.InitGasTemp, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MinGasTemp", &All.MinGasTemp, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)
    setup_parameter("ReferenceGasMass", &All.ReferenceGasMass, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)

#ifdef NAVIERSTOKES
    setup_parameter("FractionSpitzerViscosity", &All.FractionSpitzerViscosity, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
#endif // NAVIERSTOKES

#ifdef NAVIERSTOKES_CONSTANT
    setup_parameter("ShearViscosityTemperature", &All.ShearViscosityTemperature, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
#endif // NAVIERSTOKES_CONSTANT

#ifdef NAVIERSTOKES_TIMESTEP
    setup_parameter("AlphaViscTimestep", &All.AlphaViscTimestep, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // NAVIERSTOKES_TIMESTEP


#ifdef CONDUCTION
    setup_parameter("ConductionEfficiency", &All.ConductionEfficiency, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("MaxSizeConductionStep", &All.MaxSizeConductionStep, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
#endif // CONDUCTION


    setup_parameter("BlackHolesOn", &All.BlackHolesOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 3);
    setup_parameter("BlackHoleSubgridADOn", &All.BlackHoleSubgridADOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);

#if defined(BLACK_HOLES) || defined(MV_KRUMHOLZ_MOLECULAR_FRACTION_CLUMPINESS)
    setup_parameter("massDMpart", &All.massDMpart, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // defined(BLACK_HOLES) || defined(MV_KRUMHOLZ_MOLECULAR_FRACTION_CLUMPINESS)

#ifdef BLACK_HOLES
    setup_parameter("BlackHoleThermalFeedbackOn", &All.BlackHoleThermalFeedbackOn, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 1);
    setup_parameter("BlackHoleKineticFeedbackOn", &All.BlackHoleKineticFeedbackOn, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 1);
    setup_parameter("TimeBetOnTheFlyFoF", &All.TimeBetOnTheFlyFoF, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleFoFMinMassForNewSeed", &All.BlackHoleFoFMinMassForNewSeed,
                    Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleSeedMass", &All.SeedBlackHoleMass, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleSeedStarMassFraction", &All.BlackHoleSeedStarMassFraction,
                    Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleSeedDMFraction", &All.BlackHoleSeedDMFraction, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleSeedGasFraction", &All.BlackHoleSeedGasFraction, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleFormationFactor", &All.BlackHoleFormationFactor, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleNgbFactor", &All.BlackHoleNgbFactor, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleMaxAccretionRadius", &All.BlackHoleMaxAccretionRadius, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("BlackHoleSwallowGasOn", &All.BlackHoleSwallowGasOn, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleIgnoreMomentum", &All.BlackHoleIgnoreMomentum, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleAccretionSlicesOn", &All.BlackHoleAccretionSlicesOn, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleDetails", &All.BlackHoleDetails, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleRepositioningOn", &All.BlackHoleRepositioningOn, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleHotColdAccretionOn", &All.BlackHoleHotColdAccretionOn, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleAccretionFactor", &All.BlackHoleAccretionFactor, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleColdAccretionFactor", &All.BlackHoleColdAccretionFactor, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleVelocityInBondiOn", &All.BlackHoleVelocityInBondiOn, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleVariableAccretionFactorOn", &All.BlackHoleVariableAccretionFactorOn,
                    Param_Level::PARAMETER_ADVANCED, Param_Type::INT, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleVariableAccretionSlope", &All.BlackHoleVariableAccretionSlope,
                    Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleRadiativeEfficiencyTreatment", &All.BlackHoleRadiativeEfficiencyTreatment,
                    Param_Level::PARAMETER_ADVANCED, Param_Type::INT, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleRadiativeEfficiency", &All.BlackHoleRadiativeEfficiency, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("BlackHolePhysMdotUpperLimit", &All.BlackHolePhysMdotUpperLimit, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleEddingtonFactor", &All.BlackHoleEddingtonFactor, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleFeedbackFactor", &All.BlackHoleFeedbackFactor, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleRadioModeFeedbackBoost", &All.BlackHoleRadioModeFeedbackBoost,
                    Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleRadioModeTreshold", &All.BlackHoleRadioModeTreshold, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleLimitFeedbackOn", &All.BlackHoleLimitFeedbackOn, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 1);
    setup_parameter("BlackHoleLimitMaxTemp", &All.BlackHoleLimitMaxTemp, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleColdGasTemperatureTresh", &All.BlackHoleColdGasTemperatureTresh,
                    Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleFrictionForceOn", &All.BlackHoleFrictionForceOn, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_DANGEROUS, 2);
    setup_parameter("BlackHoleMergeCsndFrac", &All.BlackHoleMergeCsndFrac, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("BlackHoleMergerCondition", &All.BlackHoleMergerCondition, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::INT, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("BlackHoleMergeDistFrac", &All.BlackHoleMergeDistFrac, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("BlackHoleMergeBindingFrac", &All.BlackHoleMergeBindingFrac, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    setup_parameter("BlackHoleAccretionDiskAlpha", &All.BlackHoleAccretionDiskAlpha, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("SeedBlackHoleSpin", &All.SeedBlackHoleSpin, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map);
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION

#endif // BLACK_HOLES

#if defined(UM_CHEMISTRY) && defined(UM_CHEMISTRY_INISET)
    /* read the composition from the parameter file */
    setup_parameter("START_elec", &All.START_elec, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HI", &All.START_HI, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HII", &All.START_HII, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HM", &All.START_HM, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HeI", &All.START_HeI, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HeII", &All.START_HeII, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HeIII", &All.START_HeIII, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_H2I", &All.START_H2I, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_H2II", &All.START_H2II, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HD", &All.START_HD, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_DI", &All.START_DI, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_DII", &All.START_DII, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("START_HeHII", &All.START_HeHII, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // defined (UM_CHEMISTRY) && defined (UM_CHEMISTRY_INISET)

#ifdef SFR
    setup_parameter("CritOverDensity", &All.CritOverDensity, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#ifndef LT_STELLAREVOLUTION
    setup_parameter("CritPhysDensity", &All.CritPhysDensity, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);

    setup_parameter("FactorSN", &All.FactorSN, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("FactorEVP", &All.FactorEVP, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_STELLAREVOLUTION
    setup_parameter("TempSupernova", &All.TempSupernova, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("TempClouds", &All.TempClouds, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MaxSfrTimescale", &All.MaxSfrTimescale, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);

    setup_parameter("MuppiOn", &All.MuppiOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);

#ifdef WINDS
    setup_parameter("WindEfficiency", &All.WindEfficiency, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("WindEnergyFraction", &All.WindEnergyFraction, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("WindFreeTravelMaxTimeFactor", &All.WindFreeTravelMaxTimeFactor, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("WindFreeTravelDensFac", &All.WindFreeTravelDensFac, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
#endif // WINDS
#endif // SFR

#ifdef CS_DECOUPLING
    setup_parameter("DecouplingParam", &All.DecouplingParam, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map);
#endif // CS_DECOUPLING

#ifdef DARKENERGY
#ifdef TIMEDEPDE
    setup_parameter("DarkEnergyFile", All.DarkEnergyFile, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#else // TIMEDEPDE
    setup_parameter("DarkEnergyParam", &All.DarkEnergyParam, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // TIMEDEPDE
#endif // DARKENERGY

#ifdef RESCALEVINI
    setup_parameter("VelIniScale", &All.VelIniScale, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // RESCALEVINI

#ifdef TIME_DEP_ART_VISC
    setup_parameter("ViscosityAlphaMin", &All.ViscosityAlphaMin, Param_Level::PARAMETER_EXPERIENCED, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("ViscositySourceScaling", &All.ViscositySourceScaling, Param_Level::PARAMETER_EXPERIENCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("ViscosityDecayLength", &All.ViscosityDecayLength, Param_Level::PARAMETER_EXPERIENCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
#endif // TIME_DEP_ART_VISC

#ifdef ARTIFICIAL_MAGNETIC_DISSIPATION
    setup_parameter("ArtificialMagneticDissipationConstant", &All.ArtificialMagneticDissipationConstant,
                    Param_Level::PARAMETER_EXPERIENCED, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#ifdef TIME_DEP_MAGN_DISP
    setup_parameter("ArtificialMagneticDissipationMin", &All.ArtificialMagneticDissipationMin,
                    Param_Level::PARAMETER_EXPERIENCED, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("ArtificialMagneticDissipationSource", &All.ArtificialMagneticDissipationSource,
                    Param_Level::PARAMETER_EXPERIENCED, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("ArtificialMagneticDissipationDecaytime", &All.ArtificialMagneticDissipationDecaytime,
                    Param_Level::PARAMETER_EXPERIENCED, Param_Type::REAL, tag, addr, default_indicator, id, nt,
                    parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // TIME_DEP_MAGN_DISP
#endif // ARTIFICIAL_MAGNETIC_DISSIPATION

#ifdef DIVBCLEANING_TRICCO
    setup_parameter("DivBcleaningParabolicSigma", &All.DivBcleaningParabolicSigma, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("DivBOverCleanFac", &All.DivBOverCleanFac, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // DIVBCLEANING_TRICCO


#ifdef LMB_MFM_ENERGY_CLOSURE_LIMITER
    /* Face-closure limiter on the per-step internal-energy change */
    setup_parameter("MfmEnergyClosureMin", &All.MfmEnergyClosureMin, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    setup_parameter("MfmEnergyClosureWeight", &All.MfmEnergyClosureWeight, Param_Level::PARAMETER_ADVANCED,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("MfmEnergyTolMin", &All.MfmEnergyTolMin, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // LMB_MFM_ENERGY_CLOSURE_LIMITER

#ifdef DIVBCLEANING_DEDNER
    setup_parameter("DivBcleaningParabolicSigma", &All.DivBcleaningParabolicSigma, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
    setup_parameter("DivBcleaningHyperbolicSigma", &All.DivBcleaningHyperbolicSigma, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
#if GADGET_HYDRO != HYDRO_MFM
    setup_parameter("DivBcleaningQ", &All.DivBcleaningQ, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // GADGET_HYDRO != HYDRO_MFM
#endif // DIVBCLEANING_DEDNER

#ifdef MAGNETIC
#ifdef BINISET
    setup_parameter("BiniX", &All.BiniX, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr, default_indicator,
                    id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BiniY", &All.BiniY, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr, default_indicator,
                    id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BiniZ", &All.BiniZ, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr, default_indicator,
                    id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // BINISET

#ifdef MAGNETIC_SN_SEEDING
    setup_parameter("SnSeedRadius", &All.SnSeedRadius, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnSeedBubble", &All.SnSeedBubble, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnSeedField", &All.SnSeedField, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnSeedSoftening", &All.SnSeedSoftening, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // MAGNETIC_SN_SEEDING

#ifdef MAGNETIC_DIFFUSION
    setup_parameter("MagneticEta", &All.MagneticEta, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // MAGNETIC_DIFFUSION
#ifdef MAGNETIC_DIFFUSION_LIMIT
    setup_parameter("MagneticDiffSpeed", &All.MagneticDiffSpeed, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // MAGNETIC_DIFFUSION_LIMIT
#endif // MAGNETIC

    setup_parameter("SpectralCRsOn", &All.SpectralCRsOn, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);


#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
    setup_parameter("Epsilon", &All.Epsilon, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // defined (CHEMISTRY) || defined (UM_CHEMISTRY)
#ifdef SFR
    // the baryon fraction that you expect to end in stars
    setup_parameter("SFfactor", &All.SFfactor, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
    // Number of Generations
    setup_parameter("Generations", &All.Generations, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // SFR

#ifdef LT_STELLAREVOLUTION
#ifdef LT_STOP_COOL_BELOW_Z
    setup_parameter("CoolStop_redshift", &All.CoolStop_redshift, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_STOP_COOL_BELOW_Z

#ifdef LT_TRACK_CONTRIBUTES
    setup_parameter("PackPowerBase", &PowerBase, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_TRACK_CONTRIBUTES

#ifdef LT_POPIII
    setup_parameter("PopIII_IMF_idx", &All.PopIII_IMF_idx, Param_Level::PARAMETER_DEVELOPER, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    /* Z threshold for PIII formation
    in units of solar abundance */
    setup_parameter("PopIII_Zlimit", &All.PopIII_Zlimit, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_POPIII

#ifdef LT_STARBURSTS
    setup_parameter("StarBurstCondition", &All.StarBurstCondition, Param_Level::PARAMETER_NECESSARY, Param_Type::INT,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SB_Density_Thresh", &All.SB_Density_Thresh, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SB_DEntropy_Thresh", &All.SB_DEntropy_Thresh, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_STARBURSTS

#ifdef LT_SMOOTH_Z
#if defined(LT_SMOOTH_SIZE) && !defined(LT_SMOOTH_NGB)
    setup_parameter("SmoothRegionSize", &All.SmoothRegionSize, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SmoothRegionSizeMax", &All.SmoothRegionSizeMax, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // defined(LT_SMOOTH_SIZE) && !defined(LT_SMOOTH_NGB)
#endif // LT_SMOOTH_Z
    setup_parameter("TestSuite", &All.TestSuite, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    /* the minimum number of neighbours used to spread
     * metals from Stars.
     */
    setup_parameter("InfNeighNum", &All.InfNeighNum, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    /* the desired number of neighbours used to spread
     * metals from Stars.
     */
    setup_parameter("DesNumNgbSN", &All.DesNumNgbSN, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    /* the allowed deviation on number of neighbours used
     * to spread metals from Stars.
     */
    setup_parameter("MaxNumNgbDevRelSN", &All.MaxNumNgbDevRelSN, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    /* note: this is used just to calculate All.MaxPartMet in
     * read_file(); then, it should give a reasonable "average"
     * of the generations of each specified IMFs.
     *
     * WARNING: this will be overridden by read_imfs() !!!
     */

#ifndef LT_STELLAREVOLUTION
    /* GM NOTE: with STELLAREVOLUTION this is contained in SFs (parameter Ngen) */
    setup_parameter("Generations", &All.Generations, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_STELLAREVOLUTION
    /* the baryon fraction that you expect to end in stars
     */
    setup_parameter("SFfactor", &All.SFfactor, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("IMFsFileName", All.IMFsFileName, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SFsFileName", All.SFsFileName, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    /* the minimum mass for CC supernovae (usually 8 Msun) */
    setup_parameter("SnII_InfMass", &All.SnII_InfMass, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MinBinSystemMass", &All.MinBinSystemMass, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MaxBinSystemMass", &All.MaxBinSystemMass, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("BinarySystemFrac", &All.BinarySystemFrac, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MinStarMass_inBinSystem", &All.MinStarMass_inBinSystem, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);

#ifdef LT_HOT_EJECTA
    setup_parameter("EgySpecEjecta", &All.EgySpecEjecta, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_HOT_EJECTA

    setup_parameter("SnIaDataFileName", &All.SnIaDataFile, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnIaDataSetNum", &All.SnIaDataSetNum, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnIaEnergy", &All.SnIaEnergy, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("LongLiving_Step_Prec", &All.LongLiving_Step_Prec, Param_Level::PARAMETER_DEVELOPER,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_NEVER);
    setup_parameter("SnIIdataFileName", All.SnIIdataFileName, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnIIDataSetNum", &All.SnIIDataSetNum, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnIIEnergy", &All.SnIIEnergy, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("metIRAThMass", &All.metIRAThMass, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("egyIRAThMass", &All.egyIRAThMass, Param_Level::PARAMETER_ADVANCED, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SnII_Step_Prec", &All.SnII_Step_Prec, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AGBdataFileName", All.AGBdataFileName, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AGBDataSetNum", &All.AGBDataSetNum, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MinChemTimeStep", &All.MinChemTimeStep, Param_Level::PARAMETER_DEVELOPER, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MinSpreadLength", &All.MinSpreadLength, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("MaxSpreadLength", &All.MaxChemSpreadL, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // LT_STELLAREVOLUTION

#ifdef LT_METAL_COOLING_WAL
    setup_parameter("WalCoolTables_Path", All.WalCoolTables_Path, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // LT_METAL_COOLING_WAL

#ifdef LT_ADD_GAL_TO_SUB
    setup_parameter("BC_SED_Path", All.BC_SED_Path, Param_Level::PARAMETER_NECESSARY, Param_Type::STRING, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_ALWAYS);
#endif // LT_ADD_GAL_TO_SUB

#ifdef GENERATE_GAS_IN_ICS
#ifdef GENERATE_GAS_TG
    setup_parameter("GenGasRefFac", &All.GenGasRefFac, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // GENERATE_GAS_TG
#endif // GENERATE_GAS_IN_ICS

#ifdef ADAPTGRAVSOFT
    setup_parameter("AGS_DesNumNgb", &All.AGS_DesNumNgb, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AGS_MaxNumNgbDeviation", &All.AGS_MaxNumNgbDeviation, Param_Level::PARAMETER_NECESSARY,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
#endif // ADAPTGRAVSOFT

#ifdef SPHERICAL_BOUNDARY
    setup_parameter("SB_type", &All.spherical_boundary.type, Param_Level::PARAMETER_NECESSARY, Param_Type::INT, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("SB_radius", &All.spherical_boundary.radius, Param_Level::PARAMETER_NECESSARY, Param_Type::REAL,
                    tag, addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // SPHERICAL_BOUNDARY

#if (GADGET_HYDRO == HYDRO_MFM) && !defined(MFM_CONSERVATIVE_TOTAL_ENERGY)
    setup_parameter("EkinSwitchFraction", &All.EkinSwitchFraction, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 1);
    setup_parameter("EpotSwitchFraction", &All.EpotSwitchFraction, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag,
                    addr, default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_DANGEROUS, 1);
#endif // GADGET_HYDRO == HYDRO_MFM && !defined(MFM_CONSERVATIVE_TOTAL_ENERGY)

#ifdef ANALYTIC_POTENTIALS
    setup_parameter("AP_doHalo", &All.AP_doHalo, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_doBulge", &All.AP_doBulge, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_doThickDisk", &All.AP_doThickDisk, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_doThinDisk", &All.AP_doThinDisk, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_doDisk1", &All.AP_doDisk1, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_doDisk2", &All.AP_doDisk2, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_doKing", &All.AP_doKing, Param_Level::PARAMETER_PUBLIC, Param_Type::INT, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);

    setup_parameter("AP_HaloRho", &All.AP_HaloRho, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_HaloR", &All.AP_HaloR, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_BulgeM", &All.AP_BulgeM, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_BulgeA", &All.AP_BulgeA, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_ThickDiskM", &All.AP_ThickDiskM, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_ThickDiskA", &All.AP_ThickDiskA, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_ThickDiskB", &All.AP_ThickDiskB, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_ThinDiskM", &All.AP_ThinDiskM, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_ThinDiskA", &All.AP_ThinDiskA, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_ThinDiskB", &All.AP_ThinDiskB, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_Disk1M", &All.AP_Disk1M, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_Disk1A", &All.AP_Disk1A, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_Disk1B", &All.AP_Disk1B, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_Disk2M", &All.AP_Disk2M, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_Disk2A", &All.AP_Disk2A, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_Disk2B", &All.AP_Disk2B, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_KingRho0", &All.AP_KingRho0, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
    setup_parameter("AP_KingRc", &All.AP_KingRc, Param_Level::PARAMETER_PUBLIC, Param_Type::REAL, tag, addr,
                    default_indicator, id, nt, parameter_pointers_map, Param_Restart::RESTART_NEVER);
#endif // ANALYTIC_POTENTIALS
#ifdef GT_DUST
    setup_parameter("DeltaVelDissipationConstant", &All.DeltaVelDissipationConstant, Param_Level::PARAMETER_PUBLIC,
                    Param_Type::REAL, tag, addr, default_indicator, id, nt, parameter_pointers_map,
                    Param_Restart::RESTART_ALWAYS);
#endif // GT_DUST
}

/**
 * @brief Checks various setup consistency conditions for the code.
 *
 * This function verifies several setup consistency conditions for the code,
 * such as the power of 2 condition for the number of files written in parallel,
 * the correctness of compilation flags, and the compatibility of various options.
 * If any inconsistency is detected, the function terminates the program.
 */
void check_setup_consistency()
{
    // find the power of 2 that is greater than the number of files to write in parallel
    int pnum;
    for (pnum = 0; All.NumFilesWrittenInParallel > (1 << pnum); pnum++)
        ;

    // check if the number of files to write in parallel is a power of 2
    ENDRUN_IF(All.NumFilesWrittenInParallel != (1 << pnum), "NumFilesWrittenInParallel MUST be a power of 2 ");

    ENDRUN_IF(All.NumFilesWrittenInParallel > NTask,
              "NumFilesWrittenInParallel MUST be smaller than number of processors ");

#ifdef PERIODIC
    ENDRUN_IF(All.PeriodicBoundariesOn == 0, "Code was compiled with periodic boundary conditions switched on.\nYou "
                                             "must set `PeriodicBoundariesOn=1', or recompile the code.");
#else // PERIODIC
    ENDRUN_IF(All.PeriodicBoundariesOn == 1, "Code was compiled with periodic boundary conditions switched off.\nYou "
                                             "must set `PeriodicBoundariesOn=0', or recompile the code.");
#endif // PERIODIC

#ifdef COOLING
    ENDRUN_IF(All.CoolingOn == 0,
              "Code was compiled with cooling switched on.\nYou must set `CoolingOn=1', or recompile the code. ");
#else // COOLING
    ENDRUN_IF(All.CoolingOn == 1,
              "Code was compiled with cooling switched off.\nYou must set `CoolingOn=0', or recompile the code. ");
#endif // COOLING

    ENDRUN_IF(All.TypeOfTimestepCriterion >= 3, "The specified timestep criterion is not valid.");

#if defined(LONG_X) || defined(LONG_Y) || defined(LONG_Z)
#ifndef NOGRAVITY
    ENDRUN("Code was compiled with LONG_X/Y/Z, but not with NOGRAVITY.\nStretched periodic boxes are not "
           "implemented for gravity yet. ");
#endif // NOGRAVITY
#endif // defined(LONG_X) ||  defined(LONG_Y) || defined(LONG_Z)

#ifdef SFR
    ENDRUN_IF(All.StarformationOn == 0, "Code was compiled with star formation switched on.\nYou must set "
                                        "`StarformationOn=1', or recompile the code. ");
    ENDRUN_IF(All.CoolingOn == 0, "You try to use the code with star formation enabled, but you did not switch on "
                                  "cooling. This mode is not supported. ");
#else // SFR
    ENDRUN_IF(
        All.StarformationOn == 1,
        "Code was compiled with star formation switched off. You must set `StarformationOn=0', or recompile the code.");
#endif // SFR

#ifdef RECOMPOSE_DOMAIN_DIFFUSE
    ENDRUN_MAIN("Code was compiled with RECOMPOSE_DOMAIN_DIFFUSE, which is not working properly. We better stop.");
#endif // RECOMPOSE_DOMAIN_DIFFUSE

#ifdef TIMEDEPDE
#ifndef DARKENERGY
    ENDRUN_MAIN("Code was compiled with TIMEDEPDE, but not with DARKENERGY. This is not allowed.");
#endif // DARKENERGY
#endif // TIMEDEPDE

#ifdef ACC
#ifdef ACC_OPENMP
    if constexpr (EXPERT_LEVEL < 42)
    {
#ifdef ACC_GRAVITY_II
        ENDRUN_MAIN("ACC_GRAVITY_II currently does not work with ACC_OPENMP.");
#endif // ACC_GRAVITY_II
#ifdef ACC_CONDUCTION
        ENDRUN_MAIN("ACC_CONDUCTION currently does not work with ACC_OPENMP.");
#endif // ACC_CONDUCTION
    }
#endif // ACC_OPENMP
#endif // ACC

#ifndef MAGNETIC

#ifdef DBOUTPUT
    ENDRUN_MAIN("Code was compiled with DBOUTPUT, but not with MAGNETIC. This is not allowed.");
#endif // DBOUTPUT

#ifdef MU0_UNITY
    ENDRUN_MAIN("Code was compiled with MU0_UNITY, but not with MAGNETIC. This makes no sense.");
#endif // MU0_UNITY

#ifdef ARTIFICIAL_MAGNETIC_DISSIPATION
    ENDRUN_MAIN("Code was compiled with ARTIFICIAL_MAGNETIC_DISSIPATION, but not with MAGNETIC. This makes no sense.");
#endif // ARTIFICIAL_MAGNETIC_DISSIPATION

#ifdef TIME_DEP_MAGN_DISP
    ENDRUN_MAIN(0, "Code was compiled with TIME_DEP_MAGN_DISP, but not with MAGNETIC. This makes no sense.");
#endif // TIME_DEP_MAGN_DISP

#ifdef DIVBCLEANING_TRICCO
    ENDRUN_MAIN(0, "Code was compiled with DIVBCLEANING_TRICCO, but not with MAGNETIC. This makes no sense.");
#endif // DIVBCLEANING_TRICCO

#ifdef DIVBCLEANING_DEDNER
    ENDRUN_MAIN(0, "Code was compiled with DIVBCLEANING_DEDNER, but not with MAGNETIC. This makes no sense.");
#endif // DIVBCLEANING_DEDNER
#endif // MAGNETIC

#ifdef LT_STELLAREVOLUTION

#if (defined(LT_HYDROWINDS) || defined(LT_DECOUPLE_POSTWINDS_FROM_SF)) && !defined(WINDS)
    ENDRUN_MAIN("LT_HYDROWINDS and LT_DECOUPLE_POSTWINDS_FROM_SF require WINDS");
#endif // (defined(LT_HYDROWINDS) || defined(LT_DECOUPLE_POSTWINDS_FROM_SF)) &&
       // !defined(WINDS)
#endif // LT_STELLAREVOLUTION

#ifdef LT_METAL_COOLING
#if defined(LT_METAL_COOLING_on_SMOOTH_Z) && !defined(LT_SMOOTH_Z)
    ENDRUN_MAIN("LT_METAL_COOLING_on_SMOOTH_Z requires LT_SMOOTH_Z  ");
#endif // defined(LT_METAL_COOLING_on_SMOOTH_Z) && !defined(LT_SMOOTH_Z)

#if defined(LT_METAL_COOLING_on_SMOOTH_Z) && defined(LT_METAL_COOLING_WAL) && !defined(LT_SMOOTH_ALLMETALS)
    ENDRUN_MAIN("LT_METAL_COOLING_on_SMOOTH_Z requires LT_SMOOTH_ALLMETALS if LT_METAL_COOLING_WAL is defined  ");
#endif // defined(LT_METAL_COOLING_on_SMOOTH_Z) && defined(LT_METAL_COOLING_WAL) && !defined(LT_SMOOTH_ALLMETALS)

#if defined(NAVIERSTOKES_ANISO) && (!defined(MAGNETIC))
    ENDRUN_MAIN("NAVIERSTOKES_ANISO requires MAGNETIC  ");
#endif // defined(NAVIERSTOKES_ANISO) && (!defined(MAGNETIC))

#if defined(TMG_PLASMA_INST) && (!defined(NAVIERSTOKES_ANISO) || !defined(MAGNETIC))
    ENDRUN_MAIN("TMG_PLASMA_INST requires NAVIERSTOKES_ANISO and MAGNETIC  ");
#endif // defined(TMG_PLASMA_INST) && (!defined(NAVIERSTOKES_ANISO) || !defined(MAGNETIC))

#if defined(CONDUCTION_INCLUDEMAGNETIC) && (!defined(CONDUCTION) || !defined(MAGNETIC))
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC requires CONDUCTION and MAGNETIC  ");
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) && (!defined(CONDUCTION) || !defined(MAGNETIC))

#if defined(CONDUCTION_INCLUDEMAGNETIC_EASY) && !defined(CONDUCTION_INCLUDEMAGNETIC)
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC_EASY requires CONDUCTION_INCLUDEMAGNETIC  ");
#endif // defined(CONDUCTION_INCLUDEMAGNETIC_EASY) && !defined(CONDUCTION_INCLUDEMAGNETIC)

#if defined(CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR) && !defined(CONDUCTION_INCLUDEMAGNETIC)
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR requires CONDUCTION_INCLUDEMAGNETIC  ");
#endif // defined(CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR) && !defined(CONDUCTION_INCLUDEMAGNETIC)

#if defined(CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED) && !defined(CONDUCTION_INCLUDEMAGNETIC)
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED requires CONDUCTION_INCLUDEMAGNETIC  ");
#endif //  defined(CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED) && !defined(CONDUCTION_INCLUDEMAGNETIC)

#if defined(CONDUCTION_BICGSTAB) && !defined(CONDUCTION_INCLUDEMAGNETIC)
    ENDRUN_MAIN("CONDUCTION_BICGSTAB requires CONDUCTION_INCLUDEMAGNETIC  ");
#endif // defined(CONDUCTION_BICGSTAB) && !defined(CONDUCTION_INCLUDEMAGNETIC)

#if defined(CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE) && !defined(CONDUCTION_INCLUDEMAGNETIC)
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE requires CONDUCTION_INCLUDEMAGNETIC  ");
#endif // defined(CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE) && !defined(CONDUCTION_INCLUDEMAGNETIC)

#if defined(CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR) && defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR can not be used with CONDUCTION_INCLUDEMAGNETIC_EASY  ");
#endif // defined(CONDUCTION_INCLUDEMAGNETIC_PERPENDICULAR) && defined(CONDUCTION_INCLUDEMAGNETIC_EASY)

#if defined(CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED) && defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED can not be used with CONDUCTION_INCLUDEMAGNETIC_EASY  ");
#endif // defined(CONDUCTION_INCLUDEMAGNETIC_ISOTROPISED) && defined(CONDUCTION_INCLUDEMAGNETIC_EASY)

#if defined(CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE) && defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
    ENDRUN_MAIN("CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE can not be used with CONDUCTION_INCLUDEMAGNETIC_EASY  ");
#endif // defined(CONDUCTION_INCLUDEMAGNETIC_CHECK_ANGLE) && defined(CONDUCTION_INCLUDEMAGNETIC_EASY)
#endif // LT_METAL_COOLING

#if defined(KD_MAIN_HALO) & !defined(SUBFIND)
    ENDRUN_MAIN("KD_MAIN_HALO needs SUBFIND to be defined");
#endif // defined(KD_MAIN_HALO) & !defined(SUBFIND)

    if (ThisTask == 0)
    {
        VERBOSE(2, "BegRun: Check BlackHoleSetting ... ");
#ifdef BLACK_HOLES
        check_and_report_settings_for_blackhole();
#endif // BLACK_HOLES
#ifdef mSIDM
        msidm.check_and_report_settings();
#endif // mSIDM
    }
}

/**
 * @brief Checks if the system's data types meet the required sizes for the simulation.
 *
 * This function verifies if the fundamental data types (long long, int, float, double)
 * have the expected sizes for the simulation. If any of the types do not match the
 * required sizes (64-bit for long long and double, 32-bit for int and float), it
 * terminates the program.
 */
void check_system()
{
    ENDRUN_IF(sizeof(long long) != 8, "Type `long long' is not 64 bit on this platform. Stopping.");

    ENDRUN_IF(sizeof(int) != 4, "Type `int' is not 32 bit on this platform. Stopping.");

    ENDRUN_IF(sizeof(float) != 4, "Type `float' is not 32 bit on this platform. Stopping.");

    ENDRUN_IF(sizeof(double) != 8, "Type `double' is not 64 bit on this platform. Stopping.");
}

/**
 * @brief Check settings from parameter file
 *
 * This function performs some sanity checks on the parameters read from the parameter file.
 * It prints warnings in case when parameters are e.g. missing and sets the errorFlag to 1
 * for further error handling, which will terminate the code.
 *
 * @param errorFlag the error flag indicating error type
 */
void check_parameters(int &errorFlag)
{

    // TimerReportBalance indexes into structures sized by TimerReportLevel, so it must be smaller
    // Setting it to a TimerReportLevel or 2 automatically
    if (All.TimerReportBalance > All.TimerReportLevel)
    {
        VERBOSE_ALL(0,
                    "WARNING: TimerReportBalance (%d) is larger than TimerReportLevel (%d). \n Setting "
                    "TimerReportBalance = 2 or TimerReportBalance = TimerReportLevel if TimerReportLevel < 2",
                    All.TimerReportBalance, All.TimerReportLevel);

        if (All.TimerReportLevel > 2)
        {
            All.TimerReportBalance = 2;
        }
        else
        {
            All.TimerReportBalance = All.TimerReportLevel;
        }
    }

    const double dt_max_simulation =
        (All.ComovingIntegrationOn ? log(All.TimeMax / All.TimeBegin) : All.TimeMax - All.TimeBegin);
    // ensure the MaxSizeTimestep value is not too large
    if (All.MaxSizeTimestep > dt_max_simulation)
    {
        if constexpr (EXPERT_LEVEL < 42)
        {
            // terminate program in case MaxSizeTimestep is too large.
            VERBOSE(0, "ERROR: MaxSizeTimestep is larger than total time of simulation (%g).", dt_max_simulation);
            errorFlag = 1;
        }
        else
        {
            // just overwrite the MaxSizeTimestep variable and output a warning.
            VERBOSE(0,
                    "WARNING: MaxSizeTimestep is larger than total time of simulation. As you are running at "
                    "sufficiently high "
                    "EXPERT_LEVEL, the variable will be overwritten with the maximum simulation time: "
                    "All.MaxSizeTimestep(old)=%g to All.MaxSizeTimestep(new)=%g ",
                    All.MaxSizeTimestep, dt_max_simulation);
            All.MaxSizeTimestep = dt_max_simulation;
        }
    }
}

/**
 * @brief This function prints references to the activated modules
 *
 * This function should contain all references to all physics modules that are used in the code.
 * The references should be added to all modules used in the code here.
 */
void print_references()
{
    // add the references to the modules used in the code here:

    VERBOSE(0, " ");
    VERBOSE(0, "#################################################################################");
    VERBOSE(0, "#################################################################################");
    VERBOSE(0, "#################################################################################");
    VERBOSE(0, "##  This is OpenGadget3, which includes the work of various people.              ");
    VERBOSE(0, "##  When referring to this code, please typeset its name as \\textsc{OpenGadget3}.");
    VERBOSE(0, "##  Therefore, please always give the proper credits as listed below !           ");
    VERBOSE(0, "##");

    // General Code Infrastructure
    VERBOSE(0, "##  OG3: This is OpenGadget3 (refer to OpenGadget3 release paper, in prep)");
    VERBOSE(0, "##  OG3: Using various infrastructure and computations as introduced in Gadget2 (refer to Springel "
               "2005, MNRAS, 364, 1105, https://ui.adsabs.harvard.edu/abs/2005MNRAS.364.1105S/abstract");
    VERBOSE(0, "##  OG3: Using new GreenTREE and new Domain Decomposition (refer to Ragagnin et al. 2016, ParCo 2016, "
               "https://ui.adsabs.harvard.edu/abs/2016pcre.conf..411R/abstract)");
    VERBOSE(0, "##  SPH: Many improvements for the SPH part (summarized in Beck et al. 2016, MNRAS, 455, 2110, "
               "https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.2110B/abstract)");

    // Improvements in Hydrodynamics
    if (WAKEUP > 0)
    {
        VERBOSE(0, "##  SPH: Using particle wakeup (refer to Pakmor et al. 2012, MNARS, 424, 2222, "
                   "https://ui.adsabs.harvard.edu/abs/2012MNRAS.424.2222P/abstract)");
    }

#if defined(STANDARD_KERNEL)
    VERBOSE(0, "##  SPH: Using cubic spline kernel for SPH: (see to Monaghan & Lattanzio 1985, A&A, 149, 135, "
               "https://ui.adsabs.harvard.edu/abs/1985A%%26A...149..135M/abstract)");
#elif defined(QUINTIC_KERNEL)
    VERBOSE(0, "##  SPH: Using quintic spline kernel for SPH: (see to Morris 1996, PASA, 13, 97, "
               "doi.org/10.1017/S1323358000020610)");
#elif defined(WENDLAND_C2_KERNEL) || defined(WENDLAND_C4_KERNEL) || defined(WENDLAND_C6_KERNEL)
    VERBOSE(0, "##  SPH: Using Wendland C2/4 or C6 kernel for SPH: (see to Dehnen & Aly 2012, MNRAS, 425, 1068, "
               "https://academic.oup.com/mnras/article/425/2/1068/1187211)");
    VERBOSE(0, "##  SPH:         and for implementation in Gadget: (refer to Donnert et al. 2013, MNRAS, 429,3564, "
               "https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.3564D/abstract)");
    VERBOSE(0, "##  SPH:         and for the original functions: (refer to Wendland 1995, Adv Comput Math, 4, 389, "
               "https://doi.org/10.1007/BF02123482)");
#elif defined(WENDLAND_C8_KERNEL)
    VERBOSE(0, "##  SPH: Using Wendland C8 kernel for SPH. (see Wendland 2009, SIAM, 47, 4, "
               "https://doi.org/10.1137/080730299)");
    VERBOSE(0, "##  SPH: and for implementation in Gadget: (refer to Kummer et al. 2019, MNRAS, 487,354, "
               "https://ui.adsabs.harvard.edu/abs/2019MNRAS.487..354K/abstract)");
    VERBOSE(0, "##  SPH:         and for the original functions: (refer to Wendland 1995, Adv Comput Math, 4, 389, "
               "https://doi.org/10.1007/BF02123482)");
#endif // *_KERNEL

#ifdef TIME_DEP_ART_VISC
#if (TIME_DEP_ART_VISC == 0)
    VERBOSE(0, "##  SPH: Using time dependent artificial viscosity (see Morris & Monaghan 1997, JCoPh, 136, 41, "
               "https://ui.adsabs.harvard.edu/abs/1997JCoPh.136...41M/abstract)");
    VERBOSE(0, "##  SPH:         and for implementation in Gadget: (refer to Dolag et al. 2005, MNRAS, 364, 753, "
               "https://ui.adsabs.harvard.edu/abs/2005MNRAS.364..753D/abstract)");
#endif // TIME_DEP_ART_VISC == 0
#if (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
    VERBOSE(0, "##  SPH:                                  and for implementation in Gadget: (refer to Beck et al. "
               "2016, MNRAS, 455, 2110, https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.2110B/abstract)");
#endif // (TIME_DEP_ART_VISC == 1) || (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#if (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
    VERBOSE(0, "##  SPH:                                  and extensions as described in the OpenGadget3 presentation "
               "paper (in prep)");
#endif // (TIME_DEP_ART_VISC == 2) || (TIME_DEP_ART_VISC == 3)
#endif // TIME_DEP_ART_VISC

    VERBOSE(0, "##  SPH: Using reduced artificial viscosity based on second order gradients (see Price 2012, MNARS, "
               "420, 33, https://ui.adsabs.harvard.edu/abs/2012MNRAS.420L..33P/abstract)");
    VERBOSE(0, "##  SPH:                                  and for implementation in Gadget: (refer to Beck et al. "
               "2016, MNRAS, 455, 2110, https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.2110B/abstract)");

#ifdef ARTIFICIAL_CONDUCTIVITY
    VERBOSE(0, "##  SPH: Using time dependent artificial conductivity (see Price 2012, MNARS, 420, 33, "
               "https://ui.adsabs.harvard.edu/abs/2012MNRAS.420L..33P/abstract)");
    VERBOSE(0, "##  SPH:            and for implementation in Gadget: (refer to Beck et al. 2016, MNRAS, 455, 2110, "
               "https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.2110B/abstract)");
#endif // ARTIFICIAL_CONDUCTIVITY

#if GADGET_HYDRO == HYDRO_MFM
    VERBOSE(0, "##  MFM: The Meshless Finite Mass Hydro Solver is used instead of SPH. (refer to Groth et al. 2023, "
               "MNRAS, 526, 616, https://doi.org/10.1093/mnras/stad2717)");
#endif // GADGET_HYDRO == HYDRO_MFM

    // Additional Physics

    // Viscosity
#ifdef NAVIERSTOKES
    VERBOSE(0, "##  VISC: Using Physical Viscosity solving Navierstokes (see Sijacki & Springel 2006, MNRAS, 371, "
               "1025, https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1025S/abstract");
    VERBOSE(0, "##  VISC:                and for current implementation (refer to Marin-Gilabert et al. 2022, MNRAS, "
               "517, 5971, https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.5971M/abstract");
#endif // NAVIERSTOKES

    // Cosmology
#ifdef DARKENERGY
    VERBOSE(0, "##  DE: Using extended dark energy models (refer to Dolag et al. 2004, A&A, 416, 853, "
               "https://ui.adsabs.harvard.edu/abs/2004A&A...416..853D/abstract)");
#endif // DARKENERGY

#ifdef NEUTRINOS
    VERBOSE(0, "##  Nu: Using the treatment of NEUTRINOS (see Viel et al. 2010, JCAP, 06, 015, "
               "https://ui.adsabs.harvard.edu/abs/2010JCAP...06..015V/abstract");
    VERBOSE(0, "##  Nu:   and for current implementation (refer to Carbone et al. 2016, JCAP, 07, 034, "
               "https://ui.adsabs.harvard.edu/abs/2016JCAP...07..034C/abstract)");
#endif // NEUTRINOS

    // Self-interacting dark matter
#ifdef mSIDM
    msidm.report_credits();
#endif // mSIDM

    // Interacting dark matter

    // Cooling
#ifdef COOLING
#if defined(LT_METAL_COOLING)
    VERBOSE(0, "##  COOL: Using metalicity depending coolung based on SD tables (see Sutherland & Dopita 1993, ApjS, "
               "88, 253, https://ui.adsabs.harvard.edu/abs/1993ApJS...88..253S/abstract)");
    VERBOSE(0, "##  COOL:                     and for implementation in Gadget: (refere to Tornatore et 1l. 2007, "
               "https://ui.adsabs.harvard.edu/abs/2007MNRAS.382.1050T/abstract)");
#elif defined(LT_METAL_COOLING_WAL)
    VERBOSE(0, "##  COOL/LT: Using metal based coolig based on cloudy tables (see Wiersma et al. 2009, MNRAS, 399, "
               "574, https://ui.adsabs.harvard.edu/abs/2009MNRAS.399..574W/abstract)");
    VERBOSE(0, "##  COOL/LT:               and for implementation in Gadget: (refer to Tornatore et al. 2010, MNRAS, "
               "402, 1911, https://ui.adsabs.harvard.edu/abs/2010MNRAS.402.1911T/abstract");
#else // LT_METAL_COOLING_WAL
    VERBOSE(0, "##  COOL/LT: Using primordial composition cooling (see Katz et al. 1996, ApJS, 105, 19, "
               "https://ui.adsabs.harvard.edu/abs/1996ApJS..105...19K/abstract)");
    VERBOSE(0, "##  COOL/LT:     and for implementation in Gadget (refer to Springel 2005, MNRAS, 364, 1105, "
               "https://ui.adsabs.harvard.edu/abs/2005MNRAS.364.1105S/abstract");
#endif // cooling branches
#endif // COOLING

    // Molecular Network
#ifdef UM_CHEMISTRY
    VERBOSE(0, "##  NonEQ/LT: non-equilibrium chemistry with full cooling, star formation, additional species and "
               "metallicity evolution (refer to Maio et al. 2007, MNRAS, 379, 963, "
               "https://ui.adsabs.harvard.edu/abs/2007MNRAS.379..963M/abstract)");
    VERBOSE(0, "##  NonEQ/LT: metal cooling in non equilibrium chemistry (refer to Maio et al. 2010, MNRAS, 407, 1003, "
               "https://ui.adsabs.harvard.edu/abs/2010MNRAS.407.1003M/abstract)");
    VERBOSE(0, "##  NonEQ: modified UV background (refer to Maio et al. 2022, A&A, 657, A47, "
               "https://ui.adsabs.harvard.edu/abs/2022A&A...657A..47M/abstract)");
#endif // UM_CHEMISTRY
#ifdef UM_METAL_COOLING
    VERBOSE(0, "##  NonEQ/LT: metal cooling in non-equilibrium chemistry (refer to Maio et al. 2007, MNRAS, 379, 963, "
               "https://ui.adsabs.harvard.edu/abs/2007MNRAS.379..963M/abstract)");
    VERBOSE(0, "##  NonEQ/LT: metal cooling in non-equilibrium chemistry (refer to Maio et al. 2010, MNRAS, 407, 1003, "
               "https://ui.adsabs.harvard.edu/abs/2010MNRAS.407.1003M/abstract)");
#endif // UM_METAL_COOLING
#ifndef DONOT_EXCLUDE_WINDS_FROM_COOLING // UM wind decoupling
    VERBOSE(0, "##  Winds: cooling handling/decoupling (refer to Maio et al. 2011, MNRAS, 414, 1145, "
               "https://ui.adsabs.harvard.edu/abs/2011MNRAS.414.1145M/abstract)");
#endif // DONOT_EXCLUDE_WINDS_FROM_COOLING

    // Starformation
#ifdef SFR
    VERBOSE(0, "##  SFR: Using multiphase model based on SH2003 (refer to Springel & Hernquist 2003, MNRAS, 339, 2896, "
               "https://ui.adsabs.harvard.edu/abs/2003MNRAS.339..289S/abstract)");
#ifdef LT_STELLAREVOLUTION
    VERBOSE(0, "##  LT: Using stellar evolution model (refer to Tornatore al. 2004, MNRAS, 349, 19, "
               "https://ui.adsabs.harvard.edu/abs/2004MNRAS.349L..19T/abstract");
    VERBOSE(0, "##  LT:                               and Tornatore et al. 2007, MNRAS, 382, 1050, "
               "https://ui.adsabs.harvard.edu/abs/2007MNRAS.382.1050T/abstract)");
#ifdef LT_POPIII
    VERBOSE(0, "##  LT: Using individual star formation modes for transitiuon POP III to POP II (refer to Tornatore et "
               "al. 2007, MNRAS, 382, 945, https://ui.adsabs.harvard.edu/abs/2007MNRAS.382..945T/abstract)");
#endif // LT_POPIII
#ifdef GL_CR_DUST
    VERBOSE(0, "##  LT/DUST: Using dust treatment with two-size distributions (refer to Granato et al 2021, MNRAS, "
               "503, 511, https://ui.adsabs.harvard.edu/abs/2021MNRAS.503..511G/abstract)");
#endif // GL_CR_DUST
#endif // LT_STELLAREVOLUTION
#endif // SFR

#ifdef CS_DECOUPLING
    VERBOSE(0, "##  MULTIPHASE/CS: Multiphase gas treatment (refer to Scannapieco et al. 2006, MNRAS, 371, "
               "1125, https://ui.adsabs.harvard.edu/abs/2006MNRAS.371.1125S/abstract)");
#endif // CS_DECOUPLING

    // BlackHoles

#ifdef BLACK_HOLES
    report_credits_for_blackhole();
#endif // BLACK_HOLES

    // Shock Capturing

#ifdef SHOCKFINDER_SPH
    VERBOSE(0, "##  SHOCK: Using SPH-style on-the-fly shock finder (refer to Beck et al. 2016, MNRAS 458, 2080, "
               "https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.2080B/abstract)");
#ifdef MAGNETIC
    VERBOSE(0, "##  SHOCK: Using on the fly calculation of Alfvén Mach number (refer to Böss et al. 2023, MNRAS 519, "
               "548, https://doi.org/10.1093/mnras/stac3584)");
    VERBOSE(0, "##  SHOCK: Using on the fly calculation shock obliquity (refer to Böss et al. 2023, MNRAS 519, 548, "
               "https://doi.org/10.1093/mnras/stac3584)");
#endif // MAGNETIC
#endif // SHOCKFINDER_SPH


    // Spectral Cosmic Rays


    // Dynamical Dust
#ifdef GT_DUST
    VERBOSE(0,
            "##  DUSTFLUID: Following Dust dynamics in one fluid aproximation (refer to Tedeschi et al. 2025, subm)");
#endif // GT_DUST

    // Conduction
#ifdef CONDUCTION
    VERBOSE(0, "##  COND: Using conduction (refer to Jubelgas et al. 2004, MNRAS, 351, 423, "
               "https://ui.adsabs.harvard.edu/abs/2004MNRAS.351..423J/abstract");
    VERBOSE(0, "##  COND:                  and also to Dolag et al. 2004, ApJ, 606, 97, "
               "https://ui.adsabs.harvard.edu/abs/2004ApJ...606L..97D/abstract)");
    VERBOSE(0, "##  COND:   Note: actual implementation bases on CG solver (see Petkova & Springel 2009, MNRAS, 396, "
               "1383, https://ui.adsabs.harvard.edu/abs/2009MNRAS.396.1383P/abstract)");
#endif // CONDUCTION

    // Magneto Hydro dynamics

#ifdef MAGNETIC
    VERBOSE(0, "##  MHD: Using magneto hydrodynamics (refer to Dolag et al. 1999, A&A, 348, 351, "
               "https://ui.adsabs.harvard.edu/abs/1999A&A...348..351D/abstract)");
    VERBOSE(0, "##  MHD: Using removal of divB force on Maxwelltensor (refer to Dolag & Stasyszyn 2009, MNRAS, 398, "
               "1678, https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1678D/abstract)");

#ifdef DIVBCLEANING_TRICCO
    VERBOSE(0, "##  MHD: Using hyperbolic divergence cleaning for MHD (see Tricco et al. 2016, "
               "https://www.sciencedirect.com/science/article/abs/pii/S0021999116302789)");
    VERBOSE(0, "##  MHD:            and for implementation in Gadget: (refer to Steinwandel et al. 2024: in prep)");
#endif // DIVBCLEANING_TRICCO

#ifdef DIVBCLEANING_DEDNER
    VERBOSE(0, "##  MHD: Using Dedner divergence cleaning for MHD (see Dedner et al. 2002, "
               "https://www.sciencedirect.com/science/article/abs/pii/S002199910196961X)");
    VERBOSE(0, "##  MHD:        and for implementation in Gadget: (refer to Stasyszyn et al. 2013: MNRAS, 428, 13,  "
               "https://ui.adsabs.harvard.edu/abs/2013MNRAS.428...13S/abstract)");
#endif // DIVBCLEANING_DEDNER

#ifdef MAGNETIC_DIFFUSION
    VERBOSE(0, "##  MHD: Using non ideal MHD (refer to Bonafede et al. 2011, MNRAS, 418, 2234, "
               "https://ui.adsabs.harvard.edu/abs/2011MNRAS.418.2234B/abstract)");
#endif // MAGNETIC_DIFFUSION

#if defined(SFR) && defined(MAGNETIC_SN_SEEDING)
    VERBOSE(0, "##  MHD: Using magnetic seeding by supernovae (refer to Beck et al. 2013, MNRAS, 435, 3575, "
               "https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.3575B/abstract)");
#endif // SFR && MAGNETIC_SN_SEEDING


#if defined(CONDUCTION) && defined(CONDUCTION_BICGSTAB)
    VERBOSE(0, "##  COND/MHD: Using anisotropic conduction (refer to Arth et al. 2014, "
               "https://ui.adsabs.harvard.edu/abs/2014arXiv1412.6533A/abstract)");
#endif // CONDUCTION && CONDUCTION_BICGSTAB

#endif // MAGNETIC

#ifdef SUBFIND
    VERBOSE(0, "##  SUBFIND: Using SUBFIND for detection of haloes and sub-halos (refer to Springel et al. 2001, "
               "MNRAS, 328, 726, https://ui.adsabs.harvard.edu/abs/2001MNRAS.328..726S/abstract");
    VERBOSE(0, "##  SUBFIND:     as well as for extension to presence of baryons (refer to Dolag et al. 2009, MNRAS, "
               "399, 497, https://ui.adsabs.harvard.edu/abs/2009MNRAS.399..497D/abstract)");
#endif // SUBFIND

    VERBOSE(0, "##");
    VERBOSE(0, "################################################################################");
    VERBOSE(0, "################################################################################");
}

/**
 * @brief Read parameters from parameter file
 *
 * This function reads the parameter file line by line. It checks if the parameter is in the tag array and if so,
 * sets the value of the parameter to the value in the parameter file.
 * If the parameter is not in the tag array, it is ignored.
 * If the parameter is set to be neccessary, the code will stop if the parameter is not found in the parameter file.
 * If the user doesnt have a high enough EXPERT_LEVEL, the code will prevent the user from using the corresponding
 * parameter.
 *
 * @param fname the name of the parameter file
 * @param tag the array of tags for the parameters
 * @param addr the array of addresses for the parameters
 * @param id the array of ids for the parameters
 * @param nt the number of tags
 * @param default_indicator the array of default indicators for the parameters, to check if parameter is
 * necessary/advanced/expert-level
 * @param parameter_pointers_map a map of pointers to each parameter value
 */
void read_parameter_file(char *fname, char *tag[], void **addr, int *id, int nt, int *default_indicator,
                         std::multimap<std::string, ParamInfo> &parameter_pointers_map)
{
    int MAXTAGS = 300;

    FILE *fd_parameterfile, *fdout_paramusedvalues;
    int included = tag != NULL; // set included to false if tag is not pointing to valid adress (e.g. is empty)

    // allocate memory for tag, addr, default_indicator and id if not "included"
    if (!included)
    {
        int MAXTAGLEN = 50;
        tag = (char **)malloc(MAXTAGS * sizeof(char *));
        addr = (void **)malloc(MAXTAGS * sizeof(void *));
        default_indicator = (int *)malloc(MAXTAGS * sizeof(int));
        id = (int *)malloc(MAXTAGS * sizeof(int));
        for (int i = 0; i < MAXTAGS; i++)
        {
            tag[i] = (char *)malloc(MAXTAGLEN * sizeof(char));
        }
    }

    // check system types and exit if they dont agree with the code
    check_system();

    // set errorFlag to 0 to do full readin before breaking due to error
    int errorFlag = 0;

    // set up parameters only on process 0
    if (ThisTask == 0)
    {
        // check if parameters have to be read from parameter file or not (e.g. restart)
        if (!included)
        {
            nt = 0;
            setup_all_parameters(tag, addr, default_indicator, id, nt, parameter_pointers_map);
        } // end of seting up parameters

        /* if we are including a parameter file , we need the parent tag array because
            the code set the tag[i]==0 in order to check if the tag was assigned or not. */

        // open parameter file
        if ((fd_parameterfile = fopen(fname, "r")))
        {
            // write name of parameterfile-usedvalues file to buffer
            char paramused_fname[200];
            sprintf(paramused_fname, "%s%s", fname, "-usedvalues");
            // open parameterfile--usedvalues file
            if (!(fdout_paramusedvalues = fopen(paramused_fname, "w")))
            {
                VERBOSE_ALL(0, "error opening file '%s'  ", paramused_fname);
                errorFlag = 1;
            }
            else
            {
                VERBOSE_ALL(0, "\nObtaining parameters from file '%s': ", fname);
                int overwrite_flag = 0;

                // until end of parameterfile do...
                while (!feof(fd_parameterfile))
                {
                    // Set buffers for readin of parameter names and values with empty variables.
                    char buf[200], buf1[200], buf2[200], buf3[400];
                    *buf = 0;
                    // Read line from parameterfile
                    if (fgets(buf, 200, fd_parameterfile) == NULL)
                    {
                        // Handle error or end-of-file condition
                        VERBOSE(3, "Empty line in parameter file %s", fname);
                    }

                    // If there are less than two items, continue to next line.
                    if (sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
                    {
                        continue;
                    }

                    // If first item indicates a comment, continue to next line.
                    if ((buf1[0] == '%') || (buf1[0] == '#'))
                    {
                        continue;
                    }

                    // Iterate through all tags and check if buf1 is equal to any tag.
                    int j = -1;
                    for (int i = 0; i < nt; i++)
                    {
                        // If tag is Include, include additional file.
                        if (strcmp("Include", buf1) == 0)
                        {
                            char include_file_name[MAX_PATH_LENGTH];
                            sprintf(include_file_name, "%s/%s", dirname(strdup(fname)), buf2);

                            ENDRUN_IF(strlen(include_file_name) >= MAX_PATH_LENGTH,
                                      "Inclusion path exeeded limit of %d chars: %s ", MAX_PATH_LENGTH,
                                      include_file_name);

                            // recursively call read_parameter_file - with valid tag
                            read_parameter_file(include_file_name, tag, addr, id, nt, default_indicator,
                                                parameter_pointers_map);

                            j = -2; // j=-2 signals that we didn't find any parameter because we are including
                                    // additional files
                            break;
                        }
                        else if (strcmp(buf1, tag[i]) == 0) // if buf is equal to tag set j to i
                        {
                            j = i;
                            tag[i][0] = 0;
                            break;
                        }
                    }

                    // If j is not negative (i.e. if tag was found in tag array), do...
                    if (j >= 0)
                    {
                        // Check if  parameter is allowed to be overwritten.
                        if (default_indicator[j] <= EXPERT_LEVEL)
                        {
                            char oldstr[200], newstr[200];
                            int changed = 0;

                            // capture old value, detect change, assign, capture new value
                            switch (id[j])
                            {
                            case 1: /* double */
                                snprintf(oldstr, sizeof(oldstr), "%g", *((double *)addr[j]));
                                changed = (atof(buf2) != *((double *)addr[j]));
                                *((double *)addr[j]) = atof(buf2);
                                snprintf(newstr, sizeof(newstr), "%g", *((double *)addr[j]));
                                break;
                            case 2: /* string */
                                snprintf(oldstr, sizeof(oldstr), "%s", (char *)addr[j]);
                                changed = (strcmp(buf2, (char *)addr[j]) != 0);
                                strcpy((char *)addr[j], buf2);
                                snprintf(newstr, sizeof(newstr), "%s", buf2);
                                break;
                            case 3: /* int */
                                snprintf(oldstr, sizeof(oldstr), "%d", *((int *)addr[j]));
                                changed = (atoi(buf2) != *((int *)addr[j]));
                                *((int *)addr[j]) = atoi(buf2);
                                snprintf(newstr, sizeof(newstr), "%d", *((int *)addr[j]));
                                break;
                            }

                            // warning + output, identical for all types
                            // check if parameter not neccessary
                            if (default_indicator[j] != static_cast<int>(Param_Level::PARAMETER_NECESSARY))
                            {
                                if (!changed)
                                {
                                    VERBOSE(0,
                                            "Warning: you are overwriting %s with its default value of %s. Please "
                                            "remove parameter from parameterfile.",
                                            buf1, oldstr);
                                }
                                else
                                {
                                    // check if parameter not public
                                    if (default_indicator[j] != static_cast<int>(Param_Level::PARAMETER_PUBLIC))
                                    {
                                        VERBOSE(0, "Warning: overwriting default parameter: Default value of %s is %s",
                                                buf1, oldstr);
                                    }
                                }
                            }

                            fprintf(fdout_paramusedvalues, "%-35s%s\n", buf1, newstr);
                            fprintf(stdout, "%-35s%s\n", buf1, newstr);
                        }
                        else
                        {
                            // User wants to overwrite parameter, but hasn't set expert-level high enough.
                            VERBOSE_ALL(0, "ERROR: trying to overwrite default value of '%s'", buf1);
                            overwrite_flag = 1;
                        }
                    }
                    else if (j !=
                             -2) // in case of j==-2 an additional parameter file was included, and there we do nothing
                    {
                        VERBOSE(0, "begrun: ERROR: param file %s:   Tag '%s' not allowed or multiple defined.\n", fname,
                                buf1);
                        errorFlag = 2;
                    }
                }
                // close files
                fclose(fd_parameterfile);
                fclose(fdout_paramusedvalues);

                // check if all necessary parameters are set
                for (int i = 0; i < nt; i++)
                {
                    if (*tag[i] && (default_indicator[i] == -1))
                    {
                        VERBOSE(0, "Error. I miss a value for tag '%s' in parameter file '%s'. ", tag[i], fname);
                        errorFlag = 1;
                    }
                }

                // exit code if user wants to overwrite parameter, but hasnt set expert level high enough
                ENDRUN_IF(overwrite_flag == 1,
                          "ERROR in reading parameter file. User aimed to overwrite default parameters. Increase "
                          "EXPERT_LEVEL to allow overwriting of parameter values.");

                int i = strlen(All.OutputDir);
                if (i > 0)
                {
                    // Add forward slash to output directory if not already present
                    if (All.OutputDir[i - 1] != '/')
                    {
                        strcat(All.OutputDir, "/");
                    }
                }
            }
        }
        else
        {
            ENDRUN("Parameter file %s not found.", fname);
        }

        if (included)
        {
            return; // we have included a parameter file, our job ends here!
        }

        // check if parameters and config switches are consistent
        check_parameters(errorFlag);

        // print references to active modules
        print_references();

        if (All.OutputListOn)
        {
            int res_outputlist = read_outputlist(All.OutputListFilename);
            ENDRUN_IF(res_outputlist != 0, "Unable to process '%s'", All.OutputListFilename);
        }
        else
        {
            All.OutputListLength = 0;
        }
    }

    // ENDRUN if error was raised during reading of parameter file
    ENDRUN_IF(errorFlag == 1 || errorFlag == 2, " Problems in parameter file detected.");

    // communicate the relevant parameters to the other processes
    MPI_Bcast(&All, sizeof(struct global_data_all_processes), MPI_BYTE, 0, MYMPI_COMM_WORLD);

    // check if certain combinations of config switches are reasonable or if the code should stop
    check_setup_consistency();

    // free memory
    free(addr);
    free(id);
    for (int i = 0; i < MAXTAGS; i++)
    {
        free(tag[i]);
    }
    free(tag);
    free(default_indicator);
}

/**
 * @brief Print current parameters
 *
 * This function prints the parameters saved in the parameter_pointer_map to console and
 * the parameter-usedvalues file. It iterates through the map and prints the line, while
 * accounting for the different types of parameters.
 *
 * @param parameter_pointers_map A multimap containing the pointers to the parameters and their types
 */
void print_current_parameters(std::multimap<std::string, ParamInfo> parameter_pointers_map)
{
    // Only run on main task
    if (ThisTask == 0)
    {
        // Open file for writing
        std::ofstream fdparameterused("parameters-usedvalues");

        // Check if the file was opened successfully
        if (!fdparameterused.is_open())
        {
            std::cerr << "Failed to open parameter-usedvalues file for writing." << std::endl;
        }

        std::cout << "\nFinal Parameter values:" << std::endl;
        // Iterate through the map and print the parameters to console and file.
        for (const auto &pair : parameter_pointers_map)
        {
            std::cout << std::left << std::setw(35) << pair.first;
            fdparameterused << std::left << std::setw(35) << pair.first;

            std::string paramName = pair.first;
            void *paramAddr = pair.second.addr;
            int parameterType = pair.second.type;

            // Print value of key depending on type.
            if (parameterType == static_cast<int>(Param_Type::STRING))
            {
                std::cout << static_cast<char *>(paramAddr);
                fdparameterused << static_cast<char *>(paramAddr);
            }
            else if (parameterType == static_cast<int>(Param_Type::INT))
            {
                std::cout << *static_cast<int *>(paramAddr);
                fdparameterused << *static_cast<int *>(paramAddr);
            }
            else if (parameterType == static_cast<int>(Param_Type::REAL))
            {
                std::cout << *static_cast<double *>(paramAddr);
                fdparameterused << *static_cast<double *>(paramAddr);
            }
            std::cout << std::endl;
            fdparameterused << std::endl;
        }

        // Close file
        fdparameterused.close();
    }
}

/**
 * @brief Read output list from file
 *
 * This function reads a table with a list of desired output times. The table
 * does not have to be ordered in any way, but may not contain more than
 * MAXLEN_OUTPUTLIST entries.
 *
 * @param fname The name of the file containing the output list
 */
int read_outputlist(char *fname)
{
    FILE *fd;
    int flag;
    char buf[512];
    // Open the output file for reading
    if (!(fd = fopen(fname, "r")))
    {
        VERBOSE_ALL(0, "can't read output list in file '%s' ", fname);
        return 1;
    }
    // Initialize the output list length
    All.OutputListLength = 0;

    // Read the output list entries from the file
    while (1)
    {
        // Read a line from the file into the buffer
        if (fgets(buf, 500, fd) != buf)
        {
            // Break if unable to do so
            break;
        }

        int count = sscanf(buf, " %lg %d ", &All.OutputListTimes[All.OutputListLength], &flag);

        // If only one value is read, set the flag to 1
        if (count == 1)
        {
            flag = 1;
        }

        // Check the count to ensure valid entry format
        if (count == 1 || count == 2)
        {
            // Check if the output list is full
            ENDRUN_IF(All.OutputListLength >= MAXLEN_OUTPUTLIST,
                      "Too many entries in output-list. You should increase MAXLEN_OUTPUTLIST=%d.",
                      (int)MAXLEN_OUTPUTLIST);

            // Store the flag and increment the output list length
            All.OutputListFlag[All.OutputListLength] = flag;
            All.OutputListLength++;
        }
    }
    // Close the output file
    fclose(fd);

    // Print message with the number of times found in the output list
    VERBOSE_ALL(2, "\nfound %d times in output-list.", All.OutputListLength);

    return 0;
}

/**
 * @brief In this function the timebase is adjusted if the final time is increased
 *
 * If a restart from restart-files is carried out where the TimeMax variable
 * is increased, then the integer timeline needs to be adjusted. The approach
 * taken here is to reduce the resolution of the integer timeline by factors
 * of 2 until the new final time can be reached within TIMEBASE.
 *
 * @param TimeMax_old The old final time
 * @param TimeMax_new The new final time
 */
void readjust_timebase(double TimeMax_old, double TimeMax_new)
{
    // Check if 'long long' is 64 bit on this platform
    ENDRUN_IF(sizeof(long long) != 8, "\nType 'long long' is not 64 bit on this platform.");

    // Print verbose message indicating the change in All.TimeMax
    VERBOSE(1, "\nAll.TimeMax has been changed in the parameterfile");
    VERBOSE(1, "Need to adjust integer timeline");

    // Check if All.TimeMax is reduced and new end time is before old one
    ENDRUN_IF(TimeMax_new < TimeMax_old, "\nIt is not allowed to reduce All.TimeMax");

    // Calculate ti_end based on All.ComovingIntegrationOn flag
    long long ti_end;
    if (All.ComovingIntegrationOn)
    {
        ti_end = (long long)(log(TimeMax_new / All.TimeBegin) / All.Timebase_interval);
    }
    else
    {
        ti_end = (long long)((TimeMax_new - All.TimeBegin) / All.Timebase_interval);
    }

    // Adjust the integer timeline until ti_end is within the desired range
    while (ti_end > TIMEBASE)
    {
        // Double the All.Timebase_interval and update related variables
        All.Timebase_interval *= 2.0;
        ti_end /= 2;
        All.Ti_Current /= 2;

#ifdef PMGRID
        All.PM_Ti_begstep /= 2;
        All.PM_Ti_endstep /= 2;
#endif // PMGRID
#ifdef CONDUCTION
        All.Conduction_Ti_begstep /= 2;
        All.Conduction_Ti_endstep /= 2;
#endif // CONDUCTION

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            // Adjust Ti_begstep and Ti_current for each particle
            P[i].Ti_begstep /= 2;
            P[i].Ti_current /= 2;

            if (P[i].TimeBin > 0)
            {
                P[i].TimeBin--;
                // Error: Minimum Timebin reached for a particle
                ENDRUN_IF(P[i].TimeBin <= 0, "Error in readjust_timebase(). Minimum Timebin for particle %d reached. ",
                          i);
            }
        }
        // Adjust Ti_nextlineofsight
        All.Ti_nextlineofsight /= 2;
    }
    // Update All.TimeMax with the new value
    All.TimeMax = TimeMax_new;
}
