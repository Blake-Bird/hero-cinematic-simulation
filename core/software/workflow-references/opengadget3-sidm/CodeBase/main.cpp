/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the main function of the code.
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
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "allvars.hpp"
#include "proto.hpp"

#ifdef FPE_TRAPPING
#include <fenv.h>
#define _GNU_SOURCE
#endif // FPE_TRAPPING

#ifdef ACC
#ifdef ACC_OPENACC
#include <openacc.h>
#elif defined(ACC_OPENMP)
#include <omp.h>
#endif // ACC_OPENACC
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // ACC

/**
 * @brief main function to run the code
 *
 * This function initializes the MPI communication packages, sets
 * cpu-time counters to 0, sets up the simulation, and starts the main
 * simulation loop.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return 0 indicating successful execution.
 */
int main(int argc, char **argv)
{
    /* Initialize global variables for driftfac */
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
    {
        last_time0_drift = last_time1_drift = -1;
#ifdef MAGNETIC
        last_time0_magkick = last_time1_magkick = -1;
#endif // MAGNETIC
        last_time0_hydrokick = last_time1_hydrokick = -1;
        last_time0_gravkick = last_time1_gravkick = -1;
#if defined(mSIDM) || defined(mIDM)
        last_time0_scattering_kick = last_time1_scattering_kick = -1;
#endif // mSIDM || mIDM
    }

#ifdef IMPOSE_PINNING
    // Get the set of CPU cores to which the process should be pinned
    get_core_set();
#endif // IMPOSE_PINNING

    // Debug: Measure memory usage before MPI initialization
    long long Mem_before[6], Mem_after[6], Mem_before_max[6], Mem_after_max[6], Mem_before_min[6], Mem_after_min[6];
    Mem_before[0] = report_comittable_memory(&Mem_before[1], &Mem_before[2], &Mem_before[3], &Mem_before[4]);
    Mem_before[5] = Mem_before[1] - Mem_before[0];
    // Initialize MPI
    MPI_Init(&argc, &argv);

#ifdef TIMER_MERIC
    MERIC_Init();
#endif // TIMER_MERIC

    // Debug: Measure memory usage after MPI initialization
    Mem_after[0] = report_comittable_memory(&Mem_after[1], &Mem_after[2], &Mem_after[3], &Mem_after[4]);
    Mem_after[5] = Mem_after[1] - Mem_after[0];
    // Get MPI rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    // Print memory usage statistics (only by MPI rank 0)
    MPI_Allreduce(Mem_before, Mem_before_max, 6, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(Mem_after, Mem_after_max, 6, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(Mem_before, Mem_before_min, 6, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(Mem_after, Mem_after_min, 6, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

    for (int i = 0; i < 6; i++)
    {
        VERBOSE(1, "Mem[%i] before max: %lld, before min: %lld, after max: %lld, after min: %lld ", i,
                Mem_before_max[i], Mem_before_min[i], Mem_after_max[i], Mem_after_min[i]);
    }
    /* read memory limits */
#ifdef CHECK_OVERCOMMIT_MEMORY
    FILE *fd;
    char filename[500];
    int data;
    sprintf(filename, "/proc/sys/vm/overcommit_memory");
    if ((fd = fopen(filename, "r")))
    {
        VERBOSE(3, "reading overcommit memory ");
        fscanf(fd, "%i", &data);
        VERBOSE(3, "overcommit mem: %i ", data);
        fclose(fd);
    }
    else
    {
        ENDRUN_MAIN("could not read overcommit mem");
    }

    sprintf(filename, "/proc/sys/vm/overcommit_ratio");
    if ((fd = fopen(filename, "r")))
    {
        VERBOSE(3, "reading overcommit ratio ");
        fscanf(fd, "%i", &data);
        VERBOSE(3, "overcommit ratio: %i ", data);
        fclose(fd);
    }
    else
    {
        ENDRUN_MAIN("could not read overcommit ratio");
    }
#endif // CHECK_OVERCOMMIT_MEMORY
    // Duplicate MPI communicator
    MPI_Comm_dup(MPI_COMM_WORLD, &MYMPI_COMM_WORLD);

#ifdef IMPOSE_PINNING
    // Pin MPI tasks to CPU cores
    pin_to_core_set();
#endif // IMPOSE_PINNING

    mpi_report_comittable_memory();

    if (VERBOSE_LEVEL > 3)
    {
        report_task_list();
        write_ps_files();
    }
    // Synchronize MPI tasks
    MPI_Barrier(MYMPI_COMM_WORLD);

    /* initialize OpenMP thread pool and bind (implicitly though OpenMP runtime) */
#ifdef _OPENMP
#pragma omp parallel
    {
        ThisThread = omp_get_thread_num();
#pragma omp master
        {
            maxThreads = omp_get_num_threads();
        }
    }
#else // _OPENMP
    ThisThread = 0;
    maxThreads = 1;
#endif // _OPENMP
    // Calculate the number of PTask
    for (PTask = 0; NTask > (1 << PTask); PTask++)
    {
    };
    {
        if (argc < 2)
        {
            ENDRUN_MAIN(
                "Parameters are missing.\nCall with <ParameterFile> [<RestartFlag>] [<RestartSnapNum>]\n\n   "
                "RestartFlag    Action\n      -1          Test Models: all physical modules give report on their "
                "model\n       0          Read initial conditions and start simulation\n       1          Read "
                "restart files and resume simulation\n       2          Restart from specified snapshot dump and "
                "continue simulation\n       3          Run FOF and optionally SUBFIND if enabled\n       4         "
                " Convert snapshot file to different format\n       5          Calculate power spectrum and "
                "two-point function\n       6          Calculate velocity power spectrum for the gas particles");
        }
    }
    // Copy the parameter file name from command-line argument
    strcpy(ParameterFile, argv[1]);
    // Parse additional command-line arguments
    if (argc >= 3)
    {
        RestartFlag = atoi(argv[2]);
    }
    else
    {
        RestartFlag = 0;
    }

    if (argc >= 4)
    {
        RestartSnapNum = atoi(argv[3]);
    }
    else
    {
        RestartSnapNum = -1;
    }

    if (argc >= 5)
    {
        PostProcessDt = atof(argv[4]);
    }
    // Initialize CPU usage counter
    CPUThisRun = 0;

#ifdef PAUSE_RUN_TO_ATTACH_DEBUGGER
    pause_run_to_attach_debugger();
#endif // PAUSE_RUN_TO_ATTACH_DEBUGGER

// Nitya - to keep a track of memory allocations on GPU
#if defined(ACC_OPENMP) && defined(PERIODIC)
    NumPartOld = 0;
    LiteDInNumPartOld = 0;
    LiteDOutNumPartOld = 0;
    DAPLNumPartOld = 0;
    APLNumPartOld = 0;
#endif // defined(ACC_OPENMP) && defined(PERIODIC)

    begrun(); /* set-up run  */

    run(); /* main simulation loop */

#ifdef ACC
    // Make sure no GPU data region is left open.
    if (acc_all.inside_data_region)
    {
        update_openacc_structures_exit();
    }
#ifdef ACC_OPENACC
    // Some of the device frees issued by the "exit data delete" pragmas above (and by the
    // ones executed inside the last gravity/hydro/conduction/hsml call of the run) are only
    // queued on an async queue; without a later call that synchronizes on that queue, the
    // actual cuMemFree never happens before the process exits, which compute-sanitizer then
    // reports as a leak. Wait for everything to actually finish before tearing anything down.
    acc_wait_all();
#endif // ACC_OPENACC
#endif // ACC

    // Free MPI communicator and finalize MPI
    MPI_Comm_free(&MYMPI_COMM_WORLD);

#ifdef TIMER_MERIC
    MERIC_Close();
#endif // TIMER_MERIC

    MPI_Finalize();

#if defined(ACC) && defined(ACC_OPENACC)
    // Release the OpenACC runtime's device/pinned memory pools only after MPI_Finalize has
    // run: MPI's CUDA-aware UCX transport still touches the CUDA context during finalize, and
    // destroying the context earlier (via acc_shutdown) makes that teardown fail instead.
    acc_shutdown(acc_device_nvidia);
#endif // ACC && ACC_OPENACC

    return 0;
}
