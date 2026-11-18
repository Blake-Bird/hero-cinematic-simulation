/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
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
#include <unistd.h>

#ifdef IMPOSE_PINNING

#define __USE_GNU

#include <sched.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifndef SOCKETS
#define SOCKETS 4 /* Setting for our 4 x AMD Magny-Cours nodes */
#define MAX_CORES 48 /* Setting for our 4 x AMD Magny-Cours nodes */
#endif

/* SOCKETS   = 12 for 2 x Intel 6-core nodes with hypethreading  */
/* MAX_CORES = 24 for 2 x Intel 6-core nodes with hypethreading  */

static cpu_set_t cpuset;

/**
 * @brief Retrieves the CPU core set assigned to the current process.
 *
 * This function retrieves the set of CPU cores assigned to the current process
 * using the sched_getaffinity system call. The retrieved core set is stored
 * in the global variable cpuset.
 */
void get_core_set(void)
{
    CPU_ZERO(&cpuset);
    sched_getaffinity(getpid(), sizeof(cpuset), &cpuset);
}

/**
 * @brief Pins the current process to a specific CPU core set.
 *
 * This function pins the current process to a specific CPU core set using
 * the sched_setaffinity system call. The core set is determined based on the
 * number of OpenMP threads (if provided) and the global variables NTask and ThisTask.
 * The process is pinned to CPU cores that are set in the global variable cpuset.
 */
void pin_to_core_set(void)
{
    int core, task, num_threads;
    static cpu_set_t cpuset_new;

    char *p = getenv("OMP_NUM_THREADS");
    if (p)
    {
        num_threads = atoi(p);
    }
    else
    {
        num_threads = 1;
    }

    CPU_ZERO(&cpuset_new);

    int corestart = 0;

    for (task = 0, core = corestart; task < NTask * num_threads; task++)
    {
        while (!CPU_ISSET(core, &cpuset))
        {
            core += SOCKETS;
            if (core >= MAX_CORES)
            {
                corestart++;
                if (corestart >= SOCKETS)
                {
                    corestart = 0;
                }
                core = corestart;
            }
        }

        if ((task / num_threads) == ThisTask)
        {
            CPU_SET(core, &cpuset_new);
        }
        core += SOCKETS;
        if (core >= MAX_CORES)
        {
            corestart++;
            if (corestart >= SOCKETS)
            {
                corestart = 0;
            }
            core = corestart;
        }
    }
    sched_setaffinity(getpid(), sizeof(cpuset_new), &cpuset_new);
}

/**
 * @brief Reports the CPU core pinning information for each task.
 *
 * This function reports the CPU core pinning information for each task in the MPI communicator.
 * It retrieves the CPU affinity mask for the current process using sched_getaffinity system call
 * and prints the information for each task.
 */
void report_pinning(void)
{
    cpu_set_t cpuset;
    char buf[MAX_CORES + 1];

    CPU_ZERO(&cpuset);
    sched_getaffinity(getpid(), sizeof(cpuset), &cpuset);

    for (int i = 0; i < MAX_CORES; i++)
    {
        if (CPU_ISSET(i, &cpuset))
        {
            buf[i] = '1';
        }
        else
        {
            buf[i] = '-';
        }
    }
    buf[MAX_CORES] = 0;

    for (int i = 0; i < NTask; i++)
    {
        if (ThisTask == i)
        {
            VERBOSE_ALL(2, "Task=%02d: %s\n", ThisTask, buf);
        }
        fflush(stdout);
        MPI_Barrier(MYMPI_COMM_WORLD);
    }
}

#endif // IMPOSE_PINNING
