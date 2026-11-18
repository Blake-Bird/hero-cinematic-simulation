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

#include <stdio.h>
#include <stdlib.h>

#ifndef NUMA_HPP
#define NUMA_HPP

/**
 * @brief Holds general information about the machine architecture where the simulation is running.
 *
 * This struct provides information about the NUMA (Non-Uniform Memory Access) architecture of the system.
 */
struct NumaArchitecture
{
    int processes_per_node;
    int processes;
    int nodes;
};

/**
 * @brief Holds information about the MPI process and its corresponding NUMA node.
 *
 * This struct provides information about the MPI process index and its associated NUMA node index.
 */
struct NumaInfo
{
    int process_idx;
    int node_idx;
};

/**
 * @brief Static datastructures to store NUMA architecture and process information.
 *
 * Each compile units will have its own data, but this is ok as long as they are read-only.
 */
static NumaArchitecture *numa_architecture = NULL;
static NumaInfo *numa_info = NULL;

/**
 * @brief Initializes NUMA-related information based on global variables and environment settings.
 *
 * This function initializes NUMA-related information, such as the number of processes per node and the current
 * process's index and node index. It reads environment variables to determine the number of processes per node.
 *
 * @note The function requires the environment variable GADGET_PROCESSS_PER_NODE to be specified.
 *
 * @pre The environment variable GADGET_PROCESSS_PER_NODE must be set to use NUMA.
 */
static void numa_init()
{
    if (numa_architecture == NULL)
    {
        //
        numa_architecture = (NumaArchitecture *)calloc(1, sizeof(NumaArchitecture));
        char *GADGET_PROCESSS_PER_NODE = getenv("GADGET_PROCESSS_PER_NODE");
        ENDRUN_IF(GADGET_PROCESSS_PER_NODE == NULL,
                  "Environmnet variable GADGET_PROCESSS_PER_NODE should be specified to use NUMA");
        numa_architecture->processes_per_node = atoi(GADGET_PROCESSS_PER_NODE);
        numa_architecture->processes = NTask;
        ENDRUN_IF((numa_architecture->processes % numa_architecture->processes_per_node) != 0,
                  "Numa Library so far will only work with if all nodes have the same number of processes. "
                  "procs = %d; procs_per_node = %d",
                  numa_architecture->processes, numa_architecture->processes_per_node);
        numa_architecture->nodes = numa_architecture->processes / numa_architecture->processes_per_node;
    }

    if (numa_info == NULL)
    {
        numa_info = (NumaInfo *)calloc(1, sizeof(NumaInfo));
        numa_info->process_idx = ThisTask;
        numa_info->node_idx = numa_info->process_idx % numa_architecture->processes_per_node;
    }
}

/**
 * @brief Retrieves the NUMA architecture information.
 *
 * This function initializes NUMA-related information if not already initialized,
 * and then returns a pointer to the NUMA architecture structure.
 *
 * @return Pointer to the NUMA architecture structure.
 */
static struct NumaArchitecture *get_numa_architecture()
{
    numa_init();
    return numa_architecture;
}

/**
 * @brief Retrieves the NUMA information of a specific process.
 *
 * This function initializes NUMA-related information if not already initialized,
 * and then returns the NUMA information (process index and node index) of the specified process.
 *
 * @param process_idx The index of the process for which NUMA information is to be retrieved.
 * @return NUMA information (process index and node index) of the specified process.
 */
static struct NumaInfo get_numa_of_process_info(int process_idx)
{
    numa_init();
    struct NumaInfo numa_info;
    numa_info.process_idx = ThisTask;
    numa_info.node_idx = numa_info.process_idx % numa_architecture->processes_per_node;
    return numa_info;
}

/**
 * @brief Retrieves the NUMA information of the current process.
 *
 * This function initializes NUMA-related information if not already initialized,
 * and then returns a pointer to the NUMA information structure of the current process.
 *
 * @return Pointer to the NUMA information structure of the current process.
 */
static NumaInfo *get_numa_info()
{
    numa_init();
    return numa_info;
}

/**
 * @brief Finalizes NUMA-related resources.
 *
 * This function frees the memory allocated for NUMA-related information structures.
 * It should be called when NUMA-related resources are no longer needed.
 */
static void numa_finalize()
{
    free(numa_info);
    free(numa_architecture);
}

#endif // NUMA_HPP
