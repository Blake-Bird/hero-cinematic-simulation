/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the function that is called when the simulation is aborted.
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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

#include "allvars.hpp"
#include "proto.hpp"

#ifdef ACC
#ifdef ACC_OPENACC
#include <openacc.h>
#elif defined(ACC_OPENMP)
#include <omp.h>
#endif // ACC_OPENACC
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // ACC

/**
 * @brief Perform end-of-run procedures with a formatted error message.
 *
 * This function is called at the end of the simulation run to perform cleanup
 * and finalize MPI communication. If the `do_output` parameter is true, it prints
 * a formatted error message including the function name, file name, line number,
 * and additional information provided in the format string `f_`. It then aborts MPI
 * communication and exits the program with an error code.
 *
 * @param do_output Flag indicating whether to output an error message.
 * @param func The name of the function where the error occurred.
 * @param file The name of the file where the error occurred.
 * @param line The line number in the file where the error occurred.
 * @param f_ The format string for the error message.
 * @param ... Additional arguments to be formatted into the error message.
 */
void Endrun(const int do_output, const char *func, const char *file, const int line, const char *f_, ...)
{
    // Clean up OpenACC structures if GPU acceleration was used
#ifdef ACC
    if (acc_all.entered_gpu)
    {
        update_openacc_structures_exit();
    }
#ifdef ACC_OPENACC
    acc_shutdown(acc_device_nvidia);
#endif // ACC_OPENACC
#endif // ACC

    // If specified, output the error message
    if (do_output)
    {
        VERBOSE_ALL(0, "\n\n====== OpenGadget3 endrun. ThisTask: %d File: %s Function: %s Line: %d. Last words:\n",
                    ThisTask, file, func, line);
        va_list args;
        va_start(args, f_);
        vprintf(f_, args);
        fflush(stdout);
        va_end(args);
        VERBOSE_ALL(0, "\n\n====== End of Error-Message ======\n");
    }

    // Abort MPI communication with error code 1
    MPI_Abort(MYMPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(1); // exit 1 is more right
}
