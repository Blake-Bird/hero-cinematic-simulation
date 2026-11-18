/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Routines for periodic PM-force computation.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "pm.hpp"

#ifdef PMGRID
/**
 * @brief Divide a range into pieces for parallel processing.
 *
 * This function divides a range into multiple pieces for parallel processing
 * and calculates the starting index and count for a specific piece.
 *
 * @param N Total number of elements in the range.
 * @param pieces Number of pieces to divide the range into.
 * @param index Index of the current piece (starting from 0).
 * @param[out] first Pointer to the variable to store the starting index of the current piece.
 * @param[out] count Pointer to the variable to store the number of elements in the current piece.
 */
void pm_divide_threads(int N, int pieces, int index, int *first, int *count)
{
    int avg = (N - 1) / pieces + 1;
    int exc = pieces * avg - N;
    int indexlastsection = pieces - exc;

    if (index < indexlastsection)
    {
        *first = index * avg;
        *count = avg;
    }
    else
    {
        *first = index * avg - (index - indexlastsection);
        *count = avg - 1;
    }
}

/**
 * @brief Transpose a 3D field across MPI tasks along the x-axis.
 *
 * This function transposes a 3D field across MPI tasks along the x-axis. Each MPI task
 * holds a portion of the field, and the transposed result is stored in a scratch array,
 * which is then sent back to the corresponding tasks.
 *
 * @param mode The mode of the operation (e.g., periodic boundary conditions).
 * @param field The original 3D field to be transposed.
 * @param scratch A scratch array used for intermediate storage during transposition.
 * @param fft_params Struct containing FFT parameters and task-specific information.
 */
void pm_transposeA(int mode, fftw_real *field, fftw_real *scratch, struct fftw_parameters *fft_params)
{
    /*
       for(int task = 0; task < NTask; task++)
       for(int x = 0; x < fft_params->nslab_x; x++)
     */
    int ntask_times_nslab_x = NTask * fft_params->nslab_x;

    // Parallelize the transposition loop over task and x indices
#pragma omp parallel for
    for (int i = 0; i < ntask_times_nslab_x; i++)
    {
        int x = i % fft_params->nslab_x;
        int task = (i - x) / fft_params->nslab_x;
        // Iterate over y and z dimensions to perform transposition
        for (int y = fft_params->first_slab_of_task[task];
             y < fft_params->first_slab_of_task[task] + fft_params->slabs_per_task[task]; y++)
        {
            for (int z = 0; z < fft_params->grid; z++)
            {
                // Transpose field element and store in scratch array
                scratch[fft_params->grid *
                            (fft_params->first_slab_of_task[task] * fft_params->nslab_x +
                             x * fft_params->slabs_per_task[task] + (y - fft_params->first_slab_of_task[task])) +
                        z] = field[INDEX(fft_params, x, y, z)];
            }
        }
    }
    // Exchange transposed data among MPI tasks
    for (int ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
        int task = ThisTask ^ ngrp;

        if (task < NTask)
        {
            // Send and receive transposed data between tasks
            MPI_Sendrecv(scratch + fft_params->grid * fft_params->first_slab_of_task[task] * fft_params->nslab_x,
                         fft_params->grid * fft_params->nslab_x * fft_params->slabs_per_task[task] * sizeof(fftw_real),
                         MPI_BYTE, task, TAG_KEY,
                         field + fft_params->grid * fft_params->first_slab_of_task[task] * fft_params->nslab_x,
                         fft_params->grid * fft_params->nslab_x * fft_params->slabs_per_task[task] * sizeof(fftw_real),
                         MPI_BYTE, task, TAG_KEY, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

/**
 * @brief Transpose a 3D field across MPI tasks along the x-axis.
 *
 * This function transposes a 3D field across MPI tasks along the x-axis. Each MPI task
 * holds a portion of the field, and the transposed result is stored in a scratch array,
 * which is then sent back to the corresponding tasks.
 *
 * @param mode The mode of the operation (e.g., periodic boundary conditions).
 * @param field The original 3D field to be transposed.
 * @param scratch A scratch array used for intermediate storage during transposition.
 * @param fft_params Struct containing FFT parameters and task-specific information.
 */
void pm_transposeB(int mode, fftw_real *field, fftw_real *scratch, struct fftw_parameters *fft_params)
{
    // Exchange original data with transposed data among MPI tasks
    for (int ngrp = 0; ngrp < (1 << PTask); ngrp++)
    {
        int task = ThisTask ^ ngrp;

        if (task < NTask)
        {
            // Send and receive transposed data between tasks
            MPI_Sendrecv(field + fft_params->grid * fft_params->first_slab_of_task[task] * fft_params->nslab_x,
                         fft_params->grid * fft_params->nslab_x * fft_params->slabs_per_task[task] * sizeof(fftw_real),
                         MPI_BYTE, task, TAG_KEY,
                         scratch + fft_params->grid * fft_params->first_slab_of_task[task] * fft_params->nslab_x,
                         fft_params->grid * fft_params->nslab_x * fft_params->slabs_per_task[task] * sizeof(fftw_real),
                         MPI_BYTE, task, TAG_KEY, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    /*
       for(int task = 0; task < NTask; task++)
       for(int x = 0; x < fft_params->nslab_x; x++)
     */

    // Parallelize the transposition loop over task and x indices
    int ntask_times_nslab_x = NTask * fft_params->nslab_x;
#pragma omp parallel for
    for (int i = 0; i < ntask_times_nslab_x; i++)
    {
        // Calculate task and x indices from the loop index
        int x = i % fft_params->nslab_x;
        int task = (i - x) / fft_params->nslab_x;

        // Iterate over y and z dimensions to perform transposition
        for (int y = fft_params->first_slab_of_task[task];
             y < fft_params->first_slab_of_task[task] + fft_params->slabs_per_task[task]; y++)
        {
            for (int z = 0; z < fft_params->grid; z++)
            {
                // Transpose field element and store in original array
                field[INDEX(fft_params, x, y, z)] =
                    scratch[fft_params->grid *
                                (fft_params->first_slab_of_task[task] * fft_params->nslab_x +
                                 x * fft_params->slabs_per_task[task] + (y - fft_params->first_slab_of_task[task])) +
                            z];
            }
        }
    }
}

#endif // PMGRID
