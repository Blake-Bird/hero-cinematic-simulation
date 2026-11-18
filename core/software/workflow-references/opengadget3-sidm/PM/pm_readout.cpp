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
#include "../System/communication.hpp"

#ifdef PMGRID
/**
 * @brief Reads out the interpolated density field and assigns the values to particles.
 *
 * This function reads out the interpolated density field onto the particle positions and assigns the values
 * to the corresponding particles. It then communicates the density values between tasks and updates the
 * particles with the received values.
 *
 * @param mode_periodic Integer indicating the periodicity mode.
 *                      - 0: Non-periodic boundary conditions.
 *                      - 1: Periodic boundary conditions.
 *                      - 2: Periodic boundary conditions with wrapping at grid boundaries.
 * @param dim Integer indicating the dimension of the grid (e.g., 3 for 3D).
 * @param fac Scaling factor for the density values.
 * @param readout_grid Pointer to the density grid to be read out.
 * @param fft_params Pointer to the FFT parameters structure.
 * @param partin Pointer to the particle buffer containing imported particles.
 * @param nimport Number of imported particles.
 * @param nexport Number of exported particles.
 * @param index_task0 Pointer to the array containing the indices of task 0.
 * @param index_task1 Pointer to the array containing the indices of task 1.
 */
void pm_readout(int mode_periodic, int dim, double fac, fftw_real *readout_grid, struct fftw_parameters *fft_params,
                struct partbuf *partin, int nimport, int nexport, int *index_task0, int *index_task1)
{

    MyFloat *flistin, *flistout;

    flistin = (MyFloat *)mymalloc("flistin", nimport * sizeof(MyFloat));
    flistout = (MyFloat *)mymalloc("flistout", nexport * sizeof(MyFloat));

#ifndef KD_PM_NOSENDPOS
    MyFloat to_slab_fac = fft_params->grid / All.TotalMeshSize[mode_periodic];
#endif // KD_PM_NOSENDPOS

    double corner[3];
    for (int j = 0; j < 3; j++)
    {
        corner[j] = All.Corner[mode_periodic][j];
    }
    TIMER_PUSH(1, "PM_INTERP_FORCE");
#pragma omp parallel for
    for (int i = 0; i < nimport; i++)
    {
        flistin[i] = 0;

#ifdef KD_PM_NOSENDPOS
        int slab_x = partin[i].slab_x;
        int slab_y = partin[i].slab_y;
        int slab_z = partin[i].slab_z;
#else // KD_PM_NOSENDPOS
        int slab_x = (int)(to_slab_fac * (partin[i].Pos[0] - corner[0]));
        int slab_y = (int)(to_slab_fac * (partin[i].Pos[1] - corner[1]));
        int slab_z = (int)(to_slab_fac * (partin[i].Pos[2] - corner[2]));

        double dx = to_slab_fac * (partin[i].Pos[0] - corner[0]) - slab_x;
        double dy = to_slab_fac * (partin[i].Pos[1] - corner[1]) - slab_y;
        double dz = to_slab_fac * (partin[i].Pos[2] - corner[2]) - slab_z;

        if (mode_periodic == 2)
        {
            if (slab_x >= fft_params->grid)
            {
                slab_x -= fft_params->grid;
            }
            if (slab_y >= fft_params->grid)
            {
                slab_y -= fft_params->grid;
            }
            if (slab_z >= fft_params->grid)
            {
                slab_z -= fft_params->grid;
            }
        }
#endif // KD_PM_NOSENDPOS

        int slab_xx = slab_x + 1;
        int slab_yy = slab_y + 1;
        int slab_zz = slab_z + 1;

#ifdef KD_PM_NOSENDPOS
        MyFloat dx = partin[i].dx;
        MyFloat dy = partin[i].dy;
        MyFloat dz = partin[i].dz;
#endif // KD_PM_NOSENDPOS

        if (mode_periodic == 2)
        {
            if (slab_xx >= fft_params->grid)
            {
                slab_xx -= fft_params->grid;
            }
            if (slab_yy >= fft_params->grid)
            {
                slab_yy -= fft_params->grid;
            }
            if (slab_zz >= fft_params->grid)
            {
                slab_zz -= fft_params->grid;
            }
        }

        if (fft_params->slab_to_task[slab_x] == ThisTask)
        {
            slab_x -= fft_params->first_slab_of_task[ThisTask];

            flistin[i] +=
                (MyFloat)(readout_grid[INDEX(fft_params, slab_x, slab_y, slab_z)] * (1.0 - dx) * (1.0 - dy) *
                              (1.0 - dz) +
                          readout_grid[INDEX(fft_params, slab_x, slab_y, slab_zz)] * (1.0 - dx) * (1.0 - dy) * (dz) +
                          readout_grid[INDEX(fft_params, slab_x, slab_yy, slab_z)] * (1.0 - dx) * (dy) * (1.0 - dz) +
                          readout_grid[INDEX(fft_params, slab_x, slab_yy, slab_zz)] * (1.0 - dx) * (dy) * (dz));
        }

        if (fft_params->slab_to_task[slab_xx] == ThisTask)
        {
            slab_xx -= fft_params->first_slab_of_task[ThisTask];

            flistin[i] +=
                (MyFloat)(readout_grid[INDEX(fft_params, slab_xx, slab_y, slab_z)] * (dx) * (1.0 - dy) * (1.0 - dz) +
                          readout_grid[INDEX(fft_params, slab_xx, slab_y, slab_zz)] * (dx) * (1.0 - dy) * (dz) +
                          readout_grid[INDEX(fft_params, slab_xx, slab_yy, slab_z)] * (dx) * (dy) * (1.0 - dz) +
                          readout_grid[INDEX(fft_params, slab_xx, slab_yy, slab_zz)] * (dx) * (dy) * (dz));
        }
    }

    /* exchange the component data */

    memcpy(&flistout[Send_offset[ThisTask]], &flistin[Recv_offset[ThisTask]], Recv_count[ThisTask] * sizeof(MyFloat));
    TIMER_POP(1, "PM_INTERP_FORCE");
#ifdef USE_ISEND_IRECV
    TIMER_PUSH(1, "PM_COMM_RESULTS");
    MPI_Request *requests;
    int nrequests = 0, task;

    requests = (MPI_Request *)mymalloc("requests", 2 * NTask * sizeof(MPI_Request));

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int task = ThisTask ^ ngrp;

        if (task < NTask)
        {
            MPI_Irecv(&flistout[Send_offset[task]], Send_count[task] * sizeof(MyFloat), MPI_BYTE, task, TAG_DENS_A,
                      MYMPI_COMM_WORLD, &requests[nrequests++]);
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD);

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int task = ThisTask ^ ngrp;

        if (task < NTask)
        {
            MPI_Isend(&flistin[Recv_offset[task]], Recv_count[task] * sizeof(MyFloat), MPI_BYTE, task, TAG_DENS_A,
                      MYMPI_COMM_WORLD, &requests[nrequests++]);
        }
    }

    MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
    myfree(requests);
    TIMER_POP(1, "PM_COMM_RESULTS");
#else // USE_ISEND_IRECV
    TIMER_PUSH(1, "PM_COMM_RESULTS");
    templated_exchange_data_Sendrecv(flistin, flistout, TAG_DENS_A, Recv_offset, Recv_count, Send_offset, Send_count);
    TIMER_POP(1, "PM_COMM_RESULTS");
#endif // USE_ISEND_IRECV
    /* now assign them to the correct particles */
    TIMER_PUSH(1, "PM_ADD_FORCE");
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        if (index_task0[i] >= 0)
        {
            MyAtLeastDouble value = flistout[index_task0[i]];
            if (index_task1[i] >= 0)
            {
                value += flistout[index_task1[i]];
            }
#ifdef PLACEHIGHRESREGION
            if (mode_periodic == 1)
            {
                if (!(pmforce_is_particle_high_res(P[i].Type, P[i].Pos)))
                {
                    continue;
                }
            }
#endif // PLACEHIGHRESREGION

            if (dim < 0)
            {
#ifdef EVALPOTENTIAL
                P[i].PM_Potential += (MyFloat)(value)*fac;
#endif
            }
            else
            {
                P[i].GravPM[dim] += (MyFloat)(value);
            }
        }
    }
    TIMER_POP(1, "PM_ADD_FORCE");

    myfree(flistout);
    myfree(flistin);
}

/**
 * @brief Computes the force components by finite differencing the potential and sends back the results.
 *
 * This function calculates the force components by finite differencing the potential for each dimension
 * and sends back the results to the appropriate CPUs. It also performs transposition operations and reads
 * out the force components to assign them to particles.
 *
 * @param mode_periodic Integer indicating the periodicity mode.
 *                      - 0: Non-periodic boundary conditions.
 *                      - 1: Periodic boundary conditions.
 *                      - 2: Periodic boundary conditions with wrapping at grid boundaries.
 * @param fac Scaling factor for the force values.
 * @param rhogrid Pointer to the density grid.
 * @param fft_params Pointer to the FFT parameters structure.
 * @param partin Pointer to the particle buffer containing imported particles.
 * @param nimport Number of imported particles.
 * @param nexport Number of exported particles.
 * @param index_task0 Pointer to the array containing the indices of task 0.
 * @param index_task1 Pointer to the array containing the indices of task 1.
 */
void pm_get_forces(int mode_periodic, double fac, fftw_real *rhogrid, struct fftw_parameters *fft_params,
                   struct partbuf *partin, int nimport, int nexport, int *index_task0, int *index_task1)
{
    int start, end;

    /* for finite differencing */
    double fac_diff = fac / (2 * All.TotalMeshSize[mode_periodic] / fft_params->grid);

    fftw_real *forcegrid = (fftw_real *)mymalloc("forcegrid", fft_params->maxfftsize * sizeof(fftw_real));

    if (mode_periodic != 2)
    {
        start = 2;
        end = fft_params->grid / 2 - 2;
    }
    else
    {
        start = 0;
        end = fft_params->grid;
    }

    for (int dim = 2; dim >= 0; dim--) /* Calculate each component of the force. */
    { /* we do the x component last, because for differencing the potential in the x-direction, we nee \
     d to contruct the transpose */
        if (dim == 0)
        {
            TIMER_PUSH(1, "PM_TRANSPOSE_A");
            pm_transposeA(mode_periodic, rhogrid, forcegrid,
                          fft_params); /* compute the transpose of the potential field */
            TIMER_POP(1, "PM_TRANSPOSE_A");
        }

        TIMER_PUSH(1, "PM_FORCE_GRID");
        for (int x = 0; x < fft_params->nslab_x; x++)
#pragma omp parallel for
            for (int jk = start; jk < end * end; jk++)
            {
                int y = jk / end;
                int z = jk % end;

                if (mode_periodic != 2)
                {
                    if ((x + fft_params->slabstart_x < start) || (x + fft_params->slabstart_x >= end) || (y < start))
                        continue;
                }
                int yrr, yll, zrr, zll, zr, yr, yl, zl;
                yrr = yll = yr = yl = y;
                zrr = zll = zr = zl = z;

                switch (dim)
                {
                case 0: /* note: for the x-direction, we difference the transposed direction (y) */
                case 1:
                    yr = y + 1;
                    yl = y - 1;
                    yrr = y + 2;
                    yll = y - 2;
                    if (mode_periodic == 2)
                    {
                        if (yr >= fft_params->grid)
                        {
                            yr -= fft_params->grid;
                        }
                        if (yrr >= fft_params->grid)
                        {
                            yrr -= fft_params->grid;
                        }
                        if (yl < 0)
                        {
                            yl += fft_params->grid;
                        }
                        if (yll < 0)
                        {
                            yll += fft_params->grid;
                        }
                    }
                    break;
                case 2:
                    zr = z + 1;
                    zl = z - 1;
                    zrr = z + 2;
                    zll = z - 2;
                    if (mode_periodic == 2)
                    {
                        if (zr >= fft_params->grid)
                        {
                            zr -= fft_params->grid;
                        }
                        if (zrr >= fft_params->grid)
                        {
                            zrr -= fft_params->grid;
                        }
                        if (zl < 0)
                        {
                            zl += fft_params->grid;
                        }
                        if (zll < 0)
                        {
                            zll += fft_params->grid;
                        }
                    }
                    break;
                }

                if (dim == 0)
                {
                    forcegrid[TINDEX(fft_params, x, y, z)] =
                        fac_diff *
                        ((4.0 / 3) * (rhogrid[TINDEX(fft_params, x, yl, zl)] - rhogrid[TINDEX(fft_params, x, yr, zr)]) -
                         (1.0 / 6) *
                             (rhogrid[TINDEX(fft_params, x, yll, zll)] - rhogrid[TINDEX(fft_params, x, yrr, zrr)]));
                }
                else
                {
                    forcegrid[INDEX(fft_params, x, y, z)] =
                        fac_diff *
                        ((4.0 / 3) * (rhogrid[INDEX(fft_params, x, yl, zl)] - rhogrid[INDEX(fft_params, x, yr, zr)]) -
                         (1.0 / 6) *
                             (rhogrid[INDEX(fft_params, x, yll, zll)] - rhogrid[INDEX(fft_params, x, yrr, zrr)]));
                }
            }
        TIMER_POP(1, "PM_FORCE_GRID");

        if (dim == 0)
        {
            TIMER_PUSH(1, "PM_TRANSPOSE_B");
            pm_transposeB(mode_periodic, forcegrid, rhogrid, fft_params);
            TIMER_POP(1, "PM_TRANSPOSE_B");
        }

        /* read out the force components */
        pm_readout(mode_periodic, dim, fac, forcegrid, fft_params, partin, nimport, nexport, index_task0, index_task1);
    }

    myfree(forcegrid);
}
#endif // PMGRID
