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

struct fftw_parameters fft_params_periodic;
struct fftw_parameters fft_params_nonperiodic;

/**
 * @brief Perform the Particle-Mesh (PM) force calculation.
 *
 * This function calculates the PM forces on particles based on the given mode of operation.
 * It allocates memory for necessary structures, determines particle slabs, and distributes
 * particles among MPI tasks for force computation.
 *
 * @param mode_periodic The mode of operation for periodic boundary conditions.
 * @return Integer flag indicating success (0) or failure (non-zero).
 */
int pmforce(int mode_periodic) // , int mode, int *typelist)
{
    TIMER_PUSH(1, "PM_COLLECT_PART");
    double fac;
    int nimport, nexport;

    VERBOSE(1, "  PM: Starting PM calculation (mode=%d).  (presently allocated=%g MB)", mode_periodic,
            AllocatedBytes / (1024.0 * 1024.0));

    // Initialize FFTW plans for forward and inverse FFTs
    FFTW(plan) fft_forward_plan, fft_inverse_plan;

    struct fftw_parameters *fft_params;

    // Allocate memory for particle buffer if periodic boundary conditions are not used
#ifdef KD_PM_NOSENDPOS
    struct partbuf *part = (struct partbuf *)mymalloc("part", NumPart * sizeof(struct partbuf));
#endif // KD_PM_NOSENDPOS
    // Select appropriate FFT parameters and factor based on mode
    if (mode_periodic == 2)
    {
        fft_params = &fft_params_periodic;
        fac = All.GravityConstantInternal / (M_PI * All.BoxSize); /* to get potential */
    }
    else
    {
        fft_params = &fft_params_nonperiodic;
        fac = All.GravityConstantInternal / pow(All.TotalMeshSize[mode_periodic], 4) *
              pow(All.TotalMeshSize[mode_periodic] / fft_params->grid, 3);
    }

    MyFloat to_slab_fac = fft_params->grid / All.TotalMeshSize[mode_periodic];

    // Initialize array to count the number of particles to send to each task
    for (int j = 0; j < NTask; j++)
    {
        Send_count[j] = 0;
    }

    /* determine the slabs each particles accesses */
    int flag = 0;

    int *index_task0 = (int *)mymalloc("index_task0", NumPart * sizeof(int));
    int *index_task1 = (int *)mymalloc("index_task1", NumPart * sizeof(int));
    int *task0_for_offset = (int *)mymalloc("task0_for_offset", NumPart * sizeof(int));
    int *task1_for_offset = (int *)mymalloc("task1_for_offset", NumPart * sizeof(int));
    struct partbuf *partin;
    struct partbuf *partout;

#pragma omp parallel for reduction(+ : flag)
    for (int i = 0; i < NumPart; i++)
    {
        if (mode_periodic != 2)
#ifdef PLACEHIGHRESREGION
            if (mode_periodic == 0 || (mode_periodic == 1 && pmforce_is_particle_high_res(P[i].Type, P[i].Pos)))
#endif // PLACEHIGHRESREGION
            {
                for (int j = 0; j < 3; j++)
                {
                    if (P[i].Pos[j] < All.Xmintot[mode_periodic][j] || P[i].Pos[j] > All.Xmaxtot[mode_periodic][j])
                    {
                        VERBOSE_IF(flag == 0, 0,
                                   "  PM: Particle Id=%d on task=%d with coordinates (%g|%g|%g) lies outside PM "
                                   "mesh.\nStopping",
                                   (int)i, ThisTask, P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                        flag++;
                        break;
                    }
                }
            }

        index_task0[i] = -1;
        index_task1[i] = -1;

        if (P[i].Pos[0] < All.Corner[mode_periodic][0] || P[i].Pos[0] >= All.UpperCorner[mode_periodic][0])
        {
            continue;
        }
        if (P[i].Pos[1] < All.Corner[mode_periodic][1] || P[i].Pos[1] >= All.UpperCorner[mode_periodic][1])
        {
            continue;
        }
        if (P[i].Pos[2] < All.Corner[mode_periodic][2] || P[i].Pos[2] >= All.UpperCorner[mode_periodic][2])
        {
            continue;
        }

#ifdef KD_PM_NOSENDPOS
        part[i].slab_x = (int)(to_slab_fac * (P[i].Pos[0] - All.Corner[mode_periodic][0]));
        part[i].slab_y = (int)(to_slab_fac * (P[i].Pos[1] - All.Corner[mode_periodic][1]));
        part[i].slab_z = (int)(to_slab_fac * (P[i].Pos[2] - All.Corner[mode_periodic][2]));

        part[i].dx = to_slab_fac * (P[i].Pos[0] - All.Corner[mode_periodic][0]) - part[i].slab_x;
        part[i].dy = to_slab_fac * (P[i].Pos[1] - All.Corner[mode_periodic][1]) - part[i].slab_y;
        part[i].dz = to_slab_fac * (P[i].Pos[2] - All.Corner[mode_periodic][2]) - part[i].slab_z;

        part[i].Mass = P[i].Mass;

        if (mode_periodic == 2)
        {
            if (part[i].slab_x >= fft_params->grid)
            {
                part[i].slab_x -= fft_params->grid;
            }
            if (part[i].slab_y >= fft_params->grid)
            {
                part[i].slab_y -= fft_params->grid;
            }
            if (part[i].slab_z >= fft_params->grid)
            {
                part[i].slab_z -= fft_params->grid;
            }
        }

        int slab_xx = part[i].slab_x + 1;
#else // KD_PM_NOSENDPOS
        int slab_x = (int)(to_slab_fac * (P[i].Pos[0] - All.Corner[mode_periodic][0]));
        if (mode_periodic == 2)
        {
            if (slab_x >= fft_params->grid)
            {
                slab_x -= fft_params->grid;
            }
        }

        int slab_xx = slab_x + 1;
#endif // KD_PM_NOSENDPOS
        if (mode_periodic == 2)
        {
            if (slab_xx >= fft_params->grid)
            {
                slab_xx -= fft_params->grid;
            }
        }

#ifdef KD_PM_NOSENDPOS
        int task0 = fft_params->slab_to_task[part[i].slab_x];
#else // KD_PM_NOSENDPOS
        int task0 = fft_params->slab_to_task[slab_x];
#endif // KD_PM_NOSENDPOS
        int task1 = fft_params->slab_to_task[slab_xx];

        task0_for_offset[i] = task0;
        task1_for_offset[i] = task1;

        int send_count_task;
#pragma omp atomic capture
        send_count_task = Send_count[task0]++;
        index_task0[i] = send_count_task;
        if (task0 != task1)
        {
#pragma omp atomic capture
            send_count_task = Send_count[task1]++;
            index_task1[i] = send_count_task;
        }
    }

    int flagsum;
    MPI_Allreduce(&flag, &flagsum, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    TIMER_POP(1, "PM_COLLECT_PART");

    if (flagsum > 0)
    {
        myfree(task1_for_offset);
        myfree(task0_for_offset);
        myfree(index_task1);
        myfree(index_task0);
#ifdef KD_PM_NOSENDPOS
        myfree(part);
#endif // KD_PM_NOSENDPOS

        VERBOSE(0, "  PM: In total %d particles were outside allowed range.", flagsum);
        return 1; /* error - need to return because particles were outside allowed range */
    }

#ifdef KD_PM_FORCEOUTPUT
    VERBOSE_ALL(2, "PM OUT SendCount, Task %d: [%d,%d] [%d,%d,%d,%d]  ", ThisTask, flag, flagsum, Send_count[0],
                Send_count[1], Send_count[2], Send_count[3]);
#endif // KD_PM_FORCEOUTPUT

    TIMER_PUSH(1, "PM_CREATE_SEND_RECV");

    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

    nimport = nexport = Recv_offset[0] = Send_offset[0] = 0;
    for (int j = 0; j < NTask; j++)
    {
        nexport += Send_count[j];
        nimport += Recv_count[j];

        if (j > 0)
        {
            Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
            Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
        }
    }

#ifdef KD_PM_FORCEOUTPUT
    VERBOSE_ALL(2, "PM OUT SEND/RECV, Task %d: %d/%d [%d,%d,%d,%d] [%d,%d,%d,%d]", ThisTask, nexport, nimport,
                Send_offset[0], Send_offset[1], Send_offset[2], Send_offset[3], Recv_offset[0], Recv_offset[1],
                Recv_offset[2], Recv_offset[3]);
#endif // KD_PM_FORCEOUTPUT

#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        if (index_task0[i] >= 0)
        {
            index_task0[i] += Send_offset[task0_for_offset[i]];
            if (index_task1[i] >= 0)
            {
                index_task1[i] += Send_offset[task1_for_offset[i]];
            }
        }
    }

    myfree(task1_for_offset);
    myfree(task0_for_offset);

    /* allocate import and export buffer */

    partin = (struct partbuf *)mymalloc("partin", nimport * sizeof(struct partbuf));
    partout = (struct partbuf *)mymalloc("partout", nexport * sizeof(struct partbuf));

    TIMER_POPPUSH(1, "PM_CREATE_SEND_RECV", "PM_CREATE_PART_LIST");

#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        if (index_task0[i] >= 0)
        {
#ifdef KD_PM_NOSENDPOS
            partout[index_task0[i]] = part[i];
            if (index_task1[i] >= 0)
            {
                partout[index_task1[i]] = part[i];
            }
#else // KD_PM_NOSENDPOS
            for (int j = 0; j < 3; j++)
            {
                partout[index_task0[i]].Pos[j] = P[i].Pos[j];
            }
            partout[index_task0[i]].Mass = P[i].Mass;
            if (index_task1[i] >= 0)
            {
                for (int j = 0; j < 3; j++)
                {
                    partout[index_task1[i]].Pos[j] = P[i].Pos[j];
                }
                partout[index_task1[i]].Mass = P[i].Mass;
            }
#endif // KD_PM_NOSENDPOS
        }
    }

    /* exchange particle data */

    memcpy(&partin[Recv_offset[ThisTask]], &partout[Send_offset[ThisTask]],
           Recv_count[ThisTask] * sizeof(struct partbuf));
    TIMER_POPPUSH(1, "PM_CREATE_PART_LIST", "PM_SEND_PARTICLES");

#ifdef USE_ISEND_IRECV
    MPI_Request *requests;
    int nrequests = 0;

    requests = (MPI_Request *)mymalloc("requests", 2 * NTask * sizeof(MPI_Request));

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int task = ThisTask ^ ngrp;

        if (task < NTask)
        {
            MPI_Irecv(&partin[Recv_offset[task]], Recv_count[task] * sizeof(struct partbuf), MPI_BYTE, task, TAG_DENS_A,
                      MYMPI_COMM_WORLD, &requests[nrequests++]);
        }
    }

    MPI_Barrier(MYMPI_COMM_WORLD);

    for (int ngrp = 1; ngrp < (1 << PTask); ngrp++)
    {
        int task = ThisTask ^ ngrp;

        if (task < NTask)
        {
            MPI_Isend(&partout[Send_offset[task]], Send_count[task] * sizeof(struct partbuf), MPI_BYTE, task,
                      TAG_DENS_A, MYMPI_COMM_WORLD, &requests[nrequests++]);
        }
    }

    MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
    myfree(requests);
#else // USE_ISEND_IRECV
    templated_exchange_data_Sendrecv(partout, partin, TAG_DENS_A, Send_offset, Send_count, Recv_offset, Recv_count);
#endif // USE_ISEND_IRECV
    TIMER_POPPUSH(1, "PM_SEND_PARTICLES", "PM_ZERO_RHO_GRID");

    myfree(partout);

    /* allocate density field */
    fftw_real *rhogrid = (fftw_real *)mymalloc("rhogrid", fft_params->maxfftsize * sizeof(fftw_real));

#pragma omp parallel for
    for (large_array_offset ii = 0; ii < fft_params->maxfftsize; ii++)
    {
        rhogrid[ii] = 0;
    }

    TIMER_POPPUSH(1, "PM_ZERO_RHO_GRID", "PM_FILL_RHO_GRID");

    pm_binning(mode_periodic, fft_params, rhogrid, nimport, partin); /* bin particle data onto mesh */

#ifdef KD_PM_FORCEOUTPUT
    VERBOSE_ALL(0, "PM OUT rhogrid, Task %d: [%g,%g,%g] ", ThisTask, rhogrid[0], rhogrid[100], rhogrid[10000]);
#endif // KD_PM_FORCEOUTPUT

    /* Do the FFT of the density field */

    TIMER_POPPUSH(1, "PM_FILL_RHO_GRID", "PM_FFTW_FRWD");

    report_memory_usage(&HighMark_pmperiodic, "PM");

#ifdef _OPENMP
    FFTW(plan_with_nthreads)(omp_get_num_threads());
#endif // _OPENMP

    FFTW(complex) *fft_of_rhogrid = (FFTW(complex) *)&rhogrid[0];

    fft_forward_plan =
        FFTW(mpi_plan_dft_r2c_3d)(fft_params->grid, fft_params->grid, fft_params->grid, rhogrid, fft_of_rhogrid,
                                  MYMPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | FFTW_MPI_TRANSPOSED_OUT);
    FFTW(execute)(fft_forward_plan);

    FFTW(destroy_plan)(fft_forward_plan);

#ifdef KD_PM_FORCEOUTPUT
    VERBOSE_ALL(0, "PM OUT rhogrid_fft, Task %d: [%g,%g,%g]", ThisTask, rhogrid[0], rhogrid[100], rhogrid[10000]);
#endif // KD_PM_FORCEOUTPUT

    /* multiply with the Fourier transform of the kernel/Green's function */

    TIMER_POPPUSH(1, "PM_FFTW_FRWD", "PM_MULT_GREENS");

#ifdef PERIODIC
    if (mode_periodic == 2)
    {
        pm_greens_periodic(fft_params, fft_of_rhogrid);
    }
#endif // PERIODIC
#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
    if (mode_periodic != 2)
    {
        pm_greens_nonperiodic(mode_periodic, fft_params, fft_of_rhogrid);
    }
#endif // !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

#ifdef KD_PM_FORCEOUTPUT
    VERBOSE_ALL(0, "PM OUT rhogrid_fft_greens, Task %d: [%g,%g,%g]", ThisTask, rhogrid[0], rhogrid[100],
                rhogrid[10000]);
#endif // KD_PM_FORCEOUTPUT

    TIMER_POPPUSH(1, "PM_MULT_GREENS", "PM_FFTW_INV");

    /* Do the inverse FFT to get the potential */
    VERBOSE(2, "  PM: Do the inverse FFT to get the potential");

#ifdef _OPENMP
    FFTW(plan_with_nthreads)(omp_get_num_threads());
#endif // _OPENMP
    fft_inverse_plan =
        FFTW(mpi_plan_dft_c2r_3d)(fft_params->grid, fft_params->grid, fft_params->grid, fft_of_rhogrid, rhogrid,
                                  MYMPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | FFTW_MPI_TRANSPOSED_IN);
    FFTW(execute)(fft_inverse_plan);
    FFTW(destroy_plan)(fft_inverse_plan);
    /* Now rhogrid holds the potential */

#ifdef KD_PM_FORCEOUTPUT
    VERBOSE_ALL(0, "PM OUT rhogrid_fft_back, Task %d: [%g,%g,%g]", ThisTask, rhogrid[0], rhogrid[100], rhogrid[10000]);
#endif // KD_PM_FORCEOUTPUT

    TIMER_POP(1, "PM_FFTW_INV");

#ifdef EVALPOTENTIAL
    pm_readout(mode_periodic, -1, fac, rhogrid, fft_params, partin, nimport, nexport, index_task0, index_task1);
#endif // EVALPOTENTIAL

    /* get the force components by finite differencing the potential for each dimension,
       and send back the results to the right CPUs */
    pm_get_forces(mode_periodic, fac, rhogrid, fft_params, partin, nimport, nexport, index_task0, index_task1);

#ifdef KD_PM_FORCEOUTPUT
    VERBOSE_ALL(0, "PM OUT FORCE, Task %d: PM[%llu][%g,%g,%g] PM[%llu][%g,%g,%g] PM[%llu][%g,%g,%g]", ThisTask,
                (unsigned long long)P[0].ID, P[0].GravPM[0], P[0].GravPM[1], P[0].GravPM[2],
                (unsigned long long)P[100].ID, P[100].GravPM[0], P[100].GravPM[1], P[100].GravPM[2],
                (unsigned long long)P[10000].ID, P[10000].GravPM[0], P[10000].GravPM[1], P[10000].GravPM[2]);
    /*
       if(ThisTask == 0)
       for(int ii = 0; ii < NumPart; ii++)
       printf("PM OUT ALL force, Task %d: %g %g %g ", ThisTask, P[ii].GravPM[0],P[ii].GravPM[1],P[ii].GravPM[2]);
     */
#endif // KD_PM_FORCEOUTPUT

    myfree(rhogrid);
    myfree(partin);
    myfree(index_task1);
    myfree(index_task0);
#ifdef KD_PM_NOSENDPOS
    myfree(part);
#endif // KD_PM_NOSENDPOS

    VERBOSE(1, "  PM: done!");
    return 0;
}

#endif // PMGRID
