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
#include <vector>
#include "pm.hpp"
#include "bounds_init.hpp"

#ifdef PMGRID
/**
 * @brief Initializes the Particle-Mesh (PM) method.
 *
 * This function sets up the FFT parameters and allocates memory for the nonperiodic kernel
 * used in the PM method. It initializes FFT plans for periodic and nonperiodic cases,
 * allocates memory for the kernel, and computes the total memory allocated.
 */
void pm_init(void)
{
    double bytes_tot = 0.0;

#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
    size_t bytes;
#endif // !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

    /* Set up the FFTW plans */
#ifdef PERIODIC
    pm_fftw_init(PMGRID, &fft_params_periodic);
#endif // PERIODIC

#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
    pm_fftw_init(GRID, &fft_params_nonperiodic);
#endif // !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

#ifndef PERIODIC
    if (!(kernel[0] =
              (fftw_real *)mymalloc("kernel[0]", bytes = fft_params_nonperiodic.maxfftsize * sizeof(fftw_real))))
    {
        ENDRUN_MAIN("PM:failed to allocate memory for `FFT-kernel[0]' (%g MB).", bytes / (1024.0 * 1024.0));
    }
    bytes_tot += bytes;
    fft_of_kernel[0] = (FFTW(complex) *)kernel[0];
#endif // PERIODIC

#ifdef PLACEHIGHRESREGION
    if (!(kernel[1] =
              (fftw_real *)mymalloc("kernel[1]", bytes = fft_params_nonperiodic.maxfftsize * sizeof(fftw_real))))
    {
        ENDRUN_MAIN("PM: failed to allocate memory for `FFT-kernel[1]' (%g MB).", bytes / (1024.0 * 1024.0));
    }
    bytes_tot += bytes;
    fft_of_kernel[1] = (FFTW(complex) *)kernel[1];
#endif // PLACEHIGHRESREGION

    VERBOSE(0, "PM: Allocated %g MByte for FFT kernel(s).", bytes_tot / (1024.0 * 1024.0));
}

/**
 * @brief Initializes the FFT parameters for the Particle-Mesh method.
 *
 * This function sets up the FFT parameters required for the Particle-Mesh (PM) method,
 * including grid size, MPI communication, and memory allocation. It computes the local FFT size
 * and the slabs per task based on the grid size and MPI communicator. Additionally, it calculates
 * the smallest slab size and the maximum FFT size across all tasks.
 *
 * @param grid The size of the FFT grid.
 * @param fft_params Pointer to the FFT parameters structure to be initialized.
 */
void pm_fftw_init(int grid, struct fftw_parameters *fft_params)
{

    // int slab_to_task_local[grid];
    std::vector<int> slab_to_task_local(grid);

    /* Set up the FFTW plan files. */
    FFTW(mpi_init)();
#ifdef _OPENMP
    FFTW(init_threads)();
#endif // _OPENMP

    fft_params->grid = grid;
    fft_params->gridz = fft_params->grid / 2 + 1;
    fft_params->grid2 = 2 * fft_params->gridz;

    fft_params->fftsize = FFTW(mpi_local_size_3d_transposed)(
        fft_params->grid, fft_params->grid, fft_params->grid2, MYMPI_COMM_WORLD, &fft_params->nslab_x,
        &fft_params->slabstart_x, &fft_params->nslab_y, &fft_params->slabstart_y);

    dofftw = fft_params->nslab_x;
    VERBOSE(2, "Task %d: (pm_init) dofftw=%d ", ThisTask, dofftw);

    fft_params->fftsize *= 2;

    VERBOSE(1, "PM: init (Task 0: fftsize=%td nslab_x=%td slabstart_x=%td, nslab_y=%td slabstart_y=%td)  ",
            fft_params->fftsize, fft_params->nslab_x, fft_params->slabstart_x, fft_params->nslab_y,
            fft_params->slabstart_y);

    for (int i = 0; i < grid; i++)
    {
        slab_to_task_local[i] = 0;
    }

    for (int i = 0; i < fft_params->nslab_x; i++)
    {
        slab_to_task_local[fft_params->slabstart_x + i] = ThisTask;
    }

    fft_params->slab_to_task = (int *)mymalloc("slab_to_task", grid * sizeof(int));
    MPI_Allreduce(slab_to_task_local.data(), fft_params->slab_to_task, grid, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    fft_params->slabs_per_task = (int *)mymalloc("slabs_per_task", NTask * sizeof(int));
    fft_params->first_slab_of_task = (int *)mymalloc("first_slab_of_task", NTask * sizeof(int));

#ifndef NO_AINT_IN_MPI
    VERBOSE(
        2,
        "fftw_params->nslap_x (and others) are of type ptrdiff_t and therefore needs MPI_AINT in the "
        "Allreduce.\nHowever, some old MPI implementation seems to fail on this.\nIf the code produces a Segmentation "
        "fault here, switch on NO_AINT_IN_MPI to use MPI_INT\n(although it is formally wrong it seems to work). ");

    MPI_Allreduce(&fft_params->nslab_x, &fft_params->smallest_slab, 1, MPI_AINT, MPI_MIN, MYMPI_COMM_WORLD);
    // I m almost sure that formally once would need to use AINT for the send type here as well, but seems to fail
    MPI_Allgather(&fft_params->nslab_x, 1, MPI_INT, fft_params->slabs_per_task, 1, MPI_INT, MYMPI_COMM_WORLD);
    MPI_Allgather(&fft_params->slabstart_x, 1, MPI_INT, fft_params->first_slab_of_task, 1, MPI_INT, MYMPI_COMM_WORLD);
    MPI_Allreduce(&fft_params->fftsize, &fft_params->maxfftsize, 1, MPI_AINT, MPI_MAX, MYMPI_COMM_WORLD);
#else // NO_AINT_IN_MPI
    MPI_Allreduce(&fft_params->nslab_x, &fft_params->smallest_slab, 1, MPI_INT, MPI_MIN, MYMPI_COMM_WORLD);
    MPI_Allgather(&fft_params->nslab_x, 1, MPI_INT, fft_params->slabs_per_task, 1, MPI_INT, MYMPI_COMM_WORLD);
    MPI_Allgather(&fft_params->slabstart_x, 1, MPI_INT, fft_params->first_slab_of_task, 1, MPI_INT, MYMPI_COMM_WORLD);
    MPI_Allreduce(&fft_params->fftsize, &fft_params->maxfftsize, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
#endif // NO_AINT_IN_MPI

    VERBOSE(3, "done ... ");
}

/**
 * @brief This function initializes the FFTW library for the PM calculation
 *
 * This function determines the particle extension of all particles, and for
 * those types selected with PLACEHIGHRESREGION if this is used, and then
 * determines the boundaries of the non-periodic FFT-mesh that can be placed
 * on this region. Note that a sufficient buffer region at the rim of the
 * occupied part of the mesh needs to be reserved in order to allow a correct
 * finite differencing using a 4-point formula. In addition, to allow
 * non-periodic boundaries, the actual FFT mesh used is twice as large in
 * each dimension compared with PMGRID.
 */
void pm_init_regionsize(void)
{
    /* find enclosing rectangle */
#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
    double xmin[2][3], xmax[2][3];

    init_bounds(xmin, xmax);

    MPI_Allreduce(xmin, All.Xmintot, 6, MPI_DOUBLE, MPI_MIN, MYMPI_COMM_WORLD);
    MPI_Allreduce(xmax, All.Xmaxtot, 6, MPI_DOUBLE, MPI_MAX, MYMPI_COMM_WORLD);

    compute_and_symmetrize_mesh_region();

    /* this will produce enough room for zero-padding and buffer region to
       allow finite differencing of the potential  */
    double meshinner[2];
    init_mesh(meshinner);
#endif // !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

    init_box_corners();
    All.TotalMeshSize[2] = All.BoxSize;

#ifndef PERIODIC
    All.Asmth[0] = Asmth * All.TotalMeshSize[0] / GRID;
    All.Rcut[0] = Rcut * All.Asmth[0];
#else // PERIODIC
    All.Asmth[0] = Asmth * All.BoxSize / PMGRID;
    All.Rcut[0] = Rcut * All.Asmth[0];
#endif // PERIODIC

#ifdef PLACEHIGHRESREGION
    All.Asmth[1] = Asmth * All.TotalMeshSize[1] / GRID;
    All.Rcut[1] = Rcut * All.Asmth[1];
#endif // PLACEHIGHRESREGION

#ifdef PLACEHIGHRESREGION
    if (2 * All.TotalMeshSize[1] / GRID < All.Rcut[0])
    {
        All.TotalMeshSize[1] = 2 * (meshinner[1] + 2 * All.Rcut[0]) * (GRID) / ((double)(GRID - 2));

        init_pm_region_corners();

        if (2 * All.TotalMeshSize[1] / GRID > All.Rcut[0])
        {
            All.TotalMeshSize[1] = 2 * (meshinner[1] + 2 * All.Rcut[0]) * (GRID) / ((double)(GRID - 10));
            init_pm_region_corners2();
        }

        All.Asmth[1] = Asmth * All.TotalMeshSize[1] / GRID;
        All.Rcut[1] = Rcut * All.Asmth[1];
    }
#endif // PLACEHIGHRESREGION

#ifndef PERIODIC
    VERBOSE(2, "Allowed region for isolated PM mesh (coarse): ");
    VERBOSE(2, "(%g|%g|%g)  -> (%g|%g|%g)   ext=%g  totmeshsize=%g  meshsize=%g  ", All.Xmintot[0][0],
            All.Xmintot[0][1], All.Xmintot[0][2], All.Xmaxtot[0][0], All.Xmaxtot[0][1], All.Xmaxtot[0][2], meshinner[0],
            All.TotalMeshSize[0], All.TotalMeshSize[0] / GRID);
#endif // PERIODIC
#ifdef PLACEHIGHRESREGION
    VERBOSE(2, "Allowed region for isolated PM mesh (high-res): ");
    VERBOSE(2, "(%g|%g|%g)  -> (%g|%g|%g)   ext=%g  totmeshsize=%g  meshsize=%g  ", All.Xmintot[1][0],
            All.Xmintot[1][1], All.Xmintot[1][2], All.Xmaxtot[1][0], All.Xmaxtot[1][1], All.Xmaxtot[1][2], meshinner[1],
            All.TotalMeshSize[1], All.TotalMeshSize[1] / GRID);
    fflush(stdout);
#endif // PLACEHIGHRESREGION

#ifdef PLACEHIGHRESREGION
    if (ThisTask == 0)
    {
        fprintf(FdPM, "Time: %g Mesh0: %d Reg: (%e|%e|%e)->(%e|%e|%e) ext=%e totms=%e ms=%e\n", All.Time, PMGRID,
                All.Xmintot[0][0], All.Xmintot[0][1], All.Xmintot[0][2], All.Xmaxtot[0][0], All.Xmaxtot[0][1],
                All.Xmaxtot[0][2], meshinner[0], All.TotalMeshSize[0], All.TotalMeshSize[0] / PMGRID);
        fprintf(FdPM, "Time: %g Mesh1: %d Reg: (%e|%e|%e)->(%e|%e|%e) ext=%e totms=%e ms=%e\n", All.Time,
                HIGHRES_PMGRID, All.Xmintot[1][0], All.Xmintot[1][1], All.Xmintot[1][2], All.Xmaxtot[1][0],
                All.Xmaxtot[1][1], All.Xmaxtot[1][2], meshinner[1], All.TotalMeshSize[1], All.TotalMeshSize[1] / GRID);
        fflush(FdPM);
    }
#endif // PLACEHIGHRESREGION
}

#endif // PMGRID
