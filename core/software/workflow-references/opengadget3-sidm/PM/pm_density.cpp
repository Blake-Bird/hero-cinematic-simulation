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

#include "pm.hpp"

/*! \file pm_periodic.c
 *  \brief routines for periodic PM-force computation
 */

#ifdef PMGRID
#ifndef KD_PM_NOSENDPOS
/**
 * @brief Performs particle binning for particle mesh (PM) method.
 *
 * This function performs particle binning for the PM method. It assigns particle
 * masses to the grid cells based on their positions. The particles are distributed
 * among grid cells using linear interpolation.
 *
 * @param mode The mode of the PM grid (e.g., X, Y, Z direction).
 * @param fft_params Pointer to the FFTW parameters struct.
 * @param rhogrid Pointer to the density grid.
 * @param nimport Number of particles to be imported.
 * @param partin Pointer to the particle buffer containing particle data.
 */
void pm_binning(int mode, struct fftw_parameters *fft_params, fftw_real *rhogrid, int nimport, struct partbuf *partin)
{
    MyFloat to_slab_fac = fft_params->grid / All.TotalMeshSize[mode];

    double corner[3];
    for (int j = 0; j < 3; j++)
    {
        corner[j] = All.Corner[mode][j];
    }

#pragma omp parallel
    {
        int thread_id, nthreads;
        int flag_slab_x, flag_slab_xx;
        int flag_slab_y, flag_slab_yy;
        int y_begin, y_count;

#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#else // _OPENMP
        thread_id = 0;
        nthreads = 1;
#endif // _OPENMP

        pm_divide_threads(fft_params->grid, nthreads, thread_id, &y_begin, &y_count);

        int y_end = y_begin + y_count - 1;

        for (int i = 0; i < nimport; i++)
        {

            int slab_y = (int)(to_slab_fac * (partin[i].Pos[1] - corner[1]));
            double dy = to_slab_fac * (partin[i].Pos[1] - corner[1]) - slab_y;
            if (mode == 2)
            {
                if (slab_y >= fft_params->grid)
                {
                    slab_y -= fft_params->grid;
                }
            }

            int slab_yy = slab_y + 1;
            if (mode == 2)
            {
                if (slab_yy >= fft_params->grid)
                {
                    slab_yy -= fft_params->grid;
                }
            }

            if (slab_y >= y_begin && slab_y <= y_end)
            {
                flag_slab_y = 1;
            }
            else
            {
                flag_slab_y = 0;
            }

            if (slab_yy >= y_begin && slab_yy <= y_end)
            {
                flag_slab_yy = 1;
            }
            else
            {
                flag_slab_yy = 0;
            }

            if (flag_slab_y || flag_slab_yy)
            {
                int slab_x = (int)(to_slab_fac * (partin[i].Pos[0] - corner[0]));
                int slab_z = (int)(to_slab_fac * (partin[i].Pos[2] - corner[2]));

                double dx = to_slab_fac * (partin[i].Pos[0] - corner[0]) - slab_x;
                double dz = to_slab_fac * (partin[i].Pos[2] - corner[2]) - slab_z;

                if (mode == 2)
                {
                    if (slab_x >= fft_params->grid)
                    {
                        slab_x -= fft_params->grid;
                    }
                    if (slab_z >= fft_params->grid)
                    {
                        slab_z -= fft_params->grid;
                    }
                }

                int slab_xx = slab_x + 1;
                int slab_zz = slab_z + 1;

                if (mode == 2)
                {
                    if (slab_xx >= fft_params->grid)
                    {
                        slab_xx -= fft_params->grid;
                    }
                    if (slab_zz >= fft_params->grid)
                    {
                        slab_zz -= fft_params->grid;
                    }
                }

                MyFloat mass = partin[i].Mass;

                if (fft_params->slab_to_task[slab_x] == ThisTask)
                {
                    slab_x -= fft_params->first_slab_of_task[ThisTask];
                    flag_slab_x = 1;
                }
                else
                {
                    flag_slab_x = 0;
                }

                if (fft_params->slab_to_task[slab_xx] == ThisTask)
                {
                    slab_xx -= fft_params->first_slab_of_task[ThisTask];
                    flag_slab_xx = 1;
                }
                else
                {
                    flag_slab_xx = 0;
                }

                if (flag_slab_x)
                {
                    if (flag_slab_y)
                    {
                        rhogrid[INDEX(fft_params, slab_x, slab_y, slab_z)] +=
                            (fftw_real)(mass * ((1.0 - dx) * (1.0 - dy) * (1.0 - dz)));
                        rhogrid[INDEX(fft_params, slab_x, slab_y, slab_zz)] +=
                            (fftw_real)(mass * ((1.0 - dx) * (1.0 - dy) * (dz)));
                    }

                    if (flag_slab_yy)
                    {
                        rhogrid[INDEX(fft_params, slab_x, slab_yy, slab_z)] +=
                            (fftw_real)(mass * ((1.0 - dx) * (dy) * (1.0 - dz)));
                        rhogrid[INDEX(fft_params, slab_x, slab_yy, slab_zz)] +=
                            (fftw_real)(mass * ((1.0 - dx) * (dy) * (dz)));
                    }
                }

                if (flag_slab_xx)
                {
                    if (flag_slab_y)
                    {
                        rhogrid[INDEX(fft_params, slab_xx, slab_y, slab_z)] +=
                            (fftw_real)(mass * ((dx) * (1.0 - dy) * (1.0 - dz)));
                        rhogrid[INDEX(fft_params, slab_xx, slab_y, slab_zz)] +=
                            (fftw_real)(mass * ((dx) * (1.0 - dy) * (dz)));
                    }

                    if (flag_slab_yy)
                    {
                        rhogrid[INDEX(fft_params, slab_xx, slab_yy, slab_z)] +=
                            (fftw_real)(mass * ((dx) * (dy) * (1.0 - dz)));
                        rhogrid[INDEX(fft_params, slab_xx, slab_yy, slab_zz)] +=
                            (fftw_real)(mass * ((dx) * (dy) * (dz)));
                    }
                }
            }
        }
    }
}
#else // KD_PM_NOSENDPOS
/**
 * @brief Performs particle binning for the particle mesh (PM) method.
 *
 * This function performs particle binning for the PM method, which assigns particle masses
 * to grid cells based on their positions. The particles are distributed among grid cells
 * using linear interpolation. This parallel implementation uses OpenMP for concurrent processing
 * of particle binning tasks.
 *
 * @param mode The mode of the PM grid (e.g., X, Y, Z direction).
 * @param fft_params Pointer to the FFTW parameters struct.
 * @param rhogrid Pointer to the density grid.
 * @param nimport Number of particles to be imported.
 * @param partin Pointer to the particle buffer containing particle data.
 */
void pm_binning(int mode, struct fftw_parameters *fft_params, fftw_real *rhogrid, int nimport, struct partbuf *partin)
{
#pragma omp parallel for
    for (int i = 0; i < nimport; i++)
    {
        large_array_offset index[8];
        fftw_real dm[8];

        int slab_x = partin[i].slab_x;
        int slab_y = partin[i].slab_y;
        int slab_z = partin[i].slab_z;
        int slab_xx = slab_x + 1;
        int slab_yy = slab_y + 1;
        int slab_zz = slab_z + 1;
        MyFloat dx = partin[i].dx;
        MyFloat dy = partin[i].dy;
        MyFloat dz = partin[i].dz;

        if (mode == 2)
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

        int j = 0, flag_slab_x, flag_slab_xx;
        double mass = partin[i].Mass;

        if (fft_params->slab_to_task[slab_x] == ThisTask)
        {
            slab_x -= fft_params->first_slab_of_task[ThisTask];
            flag_slab_x = 1;
        }
        else
        {
            flag_slab_x = 0;
        }

        if (fft_params->slab_to_task[slab_xx] == ThisTask)
        {
            slab_xx -= fft_params->first_slab_of_task[ThisTask];
            flag_slab_xx = 1;
        }
        else
        {
            flag_slab_xx = 0;
        }

        if (flag_slab_x)
        {
            index[j] = INDEX(fft_params, slab_x, slab_y, slab_z);
            dm[j] = (fftw_real)(mass * ((1.0 - dx) * (1.0 - dy) * (1.0 - dz)));
            j++;
            index[j] = INDEX(fft_params, slab_x, slab_y, slab_zz);
            dm[j] = (fftw_real)(mass * ((1.0 - dx) * (1.0 - dy) * (dz)));
            j++;
            index[j] = INDEX(fft_params, slab_x, slab_yy, slab_z);
            dm[j] = (fftw_real)(mass * ((1.0 - dx) * (dy) * (1.0 - dz)));
            j++;
            index[j] = INDEX(fft_params, slab_x, slab_yy, slab_zz);
            dm[j] = (fftw_real)(mass * ((1.0 - dx) * (dy) * (dz)));
            j++;
        }

        if (flag_slab_xx)
        {
            index[j] = INDEX(fft_params, slab_xx, slab_y, slab_z);
            dm[j] = (fftw_real)(mass * ((dx) * (1.0 - dy) * (1.0 - dz)));
            j++;
            index[j] = INDEX(fft_params, slab_xx, slab_y, slab_zz);
            dm[j] = (fftw_real)(mass * ((dx) * (1.0 - dy) * (dz)));
            j++;
            index[j] = INDEX(fft_params, slab_xx, slab_yy, slab_z);
            dm[j] = (fftw_real)(mass * ((dx) * (dy) * (1.0 - dz)));
            j++;
            index[j] = INDEX(fft_params, slab_xx, slab_yy, slab_zz);
            dm[j] = (fftw_real)(mass * ((dx) * (dy) * (dz)));
            j++;
        }

        for (int k = 0; k < j; k++)
        {
#pragma omp atomic
            rhogrid[index[k]] += dm[k];
        }
    }
}
#endif // KD_PM_NOSENDPOS

#ifdef PERIODIC
/**
 * @brief Computes the Green's function for periodic particle mesh (PM) method.
 *
 * This function computes the Green's function for the periodic particle mesh (PM) method.
 * The Green's function is calculated using fast Fourier transforms (FFTs) and deconvolution
 * to account for the finite size of the computational grid and smooth the force computation.
 *
 * @param fft_params Pointer to the FFTW parameters struct.
 * @param fft_of_rhogrid Pointer to the FFT of the density grid.
 */
void pm_greens_periodic(struct fftw_parameters *fft_params, FFTW(complex) * fft_of_rhogrid)
{
    TIMER_PUSH(1, "PM_computing_lookup_tables");

    double asmth2 = (2 * M_PI) * All.Asmth[0] / All.BoxSize;
    asmth2 *= asmth2;
    int k0 = fft_params->grid / 2;

#ifdef KD_PM_FASTGREENS
    int tab_size = fft_params->grid + 1;
    double *f_tab;
    f_tab = (double *)mymalloc("PM_f_table", tab_size * sizeof(double));

#pragma omp parallel for
    for (int ik = -fft_params->grid / 2; ik <= fft_params->grid / 2; ik++)
    {
        if (ik != 0)
        {
            double f = (M_PI * ik) / fft_params->grid;
            f_tab[ik + k0] = sin(f) / f;
        }
        else
            f_tab[ik + k0] = 1;
    }
#endif // KD_PM_FASTGREENS
    TIMER_POPPUSH(1, "PM_computing_lookup_tables", "PM_multiplying_grid");

    for (int y = fft_params->slabstart_y; y < fft_params->slabstart_y + fft_params->nslab_y; y++)
    {
        int ky;
        if (y > k0)
        {
            ky = y - fft_params->grid;
        }
        else
        {
            ky = y;
        }
#ifdef KD_PM_FASTGREENS
        double fy = f_tab[ky + k0];
#else // KD_PM_FASTGREENS
        double fy = 1;
        if (ky != 0)
        {
            fy = (M_PI * ky) / fft_params->grid;
            fy = sin(fy) / fy;
        }
#endif // KD_PM_FASTGREENS

#pragma omp parallel for
        for (int jk = 0; jk < fft_params->grid * fft_params->gridz; jk++)
        {
            int x = jk / fft_params->gridz;
            int z = jk % fft_params->gridz;

            int kx;
            if (x > k0)
            {
                kx = x - fft_params->grid;
            }
            else
            {
                kx = x;
            }
#ifdef KD_PM_FASTGREENS
            double fx = f_tab[kx + k0];
#else // KD_PM_FASTGREENS
            double fx = 1;
            if (kx != 0)
            {
                fx = (M_PI * kx) / fft_params->grid;
                fx = sin(fx) / fx;
            }
#endif // KD_PM_FASTGREENS

            int kz;
            if (z > k0)
            {
                kz = z - fft_params->grid;
            }
            else
            {
                kz = z;
            }
#ifdef KD_PM_FASTGREENS
            double fz = f_tab[kz + k0];
#else // KD_PM_FASTGREENS
            double fz = 1;
            if (kz != 0)
            {
                fz = (M_PI * kz) / fft_params->grid;
                fz = sin(fz) / fz;
            }
#endif // KD_PM_FASTGREENS

            int k2 = kx * kx + ky * ky + kz * kz;
            if (k2 > 0)
            {
                MyAtLeastDouble smth = -exp(-k2 * asmth2) / k2;

                /* do deconvolution */
                double ff = 1 / (fx * fy * fz);
                smth *= ff * ff * ff * ff;

                /* end deconvolution */
                large_array_offset ip = fft_params->gridz * (fft_params->grid * (y - fft_params->slabstart_y) + x) + z;

                fft_of_rhogrid[ip][0] *= (fftw_real)smth;
                fft_of_rhogrid[ip][1] *= (fftw_real)smth;
            }
        }

        if (fft_params->slabstart_y == 0)
        {
            fft_of_rhogrid[0][0] = fft_of_rhogrid[0][1] = 0.0;
        }
    }

    TIMER_POP(1, "PM_multiplying_grid");
#ifdef KD_PM_FASTGREENS
    myfree(f_tab);
#endif // KD_PM_FASTGREENS
}
#endif // PERIODIC

#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

FFTW(complex) * fft_of_kernel[2];
fftw_real *kernel[2];

/**
 * @brief Applies the Green's function for non-periodic boundary conditions.
 *
 * This function applies the Green's function for non-periodic boundary conditions
 * to the FFT of the density grid in the PM method. It multiplies each Fourier mode
 * of the density grid by the corresponding mode of the Green's function kernel.
 *
 * @param mode_periodic Flag indicating the periodic mode.
 * @param fft_params Pointer to the FFTW parameters struct.
 * @param fft_of_rhogrid Pointer to the FFT of the density grid.
 */
void pm_greens_nonperiodic(int mode_periodic, struct fftw_parameters *fft_params, FFTW(complex) * fft_of_rhogrid)
{
    // Loop over y-slabs
    for (int y = 0; y < fft_params->nslab_y; y++)
#pragma omp parallel for
        // Loop over grid cells in x and z dimensions
        for (int jk = 0; jk < fft_params->grid * fft_params->gridz; jk++)
        {
            int x = jk / fft_params->gridz;
            int z = jk % fft_params->gridz;
            // Compute the index in the FFT array
            large_array_offset ip = fft_params->gridz * (fft_params->grid * y + x) + z;
            // Compute real and imaginary parts of the transformed density
            double re = fft_of_rhogrid[ip][0] * fft_of_kernel[mode_periodic][ip][0] -
                        fft_of_rhogrid[ip][1] * fft_of_kernel[mode_periodic][ip][1];
            double im = fft_of_rhogrid[ip][0] * fft_of_kernel[mode_periodic][ip][1] +
                        fft_of_rhogrid[ip][1] * fft_of_kernel[mode_periodic][ip][0];

            // Update the FFT of the density grid with the transformed values
            fft_of_rhogrid[ip][0] = re;
            fft_of_rhogrid[ip][1] = im;
        }
}

#ifdef PLACEHIGHRESREGION
/**
 * @brief Checks if a particle is in a high-resolution region.
 *
 * This function determines if a particle is located in a high-resolution region
 * based on its type and position. High-resolution regions are defined by a flag
 * set in the `PLACEHIGHRESREGION` variable. Additionally, the function checks if
 * the particle's position falls within the boundaries defined by `All.Xmintot` and
 * `All.Xmaxtot` arrays for the high-resolution region.
 *
 * @param type The type of the particle.
 * @param Pos The position of the particle.
 * @return Returns 1 if the particle is in a high-resolution region, 0 otherwise.
 */
int pmforce_is_particle_high_res(int type, MyLongDouble *Pos)
{
    int flag = 1;
    // Check if the particle type corresponds to a high-resolution region
    if ((1 << type) & (PLACEHIGHRESREGION))
    {
        return 1;
    }
    // Check if the particle's position falls within the high-resolution region boundaries
    for (int j = 0; j < 3; j++)
    {
        if (Pos[j] < All.Xmintot[1][j] || Pos[j] > All.Xmaxtot[1][j])
        {
            flag = 0; // Particle is outside the high-resolution region
            break;
        }
    }
    return flag;
}
#endif // PLACEHIGHRESREGION

/**
 * @brief Sets up the Greens function for the non-periodic potential in real space
 * and converts it to Fourier space by means of a Fast Fourier Transform (FFT).
 *
 * This function initializes the Greens function for the non-periodic potential
 * in real space and then converts it to Fourier space using an FFT. The Greens
 * function is computed for each slab of the FFT grid. Deconvolution with the Cloud-In-Cell
 * (CIC) kernel is applied to the Fourier transformed Greens function to account
 * for the grid spacing.
 */
void pm_setup_nonperiodic_kernel(void)
{
    /*
       int i, j, k, jk, x, y, z;
       double xx, yy, zz, r, u, fac;
       double kx, ky, kz, k2, fx, fy, fz, ff;
       size_t ip;
     */

    struct fftw_parameters *fft_params = &fft_params_nonperiodic;

    VERBOSE(0, "PM: Setting up non-periodic kernel (grid=%d)  presently allocated=%g MB)", (int)fft_params->grid,
            AllocatedBytes / (1024.0 * 1024.0));

    /* now set up kernel and its Fourier transform */

#ifndef PERIODIC

#pragma omp parallel for
    for (int i = 0; i < fft_params->maxfftsize; i++) /* clear local field */
    {
        kernel[0][i] = 0;
    }

    for (int i = fft_params->slabstart_x; i < (fft_params->slabstart_x + fft_params->nslab_x); i++)
#pragma omp parallel for
        for (int jk = 0; jk < fft_params->grid * fft_params->grid; jk++)
        {
            int j = jk / fft_params->grid;
            int k = jk % fft_params->grid;

            double xx = ((double)i) / fft_params->grid;
            double yy = ((double)j) / fft_params->grid;
            double zz = ((double)k) / fft_params->grid;

            if (xx >= 0.5)
            {
                xx -= 1.0;
            }
            if (yy >= 0.5)
            {
                yy -= 1.0;
            }
            if (zz >= 0.5)
            {
                zz -= 1.0;
            }

            double r = sqrt(xx * xx + yy * yy + zz * zz);
            double u = 0.5 * r / (Asmth / fft_params->grid);
            double fac = 1 - erfc(u);
            size_t ip = INDEX(fft_params, (i - fft_params->slabstart_x), j, k);

            if (r > 0)
            {
                kernel[0][ip] = -fac / r;
            }
            else
            {
                kernel[0][ip] = -1 / (sqrt(M_PI) * (Asmth / fft_params->grid));
            }
        }

        /* do the forward transform of the kernel */
#ifdef _OPENMP
    FFTW(plan_with_nthreads)(omp_get_num_threads());
#endif // _OPENMP

    FFTW(plan)
    fft_forward_plan =
        FFTW(mpi_plan_dft_r2c_3d)(fft_params->grid, fft_params->grid, fft_params->grid, kernel[0], fft_of_kernel[0],
                                  MYMPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | FFTW_MPI_TRANSPOSED_OUT);

    FFTW(execute)(fft_forward_plan);
    FFTW(destroy_plan)(fft_forward_plan);

#endif // PERIODIC

#ifdef PLACEHIGHRESREGION
#pragma omp parallel for
    for (int i = 0; i < fft_params->maxfftsize; i++) /* clear local field */
    {
        kernel[1][i] = 0;
    }

    for (int i = fft_params->slabstart_x; i < (fft_params->slabstart_x + fft_params->nslab_x); i++)
#pragma omp parallel for
        for (int jk = 0; jk < fft_params->grid * fft_params->grid; jk++)
        {
            int j = jk / fft_params->grid;
            int k = jk % fft_params->grid;

            double xx = ((double)i) / fft_params->grid;
            double yy = ((double)j) / fft_params->grid;
            double zz = ((double)k) / fft_params->grid;

            if (xx >= 0.5)
            {
                xx -= 1.0;
            }
            if (yy >= 0.5)
            {
                yy -= 1.0;
            }
            if (zz >= 0.5)
            {
                zz -= 1.0;
            }

            double r = sqrt(xx * xx + yy * yy + zz * zz);
            double u = 0.5 * r / ((Asmth) / fft_params->grid);
            double fac = erfc(u * All.Asmth[1] / All.Asmth[0]) - erfc(u);
            size_t ip = INDEX(fft_params, (i - fft_params->slabstart_x), j, k);

            if (r > 0)
            {
                kernel[1][ip] = -fac / r;
            }
            else
            {
                fac = 1 - All.Asmth[1] / All.Asmth[0];
                kernel[1][ip] = -fac / (sqrt(M_PI) * (Asmth / fft_params->grid));
            }
        }

        /* do the forward transform of the kernel */
#ifdef _OPENMP
    FFTW(plan_with_nthreads)(omp_get_num_threads());
#endif // _OPENMP

    FFTW(plan)
    fft_forward_plan_high_res =
        FFTW(mpi_plan_dft_r2c_3d)(fft_params->grid, fft_params->grid, fft_params->grid, kernel[1], fft_of_kernel[1],
                                  MYMPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT | FFTW_MPI_TRANSPOSED_OUT);

    FFTW(execute)(fft_forward_plan_high_res);
    FFTW(destroy_plan)(fft_forward_plan_high_res);
#endif // PLACEHIGHRESREGION

    /* deconvolve the Greens function twice with the CIC kernel */
    for (int y = fft_params->slabstart_y; y < fft_params->slabstart_y + fft_params->nslab_y; y++)
#pragma omp parallel for
        for (int jk = 0; jk < fft_params->grid * fft_params->gridz; jk++)
        {
            int x = jk / fft_params->gridz;
            int z = jk % fft_params->gridz;
            double kx, ky, kz;

            if (x > fft_params->grid / 2)
            {
                kx = x - fft_params->grid;
            }
            else
            {
                kx = x;
            }
            if (y > fft_params->grid / 2)
            {
                ky = y - fft_params->grid;
            }
            else
            {
                ky = y;
            }
            if (z > fft_params->grid / 2)
            {
                kz = z - fft_params->grid;
            }
            else
            {
                kz = z;
            }

            double k2 = kx * kx + ky * ky + kz * kz;

            if (k2 > 0)
            {
                double fx = 1, fy = 1, fz = 1;

                if (kx != 0)
                {
                    fx = (M_PI * kx) / fft_params->grid;
                    fx = sin(fx) / fx;
                }
                if (ky != 0)
                {
                    fy = (M_PI * ky) / fft_params->grid;
                    fy = sin(fy) / fy;
                }
                if (kz != 0)
                {
                    fz = (M_PI * kz) / fft_params->grid;
                    fz = sin(fz) / fz;
                }

                double ff = 1 / (fx * fy * fz);
                ff = ff * ff * ff * ff;
                size_t ip = fft_params->gridz * (fft_params->grid * (y - fft_params->slabstart_y) + x) + z;

#ifndef PERIODIC
                fft_of_kernel[0][ip][0] *= ff;
                fft_of_kernel[0][ip][1] *= ff;
#endif // PERIODIC
#ifdef PLACEHIGHRESREGION
                fft_of_kernel[1][ip][0] *= ff;
                fft_of_kernel[1][ip][1] *= ff;
#endif // PLACEHIGHRESREGION
            }
        }
    /* end deconvolution */
}

#endif // !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

#endif // PMGRID
