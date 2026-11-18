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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#ifdef PMGRID

#ifdef KD_EXTRA_TIMER_OUTPUT_PM
#undef KD_EXTRA_TIMER_OUTPUT_PM
#define KD_EXTRA_TIMER_OUTPUT_PM(f_, ...)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        t1_kd = second();                                                                                              \
        VERBOSE(0, f_, ##__VA_ARGS__);                                                                                 \
        t0_kd = second();                                                                                              \
    } while (0)
#else // KD_EXTRA_TIMER_OUTPUT_PM
#undef KD_EXTRA_TIMER_OUTPUT_PM
#define KD_EXTRA_TIMER_OUTPUT_PM(f_, ...)
#endif // KD_EXTRA_TIMER_OUTPUT_PM

#define CONCAT(prefix, name) prefix##name

#ifndef SINGLEPRECISION_FFTW
#define FFTW(x) CONCAT(fftw_, x)
#else // SINGLEPRECISION_FFTW
#define FFTW(x) CONCAT(fftwf_, x)
#endif // SINGLEPRECISION_FFTW

#include <fftw3.h>
#include <fftw3-mpi.h>

#ifndef GRIDBOOST
#define GRIDBOOST 2
#endif // GRIDBOOST

#ifndef HIGHRES_PMGRID
#define HIGHRES_PMGRID PMGRID
#else // HIGHRES_PMGRID
#ifndef PERIODIC
#error "defining HIGHRES_PMGRID without PERIODIC is not allowed !"
#endif // PERIODIC
#endif // HIGHRES_PMGRID

#ifdef PLACEHIGHRESREGION
#define GRID (GRIDBOOST * HIGHRES_PMGRID)
#else // PLACEHIGHRESREGION
#define GRID (GRIDBOOST * PMGRID)
#endif // PLACEHIGHRESREGION

#if (PMGRID > 1024) || (HIGHRES_PMGRID > 1024)
typedef long long large_array_offset;
#else // (PMGRID > 1024) || (HIGHRES_PMGRID > 1024)
typedef unsigned int large_array_offset;
#endif // (PMGRID > 1024) || (HIGHRES_PMGRID > 1024)

#ifndef SINGLEPRECISION_FFTW
typedef double fftw_real;
#else // SINGLEPRECISION_FFTW
typedef float fftw_real;
#endif // SINGLEPRECISION_FFTW

#ifdef KD_PM_NOSENDPOS
struct partbuf
{
    MyFloat dx, dy, dz, Mass;
    int slab_x, slab_y, slab_z;
};
#else // KD_PM_NOSENDPOS
struct partbuf
{
    MyLongDouble Pos[3];
    MyFloat Mass;
};
#endif // KD_PM_NOSENDPOS

/**
 * @brief Struct containing parameters for FFT computation.
 *
 * This struct holds various parameters relevant for the FFT (Fast Fourier Transform) computation,
 * particularly useful for Particle-Mesh (PM) calculations in cosmological simulations.
 */
struct fftw_parameters
{
    ptrdiff_t slabstart_x, nslab_x, slabstart_y, nslab_y, smallest_slab;
    ptrdiff_t fftsize, maxfftsize;

    int grid, gridz, grid2;

    int *slab_to_task;
    int *slabs_per_task;
    int *first_slab_of_task;
};

extern struct fftw_parameters fft_params_periodic;

extern struct fftw_parameters fft_params_nonperiodic;

#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
extern FFTW(complex) * fft_of_kernel[2];
extern fftw_real *kernel[2];
#endif // !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

/* short-cut macros for accessing different 3D arrays */
#define INDEX(p, x, y, z) ((p->grid2) * (p->grid * (x) + (y)) + (z))
#define TINDEX(p, x, y, z) ((p->grid) * ((x) + p->nslab_x * (y)) + (z))

void pm_divide_threads(int N, int pieces, int index, int *first, int *count);

void pm_fftw_init(int grid, struct fftw_parameters *fft_params);

void pm_binning(int mode, struct fftw_parameters *fft_params, fftw_real *rhogrid, int nimport, struct partbuf *partin);

void pm_transposeA(int mode, fftw_real *field, fftw_real *scratch, struct fftw_parameters *fft_params);
void pm_transposeB(int mode, fftw_real *field, fftw_real *scratch, struct fftw_parameters *fft_params);
void pm_nonperiodic_transposeA(int mode, fftw_real *field, fftw_real *scratch, struct fftw_parameters *fft_params);
void pm_nonperiodic_transposeB(int mode, fftw_real *field, fftw_real *scratch, struct fftw_parameters *fft_params);

#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
void pm_greens_nonperiodic(int mode_periodic, struct fftw_parameters *fft_params, FFTW(complex) * fft_of_rhogrid);
#endif //  !defined(PERIODIC) || defined(PLACEHIGHRESREGION)

#ifdef PERIODIC
void pm_greens_periodic(struct fftw_parameters *fft_params, FFTW(complex) * fft_of_rhogrid);
#endif // PERIODIC

void pm_readout(int mode, int dim, double fac, fftw_real *readout_grid, struct fftw_parameters *fft_params,
                struct partbuf *partin, int nimport, int nexport, int *index_task0, int *index_task1);

void pm_get_forces(int mode, double fac, fftw_real *rhogrid, struct fftw_parameters *fft_params, struct partbuf *partin,
                   int nimpot, int nexport, int *index_task0, int *index_task1);

#endif // PMGRID
