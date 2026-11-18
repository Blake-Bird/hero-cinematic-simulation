/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Variable declarations for the particle-mesh (PM) module.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef PM_VARS_HPP
#define PM_VARS_HPP

#ifndef GRIDBOOST
#define GRIDBOOST 2
#endif // GRIDBOOST

#define GRID (GRIDBOOST * PMGRID)
#define GRID2 (2 * (GRID / 2 + 1))
#define PMGRID2 (2 * (PMGRID / 2 + 1))

#if (GRID > 1024)
typedef long long large_array_offset;
#else // (GRID > 1024)
typedef unsigned int large_array_offset;
#endif // (GRID > 1024)

#define d_fftw_real fftw_real

extern rfftwnd_mpi_plan fft_forward_plan, fft_inverse_plan;

extern int slab_to_task[GRID];
extern int *slabs_per_task;
extern int *first_slab_of_task;

extern int slabstart_x, nslab_x, slabstart_y, nslab_y, smallest_slab;

extern int fftsize, maxfftsize;

extern fftw_real *kernel[2], *rhogrid, *forcegrid, *workspace;
extern fftw_complex *fft_of_kernel[2], *fft_of_rhogrid;
extern d_fftw_real *d_rhogrid, *d_forcegrid, *d_workspace;

extern int *part_sortindex;

/**
 * @brief Structure representing data associated with particles in a slab.
 *
 * This structure stores information related to particles in a slab. Each particle in the slab
 * is represented by an instance of this structure, containing the global index, particle index,
 * and local index within the slab.
 */
extern struct part_slab_data
{
    large_array_offset globalindex;
    int partindex;
    int localindex;
} *part;

#endif // PM_VARS_HPP