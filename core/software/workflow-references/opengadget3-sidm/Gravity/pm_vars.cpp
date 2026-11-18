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

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef NOTYPEPREFIX_FFTW
#include <rfftw_mpi.h>
#else // NOTYPEPREFIX_FFTW
#ifdef SINGLEPRECISION_FFTW
#include <srfftw_mpi.h> /* double precision FFTW */
#else // SINGLEPRECISION_FFTW
#include <drfftw_mpi.h>
#endif // SINGLEPRECISION_FFTW
#endif // NOTYPEPREFIX_FFTW

#include "pm_vars.hpp"

rfftwnd_mpi_plan fft_forward_plan, fft_inverse_plan;

int slab_to_task[GRID];
int *slabs_per_task;
int *first_slab_of_task;

int slabstart_x, nslab_x, slabstart_y, nslab_y, smallest_slab;

int fftsize, maxfftsize;

fftw_real *kernel[2], *rhogrid, *forcegrid, *workspace;
fftw_complex *fft_of_kernel[2], *fft_of_rhogrid;
d_fftw_real *d_rhogrid, *d_forcegrid, *d_workspace;

int *part_sortindex;

struct part_slab_data *part;
