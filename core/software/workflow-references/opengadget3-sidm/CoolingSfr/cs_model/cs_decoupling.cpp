/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Auxiliary routines for the modeling of decoupling of gas phases
 *
 * @author OpenGadget3-development team, in particular J. Badía and C. Scannapieco
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

#include "../../CodeBase/allvars.hpp"
#include "../../CodeBase/proto.hpp"
#include "../../Gravity/forcetree.hpp"
#include <algorithm>

#ifdef CS_DECOUPLING

#include "cs_model.hpp"

/**
 * @brief Preserves previous-timestep densities in CS_DensityOld so that
 *        find_hsml uses consistent values throughout its iteration.
 *
 * The densities are copied into a new variable, "CS_DensityOld", which
 * stores the density values from the previous timestep. This prevents
 * the density calculation from using a mixture of updated and
 * non-updated density values.
 * By using "CS_DensityOld" throughout the find_hsml iteration, all
 * particles consistently use the density values from the previous
 * timestep. The densities are updated for all particles only after
 * the find_hsml iteration is complete.
 */
void cs_copy_densities(void)
{
    float rho_min = 1.0e8;
    float rho_max = -1.0;

    TIMER_PUSH(1, "CS_COPY_DENSITY");
    for (int i = 0; i < N_gas; i++)
    {
        if (P[i].Type == 0)
        {
            SphP[i].CS_DensityOld = SphP[i].Density;

            rho_min = std::min(rho_min, SphP[i].CS_DensityOld);
            rho_max = std::max(rho_max, SphP[i].CS_DensityOld);

            /*      if(SphP[i].CS_DensityOld > max_rho)
                        rho_max = SphP[i].CS_DensityOld;

                    if(SphP[i].CS_DensityOld < min_rho)
                        rho_min = SphP[i].CS_DensityOld;*/
        }
    }
    VERBOSE(2, "CS DEC:...copy densities done... density min|max=%g|%g\n", rho_min, rho_max);
    TIMER_POP(1, "CS_COPY_DENSITY");
}

/**
 * @brief Fallback for the decoupling scheme that keeps the 2*NumNgb
 *        lowest-density particles available as neighbours.
 *
 * This function is only needed in special situations,
 * for example when running an isolated galaxy simulation
 * and the first particle that becomes *hot* finds no
 * neighbours. We then make sure that particles never ignore
 * the 2*NumNgb particles with lowest density to still be
 * able to continue with the density estimation using the
 * decoupling scheme.
 */
void cs_find_low_density_tail(void)
{
    float *rho, *rho_common;
    int i, contrib, listlen, *contrib_list, *contrib_offset;
    int count = 0;

    TIMER_PUSH(1, "CS_FIND_TAIL");
#define NUM_DENSITY_TAIL (2 * All.DesNumNgb)

    rho = (float *)mymalloc("rho", N_gas * sizeof(float));
    contrib_list = (int *)mymalloc("contrib_list", NTask * sizeof(int));
    contrib_offset = (int *)mymalloc("contrib_offset", NTask * sizeof(int));

    for (int i = 0; i < N_gas; i++)
    {
        if (P[i].Type == 0)
        {
            rho[count++] = SphP[i].CS_DensityOld; /*note: CS_DensityOld = Density here */
        }
    }
    qsort(rho, count, sizeof(float), cs_compare_density_values);

    contrib = NUM_DENSITY_TAIL;

    if (count < contrib)
    {
        contrib = count;
    }

    MPI_Allreduce(&contrib, &listlen, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgather(&contrib, 1, MPI_INT, contrib_list, 1, MPI_INT, MPI_COMM_WORLD);

    /* CS_NOTE
     *  Using
     *  for(int i = 1, contrib_offset[0] = 0; i < NTask; i++)
     *	does not work, so we keep the initialization outside the loop */

    contrib_offset[0] = 0;
    for (int i = 1; i < NTask; i++)
    {
        contrib_offset[i] = contrib_offset[i - 1] + contrib_list[i - 1];
    }

    if (listlen < NUM_DENSITY_TAIL)
    {
        ENDRUN("total number of gas particles is less than %d!\n", NUM_DENSITY_TAIL);
    }

    rho_common = (float *)mymalloc("rho_common", listlen * sizeof(float));

    MPI_Allgatherv(rho, contrib, MPI_FLOAT, rho_common, contrib_list, contrib_offset, MPI_FLOAT, MPI_COMM_WORLD);

    qsort(rho_common, listlen, sizeof(float), cs_compare_density_values);

    All.DensityTailThreshold = rho_common[NUM_DENSITY_TAIL - 1];
    VERBOSE(2, "CS DEC: DensityTailThreshold = %g\n", All.DensityTailThreshold);

    myfree(rho_common);
    myfree(contrib_offset);
    myfree(contrib_list);
    myfree(rho);
    TIMER_POP(1, "CS_FIND_TAIL");
}

/**
 * @brief qsort comparator that orders float density values in ascending
 *        order, used to find the DensityTailThreshold value.
 */
int cs_compare_density_values(const void *a, const void *b)
{
    if (*((float *)a) < *((float *)b))
    {
        return -1;
    }

    if (*((float *)a) > *((float *)b))
    {
        return +1;
    }

    return 0;
}

#endif // CS_DECOUPLING
