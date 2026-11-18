/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Routines for setting the boundaries of the PM-force computation.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <array>
#include "../CodeBase/allvars.hpp"
#include "pm.hpp"
#include "bounds_init.hpp"

#ifdef PMGRID
#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
/**
 * @brief Initialize 2x3 arrays for mesh min/max coordinates.
 *
 * Sets all entries of `xmin` to 1.0e36 and `xmax` to -1.0e36.
 *
 * @param xmin 2x3 array to hold minimum coordinates
 * @param xmax 2x3 array to hold maximum coordinates
 */
void init_bounds(double xmin[2][3], double xmax[2][3])
{
    constexpr double big = 1.0e36;
    constexpr double small = -1.0e36;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
        {
            xmin[i][j] = big;
            xmax[i][j] = small;
        }

    for (int i = 0; i < NumPart; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (P[i].Pos[j] > xmax[0][j])
            {
                xmax[0][j] = P[i].Pos[j];
            }
            if (P[i].Pos[j] < xmin[0][j])
            {
                xmin[0][j] = P[i].Pos[j];
            }

#ifdef PLACEHIGHRESREGION
            if (((1 << P[i].Type) & (PLACEHIGHRESREGION)))
            {
                if (P[i].Pos[j] > xmax[1][j])
                {
                    xmax[1][j] = P[i].Pos[j];
                }
                if (P[i].Pos[j] < xmin[1][j])
                {
                    xmin[1][j] = P[i].Pos[j];
                }
            }
#endif // PLACEHIGHRESREGION
        }
    }
}

/**
 * @brief Compute and symmetrize the total mesh size for each dimension.
 *
 * For each j = 0,1:
 *  - Computes TotalMeshSize[j] as the maximum extent over x/y/z
 *  - Optionally enlarges the region (ENLARGEREGION)
 *  - Recenters Xmintot/Xmaxtot symmetrically about the midpoint
 */
void compute_and_symmetrize_mesh_region()
{
    for (int j = 0; j < 2; j++)
    {
        // Compute maximum extent over the three coordinates
        All.TotalMeshSize[j] = All.Xmaxtot[j][0] - All.Xmintot[j][0];
        All.TotalMeshSize[j] = max(All.TotalMeshSize[j], All.Xmaxtot[j][1] - All.Xmintot[j][1]);
        All.TotalMeshSize[j] = max(All.TotalMeshSize[j], All.Xmaxtot[j][2] - All.Xmintot[j][2]);
#ifdef ENLARGEREGION
        All.TotalMeshSize[j] *= ENLARGEREGION;
#endif // ENLARGEREGION

        // Symmetrize the box around the center
        for (int i = 0; i < 3; i++)
        {
            const double center = 0.5 * (All.Xmintot[j][i] + All.Xmaxtot[j][i]);

            All.Xmintot[j][i] = center - 0.5 * All.TotalMeshSize[j];
            All.Xmaxtot[j][i] = All.Xmintot[j][i] + All.TotalMeshSize[j];
        }
    }
}

/**
 * @brief Initialize PM mesh geometry and update simulation box corners.
 *
 * This routine performs two related tasks for the particle–mesh (PM) solver:
 *
 * 1. Stores the original total mesh size for each PM grid (index 0 and 1)
 *    in the caller-provided array `meshinner`.
 *
 * 2. Enlarges the mesh size to include guard cells required for finite
 *    differencing of the gravitational potential. The enlargement factor
 *    accounts for the effective grid resolution and boundary padding.
 *
 * After resizing, the lower and upper corners of the PM mesh are shifted
 * such that the mesh is centered and extended by two cells in each spatial
 * direction. This ensures sufficient stencil width for a 4-point finite
 * difference operator.
 *
 * @param[out] meshinner
 *     Array of length 2 receiving the original (pre-enlargement)
 *     total mesh sizes for each PM grid.
 */
void init_mesh(double meshinner[2])
{
    for (int j = 0; j < 2; j++)
    {
        meshinner[j] = All.TotalMeshSize[j];
        All.TotalMeshSize[j] *= 2.001 * (GRID) / ((double)(GRID - 2 - 8));
    }

    /* move lower left corner by two cells to allow finite differencing of the potential by a 4-point function */
    for (int j = 0; j < 2; j++)
    {
        for (int i = 0; i < 3; i++)
        {
            All.Corner[j][i] = All.Xmintot[j][i] - 2.0005 * All.TotalMeshSize[j] / GRID;
            All.UpperCorner[j][i] = All.Corner[j][i] + (GRID / 2 - 1) * (All.TotalMeshSize[j] / GRID);
        }
    }
}
#endif // !PERIODIC || PLACEHIGHRESREGION

/**
 * @brief Initialize simulation box corners for index 2.
 *
 * Sets Corner[2][i] = 0 and UpperCorner[2][i] = BoxSize
 * for i = 0..2.
 */
void init_box_corners()
{
    for (int i = 0; i < 3; i++)
    {
        All.Corner[2][i] = 0.0;
        All.UpperCorner[2][i] = All.BoxSize;
    }
}

/**
 * @brief Initialize Corner[1] and UpperCorner[1] for the PM mesh region.
 *
 * Corner[1][i] is shifted from Xmintot by the cutoff radius.
 * UpperCorner[1][i] spans half the mesh minus guard cells.
 */
void init_pm_region_corners()
{
    for (int i = 0; i < 3; i++)
    {
        All.Corner[1][i] = All.Xmintot[1][i] - 1.0001 * All.Rcut[0];
        All.UpperCorner[1][i] = All.Corner[1][i] + (GRID / 2 - 1) * (All.TotalMeshSize[1] / GRID);
    }
}

void init_pm_region_corners2()
{
    for (int i = 0; i < 3; i++)
    {
        All.Corner[1][i] = All.Xmintot[1][i] - 1.0001 * (All.Rcut[0] + 2 * All.TotalMeshSize[1] / GRID);
        All.UpperCorner[1][i] = All.Corner[1][i] + (GRID / 2 - 1) * (All.TotalMeshSize[1] / GRID);
    }
}

#endif // PMGRID
