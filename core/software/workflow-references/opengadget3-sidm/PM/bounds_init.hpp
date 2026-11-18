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

#ifndef BOUNDS_INIT_HPP
#define BOUNDS_INIT_HPP

#ifdef PMGRID

#if !defined(PERIODIC) || defined(PLACEHIGHRESREGION)
void init_bounds(double xmin[2][3], double xmax[2][3]);
void init_mesh(double meshinner[2]);
void compute_and_symmetrize_mesh_region();
#endif // !(PERIODIC) || PLACEHIGHRESREGION

void init_box_corners();
void init_pm_region_corners();
void init_pm_region_corners2();

#endif // PMGRID
#endif // BOUNDS_INIT_HPP
