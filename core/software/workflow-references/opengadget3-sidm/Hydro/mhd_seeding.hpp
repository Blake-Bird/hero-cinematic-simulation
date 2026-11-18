/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief MHD seeding routines.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag and Ulrich Steinwandel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MHD_SEEDING_HPP
#define MHD_SEEDING_HPP

#ifdef MAGNETIC

#include "../CodeBase/allvars.hpp"
#if !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA) // we reinclude this same file if ACC is active
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // ACC_INCLUDE_HYDRA && ACC_HYDRA

#ifdef MAGNETIC_SN_SEEDING

/**
 * @brief Computes the contribution to the magnetic field from supernova seeding.
 *
 * This function calculates the contribution to the magnetic field from supernova seeding,
 * weighted by the smoothing kernel function. The result is added to the magnetic field
 * of the local particle.
 *
 * @param local Pointer to the local particle's hydrodynamic data.
 * @param kernel Pointer to the kernel parameters used for smoothing.
 * @param j Index of the neighboring particle from which the magnetic field contribution is obtained.
 * @param additional_variables Pointer to additional hydrodynamic data required for computation.
 * @param out Pointer to the output hydrodynamic data structure where the computed magnetic field contribution will be
 * stored.
 */
static inline void calculate_magseedrate_sn(int i, double sn_eff, double dtime)
{
    // convert from comoving to physical units
    double hsml_physical = P[i].Hsml * All.cf_atime / All.HubbleParam;

    // Beck+13, Eq. 9, without last fraction
    double bubble_expansion = All.SnSeedRadius / All.SnSeedBubble;
    double hsml_expansion = All.SnSeedBubble / hsml_physical;
    double mag_seed =
        All.SnSeedField * bubble_expansion * bubble_expansion * hsml_expansion * hsml_expansion * hsml_expansion;

    // Beck+13, Eq. 12
    double mag_norm = hsml_physical * hsml_physical * hsml_physical *
                      sqrt(0.5 * All.SnSeedSoftening * All.SnSeedSoftening * All.SnSeedSoftening *
                           (1.0 + All.SnSeedSoftening * All.SnSeedSoftening * All.SnSeedSoftening));

    // vector for the dipole moment
    // -> acceleration vector in the weak-field approximation where magnetic field is
    // frozen into plasma and dominated by external forces
    MyAtLeastDouble n_acc[3];
    for (int k = 0; k < 3; k++)
    {
        n_acc[k] = P[i].GravAccel[k] + SphP[i].HydroAccel[k];
    }

#ifdef PM_GRID
    // account for contribution by PM_GRID if defined
    for (int k = 0; k < 3; k++)
    {
        n_acc[k] += P[i].GravPM[k];
    }
#endif // PM_GRID
    // absolute value of acceleration to normalize vector for dipole moment
    MyAtLeastDouble n_abs = sqrt(n_acc[0] * n_acc[0] + n_acc[1] * n_acc[1] + n_acc[2] * n_acc[2]);

    // Beck+13, Eq. 9, conversion to comoving magnetic field code units per physical (!) timestep
    double magseed = mag_norm * sn_eff * mag_seed / (dtime * All.UnitMagneticField_in_gauss) * All.cf_atime *
                     All.cf_atime; // conversion to comoving code units

    // Beck+13, Eq. 11 (more or less)
    SphP[i].MagSeedRate[0] = magseed * n_acc[0] / n_abs;
    SphP[i].MagSeedRate[1] = magseed * n_acc[1] / n_abs;
    SphP[i].MagSeedRate[2] = magseed * n_acc[2] / n_abs;
}

#endif // MAGNETIC_SN_SEEDING
#endif // MAGNETIC
#endif // MHD_SEEDING_HPP