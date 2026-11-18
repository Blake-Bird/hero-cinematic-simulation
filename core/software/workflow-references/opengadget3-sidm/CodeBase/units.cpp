/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains unit class for comoving factors.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth and Ludwig Böss.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/units.hpp"
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

namespace OpenGadget3
{
/**
 * @ brief initialize the comoving factors used for conversion.
 *
 * This function should be called once per timestep to make sure the later conversion is done correctly.
 */
void UnitsComoving::init_cosmo_values(double time)
{
    atime = time;
    a2inv = 1.0 / (atime * atime);
    a3inv = 1 / (atime * atime * atime);
    afac1 = pow(atime, 3 * GAMMA_MINUS1);
    afac2 = 1 / pow(atime, 3 * GAMMA - 2);
    afac3 = pow(atime, 3 * (1 - GAMMA) / 2.0);
    hubble_a = hubble_function(atime);
    // set all comoving to physical (and vice versa) conversion factors based on the values just set
    set_comoving_physical_conversion_factors();
}
} // namespace OpenGadget3
