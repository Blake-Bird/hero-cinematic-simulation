/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for the AREPO-style MFM shock finder.
 *
 * Implements the ray-traced shock-surface reconstruction of Schaal & Springel (2015),
 * arXiv:1407.4117, adapted for the MFM hydro scheme. Coexists with the Beck+16
 * SPH-style shock finder in ../Hydro/shockfinder.hpp; both finders write the same
 * SphP output fields (Shock_Mach, Shock_Speed, Shock_Compress, Shock_Up_*,
 * Shock_Down_*) so that downstream consumers and snapshot IO blocks (MACH, SHSP,
 * SHCP, and SHRH/SHPU/SHPD/SHVU/SHVD under SHOCKFINDER_RECONSTRUCT) do not need
 * to branch on the active hydro mode. The user-facing flag is SHOCKFINDER;
 * SHOCKFINDER_MFM is auto-defined when SHOCKFINDER + HYDRO_MFM is in effect (see
 * CodeBase/switches.hpp).
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MFM_SHOCKFINDER_HPP
#define MFM_SHOCKFINDER_HPP

#include "../CodeBase/allvars.hpp"


#endif // MFM_SHOCKFINDER_HPP
