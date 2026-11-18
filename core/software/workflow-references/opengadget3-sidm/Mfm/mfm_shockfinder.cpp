/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief AREPO-style ray-traced shock finder for the MFM hydro scheme.
 *
 * Implements the algorithm of Schaal & Springel (2015), arXiv:1407.4117, adapted
 * to MFM. Activated automatically by the user-facing SHOCKFINDER flag when the
 * active hydro scheme is HYDRO_MFM (see CodeBase/switches.hpp, where
 * SHOCKFINDER_MFM is auto-derived). The Beck+16 SPH-style finder in
 * Hydro/shockfinder.hpp is the counterpart for HYDRO_SPH / HYDRO_PESPH.
 *
 * Both finders write the same SphP output fields (Shock_Mach, Shock_Speed,
 * Shock_Compress, Shock_Up_Rho, Shock_Up_Pressure, Shock_Up_V1[3], the matching
 * Shock_Down_* fields, and the gradient-pass Shock_N1) so snapshot blocks
 * (MACH, SHSP, SHCP, SHNR, plus SHRH/SHPU/SHPD/SHVU/SHVD under
 * SHOCKFINDER_RECONSTRUCT) and downstream consumers (CR injection, etc.) do
 * not need to branch on the hydro mode.
 *
 * Four logical passes run once per active-particle sync point:
 *   1. Candidate flagging   — local, no neighbor walk. Marks cells with
 *      DivVel < 0, a coarse |DivVel|·h/cs above All.ShockCandidateMin, a
 *      non-degenerate pressure normal, and ∇P·∇ρ > 0 (shear-flow filter).
 *   2. Primary-cell filter  — among kernel neighbors that share the shock's
 *      normal (cos angle > All.ShockNormalAlignCos), the cell with the largest
 *      |DivVel| is the primary. Runs in the same neighbor pass as 3.
 *   3. Single-step ray-trace — for primary cells, pick the best non-candidate
 *      neighbor in the cone of half-angle All.ShockConeAngleDeg along ±n̂ to
 *      serve as upstream / downstream state. The ray stays within the local
 *      kernel (no multi-step traversal across MPI domains).
 *   4. Rankine-Hugoniot reduction — local. Solve the temperature-jump form for
 *      the Mach number, apply the All.ShockMachMin cut to it, then
 *      v_s = c_pre · M with c_pre the reconstructed upstream signal speed, and the
 *      per-cell dissipation rate f_E · V_i^{2/3}.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../Hydro/kernel.hpp"
#include "../System/communication.hpp"
#include "mfm_shockfinder.hpp"

#include "../CodeBase/greentree.hpp"

