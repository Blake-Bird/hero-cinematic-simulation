/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Some relevant definitions for the MFM limiter calculation.
 *
 * This file declares structures for the MFM slope limiter calculation.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET3_LIMITER_HPP
#define _OPENGADGET3_LIMITER_HPP

/**
 * @brief Structure representing input data for MFM slope limiter computations.
 *
 * This struct holds information about the state and properties of a particle
 * relevant for the MFM slope limiter caclulation.
 */
struct slopedata_in
{
    MyFloat Hsml;
    MyLongDouble Pos[3];
    WFluidVector<mfm_numdims> Wprim;
    Vector<NUMDIMS> gradW[numvar];
#if MFM_SLOPE_LIMITER == LIMITER_AREPO
    WFluidVector<mfm_numdims> dWmin;
    WFluidVector<mfm_numdims> dWmax;
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO
#ifdef GT_DUST
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> Wprim_dust;
    std::array<std::array<Vector<NUMDIMS>, numvar>, GT_NUM_DUST> gradW_dust;
#if MFM_SLOPE_LIMITER == LIMITER_AREPO
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> dWmin_dust;
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> dWmax_dust;
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO
#endif // GT_DUST
};

/**
 * @brief Structure representing output data for MFM slope limiter computations.
 *
 * This struct holds information about the state and properties of a particle
 * relevant for the MFM slope limiter calculation.
 */
struct slopedata_out
{
    MyFloat AlphaSlope[numvar];
#ifdef GT_DUST
    MyFloat AlphaSlope_dust[GT_NUM_DUST][numvar];
#endif // GT_DUST
};

#endif // _OPENGADGET3_LIMITER_HPP
