/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the HllRiemannSolver class.
 *
 * This file declares member functions for the derived HllRiemannSolver class based
 * on the parent RiemannSolver class.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "RiemannSolver.hpp"
#include <memory>

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief Compute the numerical fluxes based on the HLL approximate Riemann solver.
 */
template <int ndim>
bool HllRiemannSolver<ndim>::ComputeFluxes(const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
                                           const Vector<ndim, MyAtLeastDouble> &rUnit, /**< [in] Unit vector on face */
                                           WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS primitive state */
                                           WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS primitive state */
                                           Vector<ndim, MyAtLeastDouble> &vFace, /**< [in] Face velocity vector */
                                           FluxVector<ndim, MyAtLeastDouble> &flux) const /**< [out] Flux vector */
{
    MyAtLeastDouble Sleft; // Left-travelling wave speed of star region
    MyAtLeastDouble Sright; // Right-travelling wave speed of star region

    // Compute left and right wave speeds, and the speed of the star region
    ComputeWaveSpeedEstimates(Wleft, Wright, Sleft, Sright);

    // If wave speed estimates are invalid (e.g. Sleft and Sright are converging), then signify failure of
    // approximate HLL solver (by returning false) to allow the use of the ExactRiemannSolver instead
    if (Sleft >= Sright)
    {
        return false;
    }

    // If selecting zero-mass flux fame (e.g. for MFM), then move to frame of contact discontinuity
    if (zeroMassFlux)
    {
        std::unique_ptr<HllcRiemannSolver<ndim, HLLC_RIEMANN_VERSION>> hllcSolver =
            std::make_unique<HllcRiemannSolver<ndim, HLLC_RIEMANN_VERSION>>(eos);
        const MyAtLeastDouble Sstar = hllcSolver->ComputeStarVelocity(Wleft, Wright, Sleft, Sright);
        Wleft.GetVelocity()[0] -= Sstar;
        Wright.GetVelocity()[0] -= Sstar;
        vFace += Sstar * rUnit;
    }

    // Compute flux depending on the different wave speeds in the simplified HLL shock structure
    if (Sleft >= MyAtLeastDouble(0.0))
    {
        flux = this->ComputeFlux1d(Wleft);
    }
    else if (Sright <= MyAtLeastDouble(0.0))
    {
        flux = this->ComputeFlux1d(Wright);
    }
    else
    {
        flux = ComputeHllFlux1d(Wleft, Wright, Sleft, Sright);
    }

    // Guarantee mass flux is zero (should be zero but in case of floating point rounding error)
    if (zeroMassFlux)
    {
        flux.GetMass() = MyAtLeastDouble(0.0);
    }

    return true;
}

/**
 * @brief Compute the left and right wave speed estimates for HLL Riemann solver.
 *
 * HLL Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void HllRiemannSolver<ndim>::ComputeWaveSpeedEstimates(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS fluid state vector */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS fluid state vector */
    MyAtLeastDouble &Sleft, /**< [out] Left travelling signal speed */
    MyAtLeastDouble &Sright) const /**< [out] Right travelling signal speed */
{
    Sleft = Wleft.GetVelocity()[0] - eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma));
    Sright = Wright.GetVelocity()[0] + eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma));
#ifdef GT_DUST
    Sright = std::max(abs(Wleft.GetVelocity()[0]) + eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma)),
                      abs(Wright.GetVelocity()[0]) + eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma)));
    Sleft = -Sright;
#endif // GT_DUST
}

/**
 * @brief Compute the central (HLL) state vector for HLL Riemann solver.
 *
 * HLL Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
UFluidVector<ndim, MyAtLeastDouble> HllRiemannSolver<ndim>::ComputeHllStateVector(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, ///< [in] LHS fluid state vector
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, ///< [in] RHS fluid state vector
    const MyAtLeastDouble Sleft, ///< [out] Left travelling signal speed
    const MyAtLeastDouble Sright) const ///< [out] Right travelling signal speed
{
    const FluxVector<ndim, MyAtLeastDouble> fluxLeft = this->ComputeFlux1d(Wleft);
    const FluxVector<ndim, MyAtLeastDouble> fluxRight = this->ComputeFlux1d(Wright);
    const UFluidVector<ndim, MyAtLeastDouble> Uleft(Wleft, gamma);
    const UFluidVector<ndim, MyAtLeastDouble> Uright(Wright, gamma);
    const MyAtLeastDouble invSdiff = MyAtLeastDouble(1.0) / (Sright - Sleft);
    UFluidVector<ndim, MyAtLeastDouble> Ustar;
    for (int k = 0; k < nvar; k++)
    {
        Ustar[k] = Uright[k] * Sright - Uleft[k] * Sleft + fluxLeft[k] - fluxRight[k];
        Ustar[k] *= invSdiff;
    }
    return Ustar;
}

/**
 * @brief Compute the central (HLL) flux for HLL Riemann solver.
 *
 * HLL Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
FluxVector<ndim, MyAtLeastDouble> HllRiemannSolver<ndim>::ComputeHllFlux1d(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, ///< [in] LHS fluid state vector
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, ///< [in] RHS fluid state vector
    const MyAtLeastDouble Sleft, ///< [out] Left travelling signal speed
    const MyAtLeastDouble Sright) const ///< [out] Right travelling signal speed
{
    const FluxVector<ndim, MyAtLeastDouble> fluxLeft = this->ComputeFlux1d(Wleft);
    const FluxVector<ndim, MyAtLeastDouble> fluxRight = this->ComputeFlux1d(Wright);
    const UFluidVector<ndim, MyAtLeastDouble> Uleft(Wleft, gamma);
    const UFluidVector<ndim, MyAtLeastDouble> Uright(Wright, gamma);
    const MyAtLeastDouble invSdiff = MyAtLeastDouble(1.0) / (Sright - Sleft);
    FluxVector<ndim, MyAtLeastDouble> flux;
    for (int k = 0; k < nvar; k++)
    {
        flux[k] = Sright * fluxLeft[k] - Sleft * fluxRight[k] + Sleft * Sright * (Uright[k] - Uleft[k]);
        flux[k] *= invSdiff;
    }
    return flux;
}

template class HllRiemannSolver<mfm_numdims>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
