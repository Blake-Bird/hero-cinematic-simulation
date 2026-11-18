/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the HlleRiemannSolver class.
 *
 * This file declares member functions for the derived HlleRiemannSolver class based
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
#include <iostream>
#include <memory>
// #include "../CodeBase/allvars.hpp"

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief Compute the numerical fluxes based on the HLLE approximate Riemann solver.
 */
template <int ndim>
bool HlleRiemannSolver<ndim>::ComputeFluxes(
    const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
    const Vector<ndim, MyAtLeastDouble> &rUnit, /**< [in] Unit vector on face */
    WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS primitive state */
    WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS primitive state */
    Vector<ndim, MyAtLeastDouble> &vFace, /**< [in] Face velocity vector */
    FluxVector<ndim, MyAtLeastDouble> &flux) const /**< [out] Flux vector */
{
    Vector<nvar, MyAtLeastDouble> valtilde; // Roe averaged values: soundspeed, velocity, entalpy

    static std::unique_ptr<RoeRiemannSolver<ndim>> roeSolver = std::make_unique<RoeRiemannSolver<ndim>>(eos);

    // Compute Roe averaged values
    roeSolver->ComputeAveragedValues(Wleft, Wright, valtilde);
    if (valtilde[irho] < 0.0)
    {
        valtilde[irho] = (MyAtLeastDouble)0.0;
    }

    MyAtLeastDouble bp, bm;
    ComputeWaveSpeedEstimates(Wleft, Wright, valtilde, bm, bp);

    if (bm >= bp)
    {
        return false;
    }

    // If selecting zero-mass flux fame (e.g. for MFM), then move to frame of contact discontinuity
    if (zeroMassFlux)
    {
        const MyAtLeastDouble Sstar = 0.5 * (bp + bm); // get this from Einfeldt+1991  // Toro Eqn 11.139 / 11.136
        Wleft.GetVelocity()[0] -= Sstar;
        Wright.GetVelocity()[0] -= Sstar;
        vFace += Sstar * rUnit;
    }

    // Compute flux depending on the different wave speeds in the simplified HLLE shock structure
    if (bm >= (MyAtLeastDouble)0.0)
    {
        flux = this->ComputeFlux1d(Wleft);
    }
    else if (bp <= (MyAtLeastDouble)0.0)
    {
        flux = this->ComputeFlux1d(Wright);
    }
    else
    {
        flux = ComputeHlleFlux1d(Wleft, Wright, bp, bm);
    }

    // Guarantee mass flux is zero (should be zero but in case of floating point rounding error)
    if (zeroMassFlux)
    {
        flux.GetMass() = (MyAtLeastDouble)0.0;
    }

    return true;
}

/**
 * @brief Compute central (Hlle) flux.
 *
 * Hlle Riemann solver, based on approach outlined by Einfeldt (1991).
 * Variable names often follow Einfeldt or Toro naming conventions so read paper/chapter for further details.
 */
template <int ndim>
FluxVector<ndim, MyAtLeastDouble> HlleRiemannSolver<ndim>::ComputeHlleFlux1d(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS fluid state vector */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS fluid state vector */
    const MyAtLeastDouble bp, /**< [in] right signal velocity */
    const MyAtLeastDouble bm) const /**< [in] left signal velocity */
{
    const FluxVector<ndim, MyAtLeastDouble> fluxLeft = this->ComputeFlux1d(Wleft);
    const FluxVector<ndim, MyAtLeastDouble> fluxRight = this->ComputeFlux1d(Wright);
    const UFluidVector<ndim, MyAtLeastDouble> Uleft(Wleft, gamma);
    const UFluidVector<ndim, MyAtLeastDouble> Uright(Wright, gamma);
    FluxVector<ndim, MyAtLeastDouble> flux;

#ifdef HLLE_DIFFUSIVE
    for (int k = 0; k < nvar; k++)
    {
        flux[k] = (MyAtLeastDouble)0.5 * ((fluxLeft[k] + fluxRight[k]) - bp * (Uright[k] - Uleft[k]));
    }
#else // HLLE_DIFFUSIVE
    const MyAtLeastDouble invbdiff = (MyAtLeastDouble)1.0 / (bp - bm);
    for (int k = 0; k < nvar; k++)
    {
        flux[k] = (bp * fluxLeft[k] - bm * fluxRight[k]) + bp * bm * (Uright[k] - Uleft[k]);
        flux[k] *= invbdiff;
    }
#endif // HLLE_DIFFUSIVE
    return flux;
}

/**
 * @brief Compute left and right wave speed estimates.
 *
 * Hlle Riemann solver, based on approach outlined by Einfeldt (1991).
 * Variable names often follow Einfeldt or Toro naming conventions so read paper/chapter for further details.
 */
template <int ndim>
void HlleRiemannSolver<ndim>::ComputeWaveSpeedEstimates(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS fluid state vector */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS fluid state vector */
    const Vector<nvar, MyAtLeastDouble> &val, /**< [in] average values */
    MyAtLeastDouble &Sleft, /**< [out] Left travelling signal speed */
    MyAtLeastDouble &Sright) const /**< [out] Right travelling signal speed */
{
    // MyAtLeastDouble sqrtrhol = sqrt(Wleft.GetDensity());
    // MyAtLeastDouble sqrtrhor = sqrt(Wright.GetDensity());
    // MyAtLeastDouble invsqrtrhosum = 1.0 / (sqrtrhol + sqrtrhor);
    // MyAtLeastDouble dbar;
    // dbar = (sqrtrhol * pow(eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma)),2)
    //      + sqrtrhor * pow(eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma)),2)) * invsqrtrhosum;
    // dbar += (MyAtLeastDouble)0.5 * sqrtrhol*sqrtrhor * pow(invsqrtrhosum,2);
    // dbar = sqrt(dbar);
    // Sleft = val[ivx] - dbar;
    // Sright = val[ivx] + dbar;
#ifdef HLLE_DIFFUSIVE
    Sright =
        std::max(fabs(val[ivx]) + val[irho],
                 std::max(fabs(Wleft.GetVelocity()[0]) - eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma), 1),
                          fabs(Wright.GetVelocity()[0]) + eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma))));
    Sleft = -Sright;
#else // HLLE_DIFFUSIVE
    Sright = std::max(val[ivx] + val[irho], // convert the average eigenvalues to the last eigenvalue (also compare Roe
                                            // solver implementation)
                      Wright.GetVelocity()[0] + eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma)));
    Sleft = std::min(val[ivx] - val[irho], // convert the average eigenvalues to the first eigenvalue (also compare Roe
                                           // solver implementation)
                     Wleft.GetVelocity()[0] - eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma)));
#endif // HLLE_DIFFUSIVE
}

template class HlleRiemannSolver<mfm_numdims>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
