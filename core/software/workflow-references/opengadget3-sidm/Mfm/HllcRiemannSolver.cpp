/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the HllcRiemannSolver class.
 *
 * This file declares member functions for the derived HllcRiemannSolver class based
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
 * @brief Compute the numerical fluxes based on the HLLC approximate Riemann solver.
 */
template <int ndim, int version>
bool HllcRiemannSolver<ndim, version>::ComputeFluxes(
    const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
    const Vector<ndim, MyAtLeastDouble> &rUnit, /**< [in] Unit vector on face */
    WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS primitive state */
    WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS primitive state */
    Vector<ndim, MyAtLeastDouble> &vFace, /**< [in] Face velocity vector */
    FluxVector<ndim, MyAtLeastDouble> &flux) const /**< [out] Flux vector */
{
    MyAtLeastDouble Sleft; // Left-travelling wave speed of star region
    MyAtLeastDouble Sright; // Right-travelling wave speed of star region

    // Compute left and right wave speeds, and the speed of the star region
    std::unique_ptr<HllRiemannSolver<ndim>> hllSolver = std::make_unique<HllRiemannSolver<ndim>>(eos);
    hllSolver->ComputeWaveSpeedEstimates(Wleft, Wright, Sleft, Sright);

    // If wave speed estimates are invalid (e.g. Sleft and Sright are converging), then signify failure of
    // approximate HLLC solver (by returning false) to allow the use of the ExactRiemannSolver instead
    if (Sleft >= Sright)
    {
        return false;
    }

    // If selecting zero-mass flux frame (e.g. for MFM), then move to frame of contact discontinuity
    const MyAtLeastDouble Sstar = ComputeStarVelocity(Wleft, Wright, Sleft, Sright);
    if (!std::isfinite(Sstar))
    {
        return false;
    }
    if (zeroMassFlux)
    {
        Wleft.GetVelocity()[0] -= Sstar;
        Wright.GetVelocity()[0] -= Sstar;
        vFace += Sstar * rUnit;
    }

    // Compute flux depending on the different wave speeds in the simplified HLLC shock structure
    if (Sleft >= (MyAtLeastDouble)0.0)
    {
        flux = this->ComputeFlux1d(Wleft);
    }
    else if ((Sleft <= (MyAtLeastDouble)0.0) && ((MyAtLeastDouble)0.0 <= Sstar))
    {
        flux = ComputeHllcFlux1d(Wleft, Wright, Sleft, Sright, Sstar);
        // flux = ComputeHllcFlux1d(Wright, Wleft, Sright, Sleft, Sstar);
    }
    else if ((Sstar <= (MyAtLeastDouble)0.0) && ((MyAtLeastDouble)0.0 <= Sright))
    {
        // flux = ComputeHllcFlux1d(Wleft, Wright, Sleft, Sright, Sstar);
        flux = ComputeHllcFlux1d(Wright, Wleft, Sright, Sleft, Sstar);
    }
    else if (Sright <= (MyAtLeastDouble)0.0)
    {
        flux = this->ComputeFlux1d(Wright);
    }

    // Guarantee mass flux is zero (should be zero but in case of floating point rounding error)
    if (zeroMassFlux)
    {
        flux.GetMass() = (MyAtLeastDouble)0.0;
    }

    return true;
}

/**
 * @brief Compute the central (star) velocity.
 *
 * HLLC Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim, int version>
MyAtLeastDouble HllcRiemannSolver<ndim, version>::ComputeStarVelocity(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, ///< [in] LHS fluid state vector
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, ///< [in] RHS fluid state vector
    const MyAtLeastDouble Sleft, ///< [out] Left travelling signal speed
    const MyAtLeastDouble Sright) const ///< [out] Right travelling signal speed
{
    const MyAtLeastDouble jLeft = Wleft.GetDensity() * (Sleft - Wleft.GetVelocity()[0]);
    const MyAtLeastDouble jRight = Wright.GetDensity() * (Sright - Wright.GetVelocity()[0]);
    MyAtLeastDouble Sstar = (jRight * Wright.GetVelocity()[0] - jLeft * Wleft.GetVelocity()[0] + Wleft.GetPressure() -
                             Wright.GetPressure());
    Sstar /= (jRight - jLeft);
    return Sstar;
}

/**
 * @brief Compute the central (Hllc) flux.
 *
 * HLLC Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim, int version>
FluxVector<ndim, MyAtLeastDouble> HllcRiemannSolver<ndim, version>::ComputeHllcFlux1d(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, ///< [in] LHS fluid state vector
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, ///< [in] RHS fluid state vector
    const MyAtLeastDouble Sleft, ///< [in] Left travelling signal speed
    const MyAtLeastDouble Sright, ///< [in] Right travelling signal speed
    const MyAtLeastDouble Sstar) const ///< [in] signal speed of contact disconstinuity
{
    const FluxVector<ndim, MyAtLeastDouble> fluxLeft = this->ComputeFlux1d(Wleft);
    // const FluxVector<ndim, MyAtLeastDouble> fluxRight = this->ComputeFlux1d(Wright);
    const UFluidVector<ndim, MyAtLeastDouble> Uleft(Wleft, gamma);
    // const UFluidVector<ndim, MyAtLeastDouble> Uright(Wright, gamma);
    const MyAtLeastDouble invSdiff = (MyAtLeastDouble)1.0 / (Sleft - Sstar);
    FluxVector<ndim, MyAtLeastDouble> flux;
    MyAtLeastDouble PLR;
    if constexpr (version == 3)
    {
        PLR = (MyAtLeastDouble)0.5 *
              (Wleft.GetPressure() + Wright.GetPressure() +
               Wleft.GetDensity() * (Sleft - Wleft.GetVelocity()[0]) * (Sstar - Wleft.GetVelocity()[0]) +
               Wright.GetDensity() * (Sright - Wright.GetVelocity()[0]) * (Sstar - Wright.GetVelocity()[0]));
    }
    for (int k = 0; k < nvar; k++)
    {
        if constexpr (version == 1)
        { // Toro Eqn. (10.38) and (10.39)
            MyAtLeastDouble fac;
            if (k == irho)
            {
                fac = 1;
            }
            else if (k == ipress)
            {
                fac = Uleft.GetEnergyDensity() / Wleft.GetDensity() +
                      (Sstar - Wleft.GetVelocity()[0]) *
                          (Sstar + Wleft.GetPressure() / (Wleft.GetDensity() * (Sleft - Wleft.GetVelocity()[0])));
            }
            else if (k == ivx)
            {
                fac = Sstar;
            }
            else if (k == ivy)
            { // because of the order automatically ensures numdims > 1
                fac = Wleft[ivy];
            }
            else if (k == ivz)
            {
                fac = Wleft[ivz];
            }
            else
            {
                fac = 1.0;
            }
            MyAtLeastDouble Ustarleftk = Wleft.GetDensity() * (Sleft - Wleft.GetVelocity()[0]) * invSdiff * fac;
            flux[k] = fluxLeft[k] + Sleft * (Ustarleftk - Uleft[k]);
        }
        else if constexpr (version == 2)
        { // Toro Eqn. (10.41)
            MyAtLeastDouble Dstar;
            if (k == irho)
            {
                Dstar = 0;
            }
            else if (k == ipress)
            {
                Dstar = Sstar;
            }
            else if (k == ivx)
            {
                Dstar = 1;
            }
            else if (k == ivy)
            {
                Dstar = 0;
            }
            else if (k == ivz)
            {
                Dstar = 0;
            }
            else
            {
                Dstar = 0;
            }
            flux[k] = Sstar * (Sleft * Uleft[k] - fluxLeft[k]) +
                      Sleft *
                          (Wleft.GetPressure() +
                           Wleft.GetDensity() * (Sleft - Wleft.GetVelocity()[0]) * (Sstar - Wleft.GetVelocity()[0])) *
                          Dstar;
            flux[k] *= invSdiff;
        }
        else
        { // version == 3 // Toro Eqn. (10.44)
            MyAtLeastDouble Dstar;
            if (k == irho)
            {
                Dstar = 0;
            }
            else if (k == ipress)
            {
                Dstar = Sstar;
            }
            else if (k == ivx)
            {
                Dstar = 1;
            }
            else if (k == ivy)
            {
                Dstar = 0;
            }
            else if (k == ivz)
            {
                Dstar = 0;
            }
            else
            {
                Dstar = 0;
            }
            flux[k] = Sstar * (Sleft * Uleft[k] - fluxLeft[k]) + Sleft * PLR * Dstar;
            flux[k] *= invSdiff;
        }
    }
    return flux;
}

template class HllcRiemannSolver<mfm_numdims, HLLC_RIEMANN_VERSION>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
