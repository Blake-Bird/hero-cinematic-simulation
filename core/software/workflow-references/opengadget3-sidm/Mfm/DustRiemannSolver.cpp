/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the DustRiemannSolver class.
 *
 * @author OpenGadget3-development team, in particular Giovanni Tedeschi.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/verbose.hpp"
#include "RiemannSolver.hpp"

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{
/**
 * @brief Compute the numerical fluxes for a dusty fluid
 */
template <int ndim>
bool DustRiemannSolver<ndim>::ComputeFluxes(const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
                                            const Vector<ndim> &rUnit, /**< [in] Unit vector on face */
                                            WFluidVector<ndim> &Wleft, /**< [in] LHS primitive state */
                                            WFluidVector<ndim> &Wright, /**< [in] RHS primitive state */
                                            Vector<ndim> &vFace, /**< [in] Face velocity vector */
                                            FluxVector<ndim> &flux) const /**< [out] Flux vector */
{

    if ((Wleft[ivx] >= 0.) && (Wright[ivx] >= 0.))
    {
        flux[irho] = Wleft[irho] * Wleft[ivx];
        flux[ivx] = Wleft[irho] * Wleft[ivx] * Wleft[ivx];
        if constexpr (ndim >= 2)
        {
            flux[ivy] = Wleft[irho] * Wleft[ivx] * Wleft[ivy];
        }
        if constexpr (ndim == 3)
        {
            flux[ivz] = Wleft[irho] * Wleft[ivx] * Wleft[ivz];
        }
    }
    else if ((Wleft[ivx] <= 0.) && (Wright[ivx] <= 0.))
    {
        flux[irho] = Wright[irho] * Wright[ivx];
        flux[ivx] = Wright[irho] * Wright[ivx] * Wright[ivx];
        if constexpr (ndim >= 2)
        {
            flux[ivy] = Wright[irho] * Wright[ivx] * Wright[ivy];
        }
        if constexpr (ndim == 3)
        {
            flux[ivz] = Wright[irho] * Wright[ivx] * Wright[ivz];
        }
    }
    else if ((Wleft[ivx] < 0.) && (Wright[ivx] > 0.))
    {
        flux[irho] = 0.;
        flux[ivx] = 0.;
        if constexpr (ndim >= 2)
        {
            flux[ivy] = 0.;
        }
        if constexpr (ndim == 3)
        {
            flux[ivz] = 0.;
        }
    }
    else if ((Wleft[ivx] > 0.) && (Wright[ivx] < 0.))
    {
        flux[irho] = Wleft[irho] * Wleft[ivx] + Wright[irho] * Wright[ivx];
        flux[ivx] = Wleft[irho] * Wleft[ivx] * Wleft[ivx] + Wright[irho] * Wright[ivx] * Wright[ivx];
        if constexpr (ndim >= 2)
        {
            flux[ivy] = Wleft[irho] * Wleft[ivx] * Wleft[ivy] + Wright[irho] * Wright[ivx] * Wright[ivy];
        }
        if constexpr (ndim == 3)
        {
            flux[ivz] = Wleft[irho] * Wleft[ivx] * Wleft[ivz] + Wright[irho] * Wright[ivx] * Wright[ivz];
        }
    }

    return true;
}

template class DustRiemannSolver<mfm_numdims>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
