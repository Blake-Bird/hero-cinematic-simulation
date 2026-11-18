/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the ExactRiemannSovler class.
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
#include "../System/assert.hpp"

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief Compute fluxes for Exact Riemann solver, based on approach outlined by Toro (1999).
 */
template <int ndim>
bool ExactRiemannSolver<ndim>::ComputeFluxes(
    const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
    const Vector<ndim, MyAtLeastDouble> &rUnit, /**< [in] Unit vector on face */
    WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS primitive state */
    WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS primitive state */
    Vector<ndim, MyAtLeastDouble> &vFace, /**< [inout] Face velocity vector */
    FluxVector<ndim, MyAtLeastDouble> &flux) const /**< [out] Flux vector */
{
    const MyAtLeastDouble cl = eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma));
    const MyAtLeastDouble cr = eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma));
    MyAtLeastDouble uStar; // Velocity in star region
    MyAtLeastDouble pStar; // Pressure in star region
    MyAtLeastDouble u; // Riemann solver state velocity
    MyAtLeastDouble d; // Riemann solver state density
    MyAtLeastDouble p; // Riemann solver state pressure
    WFluidVector<ndim, MyAtLeastDouble> Wface; // Riemann solver state vector

    // Compute p and u values at interface (in star region)
    ComputeStarRegion(Wleft.GetDensity(), Wright.GetDensity(), Wleft.GetVelocity()[0], Wright.GetVelocity()[0],
                      Wleft.GetPressure(), Wright.GetPressure(), cl, cr, uStar, pStar);

    // If not a vacuum state (i.e. pStar > 0), then sample solution
    //---------------------------------------------------------------------------------------------
    if (pStar > small_float)
    {
        SampleExactSolution(MyAtLeastDouble(0.0), pStar, uStar, Wleft.GetPressure(), Wright.GetPressure(),
                            Wleft.GetDensity(), Wright.GetDensity(), cl, cr, Wleft.GetVelocity()[0],
                            Wright.GetVelocity()[0], p, d, u);

        // Copy Riemann solver state
        Wface.GetVelocity()[0] = u;
        if constexpr (ndim >= 2)
        {
            Wface.GetVelocity()[1] = 0.0;
        }
        if constexpr (ndim == 3)
        {
            Wface.GetVelocity()[2] = 0.0;
        }
        Wface.GetDensity() = d;
        Wface.GetPressure() = p;

        // If selecting face that moves with the star region velocity (i.e. so mass flux is zero),
        // then re-set face velocity to match star region (in x-direction)
        if (zeroMassFlux)
        {
            Wface.GetVelocity()[0] = MyAtLeastDouble(0.0);
            for (int k = 0; k < ndim; k++)
            {
                vFace[k] += u * rUnit[k];
            }
        }
        // Otherwise, calculate transverse velocity fluxes from upwind direction
        else
        {
            if (u > MyAtLeastDouble(0.0))
            {
                if constexpr (ndim >= 2)
                {
                    Wface.GetVelocity()[1] = Wleft.GetVelocity()[1];
                }
                if constexpr (ndim == 3)
                {
                    Wface.GetVelocity()[2] = Wleft.GetVelocity()[2];
                }
            }
            else
            {
                if constexpr (ndim >= 2)
                {
                    Wface.GetVelocity()[1] = Wright.GetVelocity()[1];
                }
                if constexpr (ndim == 3)
                {
                    Wface.GetVelocity()[2] = Wright.GetVelocity()[2];
                }
            }
        }
    }
    // Otherwise assume vacuum state conditions (compare Toro (1999), Chapter 4.6)
    //---------------------------------------------------------------------------------------------
    else
    {
        if ((Wleft.GetDensity() > 0.0) && (Wright.GetDensity() > 0.0))
        {
            // internal vacuum, Toro (1999), chapter 4.6.3
            SampleInternalVacuum((MyAtLeastDouble)0.0, Wleft, Wright, cl, cr, Wface);
        }
        else if (Wleft.GetDensity() > 0)
        {
            // right vacuum, Toro (1999), chapter 4.6.1
            SampleOneSidedVacuum((MyAtLeastDouble)0.0, Wleft, cl, Wface);
        }
        else if (Wright.GetDensity() > 0)
        {
            // left vacuum, Toro (1999), chapter 4.6.2
            SampleOneSidedVacuum((MyAtLeastDouble)0.0, Wright, -cr, Wface);
        }
        else
        {
            // vacuum states on both sides.
            flux = MyAtLeastDouble(0.0);
            return false;
        }
    }
    //---------------------------------------------------------------------------------------------

    // use the sampled solution to calculate the fluxes.
    flux = this->ComputeFlux1d(Wface);
    Wface.GetVelocity()[0] += vFace[0];

    return true;
}

/**
 * @brief Compute the face state solution for an internal vacuum.
 */
template <int ndim>
void ExactRiemannSolver<ndim>::SampleInternalVacuum(
    const MyAtLeastDouble S, /**< [in] (Dimensionless) position in shock */
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS primitive state */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS primitive state */
    const MyAtLeastDouble cl, /**< [in] LHS sound velocity */
    const MyAtLeastDouble cr, /**< [in] RHS sound velocity */
    WFluidVector<ndim, MyAtLeastDouble> &Wface) const /**< [out] Intermediate primitive state */
{
    MyAtLeastDouble SstarL = Wleft.GetVelocity()[0] + g9 * cl;
    MyAtLeastDouble SstarR = Wright.GetVelocity()[0] - g9 * cr;
    if (SstarL >= S)
    {
        // left state
        Wface = Wleft;
    }
    else if (SstarR <= S)
    {
        // right state
        Wface = Wright;
    }
    else
    {
        // vacuum
        Wface.GetVelocity()[0] = 0.0; // in principle this can be non-zero, but as the flux calculated from it will be
                                      // zero we just leave tis for now.
        if constexpr (ndim >= 2)
        {
            Wface.GetVelocity()[1] = 0.0;
        }
        if constexpr (ndim == 3)
        {
            Wface.GetVelocity()[2] = 0.0;
        }
        Wface.GetDensity() = 0.0;
        Wface.GetPressure() = 0.0;
    }
}

/**
 * @brief Compute the face state solution for a single-sided (left or right) vacuum.
 */
template <int ndim>
void ExactRiemannSolver<ndim>::SampleOneSidedVacuum(
    const MyAtLeastDouble S, /**< [in] (Dimensionless) position in shock */
    const WFluidVector<ndim, MyAtLeastDouble> &W, /**< [in] non-vacuum primitive state */
    const MyAtLeastDouble csound, /**< [in] sound velocity */
    WFluidVector<ndim, MyAtLeastDouble> &Wface) const /**< [out] Intermediate primitive state */
{
    MyAtLeastDouble Sstar = W.GetVelocity()[0] + g9 * csound;
    if (Sstar <= S)
    {
        // vacuum
        Wface.GetVelocity()[0] = Sstar;
        if constexpr (ndim >= 2)
        {
            Wface.GetVelocity()[1] = W.GetVelocity()[1];
        }
        if constexpr (ndim == 3)
        {
            Wface.GetVelocity()[2] = W.GetVelocity()[2];
        }
        Wface.GetDensity() = 0.0;
        Wface.GetPressure() = 0.0;
    }
    else if (S <= W.GetVelocity()[0] - csound)
    {
        // rarefaction
        Wface.GetVelocity()[0] = g5 * (csound + g7 * W.GetVelocity()[0] + S);
        if constexpr (ndim >= 2)
        {
            Wface.GetVelocity()[1] = W.GetVelocity()[1];
        }
        if constexpr (ndim == 3)
        {
            Wface.GetVelocity()[2] = W.GetVelocity()[2];
        }
        const MyAtLeastDouble c = g5 * (1 + g7 * (W.GetVelocity()[0] - S) / csound);
        Wface.GetDensity() = W.GetDensity() * pow(c, g4);
        Wface.GetPressure() = W.GetPressure() * pow(c, g3);
    }
    else
    {
        // original state
        Wface = W;
    }
}

/**
 * @brief Compute the star region for Exact Riemann solver, based on approach outlined by Toro (1999).
 *
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void ExactRiemannSolver<ndim>::ComputeStarRegion(const MyAtLeastDouble dl, /**< [in] LHS density */
                                                 const MyAtLeastDouble dr, /**< [in] RHS density */
                                                 const MyAtLeastDouble ul, /**< [in] LHS velocity */
                                                 const MyAtLeastDouble ur, /**< [in] RHS velocity */
                                                 const MyAtLeastDouble pl, /**< [in] LHS pressure */
                                                 const MyAtLeastDouble pr, /**< [in] RHS pressure */
                                                 const MyAtLeastDouble cl, /**< [in] LHS sound speed */
                                                 const MyAtLeastDouble cr, /**< [in] RHS sound speed */
                                                 MyAtLeastDouble &uStar, /**< [out] Velocity of intermediate state */
                                                 MyAtLeastDouble &pStar) const /**< [out] Intermediate pressure state */
{
    const MyAtLeastDouble Al = g5 / dl; // ..
    const MyAtLeastDouble Ar = g5 / dr; // ..
    const MyAtLeastDouble Bl = pl * g6; // ..
    const MyAtLeastDouble Br = pr * g6; // ..
    const MyAtLeastDouble quser = (MyAtLeastDouble)2.0; // First guess of solution
    const MyAtLeastDouble cup = (MyAtLeastDouble)0.25 * (dl + dr) * (cl + cr); // ..
    const MyAtLeastDouble pmin = std::min(pl, pr); // ..
    const MyAtLeastDouble pmax = std::max(pl, pr); // ..
    int iteration = 0; // Iteration counter
    MyAtLeastDouble fl; // Left-state variable
    MyAtLeastDouble flprime; // Left-state variable gradient
    MyAtLeastDouble fr; // Right-state variable
    MyAtLeastDouble frprime; // Right-state variable gradient
    MyAtLeastDouble pold; // Old intermediate pressure
    MyAtLeastDouble ppv; // Primitive Variable pressure

    // Check all input variables are valid numbers
    assert(std::isnormal(dl));
    assert(std::isnormal(dr));
    assert(std::isnormal(pl));
    assert(std::isnormal(pr));
    assert(std::isnormal(cl));
    assert(std::isnormal(cr));

    // Check for vacuum condition
    if (cl + cr - g7 * (ur - ul) <= MyAtLeastDouble(0.0))
    {
        pStar = (MyAtLeastDouble)0.0;
        uStar = (MyAtLeastDouble)0.0;
        return;
    }

    // Compute guess for pStar from Primitive Variable Riemann Solver
    ppv = MyAtLeastDouble(0.5) * (pl + pr) + MyAtLeastDouble(0.5) * (ul - ur) * cup;
    ppv = std::max(MyAtLeastDouble(0.0), ppv);

    // Select PVRS Riemann Solver
    if (pmax / pmin <= quser && pmin <= ppv && ppv <= pmax)
    {
        pStar = ppv;
    }
    // Select two-rarefaction Riemann Solver
    else if (ppv < pmin)
    {
        const MyAtLeastDouble pq = pow(pl / pr, g1);
        const MyAtLeastDouble um =
            (pq * ul / cl + ur / cr + g4 * (pq - MyAtLeastDouble(1.0))) / (pq / cl + MyAtLeastDouble(1.0) / cr);
        const MyAtLeastDouble ptl = MyAtLeastDouble(1.0) + g7 * (ul - um) / cl;
        const MyAtLeastDouble ptr = MyAtLeastDouble(1.0) + g7 * (um - ur) / cr;
        pStar = MyAtLeastDouble(0.5) * (pl * pow(ptl, g3) + pr * pow(ptr, g3));
    }
    // Select two-shock Riemann Solver with PVRS as estimate
    else
    {
        const MyAtLeastDouble gel = sqrt((g5 / dl) / (g6 * pl + ppv));
        const MyAtLeastDouble ger = sqrt((g5 / dr) / (g6 * pr + ppv));
        pStar = (gel * pl + ger * pr - (ur - ul)) / (gel + ger);
    }

    pStar = std::max(pStar, MyAtLeastDouble(small_float));
    assert(std::isnormal(pStar));

    // Main iteration loop
    //---------------------------------------------------------------------------------------------
    do
    {
        iteration++;

        // Calculate contribution to f and fprime for LHS
        if (pStar > pl)
        {
            fl = (pStar - pl) * sqrt(Al / (pStar + Bl));
            flprime =
                sqrt(Al / (pStar + Bl)) * (MyAtLeastDouble(1.0) - MyAtLeastDouble(0.5) * (pStar - pl) / (pStar + Bl));
        }
        else
        {
            fl = g4 * cl * (pow(pStar / pl, g1) - MyAtLeastDouble(1.0));
            flprime = pow(pStar / pl, -g2) / (dl * cl);
        }

        // Calculate contribution to f and fprime for RHS
        if (pStar > pr)
        {
            fr = (pStar - pr) * sqrt(Ar / (pStar + Br));
            frprime =
                sqrt(Ar / (pStar + Br)) * (MyAtLeastDouble(1.0) - MyAtLeastDouble(0.5) * (pStar - pr) / (pStar + Br));
        }
        else
        {
            fr = g4 * cr * (pow(pStar / pr, g1) - MyAtLeastDouble(1.0));
            frprime = pow(pStar / pr, -g2) / (dr * cr);
        }

        // Perform Newton-Raphson iteration
        pold = pStar;
        pStar = pStar - (fl + fr + ur - ul) / (flprime + frprime);

        // Check if convergence has been achieved
        if (pStar < MyAtLeastDouble(1.0e-30))
        {
            pStar = MyAtLeastDouble(1.0e-30); // this could also be smaller than zero
        }
        else if ((MyAtLeastDouble)2.0 * abs(pStar - pold) / (pStar + pold) < tolerance)
        {
            break;
        }

        assert(std::isnormal(pStar));

    } while (iteration < maxNumIterations);
    //---------------------------------------------------------------------------------------------

    // Compute velocity of star region
    uStar = MyAtLeastDouble(0.5) * (ul + ur) + MyAtLeastDouble(0.5) * (fr - fl);
    assert(std::isfinite(uStar));

    return;
}

/**
 * @brief Samples the solution of the Exact Riemann solver at the given LHS and RHS states.
 */
template <int ndim>
void ExactRiemannSolver<ndim>::SampleExactSolution(const MyAtLeastDouble s, ///< [in] (Dimensionless) position in shock
                                                   const MyAtLeastDouble pStar, ///< [in] Pressure of star region
                                                   const MyAtLeastDouble uStar, ///< [in] Velocity of star region
                                                   const MyAtLeastDouble pl, ///< [in] LHS pressure
                                                   const MyAtLeastDouble pr, ///< [in] RHS pressure
                                                   const MyAtLeastDouble dl, ///< [in] LHS density
                                                   const MyAtLeastDouble dr, ///< [in] RHS density
                                                   const MyAtLeastDouble cl, ///< [in] LHS sound speed
                                                   const MyAtLeastDouble cr, ///< [in] RHS sound speed
                                                   const MyAtLeastDouble ul, ///< [in] LHS velocity
                                                   const MyAtLeastDouble ur, ///< [in] RHS velocity
                                                   MyAtLeastDouble &p, ///< [out] Pressure at s
                                                   MyAtLeastDouble &d, ///< [out] Density at s
                                                   MyAtLeastDouble &u) const ///< [out] Velocity at s
{

    // If sampling point lies to the left of the contact discontinuity
    //---------------------------------------------------------------------------------------------
    if (s <= uStar)
    {

        // Left rarefaction
        //-------------------------------------------------------------------------------------------
        if (pStar <= pl)
        {
            const MyAtLeastDouble sh = ul - cl;

            // Sampled point is left data state
            if (s <= sh)
            {
                d = dl;
                u = ul;
                p = pl;
            }
            else
            {
                const MyAtLeastDouble cm = cl * pow(pStar / pl, g1);
                const MyAtLeastDouble st = uStar - cm;

                // Sample point is star left state
                if (s > st)
                {
                    d = dl * pow(pStar / pl, invgamma);
                    u = uStar;
                    p = pStar;
                }
                // Sampled point is inside left fan
                else
                {
                    const MyAtLeastDouble c = g5 * (cl + g7 * (ul - s));
                    u = g5 * (cl + g7 * ul + s);
                    d = dl * pow(c / cl, g4);
                    p = pl * pow(c / cl, g3);
                }
            }
        }

        // Left shock
        //-------------------------------------------------------------------------------------------
        else
        {
            const MyAtLeastDouble pml = pStar / pl;
            const MyAtLeastDouble sl = ul - cl * sqrt(g2 * pml + g1);

            // Sampled point is left data state
            if (s <= sl)
            {
                d = dl;
                u = ul;
                p = pl;
            }
            // Sampled point is star left state
            else
            {
                d = dl * (pml + g6) / (pml * g6 + MyAtLeastDouble(1.0));
                u = uStar;
                p = pStar;
            }
        }
        //-------------------------------------------------------------------------------------------
    }
    // Sampling point lies to the right of the contact discontinuity
    //---------------------------------------------------------------------------------------------
    else
    {

        // Right shock
        //-------------------------------------------------------------------------------------------
        if (pStar >= pr)
        {
            const MyAtLeastDouble pmr = pStar / pr;
            const MyAtLeastDouble sr = ur + cr * sqrt(g2 * pmr + g1);

            // Sampled point is right data state
            if (s >= sr)
            {
                d = dr;
                u = ur;
                p = pr;
            }
            // Sampled point is star right state
            else
            {
                d = dr * (pmr + g6) / (pmr * g6 + MyAtLeastDouble(1.0));
                u = uStar;
                p = pStar;
            }
        }
        // Right rarefaction
        //-------------------------------------------------------------------------------------------
        else
        {
            const MyAtLeastDouble sh = ur + cr;

            // Sampled point is left data state
            if (s >= sh)
            {
                d = dr;
                u = ur;
                p = pr;
            }
            else
            {
                const MyAtLeastDouble cm = cr * pow(pStar / pr, g1);
                const MyAtLeastDouble st = uStar + cm;

                // Sample point is star right state
                if (s <= st)
                {
                    d = dr * pow(pStar / pr, invgamma);
                    u = uStar;
                    p = pStar;
                }
                // Sampled point is inside left fan
                else
                {
                    const MyAtLeastDouble c = g5 * (cr - g7 * (ur - s));
                    u = g5 * (-cr + g7 * ur + s);
                    d = dr * pow(c / cr, g4);
                    p = pr * pow(c / cr, g3);
                }
            }
        }
    }
    //---------------------------------------------------------------------------------------------

    assert(std::isnormal(p));
    assert(std::isnormal(d));
    assert(std::isfinite(u));

    return;
}

template class ExactRiemannSolver<mfm_numdims>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
