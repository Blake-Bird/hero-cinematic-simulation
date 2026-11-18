/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the RoeRiemannSolver class.
 *
 * This file declares member functions for the derived RoeRiemannSolver class based
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

#include "Matrix.hpp"
#include "RiemannSolver.hpp"
#include <memory>
// #include "../CodeBase/allvars.hpp"

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief Compute the numerical fluxes based on the Roe approximate Riemann solver.
 */
template <int ndim>
bool RoeRiemannSolver<ndim>::ComputeFluxes(const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
                                           const Vector<ndim, MyAtLeastDouble> &rUnit, /**< [in] Unit vector on face */
                                           WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS primitive state */
                                           WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS primitive state */
                                           Vector<ndim, MyAtLeastDouble> &vFace, /**< [in] Face velocity vector */
                                           FluxVector<ndim, MyAtLeastDouble> &flux) const /**< [out] Flux vector */
{
    Vector<nvar, MyAtLeastDouble> valtilde; // averaged values: soundspeed, velocity, entalpy
    Vector<nvar, MyAtLeastDouble> lambda; // eigenvalues
    SquareMatrix<nvar, MyAtLeastDouble> Ktilde; // eigenvectors
    Vector<nvar, MyAtLeastDouble> alpha; // wave strength

    // Compute averaged values
    ComputeAveragedValues(Wleft, Wright, valtilde);

    ComputeEigenvalues(valtilde, lambda);

    if (zeroMassFlux)
    {
        MyAtLeastDouble bm, bp;
        std::unique_ptr<HlleRiemannSolver<ndim>> hlleSolver = std::make_unique<HlleRiemannSolver<ndim>>(eos);
        hlleSolver->ComputeWaveSpeedEstimates(Wleft, Wright, valtilde, bm, bp);
        const MyAtLeastDouble Sstar = 0.5 * (bm + bp);
        Wleft.GetVelocity()[0] -= Sstar;
        Wright.GetVelocity()[0] -= Sstar;
        vFace += Sstar * rUnit;
    }

    ComputeRightEigenvectors(valtilde, Ktilde);
    ComputeWaveStrength(Wleft, Wright, valtilde, alpha);

    // Compute flux depending on the different wave speeds in the simplified shock structure
    if (lambda[irho] >= (MyAtLeastDouble)0.0)
    {
        flux = this->ComputeFlux1d(Wleft);
    }
    // else if ((valtilde[irho] <= (MyAtLeastDouble)0.0)&&((MyAtLeastDouble)0.0 <= valtilde[ivx])) {
    //   flux = ComputeRoeFlux1d(Wleft, Wright, alpha, lambda, Ktilde);
    // } else if ((valtilde[ivx] <= (MyAtLeastDouble)0.0)&&((MyAtLeastDouble)0.0 <= valtilde[ipress])) {
    //   flux = ComputeRoeFlux1d(Wright, Wleft, alpha, lambda, Ktilde);
    else if ((lambda[irho] < (MyAtLeastDouble)0.0) && ((MyAtLeastDouble)0.0 < lambda[ipress]))
    {
        flux = ComputeRoeFlux1d(Wleft, Wright, alpha, lambda, Ktilde);
    }
    else if (lambda[ipress] <= (MyAtLeastDouble)0.0)
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
 * @brief Compute Roe averages.
 *
 * Roe Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void RoeRiemannSolver<ndim>::ComputeAveragedValues(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS fluid state vector */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS fluid state vector */
    Vector<nvar, MyAtLeastDouble> &valtilde) const /**< [out] average values */
{
    MyAtLeastDouble sqrtrhol = sqrt(Wleft.GetDensity());
    MyAtLeastDouble sqrtrhor = sqrt(Wright.GetDensity());
    MyAtLeastDouble invsqrtrhosum = 1.0 / (sqrtrhol + sqrtrhor);
    valtilde[ivx] = (sqrtrhol * Wleft.GetVelocity()[0] + sqrtrhor * Wright.GetVelocity()[0]) * invsqrtrhosum;
    if constexpr (ndim > 1)
    {
        valtilde[ivy] = (sqrtrhol * Wleft.GetVelocity()[1] + sqrtrhor * Wright.GetVelocity()[1]) * invsqrtrhosum;
    }
    if constexpr (ndim > 2)
    {
        valtilde[ivz] = (sqrtrhol * Wleft.GetVelocity()[2] + sqrtrhor * Wright.GetVelocity()[2]) * invsqrtrhosum;
    }
    MyAtLeastDouble Hleft = Wleft.GetSpecificInternalEnergy(gamma) + Wleft.GetPressure() / Wleft.GetDensity();
    MyAtLeastDouble Hright = Wright.GetSpecificInternalEnergy(gamma) + Wright.GetPressure() / Wright.GetDensity();
    for (int i = 0; i < ndim; i++)
    {
        Hleft += (MyAtLeastDouble)0.5 * Wleft.GetVelocity()[i] * Wleft.GetVelocity()[i];
        Hright += (MyAtLeastDouble)0.5 * Wright.GetVelocity()[i] * Wright.GetVelocity()[i];
    }
    valtilde[ipress] = (sqrtrhol * Hleft + sqrtrhor * Hright) * invsqrtrhosum; // Htilde
    valtilde[irho] = valtilde[ipress];
    for (int i = 0; i < ndim; i++)
    {
        valtilde[irho] -= (MyAtLeastDouble)0.5 * valtilde[i] * valtilde[i]; // atilde
    }
    valtilde[irho] = sqrt((gamma - (MyAtLeastDouble)1.0) * valtilde[irho]);
    // if (valtilde[irho] < 0)
    //   valtilde[irho] = (MyAtLeastDouble)0.0;
}

/**
 * @brief Compute Roe eigenvalues.
 *
 * Roe Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void RoeRiemannSolver<ndim>::ComputeEigenvalues(const Vector<nvar, MyAtLeastDouble> &val, /**< [in] values */
                                                Vector<nvar, MyAtLeastDouble> &lambda) const /**< [out] eigenvalues */
{
    lambda[ivx] = val[ivx];
    if constexpr (ndim > 1)
    {
        lambda[ivy] = val[ivx];
    }
    if constexpr (ndim > 2)
    {
        lambda[ivz] = val[ivx];
    }
    lambda[irho] = val[ivx] - val[irho];
    lambda[ipress] = val[ivx] + val[irho];
}

/**
 * @brief Compute Roe right eigenvectors.
 *
 * Roe Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void RoeRiemannSolver<ndim>::ComputeRightEigenvectors(
    const Vector<nvar, MyAtLeastDouble> &val, /**< [in] values */
    SquareMatrix<nvar, MyAtLeastDouble> &K) const /**< [out] eigenvectors */
{
    K = (MyAtLeastDouble)0.0;

    K(ivx, ivx) = val[ivx];
    if constexpr (ndim > 1)
    {
        K(ivy, ivx) = val[ivy];
    }
    if constexpr (ndim > 2)
    {
        K(ivz, ivx) = val[ivz];
    }
    K(irho, ivx) = (MyAtLeastDouble)1.0;
    for (int i = 0; i < ndim; i++)
    {
        K(ipress, ivx) += (MyAtLeastDouble)0.5 * val[i] * val[i];
    }

    if constexpr (ndim > 1)
    {
        K(ivy, ivy) = (MyAtLeastDouble)1.0;
        K(ipress, ivy) = val[ivy];
    }
    if constexpr (ndim > 2)
    {
        K(ivz, ivz) = (MyAtLeastDouble)1.0;
        K(ipress, ivz) = val[ivz];
    }

    K(ivx, irho) = val[ivx] - val[irho];
    if constexpr (ndim > 1)
    {
        K(ivy, irho) = val[ivy];
    }
    if constexpr (ndim > 2)
    {
        K(ivz, irho) = val[ivz];
    }
    K(irho, irho) = (MyAtLeastDouble)1.0;
    K(ipress, irho) = val[ipress] - val[ivx] * val[irho];

    K(ivx, ipress) = val[ivx] + val[irho];
    if constexpr (ndim > 1)
    {
        K(ivy, ipress) = val[ivy];
    }
    if constexpr (ndim > 2)
    {
        K(ivz, ipress) = val[ivz];
    }
    K(irho, ipress) = (MyAtLeastDouble)1.0;
    K(ipress, ipress) = val[ipress] + val[ivx] * val[irho];
}

/**
 * @brief Compute Roe left and right wave speed estimates.
 *
 * Roe Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void RoeRiemannSolver<ndim>::ComputeWaveSpeedEstimates(
    const Vector<nvar, MyAtLeastDouble> &lambda, /**< [in] Roe averaged eigenvalues */
    MyAtLeastDouble &Sleft, /**< [out] Left travelling signal speed */
    MyAtLeastDouble &Sright) const /**< [out] Right travelling signal speed */
{
    Sleft = lambda[irho];
    Sright = lambda[ipress];
}

/**
 * @brief Compute Roe wave strengths.
 *
 * Roe Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void RoeRiemannSolver<ndim>::ComputeWaveStrength(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS fluid state vector */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS fluid state vector */
    const Vector<nvar, MyAtLeastDouble> val, /**< [in] average velocity */
    Vector<nvar, MyAtLeastDouble> &alpha) const /**< [out] wave strength values */
{
    const UFluidVector<ndim, MyAtLeastDouble> Uleft(Wleft, gamma);
    const UFluidVector<ndim, MyAtLeastDouble> Uright(Wright, gamma);
    Vector<nvar, MyAtLeastDouble> deltau{};
    for (int i = 0; i < nvar; i++)
    {
        deltau[i] = Uright[i] - Uleft[i];
    }
    MyAtLeastDouble bardeltau5 = deltau[ipress]; // Uright[ipress] - Uleft[ipress]; // other terms are zero
    alpha[ivx] = (gamma - (MyAtLeastDouble)1.0) / (val[irho] * val[irho]) *
                 (deltau[irho] * (val[ipress] - val[ivx] * val[ivx]) + val[ivx] * deltau[ivx] - bardeltau5);
    if constexpr (ndim > 1)
    {
        alpha[ivy] = deltau[ivy] - deltau[irho] * val[ivy];
    }
    if constexpr (ndim > 2)
    {
        alpha[ivz] = deltau[ivz] - deltau[irho] * val[ivz];
    }
    alpha[irho] = (MyAtLeastDouble)0.5 / val[irho] *
                  (deltau[irho] * (val[ivx] + val[irho]) - deltau[ivx] - val[irho] * alpha[ivx]);
    alpha[ipress] = deltau[irho] - (alpha[irho] + alpha[ivx]);
}

/**
 * @brief Compute central Roe flux.
 *
 * Roe Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
FluxVector<ndim, MyAtLeastDouble> RoeRiemannSolver<ndim>::ComputeRoeFlux1d(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS fluid state vector */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS fluid state vector */
    const Vector<nvar, MyAtLeastDouble> &alpha, /**< [in] wave strength */
    const Vector<nvar, MyAtLeastDouble> &lambda, /**< [in] eigenvalues */
    const SquareMatrix<nvar, MyAtLeastDouble> &Ktilde) const /**< [in] eigenvectors */
{
    MyAtLeastDouble uStarLeft, uStarRight, aStarLeft, aStarRight;
    ComputeStarVelocities(Wleft, Wright, alpha, lambda, Ktilde, uStarLeft, uStarRight, aStarLeft, aStarRight);

#ifndef ROE_RIGHT_FLUXES
    // left
    const FluxVector<ndim, MyAtLeastDouble> fluxLeft = this->ComputeFlux1d(Wleft);
    MyAtLeastDouble lambda1Left = Wleft.GetVelocity()[0] - eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma));
    MyAtLeastDouble lambda1Right = uStarLeft - aStarLeft;
    bool LeftIsTranssonic = ((lambda1Left < 0.0) && (0 < lambda1Right));
    MyAtLeastDouble barlambda1;
    if (LeftIsTranssonic)
    {
        barlambda1 = lambda1Left * (lambda1Right - lambda[irho]) / (lambda1Right - lambda1Left);
    }
#endif // #ifndef ROE_RIGHT_FLUXES
#ifndef ROE_LEFT_FLUXES
    // right
    const FluxVector<ndim, MyAtLeastDouble> fluxRight = this->ComputeFlux1d(Wright);
    MyAtLeastDouble lambda5Left = uStarRight + aStarRight;
    MyAtLeastDouble lambda5Right = Wright.GetVelocity()[0] + eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma));
    bool RightIsTranssonic = ((lambda5Left < 0.0) && (0 < lambda5Right));
    MyAtLeastDouble barlambda5;
    if (RightIsTranssonic)
    {
        barlambda5 = lambda5Right * (lambda[ipress] - lambda5Left) / (lambda5Right - lambda5Left);
    }
#endif // #ifndef ROE_LEFT_FLUXES

    FluxVector<ndim, MyAtLeastDouble> flux{};

#ifndef ROE_RIGHT_FLUXES
    flux += fluxLeft;
#endif // #ifndef ROE_RIGHT_FLUXES
#ifndef ROE_LEFT_FLUXES
    flux += fluxRight;
#endif // #ifndef ROE_LEFT_FLUXES
#ifndef ROE_RIGHT_FLUXES
    // left-sided flux
    if (LeftIsTranssonic)
    {
        for (int kk = 0; kk < nvar; kk++)
        {
            flux[kk] += alpha[irho] * barlambda1 * Ktilde(kk, irho);
        }
    }
    else
    {
        for (int k = 0; k < nvar; k++)
        {
            if (lambda[k] <= (MyAtLeastDouble)0.0)
            {
                for (int kk = 0; kk < nvar; kk++)
                {
                    flux[kk] += alpha[k] * lambda[k] * Ktilde(kk, k);
                }
            }
        }
    }

#endif // ifndef ROE_RIGHT_FLUXES
#ifndef ROE_LEFT_FLUXES
    // right-sided flux
    if (RightIsTranssonic)
    {
        for (int kk = 0; kk < nvar; kk++)
        {
            flux[kk] += alpha[ipress] * barlambda5 * Ktilde(kk, ipress);
        }
    }
    else
    {
        for (int k = 0; k < nvar; k++)
        {
            if (lambda[k] >= (MyAtLeastDouble)0.0)
            {
                for (int kk = 0; kk < nvar; kk++)
                {
                    flux[kk] -= alpha[k] * lambda[k] * Ktilde(kk, k);
                }
            }
        }
    }
#endif // #ifndef ROE_LEFT_FLUXES

#if (!defined(ROE_LEFT_FLUXES) && !defined(ROE_LEFT_FLUXES))
    flux *= (MyAtLeastDouble)0.5;
#endif // #if(!defined(ROE_LEFT_FLUXES) && !defined(ROE_LEFT_FLUXES))
    return flux;
}

/**
 * @brief Compute Roe central (star) velocities.
 *
 * Roe Riemann solver, based on approach outlined by Toro (1999).
 * Variable names often follow Toro naming conventions so read chapter for further details.
 */
template <int ndim>
void RoeRiemannSolver<ndim>::ComputeStarVelocities(
    const WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [in] LHS fluid state vector */
    const WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [in] RHS fluid state vector */
    const Vector<nvar, MyAtLeastDouble> &alpha, /**< [in] wave strength */
    const Vector<nvar, MyAtLeastDouble> &lambda, /**< [in] eigenvalues */
    const SquareMatrix<nvar, MyAtLeastDouble> &Ktilde, /**< [in] eigenvectors */
    MyAtLeastDouble &uStarLeft, /**< [out] primitive fluid variables in star region, derived from left-sided values */
    MyAtLeastDouble &uStarRight, /**< [out] primitive fluid variables in star region, derived from right-sided values */
    MyAtLeastDouble &aStarLeft, /**< [out] left sound speed in star region */
    MyAtLeastDouble &aStarRight) const /**< [out] right sound speed in star region */
{
#ifdef ROE_AVERAGE
    uStarLeft =
        (Wleft.GetDensity() * Wleft.GetVelocity()[0] + alpha[irho] * lambda[irho]) / (Wleft.GetDensity() + alpha[irho]);
    MyAtLeastDouble rhoStar = Wleft.GetDensity() + alpha[irho];
    MyAtLeastDouble pStar = (gamma - (MyAtLeastDouble)1.0) * (Wleft.GetDensity() * Wleft.GetSpecificTotalEnergy(gamma) +
                                                              alpha[irho] * Ktilde(ipress, irho) -
                                                              (MyAtLeastDouble)0.5 * rhoStar * uStarLeft * uStarLeft);
    aStarLeft = sqrt(gamma * pStar / rhoStar); // replace with sound speed using eos???

    uStarRight = (Wright.GetDensity() * Wright.GetVelocity()[0] - alpha[ipress] * lambda[ipress]) /
                 (Wright.GetDensity() - alpha[ipress]);
    rhoStar = Wright.GetDensity() - alpha[ipress];
    pStar = (gamma - (MyAtLeastDouble)1.0) *
            (Wright.GetDensity() * Wright.GetSpecificTotalEnergy(gamma) + alpha[ipress] * Ktilde(ipress, ipress) -
             (MyAtLeastDouble)0.5 * rhoStar * uStarRight * uStarRight);
    aStarRight = sqrt(gamma * pStar / rhoStar); // replace with sound speed using eos???
#elif defined(PVRS_APPROXIMATION)
    MyAtLeastDouble barrho = (MyAtLeastDouble)0.5 * (Wleft.GetDensity() + Wright.GetDensity());
    MyAtLeastDouble bara = (MyAtLeastDouble)0.5 * (eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma)) +
                                                   eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma)));
    MyAtLeastDouble pStar = (MyAtLeastDouble)0.5 * ((Wleft.GetPressure() + Wright.GetPressure()) +
                                                    (Wleft.GetVelocity()[0] - Wright.GetVelocity()[0]) * barrho * bara);
    pstar = max(0, pStar);
    uStarLeft = (MyAtLeastDouble)0.5 * ((Wleft.GetVelocity()[0] + Wright.GetVelocity()[0]) +
                                        (Wleft.GetPressure() - Wright.GetPressure()) / (rhobar * abar));
    uStarRight = uStarLeft;
    MyAtLeastDouble rhoStarLeft = Wleft.GetDensity() + (Wleft.GetVelocity()[0] - uStar) * barrho / bara;
    MyAtLeastDouble rhoStarRight = Wright.GetDensity() + (uStar - Wright.GetVelocity()[0]) * barrho / bara;
    aStarLeft = sqrt(gamma * pStar / rhoStarLeft); // replace with sound speed using eos???
    aStarRight = sqrt(gamma * pStar / rhoStarRight); // replace with sound speed using eos???
#elif defined(TRRS_APPROXIMATION)
    MyAtLeastDouble z = (gamma - (MyAtLeastDouble)1.0) / ((MyAtLeastDouble)2.0 * gamma);
    MyAtLeastDouble pStar =
        pow(eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma)) +
                eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma)) -
                (MyAtLeastDouble)0.5 * (gamma - (MyAtLeastDouble)1.0) *
                    (Wright.GetVelocity()[0] - Wleft.GetVelocity()[0]) / pow(aLeft / Wleft.GetPressure(), z) +
                pow(aRight / Wright.GetPressure(), z),
            1 / z);
    aStarLeft = eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma)) * pow(pStar / Wleft.GetPressure(), z);
    uStarLeft = Wleft.GetPressure() + (MyAtLeastDouble)2.0 / (gamma - (MyAtLeastDouble)1.0) *
                                          (eos->SoundSpeed(Wleft.GetSpecificInternalEnergy(gamma)) - aStarLeft);
    aStarRight = eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma)) * pow(pStar / Wright.GetPressure(), z);
    uStarRight = Wright.GetPressure() + (MyAtLeastDouble)2.0 / (gamma - (MyAtLeastDouble)1.0) *
                                            (aStarRight - eos->SoundSpeed(Wright.GetSpecificInternalEnergy(gamma)));
#endif // defined(ROE_AVERAGE) || defined(PVRS_APPROXIMATION)
}

template class RoeRiemannSolver<mfm_numdims>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
