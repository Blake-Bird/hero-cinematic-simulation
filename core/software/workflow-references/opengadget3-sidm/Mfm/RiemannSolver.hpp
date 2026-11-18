/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the RiemannSolver class.
 *
 * This file declares the RiemannSolver class including its derived classes.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET3_RIEMANN_SOLVER_HPP_
#define _OPENGADGET3_RIEMANN_SOLVER_HPP_

#include "../CodeBase/precision.hpp"
#include "Eos.hpp"
#include "FluidVector.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include <cmath>
#include <cstdlib>
#include <math.h>

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief   Base Riemann solver class.
 * @author  D. A. Hubber, updated by F. Groth
 * @date    13/02/2018
 */
template <int ndim> class RiemannSolver
{
  protected:
    const MyAtLeastDouble gamma;
    const MyAtLeastDouble invgamma;
    const MyAtLeastDouble g1;
    const MyAtLeastDouble g2;
    const MyAtLeastDouble g3;
    const MyAtLeastDouble g4;
    const MyAtLeastDouble g5;
    const MyAtLeastDouble g6;
    const MyAtLeastDouble g7;
    const MyAtLeastDouble g8;
    const MyAtLeastDouble g9;
    Eos<ndim> *const eos;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief Base constructor for Riemann solver class.
     *
     * Sets all gamma const variables.
     */
    RiemannSolver(Eos<ndim> *const);

    /**
     * @brief Compute fluxes for given Riemann solver.
     */
    virtual bool ComputeFluxes(const bool, const Vector<ndim, MyAtLeastDouble> &, WFluidVector<ndim, MyAtLeastDouble> &,
                               WFluidVector<ndim, MyAtLeastDouble> &, Vector<ndim, MyAtLeastDouble> &,
                               FluxVector<ndim, MyAtLeastDouble> &) const = 0;

    /**
     * @brief Compute fluxes in 1d direction.
     */
    inline FluxVector<ndim, MyAtLeastDouble> ComputeFlux1d(const WFluidVector<ndim, MyAtLeastDouble> &W) const
    {
        FluxVector<ndim, MyAtLeastDouble> flux;
        flux.GetMass() = W.GetDensity() * W.GetVelocity()[0];
        for (int k = 0; k < ndim; k++)
        {
            flux.GetMomentum()[k] = flux.GetMass() * W.GetVelocity()[k];
        }
        flux.GetMomentum()[0] += W.GetPressure();
        flux.GetEnergy() = W.GetVelocity()[0] * (W.GetDensity() * W.GetSpecificTotalEnergy(gamma) + W.GetPressure());
        return flux;
    }

    /**
     * @brief Compute flux in 1d face reference frame
     */
    inline FluxVector<ndim, MyAtLeastDouble> ComputeFlux1dAtFace(const WFluidVector<ndim, MyAtLeastDouble> &W,
                                                                 const MyAtLeastDouble vFace) const
    {
        FluxVector<ndim, MyAtLeastDouble> flux;
        for (int k = 0; k < ndim; k++)
        {
            flux.GetMomentum()[k] = W.GetDensity() * (W.GetVelocity()[0] - vFace) * W.GetVelocity()[k];
        }
        flux.GetMomentum()[0] = W.GetDensity() * (W.GetVelocity()[0] - vFace) * W.GetVelocity()[0] + W.GetPressure();
        flux.GetMass() = W.GetDensity() * (W.GetVelocity()[0] - vFace);
        flux.GetEnergy() = (W.GetVelocity()[0] - vFace) * W.GetDensity() * W.GetSpecificTotalEnergy(gamma) +
                           W.GetVelocity()[0] * W.GetPressure();
        return flux;
    }
};

/**
 * @brief   Exact Riemann solver, based on algorithm described in Toro (1999).
 * @author  D. A. Hubber, updated by F. Groth
 * @date    13/02/2018
 */
template <int ndim> class ExactRiemannSolver final : public RiemannSolver<ndim>
{
  protected:
    using RiemannSolver<ndim>::gamma;
    using RiemannSolver<ndim>::invgamma;
    using RiemannSolver<ndim>::g1;
    using RiemannSolver<ndim>::g2;
    using RiemannSolver<ndim>::g3;
    using RiemannSolver<ndim>::g4;
    using RiemannSolver<ndim>::g5;
    using RiemannSolver<ndim>::g6;
    using RiemannSolver<ndim>::g7;
    using RiemannSolver<ndim>::g8;
    using RiemannSolver<ndim>::g9;
    using RiemannSolver<ndim>::eos;

    const int maxNumIterations;
    const MyAtLeastDouble tolerance;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief Constructor for Exact Riemann solver class.
     *
     * Sets all gamma const variables.
     */
    ExactRiemannSolver(Eos<ndim> *const _eos, const int _maxNumIterations, const MyAtLeastDouble _tolerance)
        : RiemannSolver<ndim>(_eos), maxNumIterations(_maxNumIterations), tolerance(_tolerance){};

    /**
     * @brief Compute fluxes for Exact Riemann solver.
     */
    virtual bool ComputeFluxes(const bool, const Vector<ndim, MyAtLeastDouble> &, WFluidVector<ndim, MyAtLeastDouble> &,
                               WFluidVector<ndim, MyAtLeastDouble> &, Vector<ndim, MyAtLeastDouble> &,
                               FluxVector<ndim, MyAtLeastDouble> &) const override;

    void ComputeStarRegion(const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble,
                           const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble,
                           MyAtLeastDouble &, MyAtLeastDouble &) const;

    void SampleExactSolution(const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble,
                             const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble,
                             const MyAtLeastDouble, const MyAtLeastDouble, const MyAtLeastDouble, MyAtLeastDouble &,
                             MyAtLeastDouble &, MyAtLeastDouble &) const;

    void SampleInternalVacuum(const MyAtLeastDouble, const WFluidVector<ndim, MyAtLeastDouble> &,
                              const WFluidVector<ndim, MyAtLeastDouble> &, const MyAtLeastDouble, const MyAtLeastDouble,
                              WFluidVector<ndim, MyAtLeastDouble> &) const;

    void SampleOneSidedVacuum(const MyAtLeastDouble, const WFluidVector<ndim, MyAtLeastDouble> &, const MyAtLeastDouble,
                              WFluidVector<ndim, MyAtLeastDouble> &) const;
};

/**
 * @brief   HLL approximate Riemann solver
 * @author  D. A. Hubber, updated by F. Groth
 * @date    13/02/2018
 */
template <int ndim> class HllRiemannSolver final : public RiemannSolver<ndim>
{
  protected:
    using RiemannSolver<ndim>::gamma;
    using RiemannSolver<ndim>::invgamma;
    using RiemannSolver<ndim>::g1;
    using RiemannSolver<ndim>::g2;
    using RiemannSolver<ndim>::g3;
    using RiemannSolver<ndim>::g4;
    using RiemannSolver<ndim>::g5;
    using RiemannSolver<ndim>::g6;
    using RiemannSolver<ndim>::g7;
    using RiemannSolver<ndim>::g8;
    using RiemannSolver<ndim>::g9;
    using RiemannSolver<ndim>::eos;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief Constructor for Hll Riemann solver class.
     *
     * Sets all gamma const variables.
     */
    HllRiemannSolver(Eos<ndim> *const _eos) : RiemannSolver<ndim>(_eos){};

    /**
     * @brief Compute the numerical fluxes based on the HLL approximate Riemann solver.
     */
    virtual bool ComputeFluxes(const bool, const Vector<ndim, MyAtLeastDouble> &, WFluidVector<ndim, MyAtLeastDouble> &,
                               WFluidVector<ndim, MyAtLeastDouble> &, Vector<ndim, MyAtLeastDouble> &,
                               FluxVector<ndim, MyAtLeastDouble> &) const override;

    /**
     * @brief Compute the left and right wave speed estimates for HLL Riemann solver.
     *
     * HLL Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    void ComputeWaveSpeedEstimates(const WFluidVector<ndim, MyAtLeastDouble> &,
                                   const WFluidVector<ndim, MyAtLeastDouble> &, MyAtLeastDouble &,
                                   MyAtLeastDouble &) const;
    /**
     * @brief Compute the central (HLL) state vector for HLL Riemann solver.
     *
     * HLL Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    UFluidVector<ndim, MyAtLeastDouble> ComputeHllStateVector(const WFluidVector<ndim, MyAtLeastDouble> &,
                                                              const WFluidVector<ndim, MyAtLeastDouble> &,
                                                              const MyAtLeastDouble, const MyAtLeastDouble) const;
    /**
     * @brief Compute the central (HLL) flux for HLL Riemann solver.
     *
     * HLL Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    FluxVector<ndim, MyAtLeastDouble> ComputeHllFlux1d(const WFluidVector<ndim, MyAtLeastDouble> &,
                                                       const WFluidVector<ndim, MyAtLeastDouble> &,
                                                       const MyAtLeastDouble, const MyAtLeastDouble) const;
};

/**
 * @brief   HLLC approximate Riemann solver
 * @author  F. Groth, based on HLL solver of D. A. Hubber
 * @date    02/12/2022
 */
template <int ndim, int version> class HllcRiemannSolver final : public RiemannSolver<ndim>
{
  protected:
    using RiemannSolver<ndim>::gamma;
    using RiemannSolver<ndim>::invgamma;
    using RiemannSolver<ndim>::g1;
    using RiemannSolver<ndim>::g2;
    using RiemannSolver<ndim>::g3;
    using RiemannSolver<ndim>::g4;
    using RiemannSolver<ndim>::g5;
    using RiemannSolver<ndim>::g6;
    using RiemannSolver<ndim>::g7;
    using RiemannSolver<ndim>::g8;
    using RiemannSolver<ndim>::g9;
    using RiemannSolver<ndim>::eos;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief Constructor for Hllc Riemann solver class.
     *
     * Sets all gamma const variables.
     */
    HllcRiemannSolver(Eos<ndim> *const _eos) : RiemannSolver<ndim>(_eos){};

    /**
     * @brief Compute the numerical fluxes based on the HLLC approximate Riemann solver.
     */
    virtual bool ComputeFluxes(const bool, const Vector<ndim, MyAtLeastDouble> &, WFluidVector<ndim, MyAtLeastDouble> &,
                               WFluidVector<ndim, MyAtLeastDouble> &, Vector<ndim, MyAtLeastDouble> &,
                               FluxVector<ndim, MyAtLeastDouble> &) const override;

    /**
     * @brief Compute the central (star) velocity.
     *
     * HLLC Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    MyAtLeastDouble ComputeStarVelocity(const WFluidVector<ndim, MyAtLeastDouble> &,
                                        const WFluidVector<ndim, MyAtLeastDouble> &, const MyAtLeastDouble,
                                        const MyAtLeastDouble) const;
    /**
     * @brief Compute the central (Hllc) flux.
     *
     * HLLC Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    FluxVector<ndim, MyAtLeastDouble> ComputeHllcFlux1d(const WFluidVector<ndim, MyAtLeastDouble> &,
                                                        const WFluidVector<ndim, MyAtLeastDouble> &,
                                                        const MyAtLeastDouble, const MyAtLeastDouble,
                                                        const MyAtLeastDouble) const;
};

/**
 * @brief   HLLE approximate Riemann solver
 * @author  F. Groth
 * @date    17/11/2022
 */
template <int ndim> class HlleRiemannSolver final : public RiemannSolver<ndim>
{
  protected:
    using RiemannSolver<ndim>::gamma;
    using RiemannSolver<ndim>::invgamma;
    using RiemannSolver<ndim>::g1;
    using RiemannSolver<ndim>::g2;
    using RiemannSolver<ndim>::g3;
    using RiemannSolver<ndim>::g4;
    using RiemannSolver<ndim>::g5;
    using RiemannSolver<ndim>::g6;
    using RiemannSolver<ndim>::g7;
    using RiemannSolver<ndim>::g8;
    using RiemannSolver<ndim>::g9;
    using RiemannSolver<ndim>::eos;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief Constructor for Hlle Riemann solver class.
     *
     * Sets all gamma const variables.
     */
    HlleRiemannSolver(Eos<ndim> *const _eos) : RiemannSolver<ndim>(_eos){};

    /**
     * @brief Compute the numerical fluxes based on the HLLE approximate Riemann solver.
     */
    virtual bool ComputeFluxes(const bool, const Vector<ndim, MyAtLeastDouble> &, WFluidVector<ndim, MyAtLeastDouble> &,
                               WFluidVector<ndim, MyAtLeastDouble> &, Vector<ndim, MyAtLeastDouble> &,
                               FluxVector<ndim, MyAtLeastDouble> &) const override;

    /**
     * @brief Compute left and right wave speed estimates.
     *
     * Hlle Riemann solver, based on approach outlined by Einfeldt (1991).
     * Variable names often follow Einfeldt or Toro naming conventions so read paper/chapter for further details.
     */
    void ComputeWaveSpeedEstimates(const WFluidVector<ndim, MyAtLeastDouble> &,
                                   const WFluidVector<ndim, MyAtLeastDouble> &, const Vector<nvar, MyAtLeastDouble> &,
                                   MyAtLeastDouble &, MyAtLeastDouble &) const;
    /**
     * @brief Compute central (Hlle) flux.
     *
     * Hlle Riemann solver, based on approach outlined by Einfeldt (1991).
     * Variable names often follow Einfeldt or Toro naming conventions so read paper/chapter for further details.
     */
    FluxVector<ndim, MyAtLeastDouble> ComputeHlleFlux1d(const WFluidVector<ndim, MyAtLeastDouble> &,
                                                        const WFluidVector<ndim, MyAtLeastDouble> &,
                                                        const MyAtLeastDouble, const MyAtLeastDouble) const;
};

/**
 * @brief   Roe approximate Riemann solver
 * @author  F. Groth
 * @date    02/12/2022
 */
template <int ndim> class RoeRiemannSolver final : public RiemannSolver<ndim>
{
  protected:
    using RiemannSolver<ndim>::gamma;
    using RiemannSolver<ndim>::invgamma;
    using RiemannSolver<ndim>::g1;
    using RiemannSolver<ndim>::g2;
    using RiemannSolver<ndim>::g3;
    using RiemannSolver<ndim>::g4;
    using RiemannSolver<ndim>::g5;
    using RiemannSolver<ndim>::g6;
    using RiemannSolver<ndim>::g7;
    using RiemannSolver<ndim>::g8;
    using RiemannSolver<ndim>::g9;
    using RiemannSolver<ndim>::eos;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief Constructor for Roe Riemann solver class.
     *
     * Sets all gamma const variables.
     */
    RoeRiemannSolver(Eos<ndim> *const _eos) : RiemannSolver<ndim>(_eos){};

    /**
     * @brief Compute the numerical fluxes based on the Roe approximate Riemann solver.
     */
    virtual bool ComputeFluxes(const bool, const Vector<ndim, MyAtLeastDouble> &, WFluidVector<ndim, MyAtLeastDouble> &,
                               WFluidVector<ndim, MyAtLeastDouble> &, Vector<ndim, MyAtLeastDouble> &,
                               FluxVector<ndim, MyAtLeastDouble> &) const override;

    /**
     * @brief Compute Roe averages.
     *
     * Roe Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    void ComputeAveragedValues(const WFluidVector<ndim, MyAtLeastDouble> &, const WFluidVector<ndim, MyAtLeastDouble> &,
                               Vector<nvar, MyAtLeastDouble> &) const;
    /**
     * @brief Compute Roe eigenvalues.
     *
     * Roe Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    void ComputeEigenvalues(const Vector<nvar, MyAtLeastDouble> &, Vector<nvar, MyAtLeastDouble> &) const;
    /**
     * @brief Compute Roe right eigenvectors.
     *
     * Roe Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    void ComputeRightEigenvectors(const Vector<nvar, MyAtLeastDouble> &, SquareMatrix<nvar, MyAtLeastDouble> &) const;
    /**
     * @brief Compute Roe wave strengths.
     *
     * Roe Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    void ComputeWaveStrength(const WFluidVector<ndim, MyAtLeastDouble> &, const WFluidVector<ndim, MyAtLeastDouble> &,
                             const Vector<nvar, MyAtLeastDouble>, Vector<nvar, MyAtLeastDouble> &) const;
    /**
     * @brief Compute Roe left and right wave speed estimates.
     *
     * Roe Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    void ComputeWaveSpeedEstimates(const Vector<nvar, MyAtLeastDouble> &, MyAtLeastDouble &, MyAtLeastDouble &) const;
    /**
     * @brief Compute Roe central (star) velocities.
     *
     * Roe Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    void ComputeStarVelocities(const WFluidVector<ndim, MyAtLeastDouble> &, const WFluidVector<ndim, MyAtLeastDouble> &,
                               const Vector<nvar, MyAtLeastDouble> &, const Vector<nvar, MyAtLeastDouble> &,
                               const SquareMatrix<nvar, MyAtLeastDouble> &, MyAtLeastDouble &, MyAtLeastDouble &,
                               MyAtLeastDouble &, MyAtLeastDouble &) const;
    /**
     * @brief Compute central Roe flux.
     *
     * Roe Riemann solver, based on approach outlined by Toro (1999).
     * Variable names often follow Toro naming conventions so read chapter for further details.
     */
    FluxVector<ndim, MyAtLeastDouble> ComputeRoeFlux1d(const WFluidVector<ndim, MyAtLeastDouble> &,
                                                       const WFluidVector<ndim, MyAtLeastDouble> &,
                                                       const Vector<nvar, MyAtLeastDouble> &,
                                                       const Vector<nvar, MyAtLeastDouble> &,
                                                       const SquareMatrix<nvar, MyAtLeastDouble> &) const;
};


#ifdef GT_DUST
/**
 * @brief   Dust Riemann solver
 * @author  G. Tedeschi Prades
 * @date    08/01/2025
 */
template <int ndim> class DustRiemannSolver final : public RiemannSolver<ndim>
{
  protected:
    using RiemannSolver<ndim>::gamma;
    using RiemannSolver<ndim>::invgamma;
    using RiemannSolver<ndim>::g1;
    using RiemannSolver<ndim>::g2;
    using RiemannSolver<ndim>::g3;
    using RiemannSolver<ndim>::g4;
    using RiemannSolver<ndim>::g5;
    using RiemannSolver<ndim>::g6;
    using RiemannSolver<ndim>::g7;
    using RiemannSolver<ndim>::g8;
    using RiemannSolver<ndim>::g9;
    using RiemannSolver<ndim>::eos;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief Constructor for Dust Riemann solver class.
     *
     * Sets all gamma const variables.
     */
    DustRiemannSolver(Eos<ndim> *const _eos) : RiemannSolver<ndim>(_eos){};

    /**
     * @brief Compute fluxes for Dust Riemann solver.
     */
    virtual bool ComputeFluxes(const bool, const Vector<ndim> &, WFluidVector<ndim> &, WFluidVector<ndim> &,
                               Vector<ndim> &, FluxVector<ndim> &) const override;
};
#endif // GT_DUST

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
#endif // _OPENGADGET3_RIEMANN_SOLVER_HPP_
