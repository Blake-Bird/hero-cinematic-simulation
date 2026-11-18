/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares the FluxSolver class.
 *
 * This file declares the FluxSolver class to compute Godunov fluxes using a
 * Riemann solver.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET3_FLUX_SOLVER_HPP_
#define _OPENGADGET3_FLUX_SOLVER_HPP_

#include "../CodeBase/precision.hpp"
#include "FluidVector.hpp"
#include "RiemannSolver.hpp"
#include <cstdlib>
#include <string>

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief   Class to compute the Godunov flux using the chosen Riemann solver algorithm.
 * @author  S. Heigl & D. A. Hubber
 * @date    13/02/2018
 */
template <int ndim> class FluxSolver
{
  private:
    Eos<ndim> *const eos;
    ExactRiemannSolver<ndim> *exactRiemann;
    RoeRiemannSolver<ndim> *roeRiemann;
    HlleRiemannSolver<ndim> *hlleRiemann;
    HllcRiemannSolver<ndim, HLLC_RIEMANN_VERSION> *hllcRiemann;
    HllRiemannSolver<ndim> *hllRiemann;
#ifdef GT_DUST
    DustRiemannSolver<ndim> *dustRiemann;
#endif // GT_DUST

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;

    /**
     * @brief Constructor for FluxSolver class.
     *
     * Initialize Riemann solvers based on eos.
     */
    FluxSolver(Eos<ndim> *const);
    /**
     * @brief Destructor for FluxSolver class.
     */
    ~FluxSolver();
    // follow the rule of three: as we user define a destructor, we should also define the copy constructor and copy
    // assignment operator:
    /**
     * @brief Copy constructor for FluxSolver class.
     */
    FluxSolver(FluxSolver<ndim> const &other)
        : eos{other.eos}, exactRiemann{other.exactRiemann}, roeRiemann{other.roeRiemann},
          hlleRiemann{other.hlleRiemann}, hllcRiemann{other.hllcRiemann}, hllRiemann{other.hllRiemann}
#ifdef GT_DUST
          ,
          dustRiemann{other.dustRiemann}
#endif // GT_DUST
    {
    }
    /**
     * @brief Copy assignment operator for FluxSolver class (deleted).
     */
    FluxSolver<ndim> operator=(const FluxSolver<ndim> &other) =
        delete; // as eos is const, we delete the copy assignment operator

    /**
     * @brief Compute the numerical Godunov fluxes for a given Riemann problem with left and right states.
     */
    FluxVector<ndim, MyAtLeastDouble> ComputeFluxes(const bool, const Vector<ndim, MyAtLeastDouble> &,
                                                    Vector<ndim, MyAtLeastDouble> &,
                                                    WFluidVector<ndim, MyAtLeastDouble> &,
                                                    WFluidVector<ndim, MyAtLeastDouble> &);

    /**
     * @brief Compute the numerical Godunov fluxes for multiple fluids at once, and find the zero-mass-flux reference
     * frame.
     */
#ifdef GT_DUST
    void ComputeFluxesMultifluid(const bool, const Vector<ndim, MyAtLeastDouble> &, Vector<ndim, MyAtLeastDouble> &,
                                 WFluidVector<ndim, MyAtLeastDouble> &, WFluidVector<ndim, MyAtLeastDouble> &,
                                 std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> &,
                                 std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> &,
                                 FluxVector<ndim, MyAtLeastDouble> &,
                                 std::array<FluxVector<ndim, MyAtLeastDouble>, GT_NUM_DUST> &);
#endif // GT_DUST
};
} // namespace OpenGadget3
//-------------------------------------------------------------------------------------------------
#endif // _OPENGADGET3_FLUX_SOLVER_HPP_
