/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the FluxSolver class.
 *
 * This file declares member functions of the FluxSolver class to compute
 * Godunov fluxes using a Riemann solver.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "FluxSolver.hpp"
#include "../CodeBase/allvars.hpp"
#include "../System/assert.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include <map>
#include <string>

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief Constructor for FluxSolver class.
 *
 * Initialize Riemann solvers based on eos.
 */
template <int ndim> FluxSolver<ndim>::FluxSolver(Eos<ndim> *const _eos) : eos(_eos)
{
    exactRiemann = new ExactRiemannSolver<ndim>(_eos, 8, (MyFloat)1.0e-4);
    roeRiemann = new RoeRiemannSolver<ndim>(_eos);
    hlleRiemann = new HlleRiemannSolver<ndim>(_eos);
    hllcRiemann = new HllcRiemannSolver<ndim, HLLC_RIEMANN_VERSION>(_eos);
    hllRiemann = new HllRiemannSolver<ndim>(_eos);
#ifdef GT_DUST
    dustRiemann = new DustRiemannSolver<ndim>(_eos);
#endif // GT_DUST
}

/**
 * @brief Destructor for FluxSolver class.
 */
template <int ndim> FluxSolver<ndim>::~FluxSolver()
{
#ifdef GT_DUST
    delete dustRiemann;
#endif // GT_DUST
    delete hllRiemann;
    delete hllcRiemann;
    delete hlleRiemann;
    delete roeRiemann;
    delete exactRiemann;
}

/**
 * @brief Compute the numerical Godunov fluxes for a given Riemann problem with left and right states.
 *
 * The energy component is boosted back to the lab frame below, which makes it ~A P |v| per face
 * while the internal-energy rate that convert_total_to_internal_energy_flux later recovers from it
 * is only ~A P |dv|, dv being the velocity variation across the cell. The flux is therefore
 * returned in MyAtLeastDouble, so that cancellation of ~|v|/|dv| does not eat MyFloat's mantissa.
 */
template <int ndim>
FluxVector<ndim, MyAtLeastDouble> FluxSolver<ndim>::ComputeFluxes(
    const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
    const Vector<ndim, MyAtLeastDouble> &rUnit, /**< [in] Unit vector on face */
    Vector<ndim, MyAtLeastDouble> &vFace, /**< [inout] Face velocity vector */
    WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [inout] LHS primitive state */
    WFluidVector<ndim, MyAtLeastDouble> &Wright) /**< [inout] RHS primitive state */
{
    FluxVector<ndim, MyAtLeastDouble> flux;

    // Compute rotation and inverse rotation matrices
    const RotationMatrix<ndim, MyAtLeastDouble> rotMatrix =
        RotationMatrix<ndim, MyAtLeastDouble>::GetRotationMatrixFromVector(rUnit);
    const RotationMatrix<ndim, MyAtLeastDouble> invRotMatrix = rotMatrix.GetInverseMatrix();

    // Boost velocities to frame of moving frame
    for (int k = 0; k < ndim; k++)
    {
        Wleft[k] -= vFace[k];
        Wright[k] -= vFace[k];
    }

    // Rotate velocity component so Riemann problem is now a 1d problem set along the x-axis
    Wleft = invRotMatrix * Wleft;
    Wright = invRotMatrix * Wright;

    assert(std::isfinite(Wleft[0]));
    assert(std::isfinite(Wright[0]));

#if   defined(RIEMANN_SOLVER_HLL)
    // First attempt to use the HLL solver; if an error is generated, then use the exact solver
    if (!hllRiemann->ComputeFluxes(zeroMassFlux, rUnit, Wleft, Wright, vFace, flux))
#elif defined(RIEMANN_SOLVER_HLLC)
    if (!hllcRiemann->ComputeFluxes(zeroMassFlux, rUnit, Wleft, Wright, vFace, flux))
#elif defined(RIEMANN_SOLVER_ROE)
    if (!roeRiemann->ComputeFluxes(zeroMassFlux, rUnit, Wleft, Wright, vFace, flux))
#elif defined(RIEMANN_SOLVER_HLLE)
    if (!hlleRiemann->ComputeFluxes(zeroMassFlux, rUnit, Wleft, Wright, vFace, flux))
#else // not any specif RIEMANN_SOLVER defined
      // use exact Riemann solver by default
#endif // RIEMANN_SOLVER_...
    {
        // always fall back to exact Riemann solver
        assert(std::isfinite(Wleft[0]));
        assert(std::isfinite(Wright[0]));
        exactRiemann->ComputeFluxes(zeroMassFlux, rUnit, Wleft, Wright, vFace, flux);
    }

    // Rotate velocity (and magnetic field) component of flux back to orientation of original frame
    flux = rotMatrix * flux;

    if (zeroMassFlux)
    {
        flux.GetMass() = 0; // explicetely set to zero for MFM
    }

    // Add corrections to flux for boosting back to original lab frame
    // compare Pakmor+2011
    Vector<ndim, MyAtLeastDouble> fv;
    for (int k = 0; k < ndim; k++)
    {
        fv[k] = flux[k];
    }
    flux.GetEnergy() += Vector<ndim, MyAtLeastDouble>::DotProduct(vFace, fv);
    // for (int k=0; k<ndim; k++) flux[k] = flux.GetEnergy()*rUnit[k];
    if (!zeroMassFlux)
    {
        flux.GetEnergy() +=
            (MyAtLeastDouble)0.5 * vFace.GetMagnitudeSquared() * flux.GetMass(); // this term is zero for MFM
        for (int k = 0; k < ndim; k++)
        {
            flux[k] += vFace[k] * flux.GetMass(); // zero for MFM
        }
    }
    return flux;
}

#ifdef GT_DUST
/**
 * @brief Compute the numerical Godunov fluxes for a given Riemann problem with left and right states.
 */
template <int ndim>
void FluxSolver<ndim>::ComputeFluxesMultifluid(
    const bool zeroMassFlux, /**< [in] Compute fluxes in zero-mass flux frame */
    const Vector<ndim, MyAtLeastDouble> &rUnit, /**< [in] Unit vector on face */
    Vector<ndim, MyAtLeastDouble> &vFace, /**< [inout] Face velocity vector */
    WFluidVector<ndim, MyAtLeastDouble> &Wleft, /**< [inout] LHS primitive state */
    WFluidVector<ndim, MyAtLeastDouble> &Wright, /**< [inout] RHS primitive state */
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> &Wleft_dust, /**< [inout] LHS primitive state for dust */
    std::array<WFluidVector<mfm_numdims>, GT_NUM_DUST> &Wright_dust, /**< [inout] RHS primitive state for dust */
    FluxVector<ndim, MyAtLeastDouble> &flux_gas, /**< [inout] gas flux vector */
    std::array<FluxVector<ndim, MyAtLeastDouble>, GT_NUM_DUST> &flux_dust) /**< [inout] dust flux vector */
{

    // Compute rotation and inverse rotation matrices
    const RotationMatrix<ndim, MyAtLeastDouble> rotMatrix =
        RotationMatrix<ndim, MyAtLeastDouble>::GetRotationMatrixFromVector(rUnit);
    const RotationMatrix<ndim, MyAtLeastDouble> invRotMatrix = rotMatrix.GetInverseMatrix();

    // Boost velocities to frame of moving frame
    for (int k = 0; k < ndim; k++)
    {
        Wleft[k] -= vFace[k];
        Wright[k] -= vFace[k];
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            Wleft_dust[idust][k] -= vFace[k];
            Wright_dust[idust][k] -= vFace[k];
        }
    }
    // Rotate velocity component so Riemann problem is now a 1d problem set along the x-axis
    Wleft = invRotMatrix * Wleft;
    Wright = invRotMatrix * Wright;
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        Wleft_dust[idust] = invRotMatrix * Wleft_dust[idust];
        Wright_dust[idust] = invRotMatrix * Wright_dust[idust];
    }

    // Define and set variables for the Newton iteration
    int max_iter = 20;
    int i = 0;
    MyAtLeastDouble tol = 1e-6;
    MyAtLeastDouble v_RFT = 0.0;
    MyAtLeastDouble v_RFT_old = v_RFT;
    MyAtLeastDouble dv = 1e-12;
    MyAtLeastDouble dFdv = 0.0;
    FluxVector<mfm_numdims> flux_gas_old;
    std::array<FluxVector<ndim, MyAtLeastDouble>, GT_NUM_DUST> flux_dust_old;
    MyAtLeastDouble TotMassFlux_old, TotMassFlux;
    MyAtLeastDouble vgL_init = Wleft.GetVelocity()[0];
    MyAtLeastDouble vgR_init = Wright.GetVelocity()[0];
    MyAtLeastDouble vdL_init[GT_NUM_DUST];
    MyAtLeastDouble vdR_init[GT_NUM_DUST];
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        vdL_init[idust] = Wleft_dust[idust].GetVelocity()[0];
        vdR_init[idust] = Wright_dust[idust].GetVelocity()[0];
    }

    // Helper lambda functions
    auto set_boosted_velocities = [&](MyAtLeastDouble vBoost) {
        Wleft.GetVelocity()[0] = vgL_init - vBoost;
        Wright.GetVelocity()[0] = vgR_init - vBoost;
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            Wleft_dust[idust].GetVelocity()[0] = vdL_init[idust] - vBoost;
            Wright_dust[idust].GetVelocity()[0] = vdR_init[idust] - vBoost;
        }
    };

    auto compute_all_fluxes = [&]() {
        exactRiemann->ComputeFluxes(zeroMassFlux, rUnit, Wleft, Wright, vFace, flux_gas);
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            dustRiemann->ComputeFluxes(zeroMassFlux, rUnit, Wleft_dust[idust], Wright_dust[idust], vFace,
                                       flux_dust[idust]);
        }
    };

    auto compute_total_mass_flux = [&]() -> MyAtLeastDouble {
        MyAtLeastDouble tot = flux_gas.GetMass();
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            tot += flux_dust[idust].GetMass();
        }
        return tot;
    };

    auto save_old_variables = [&]() {
        flux_gas_old = flux_gas;
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            flux_dust_old[idust] = flux_dust[idust];
        }
        TotMassFlux_old = TotMassFlux;
        v_RFT_old = v_RFT;
    };

    set_boosted_velocities(v_RFT);
    compute_all_fluxes();
    TotMassFlux = compute_total_mass_flux();
    save_old_variables();

    while (abs(TotMassFlux) > tol && i <= max_iter)
    {
        set_boosted_velocities(v_RFT + dv);
        compute_all_fluxes();
        TotMassFlux = compute_total_mass_flux();

        // Newton update
        dFdv = (TotMassFlux - TotMassFlux_old) / dv;
        v_RFT = v_RFT_old - TotMassFlux_old / dFdv;

        set_boosted_velocities(v_RFT);
        compute_all_fluxes();
        TotMassFlux = compute_total_mass_flux();
        save_old_variables();
        i++;
    }

    for (int k = 0; k < ndim; k++)
    {
        vFace[k] += v_RFT * rUnit[k];
    }

    // Rotate velocity (and magnetic field) component of flux back to orientation of original frame
    flux_gas = rotMatrix * flux_gas;
    for (int idust = 0; idust < GT_NUM_DUST; idust++)
    {
        flux_dust[idust] = rotMatrix * flux_dust[idust];
    }

    // Add corrections to flux for boosting back to original lab frame
    // compare Pakmor+2011
    Vector<ndim, MyAtLeastDouble> fv;
    for (int k = 0; k < ndim; k++)
    {
        fv[k] = flux_gas[k];
    }
    flux_gas.GetEnergy() += Vector<ndim, MyAtLeastDouble>::DotProduct(vFace, fv);

    flux_gas.GetEnergy() +=
        (MyAtLeastDouble)0.5 * vFace.GetMagnitudeSquared() * flux_gas.GetMass(); // this term is zero for MFM
    for (int k = 0; k < ndim; k++)
    {
        flux_gas[k] += vFace[k] * flux_gas.GetMass(); // zero for MFM
        for (int idust = 0; idust < GT_NUM_DUST; idust++)
        {
            flux_dust[idust][k] += vFace[k] * flux_dust[idust].GetMass(); // zero for MFM
        }
    }
}

#endif // GT_DUST

template class FluxSolver<mfm_numdims>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
