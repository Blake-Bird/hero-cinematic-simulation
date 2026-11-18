/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Desclares EOS classes.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/precision.hpp"
#include "../CodeBase/units.hpp"
#include "../System/assert.hpp"
#include <cmath>

#ifndef _OPENGADGET3_EOS_HPP_
#define _OPENGADGET3_EOS_HPP_

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @class
 * @author D. A. Hubber, Updated by F. Groth
 * @date 12/03/2018
 *
 * @brief Base class for defining all equations-of-state classes in hydro computations.

 * Define functions to compute derived hydrodynamical properties such as Pressure, SooundSpeed, etc.
 */
template <int ndim, typename T = MyFloat> class Eos
{
  protected:
    const T gamma; /**< Adiabatic index / Ratio of specific heats */
    const T gammam1; /**< gamma - 1 */
    const T invgammam1; /**< 1 / (gamma - 1) */

  public:
    /**
     * @ brief constructor to set all internal variables based on adiabatic index.
     */
    Eos(const T _gamma) : gamma(_gamma), gammam1(_gamma - T(1.0)), invgammam1(T(1.0) / (_gamma - T(1.0)))
    {
        assert(_gamma > T(1.0));
    };
    /**
     * @ brief Calculate the Pressure.
     */
    virtual T Pressure(const T rho, const T u) const = 0;
    /**
     * @ brief Calculate the speed of sound.
     */
    virtual T SoundSpeed(const T u) const = 0;

    /**
     * @ brief Calculate the specific internal energy.
     */
    virtual T SpecificInternalEnergy(const T u) const = 0;
    /**
     * @ brief Calculate the temperature.
     */
    virtual T Temperature(const T u) const = 0;

    /**
     * @ brief Return the adiabatic index gamma.
     */
    inline T GetGamma() const
    {
        return gamma;
    }
    /**
     * @ brief Return gammam1 = gamma-1.
     */
    inline T GetGammaMinusOne() const
    {
        return gammam1;
    }
    /**
     * @ brief Return invgammam1 = 1 / (gamma - 1).
     */
    inline T GetInvGammaMinusOne() const
    {
        return invgammam1;
    }

    /**
     * @ brief Calculate the internal energy based on entropy and other input quantities.
     */
    inline T GetUFromEntropy(const T rho, const T entropy, const Units *units) const
    {
        return invgammam1 * entropy * pow(units->density_to_physical * rho, gammam1);
    }
    /**
     * @ brief Calculate the entropy based on internal energy and other input quantities.
     */
    inline T GetEntropyFromU(const T rho, const T u, const Units *units) const
    {
        return gammam1 * u / pow(units->density_to_physical * rho, gammam1);
    }

    /**
     * @ brief Calculate the time derivative of the (reduced) conserved fluid vector (kinetic energy already
     * substracted, only internal energy component left, in physical units) based on time derivative of entropy and
     * other input quantities.
     */
    inline T GetDtQFromDtEntropy(const T dtentropy, const T entropy, const T u, const T mass, const T divvel,
                                 const T hubble_a) const
    {
        return u * (dtentropy / entropy - gammam1 * divvel) * hubble_a * mass;
    }
    /**
     * @ brief Calculate the time derivative of internal energy based on time derivative of entropy and other input
     * quantities.
     */
    inline T GetDtUFromDtEntropy(const T dtentropy, const T entropy, const T u, const T divvel, const T hubble_a,
                                 const int comoving_integration_on) const
    {
        T DtU = u * (dtentropy / entropy - gammam1 * divvel);
        if (comoving_integration_on)
        {
            DtU -= u * 3 * gammam1 * hubble_a;
        }
        return DtU;
    }
    /**
     * @ brief Calculate the time derivative of entropy based on time derivative of internal energy and other input
     * quantities.
     */
    inline T GetDtEntropyFromDtU(const T dtu, const T entropy, const T u, const T divvel, const T hubble_a,
                                 const int comoving_integration_on) const
    {
        T DtEntropy = entropy * dtu / u + gammam1 * divvel;
        if (comoving_integration_on)
        {
            DtEntropy += entropy * 3 * gammam1 * hubble_a;
        }
        return DtEntropy;
    }
};

/**
 * @class
 * @author D. A. Hubber, Updated by F. Groth
 * @date 12/03/2018
 *
 * @brief Adiabatic equation-of-state class.
 *
 * Define functions to compute derived hydrodynamical properties such as Pressure, SooundSpeed, etc for an
 * adiabatic equation of state with adiabaticc index gamma.
 */
template <int ndim, typename T = MyFloat> class AdiabaticEos final : public Eos<ndim, T>
{
  protected:
    using Eos<ndim, T>::gamma;
    using Eos<ndim, T>::gammam1;

  public:
    /**
     * @ brief constructor to set all internal variables based on adiabatic index.
     */
    AdiabaticEos(const T _gamma) : Eos<ndim, T>(_gamma){};

    /**
     * @ brief Calculate the adiabatic Pressure.
     */
    virtual T Pressure(const T rho, const T u) const override
    {
        return gammam1 * rho * u;
    }
    /**
     * @ brief Calculate the adiabatic speed of sound.
     */
    virtual T SoundSpeed(const T u) const override
    {
        return sqrt(gamma * gammam1 * u);
    }

    /**
     * @ brief Calculate the adiabatic specific internal energy.
     */
    virtual T SpecificInternalEnergy(const T u) const override
    {
        return u;
    }
    /**
     * @ brief Calculate the adiabatic temperature.
     */
    virtual T Temperature(const T u) const override
    {
        return gammam1 * u;
    }
};

/**
 * @class
 * @author D. A. Hubber, Updated by F. Groth
 * @date 06/06/2018
 *
 * @brief Isothermal equation-of-state class..
 *
 * Define functions to compute derived hydrodynamical properties such as Pressure, SooundSpeed, etc for an
 * isothermal equation of state.
 */
template <int ndim, typename T = MyFloat> class IsothermalEos final : public Eos<ndim, T>
{
  protected:
    using Eos<ndim, T>::gamma;
    using Eos<ndim, T>::gammam1;
    using Eos<ndim, T>::invgammam1;

    const T temp0; /**< Isothermal temperature */
    const T muBar; /**< Mean gas particle mass */
    const T invMuBar; /**< 1 / muBar */
    const T cSound; /**< Isothermal sound speed */

  public:
    /**
     * @ brief constructor to set all internal variables (adiabatic index and isothermal variables) based on adiabatic
     * index, temperature, and mean particle mass.
     */
    IsothermalEos(const T _gamma, const T _temp0, const T _muBar)
        : Eos<ndim, T>(_gamma), temp0(_temp0), muBar(_muBar), invMuBar(T(1.0) / _muBar), cSound(sqrt(_temp0 / _muBar))
    {
        assert(std::isnormal(_temp0));
        assert(std::isnormal(_muBar));
    }

    /**
     * @ brief Calculate the isothermal Pressure.
     */
    virtual T Pressure(const T rho, const T u) const override
    {
        return rho * temp0 * invMuBar;
    }
    /**
     * @ brief Calculate the isothermal speed of sound.
     */
    virtual T SoundSpeed(const T u) const override
    {
        return cSound;
    }

    /**
     * @ brief Calculate the isothermal specific internal energy.
     */
    virtual T SpecificInternalEnergy(const T u) const override
    {
        return temp0 * invgammam1 * invMuBar;
    }
    /**
     * @ brief Return the isothermal temperature.
     */
    virtual T Temperature(const T u) const override
    {
        return temp0;
    }
};

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
#endif // _OPENGADGET3_EOS_HPP_
