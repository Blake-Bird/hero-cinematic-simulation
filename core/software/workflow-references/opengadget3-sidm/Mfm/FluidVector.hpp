/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares the FluidVector class.
 *
 * This file declares a parent FluidVector class with its derived classes for
 * W/U/QFluidVector.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET3_FLUID_VECTOR_HPP_
#define _OPENGADGET3_FLUID_VECTOR_HPP_

#include <cmath>
#include <cstdlib>
#include <math.h>
#include <ostream>

#include "Vector.hpp"

#include "../CodeBase/precision.hpp"
#include "../System/assert.hpp"

namespace OpenGadget3
{

/**
 * @brief   General fluid vector class.
 * @author  S. Heigl & D. A. Hubber, update by F. Groth
 * @date    13/02/2018
 */
template <int ndim, typename T = MyFloat> class FluidVector
{

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

  protected:
    T data[nvar]; /**< Data of Fluid vector in various forms (e.g. W, Q, U) */

  public:
    /**
     * @brief default constructor (initialize all entries to zero).
     */
    FluidVector()
    {
        for (int k = 0; k < nvar; k++)
        {
            data[k] = T(0.0);
        }
    }
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     */
    FluidVector(const Vector<ndim, T> vec, const T s1, const T s2
    )
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] = vec[k];
        }
        data[irho] = s1;
        data[ipress] = s2;
    }
    /**
     * @brief copy constructor.
     */
    template <typename U> FluidVector(FluidVector<ndim, U> other)
    {
        for (int k = 0; k < nvar; ++k)
        {
            data[k] = other[k];
        }
    }

    /**
     * @brief operator to set all entries to given value.
     */
    inline FluidVector<ndim, T> &operator=(const T value)
    {
        assert(std::isfinite(value));
        for (int k = 0; k < nvar; k++)
        {
            data[k] = value;
        }
        return *this;
    }

    /**
     * @brief operator to set all as for other FluidVector.
     */
    template <typename U> inline FluidVector<ndim, T> &operator=(const FluidVector<ndim, U> &other)
    {
        for (int k = 0; k < nvar; k++)
        {
            data[k] = other[k];
        }
        return *this;
    }

    /**
     * @brief operator to access specific entry.
     */
    inline T &operator[](const int index)
    {
        assert(index >= 0 && index < nvar);
        return data[index];
    }

    /**
     * @brief const operator to access specific entry.
     */
    inline const T &operator[](const int index) const
    {
        assert(index >= 0 && index < nvar);
        return data[index];
    }

    /**
     * @brief vector addition operator.
     */
    inline FluidVector<ndim, T> operator+(const FluidVector<ndim, T> &other) const
    {
        FluidVector<ndim, T> result;
        for (int k = 0; k < nvar; k++)
        {
            result[k] = this->data[k] + other[k];
        }
        return result;
    }

    /**
     * @brief vector substraction operator.
     */
    inline FluidVector<ndim, T> operator-(const FluidVector<ndim, T> &otherFluidVector) const
    {
        FluidVector<ndim, T> result;
        for (int k = 0; k < nvar; k++)
        {
            result[k] = this->data[k] - otherFluidVector[k];
        }
        return result;
    }

    /**
     * @brief scalar multiplication operator.
     */
    inline FluidVector<ndim, T> operator*(const T multFac) const
    {
        assert(std::isfinite(multFac));
        FluidVector<ndim, T> result;
        for (int k = 0; k < nvar; k++)
        {
            result[k] = this->data[k] * multFac;
        }
        return result;
    }

    /**
     * @brief vector addition assignment operator.
     */
    inline FluidVector<ndim, T> &operator+=(const FluidVector<ndim, T> &other)
    {
        for (int k = 0; k < nvar; k++)
        {
            data[k] += other[k];
        }
        return *this;
    }

    /**
     * @brief vector substraction assignment operator.
     */
    template <typename U> inline FluidVector<ndim, T> &operator-=(const FluidVector<ndim, U> &other)
    {
        for (int k = 0; k < nvar; k++)
        {
            data[k] -= other[k];
        }
        return *this;
    }

    /**
     * @brief scalar multiplication assignment operator.
     */
    inline FluidVector<ndim, T> &operator*=(const T multFac)
    {
        assert(std::isfinite(multFac));
        for (int k = 0; k < nvar; k++)
        {
            data[k] *= multFac;
        }
        return *this;
    }

    /**
     * @brief output operator (add all entries to output stream).
     */
    inline friend std::ostream &operator<<(std::ostream &out, const FluidVector<ndim, T> &FluidVector)
    {
        out << "FluidVector: (" << FluidVector[0];
        for (int k = 1; k < nvar; k++)
        {
            out << ", " << FluidVector[k];
        }
        out << ")";
        return out;
    }

    /**
     * @brief static function to generate random fluid vector.
     */
    static inline FluidVector<ndim, T> GenerateRandomFluidVector()
    {
        FluidVector<ndim, T> fv;
        for (int k = 0; k < ndim; k++)
        {
            fv[k] = T(2.0) * T(rand()) / T(RAND_MAX) - T(1.0);
        }
        fv[irho] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        fv[ipress] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        return fv;
    }

    /**
     * @brief limit pressure and density to values greater than zero.
     */
    inline void enforce_positivity()
    {
        if (data[irho] <= (MyFloat)1.0e-30 or !std::isfinite(data[irho]))
        {
            data[irho] = 1.0e-30;
        }
        if (data[ipress] <= (MyFloat)1.0e-30 or !std::isfinite(data[ipress]))
        {
            data[ipress] = 1.0e-30;
        }
    }
};

// forward declare the other fluid vector types.
template <int ndim, typename T> class QFluidVector;

template <int ndim, typename T> class UFluidVector;

/**
 * @brief   Fluid vector of primitive quantities, W = (velocity, density, pressure)
 * @author  S. Heigl & D. A. Hubber, update by F. Groth
 * @date    14/02/2018
 */
template <int ndim, typename T = MyFloat> class WFluidVector final : public FluidVector<ndim, T>
{
  protected:
    using FluidVector<ndim, T>::data;

  public:
    using FluidVector<ndim, T>::ivx;
    using FluidVector<ndim, T>::ivy;
    using FluidVector<ndim, T>::ivz;
    using FluidVector<ndim, T>::irho;
    using FluidVector<ndim, T>::ipress;
    using FluidVector<ndim, T>::nvar;
    /**
     * @brief default constructor (initialize all entries to zero).
     */
    WFluidVector() : FluidVector<ndim, T>(){};
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     *
     * Convert (Sph)P variables to primitive fluid vector (W).
     */
    WFluidVector(const Vector<ndim, T> v,
                 const T rho, const T p)
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] = v[k];
        }
        data[irho] = rho;
        data[ipress] = p;
    }
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     */
    WFluidVector(const T *v,
                 const T rho, const T p);
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     *
     * Convert from conserved fluid vector (Q) to primitive fluid vector (W).
     */
    WFluidVector(const QFluidVector<ndim, T> &, const T, const T, const T);
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     *
     * Convert from conserved fluid vector (U) to primitive fluid vector (W).
     */
    WFluidVector(const UFluidVector<ndim, T> &, const T);
    /**
     * @brief copy constructor.
     */
    WFluidVector(const FluidVector<ndim, T> &other)
    {
        for (int k = 0; k < nvar; k++)
        {
            data[k] = other[k];
        }
    }

    /**
     * @brief cross-precision converting constructor (needed when MyFloat != MyAtLeastDouble,
     * e.g. DOUBLEPRECISION=3; a no-op conversion otherwise).
     */
    template <typename U> WFluidVector(const FluidVector<ndim, U> &other) : FluidVector<ndim, T>(other)
    {
    }

    using FluidVector<ndim, T>::operator=;

    /**
     * @brief Return density.
     */
    inline T const GetDensity() const
    {
        return data[irho];
    }
    /**
     * @brief Return modifyable density.
     */
    inline T &GetDensity()
    {
        return data[irho];
    }
    /**
     * @brief Return pressure.
     */
    inline T const GetPressure() const
    {
        return data[ipress];
    }
    /**
     * @brief Return modifyable pressure.
     */
    inline T &GetPressure()
    {
        return data[ipress];
    }
    /**
     * @brief Return specific internal energy (in physical units!).
     *
     * A vacuum state carries zero pressure and zero density, and the callers multiply this back by
     * the density to build a flux, so report zero there rather than evaluating 0/0.
     */
    inline T GetSpecificInternalEnergy(const T gamma) const
    {
        if (!(data[irho] > T(0.0)))
        {
            return T(0.0);
        }
        return data[ipress] / (gamma - T(1.0)) / data[irho];
    }
    /**
     * @brief Return specific kinetic energy.
     */
    inline T GetSpecificKineticEnergy() const
    {
        if (!(data[irho] > T(0.0)))
        {
            return T(0.0);
        }
        T keTot = T(0.0);
        for (int k = 0; k < ndim; k++)
        {
            keTot += data[k] * data[k];
        }
        return T(0.5) * keTot;
    }
    /**
     * @brief Return specific total energy.
     */
    inline T GetSpecificTotalEnergy(const T gamma) const
    {
        T eTot = GetSpecificKineticEnergy() + GetSpecificInternalEnergy(gamma);
        return eTot;
    }
    /**
     * @brief Return velocity.
     */
    inline T const *GetVelocity() const
    {
        return &data[ivx];
    }
    /**
     * @brief Return modifyable velocity.
     */
    inline T *GetVelocity()
    {
        return &data[ivx];
    }
    /**
     * @brief Function to compute the time derivative of the primitive fluid vector.
     *
     * Used in the MUSCL scheme to predict primitive properties at the half timestep.
     */
    static WFluidVector<ndim, T> ComputePrimitiveTimeDerivative(
        const T cSoundSqd, /**< [in] Sound speed squared */
        const WFluidVector<ndim, T> &W, /**< [in] Primitive fluid vector */
        const Vector<ndim, T> *gradW, /**< [in] (Slope-limited) gradients */
        const Vector<ndim, T> Vel_smooth) /**< [in] smmothed velocity over neighbours */
    {
        WFluidVector<ndim, T> Wdot;
        for (int k = 0; k < nvar; ++k)
        {
            Wdot[k] = (T)0.0;
        }
        T divV = (T)0.0;
        for (int k = 0; k < ndim; ++k)
        {
            divV += gradW[ivx + k][k];
        }
        const T invrho = (T)1.0 / W[irho];

        for (int k = 0; k < ndim; ++k)
        {
            for (int kk = 0; kk < ndim; ++kk)
            {
                Wdot[ivx + kk] -= (W[ivx + k] - Vel_smooth[k]) * gradW[ivx + kk][k];
            }
            Wdot[ivx + k] -= gradW[ipress][k] * invrho;

            Wdot[irho] -= (W[ivx + k] - Vel_smooth[k]) * gradW[irho][k];

            Wdot[ipress] -= (W[ivx + k] - Vel_smooth[k]) * gradW[ipress][k];
        }
        Wdot[irho] -= W[irho] * divV;
        Wdot[ipress] -= W[irho] * cSoundSqd * divV;
        return Wdot;
    }

    static inline WFluidVector<ndim, T> GenerateRandomWFluidVector()
    {
        WFluidVector<ndim, T> W;
        for (int k = 0; k < ndim; k++)
        {
            W[k] = T(2.0) * T(rand()) / T(RAND_MAX) - T(1.0);
        }
        W[irho] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        W[ipress] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        return W;
    }
};

/**
 * @brief   Fluid vector of conserved quantities per volume, U = (momentum density, density, energy density)
 * @author  S. Heigl & D. A. Hubber, update by F. Groth
 * @date    14/02/2018
 */
template <int ndim, typename T = MyFloat> class UFluidVector final : public FluidVector<ndim, T>
{
  protected:
    using FluidVector<ndim, T>::data;

  public:
    using FluidVector<ndim, T>::ivx;
    using FluidVector<ndim, T>::ivy;
    using FluidVector<ndim, T>::ivz;
    using FluidVector<ndim, T>::irho;
    using FluidVector<ndim, T>::ipress;
    using FluidVector<ndim, T>::nvar;
    /**
     * @brief default constructor (initialize all entries to zero).
     */
    UFluidVector(const FluidVector<ndim, T> &other)
    {
        for (int k = 0; k < nvar; k++)
        {
            data[k] = other[k];
        }
    }
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     *
     * Convert from conserved fluid vector (Q) to conserved fluid vector (U).
     */
    UFluidVector(const QFluidVector<ndim, T> &Q, const T vol)
    {
        assert(std::isnormal(vol));
        const T invVol = T(1.0) / vol;
        for (int k = 0; k < nvar; k++)
        {
            data[k] = Q[k] * invVol;
        }
    }
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     *
     * Convert from primitive fluid vector (W) to conserved fluid vector (U).
     */
    UFluidVector(const WFluidVector<ndim, T> &, const T);
    /**
     * @brief copy constructor.
     */
    UFluidVector() : FluidVector<ndim, T>(){};

    /**
     * @brief cross-precision converting constructor (needed when MyFloat != MyAtLeastDouble,
     * e.g. DOUBLEPRECISION=3; a no-op conversion otherwise).
     */
    template <typename U> UFluidVector(const FluidVector<ndim, U> &other) : FluidVector<ndim, T>(other)
    {
    }

    using FluidVector<ndim, T>::operator=;

    /**
     * @brief Return density.
     */
    inline T const GetDensity() const
    {
        return data[irho];
    }
    /**
     * @brief Return modifyable density.
     */
    inline T &GetDensity()
    {
        return data[irho];
    }
    /**
     * @brief Return energy density.
     */
    inline T const GetEnergyDensity() const
    {
        return data[ipress];
    }
    /**
     * @brief Return modifyable energy density.
     */
    inline T &GetEnergyDensity()
    {
        return data[ipress];
    }
    /**
     * @brief Return specific internal energy (in physical units!).
     */
    T GetSpecificInternalEnergy() const;

    /**
     * @brief Return momentum density.
     */
    inline const T *GetMomentumDensity() const
    {
        return &data[ivx];
    }
    /**
     * @brief Return modifyable momentum density.
     */
    inline T *GetMomentumDensity()
    {
        return &data[ivx];
    }
    /**
     * @brief Return velocity.
     */
    inline Vector<ndim, T> GetVelocity() const
    {
        const T invRho = T(1.0) / data[irho];
        Vector<ndim, T> v;
        for (int k = 0; k < ndim; k++)
        {
            v[k] = data[k] * invRho;
        }
        return v;
    }

    /**
     * @brief static function to generate random fluid vector.
     */
    static inline UFluidVector<ndim, T> GenerateRandomUFluidVector()
    {
        UFluidVector<ndim, T> U;
        for (int k = 0; k < ndim; k++)
        {
            U[k] = T(2.0) * T(rand()) / T(RAND_MAX) - T(1.0);
        }
        U[irho] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        U[ipress] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        return U;
    }
};

/**
 * @brief   Fluid vector of conserved quantities, Q = (momentum, mass, total energy)
 * @author  S. Heigl & D. A. Hubber, update by F. Groth
 * @date    14/02/2018
 */
template <int ndim, typename T = MyFloat> class QFluidVector final : public FluidVector<ndim, T>
{
  protected:
    using FluidVector<ndim, T>::data;

  public:
    static const int ivx = 0;
    static const int ivy = 1;
    static const int ivz = 2;
    static const int irho = ndim;
    static const int ipress = ndim + 1;
    static const int nvar = ndim + 2;

    /**
     * @brief default constructor (initialize all entries to zero).
     */
    QFluidVector() : FluidVector<ndim, T>(){};
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     */
    QFluidVector(const Vector<ndim, T> p, const T m, const T E)
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] = p[k];
        }
        data[irho] = m;
        data[ipress] = E;
    }
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     *
     * Convert from conserved fluid vector (U) to conserved fluid vector (Q).
     */
    QFluidVector(const UFluidVector<ndim, T> &U, const T vol)
    {
        assert(std::isnormal(vol));
        for (int k = 0; k < nvar; k++)
        {
            data[k] = U[k] * vol;
        }
    }
    /**
     * @brief value-based constructor (initialize all entries using the given values).
     *
     * Convert from primitive fluid vector (W) to conserved fluid vector (Q).
     */
    QFluidVector(const WFluidVector<ndim, T> &, const T, const T, const T);
    /**
     * @brief copy constructor.
     */
    QFluidVector(const FluidVector<ndim, T> &other)
    {
        for (int k = 0; k < nvar; k++)
        {
            data[k] = other[k];
        }
    }

    /**
     * @brief cross-precision converting constructor (needed when MyFloat != MyAtLeastDouble,
     * e.g. DOUBLEPRECISION=3; a no-op conversion otherwise).
     */
    template <typename U> QFluidVector(const FluidVector<ndim, U> &other) : FluidVector<ndim, T>(other)
    {
    }

    using FluidVector<ndim, T>::operator=;

    /**
     * @brief Return mass.
     */
    inline T const GetMass() const
    {
        return data[irho];
    }
    /**
     * @brief Return modifyable mass.
     */
    inline T &GetMass()
    {
        return data[irho];
    }
    /**
     * @brief Return specific internal energy.
     */
    T GetSpecificInternalEnergy() const
    {
        assert(std::isnormal(data[irho]));
        const T invm = (T)1.0 / data[irho];
        // so far, this is the total energy
        T u = data[ipress] * invm;
        // kinetic energy
        T keTot = (T)0.0;
        for (int k = 0; k < ndim; ++k)
        {
            keTot += data[k] * data[k];
        }
        keTot *= T(0.5) * invm * invm;
        // substract kinetic contribution
        u -= keTot;
        return u;
    }

    /**
     * @brief Return energy.
     */
    inline T const GetEnergy() const
    {
        return data[ipress];
    }
    /**
     * @brief Return modifyable energy.
     */
    inline T &GetEnergy()
    {
        return data[ipress];
    }
    /**
     * @brief Return momentum.
     */
    inline const T *GetMomentum() const
    {
        return &data[ivx];
    }
    /**
     * @brief Return modifyable momentum.
     */
    inline T *GetMomentum()
    {
        return &data[ivx];
    }
    /**
     * @brief Return velocity.
     */
    inline Vector<ndim, T> GetVelocity() const
    {
        assert(data[irho] > T(0.0));
        const T invm = T(1.0) / data[irho];
        Vector<ndim, T> v;
        for (int k = 0; k < ndim; k++)
        {
            v[k] = data[k] * invm;
        }
        return v;
    }

    /**
     * @brief static function to generate random fluid vector.
     */
    static inline QFluidVector<ndim, T> GenerateRandomQFluidVector()
    {
        QFluidVector<ndim, T> Q;
        for (int k = 0; k < ndim; k++)
        {
            Q[k] = T(2.0) * T(rand()) / T(RAND_MAX) - T(1.0);
        }
        Q[irho] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        Q[ipress] = (T(rand()) + T(1.0)) / (T(RAND_MAX) + T(1.0));
        return Q;
    }
};

// We want to access the flux vector with the names of conserved quantities. Thus, we define it as QFluidVector, even
// though in theory it is not of that type!
template <int ndim, typename T = MyFloat> using FluxVector = QFluidVector<ndim, T>;

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
#endif // _OPENGADGET3_FLUID_VECTOR_HPP_
