/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file declares a Vector class.
 *
 * This file declares a parent SlopeLimiter class and derived classes for the
 * different limiters implemented.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET3_VECTOR_HPP_
#define _OPENGADGET3_VECTOR_HPP_

#include <cmath>
#include <cstdlib>
#include <ostream>

#include "../CodeBase/precision.hpp"
#include "../System/assert.hpp"

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief   Basic 1d/2d/3d vector class for simplifying storage and calculations with vectors.
 * @author  D. A. Hubber, updated by F. Groth
 * @date    27/10/2017
 */
template <int ndim, typename T = MyFloat> class Vector
{
  private:
    T data[ndim]; /**< Basic array containing vector components */

  public:
    /**
     * @brief default constructor (initialize all entries to zero).
     */
    Vector()
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] = T(0.0);
        }
    }

    /**
     * @brief constructor (initialize all entries to given value).
     */
    Vector(const T value)
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] = value;
        }
    }

    /**
     * @brief constructor for 2d vector (initialize all entries to given values).
     */
    Vector(const T v0, const T v1)
    {
        static_assert(ndim == 2, "2-argument Vector constructor requires ndim=2");
        data[0] = v0;
        data[1] = v1;
    }

    /**
     * @brief constructor for 3d vector (initialize all entries to given values).
     */
    Vector(const T v0, const T v1, const T v2)
    {
        static_assert(ndim == 3, "3-argument Vector constructor requires ndim=3");
        data[0] = v0;
        data[1] = v1;
        data[2] = v2;
    }
    /**
     * @brief copy constructor.
     */
    template <typename U> Vector(const Vector<ndim, U> &v)
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] = v[k];
        }
    }

    /**
     * @brief assignment operator (set all entries to given value).
     */
    inline Vector<ndim, T> &operator=(const T value)
    {
        assert(std::isfinite(value));
        for (int k = 0; k < ndim; k++)
        {
            data[k] = value;
        }
        return *this;
    }

    /**
     * @brief Return element at given index.
     */
    inline T &operator[](const int index)
    {
        assert(index >= 0 && index < ndim);
        return data[index];
    }

    /**
     * @brief Return element at given index.
     */
    inline const T &operator[](const int index) const
    {
        assert(index >= 0 && index < ndim);
        return data[index];
    }

    /**
     * @brief Vector (element-wise) addition operator.
     */
    template <typename U> inline Vector<ndim, T> operator+(const Vector<ndim, U> &otherVector) const
    {
        Vector<ndim, T> result;
        for (int k = 0; k < ndim; k++)
        {
            result[k] = this->data[k] + otherVector[k];
        }
        return result;
    }

    /**
     * @brief Vector (element-wise) substraction operator.
     */
    inline Vector<ndim, T> operator-(const Vector<ndim, T> &otherVector) const
    {
        Vector<ndim, T> result;
        for (int k = 0; k < ndim; k++)
        {
            result[k] = this->data[k] - otherVector[k];
        }
        return result;
    }

    /**
     * @brief Vector (element-wise) addition assignment operator.
     */
    inline Vector<ndim, T> &operator+=(const Vector<ndim, T> &otherVector)
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] += otherVector[k];
        }
        return *this;
    }

    /**
     * @brief Vector (element-wise) substraction assignment operator.
     */
    template <typename U> inline Vector<ndim, T> &operator-=(const Vector<ndim, U> &otherVector)
    {
        for (int k = 0; k < ndim; k++)
        {
            data[k] -= otherVector[k];
        }
        return *this;
    }

    /**
     * @brief Vector scalar multiplication operator.
     */
    inline Vector<ndim, T> operator*(const T multFac) const
    {
        assert(std::isfinite(multFac));
        Vector<ndim, T> result;
        for (int k = 0; k < ndim; k++)
        {
            result[k] = this->data[k] * multFac;
        }
        return result;
    }

    /**
     * @brief Vector scalar multiplication operator (inverse order).
     */
    friend inline Vector<ndim, T> operator*(const T multFac, const Vector<ndim, T> &vec)
    {
        assert(std::isfinite(multFac));
        Vector<ndim, T> result;
        for (int k = 0; k < ndim; k++)
        {
            result[k] = vec[k] * multFac;
        }
        return result;
    }

    /**
     * @brief Vector scalar multiplication assignment operator.
     */
    Vector<ndim, T> &operator*=(const T multFac)
    {
        assert(std::isfinite(multFac));
        for (int k = 0; k < ndim; k++)
        {
            data[k] *= multFac;
        }
        return *this;
    }

    /**
     * @brief Vector scalar division assignment operator.
     */
    Vector<ndim, T> &operator/=(const T divFac)
    {
        assert(std::isnormal(divFac));
        const T invFac = T(1.0) / divFac;
        for (int k = 0; k < ndim; k++)
        {
            data[k] *= invFac;
        }
        return *this;
    }

    /**
     * @brief Vector output operator. Add all entries to output stream.
     */
    inline friend std::ostream &operator<<(std::ostream &out, const Vector<ndim, T> &vector)
    {
        for (int k = 0; k < ndim; k++)
        {
            out << vector[k] << " ";
        }
        // out << vector[ndim-1];
        return out;
    }

    /**
     * @brief Return vector magnitude (2-norm) squared.
     */
    inline T GetMagnitudeSquared() const
    {
        T magSqd = T(0.0);
        for (int k = 0; k < ndim; k++)
        {
            magSqd += data[k] * data[k];
        }
        return magSqd;
    }

    /**
     * @brief Return vector magnitude (2-norm).
     */
    inline T GetMagnitude() const
    {
        return sqrt(GetMagnitudeSquared());
    }

    /**
     * @brief Return unit vector in the same direction.
     */
    inline Vector<ndim, T> GetUnitVector() const
    {
        Vector<ndim, T> unitVector(*this);
        unitVector.Normalise();
        return unitVector;
    }

    /**
     * @brief Normalize the vector to become a unit vector.
     */
    inline void Normalise()
    {
        const T magSqd = GetMagnitudeSquared();
        if (std::isnormal(magSqd))
        {
            const T invMag = T(1.0) / sqrt(magSqd);
            for (int k = 0; k < ndim; k++)
            {
                data[k] *= invMag;
            }
        }
    }

    /**
     * @brief Return scalar (dot) product of two 1d vectors.
     */
    inline static T DotProduct(const Vector<1> &v1, const Vector<1> &v2)
    {
        return v1[0] * v2[0];
    }

    /**
     * @brief Return scalar (dot) product of two 2d vectors.
     */
    inline static T DotProduct(const Vector<2> &v1, const Vector<2> &v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1];
    }

    /**
     * @brief Return scalar (dot) product of two 3d vectors.
     */
    inline static T DotProduct(const Vector<3> &v1, const Vector<3> &v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    /**
     * @brief Return magnitude (2-norm) squared of difference between two 1d vectors.
     */
    inline static T DistanceSquared(const Vector<1> &v1, const Vector<1> &v2)
    {
        return (v2[0] - v1[0]) * (v2[0] - v1[0]);
    }

    /**
     * @brief Return magnitude (2-norm) squared of difference between two 2d vectors.
     */
    inline static T DistanceSquared(const Vector<2> &v1, const Vector<2> &v2)
    {
        return (v2[0] - v1[0]) * (v2[0] - v1[0]) + (v2[1] - v1[1]) * (v2[1] - v1[1]);
    }

    /**
     * @brief Return magnitude (2-norm) squared of difference between two 3d vectors.
     */
    inline static T DistanceSquared(const Vector<3> &v1, const Vector<3> &v2)
    {
        return (v2[0] - v1[0]) * (v2[0] - v1[0]) + (v2[1] - v1[1]) * (v2[1] - v1[1]) +
               (v2[2] - v1[2]) * (v2[2] - v1[2]);
    }

    /**
     * @brief Return magnitude (2-norm) of difference between two 1d vectors.
     */
    inline static T Distance(const Vector<ndim, T> &v1, const Vector<ndim, T> &v2)
    {
        return sqrt(DistanceSquared(v1, v2));
    }

    /**
     * @brief Return random vector (each component is a random value).
     */
    static inline Vector<ndim, T> GenerateRandomVector()
    {
        Vector<ndim, T> v;
        for (int k = 0; k < ndim; k++)
        {
            v[k] = T(2.0) * T(rand()) / T(RAND_MAX) - T(1.0);
        }
        return v;
    }

    /**
     * @brief Return random unit vector (normalized vector where each component is a random value).
     */
    static inline Vector<ndim, T> GenerateRandomUnitVector()
    {
        Vector<ndim, T> v = GenerateRandomVector();
        v.Normalise();
        return v;
    }
#ifdef ACC
    T *get_raw()
    {
        return data;
    }
#endif
};

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
#endif // _OPENGADGET3_VECTOR_HPP_
