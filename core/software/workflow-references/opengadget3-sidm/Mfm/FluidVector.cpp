/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares member functions of the FluidVector class.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */
#include "FluidVector.hpp"
#include "../System/assert.hpp"

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief value-based constructor (initialize all entries using the given values).
 *
 * Convert from primitive fluid vector (W) to conserved fluid vector (Q).
 */
template <int ndim, typename T>
QFluidVector<ndim, T>::QFluidVector(const WFluidVector<ndim, T> &W, /**< [in] Primtive fluid vector to convert to Q */
                                    const T vol, /**< [in] Volume of particle */
                                    const T gamma, /**< [in] Ratio of specific heats */
                                    const T a) /**< [in] Scale factor */
{
    assert(std::isnormal(vol));
    assert(std::isnormal(gamma));
    assert(std::isnormal(W[irho]));
    assert(std::isnormal(W[ipress]));
    const T mass = W[irho] * vol;
    const T gammam1 = gamma - (T)1.0;
    T vSqd = (T)0.0;
    for (int k = 0; k < ndim; k++)
    {
        vSqd += (W[k] * W[k]);
    }
    for (int k = 0; k < ndim; k++)
    {
        data[k] = W[k] * mass;
    }
    data[irho] = mass;
    data[ipress] = W[ipress] * vol / gammam1 / pow(a, 3 * gammam1) + (T)0.5 * mass * vSqd;
}

/**
 * @brief value-based constructor (initialize all entries using the given values).
 *
 * Convert from primitive fluid vector (W) to conserved fluid vector (U).
 */
template <int ndim, typename T>
UFluidVector<ndim, T>::UFluidVector(const WFluidVector<ndim, T> &W, /**< [in] Primtive fluid vector to convert to U */
                                    const T gamma) /**< [in] Ratio of specific heats */
{
    assert(std::isnormal(gamma));
    const T gammam1 = gamma - (T)1.0;
    T vSqd = (T)0.0;
    for (int k = 0; k < ndim; k++)
    {
        vSqd += (W[k] * W[k]);
    }
    for (int k = 0; k < ndim; k++)
    {
        data[k] = W[irho] * W[k];
    }
    data[irho] = W[irho];
    data[ipress] = W[ipress] / gammam1 + (T)0.5 * W[irho] * vSqd;
}

/**
 * @brief value-based constructor (initialize all entries using the given values).
 *
 * Convert from conserved fluid vector (Q) to primitive fluid vector (W).
 */
template <int ndim, typename T>
WFluidVector<ndim, T>::WFluidVector(const QFluidVector<ndim, T> &Q, /**< [in] Conserved fluid vector to be converted */
                                    const T vol, /**< [in] Volume of particle */
                                    const T gamma, /**< [in] Ratio of specific heats */
                                    const T a) /**< [in] Scale factor */
{
    assert(std::isnormal(gamma));
    assert(std::isnormal(vol));
    assert(std::isnormal(Q[irho]));
    const T gammam1 = gamma - (T)1.0;
    const T invMass = (T)1.0 / Q[irho];
    const T invVol = (T)1.0 / vol;
    T pSqd = (T)0.0;
    for (int k = 0; k < ndim; k++)
    {
        pSqd += (Q[k] * Q[k]);
    }
    for (int k = 0; k < ndim; k++)
    {
        data[k] = Q[k] * invMass;
    }
    data[irho] = Q[irho] * invVol;
    data[ipress] = (Q[ipress] - (T)0.5 * pSqd * invMass);
    // data[ipress] = (Q[ipress] - (T) 0.5*pSqd*invMass)*invMass*pow(data[irho],gamma);
    data[ipress] = data[ipress] * gammam1 * invVol * pow(a, 3 * gammam1);
}

/**
 * @brief value-based constructor (initialize all entries using the given values).
 *
 * Convert from conserved fluid vector (U) to primitive fluid vector (W).
 */
template <int ndim, typename T>
WFluidVector<ndim, T>::WFluidVector(const UFluidVector<ndim, T> &U, /**< [in] Conserved fluid vector to be converted */
                                    const T gamma) /**< [in] Ratio of specific heats */
{
    assert(std::isnormal(U[irho]));
    const T gammam1 = gamma - (T)1.0;
    const T invRho = (T)1.0 / U[irho];
    T pSqd = (T)0.0;
    for (int k = 0; k < ndim; k++)
    {
        pSqd += (U[k] * U[k]);
    }
    for (int k = 0; k < ndim; k++)
    {
        data[k] = U[k] * invRho;
    }
    data[irho] = U[irho];
    data[ipress] = (U[ipress] - (T)0.5 * pSqd * invRho) * gammam1;
}

/**
 * @brief value-based constructor (initialize all entries using the given values).
 *
 * Convert (Sph)P variables to primitive fluid vector (W).
 */
template <int ndim, typename T>
WFluidVector<ndim, T>::WFluidVector(const T *Vel, /**< [in] velocity */
                                    const T Density, /**< [in] density */
                                    const T Pressure) /**< [in] pressure */
{
    assert(std::isnormal(Density));
    for (int k = 0; k < ndim; k++)
    {
        data[k] = Vel[k];
    }
    data[irho] = Density;
    data[ipress] = Pressure;
}

template class QFluidVector<mfm_numdims, MyFloat>;
template class UFluidVector<mfm_numdims, MyFloat>;
template class WFluidVector<mfm_numdims, MyFloat>;
#ifndef FLOAT_IS_DOUBLE
// avoid duplicate explicit instantiation
template class QFluidVector<mfm_numdims, MyAtLeastDouble>;
template class UFluidVector<mfm_numdims, MyAtLeastDouble>;
template class WFluidVector<mfm_numdims, MyAtLeastDouble>;
#endif // FLOAT_IS_DOUBLE

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
