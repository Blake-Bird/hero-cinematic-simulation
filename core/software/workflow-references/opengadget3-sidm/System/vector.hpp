/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP

#if defined(VECTOR_SSE)
#include "vector_sse2.h"
#elif defined(VECTOR_AVX)
#include "vector_avx.h"
#elif defined(VECTOR_VSX)
#include "vector_vsx.h"
#elif defined(VECTOR_QPX)
#include "vector_qpx.h"
#else

// generic scalar code
// Nitya - target variable
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
#ifdef DOUBLEPRECISION
/**
 * @brief Typedef for a 4-element vector with 32-byte alignment.
 *
 * This typedef defines a union named t_vector, which can hold either four unsigned long long integers or four
 * double-precision floating-point numbers. The union ensures that the memory for the elements is shared, meaning that
 * modifying one element will affect the others. The elements are accessed either as an array of unsigned long long
 * integers (i) or as an array of double-precision floating-point numbers (d). The alignment specifier ALIGN(32) ensures
 * that the union is aligned to a 32-byte boundary in memory.
 */
typedef ALIGN(32) union {
    unsigned long long i[4];
    double d[4];
} t_vector;
#else // DOUBLEPRECISION
/**
 * @brief Typedef for a 4-element vector with 16-byte alignment.
 *
 * This typedef defines a union named t_vector, which can hold either four unsigned integers or four floating-point
 * numbers. The union ensures that the memory for the elements is shared, meaning that modifying one element will affect
 * the others. The elements are accessed either as an array of unsigned integers (i) or as an array of floating-point
 * numbers (d). The alignment specifier ALIGN(16) ensures that the union is aligned to a 16-byte boundary in memory.
 */
typedef ALIGN(16) union {
    unsigned int i[4];
    float d[4];
} t_vector;
#endif // DOUBLEPRECISION
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP
/**
 * @brief Load the first 3 components of a vector from an array of MyLongDouble values.
 *
 * This function loads the first 3 components of a vector from an array of MyLongDouble values into a t_vector
 * structure. The 4th component of the t_vector structure remains undefined after this operation.
 *
 * @param src Pointer to the array of MyLongDouble values containing the vector components.
 * @param v Pointer to the t_vector structure where the components will be loaded.
 */
static inline void LOAD_VECTOR3(MyLongDouble *src, t_vector *v)
{
    v->d[0] = src[0];
    v->d[1] = src[1];
    v->d[2] = src[2];
}

/**
 * @brief Load all 4 components of a vector from an array of MyLongDouble values.
 *
 * This function loads all 4 components of a vector from an array of MyLongDouble values into a t_vector structure.
 *
 * @param src Pointer to the array of MyLongDouble values containing the vector components.
 * @param v Pointer to the t_vector structure where the components will be loaded.
 */
static inline void LOAD_VECTOR4(MyLongDouble *src, t_vector *v)
{
    v->d[0] = src[0];
    v->d[1] = src[1];
    v->d[2] = src[2];
    v->d[3] = src[3];
}

/**
 * @brief Store only the first 3 components of a vector into an array of MyLongDouble values.
 *
 * This function stores only the first 3 components of a vector from a t_vector structure into an array of MyLongDouble
 * values.
 *
 * @param dst Pointer to the array of MyLongDouble values where the vector components will be stored.
 * @param v Pointer to the t_vector structure containing the vector components.
 *
 * @note This function is slow and should be avoided.
 */
static inline void STORE_VECTOR3(MyLongDouble *dst, t_vector *v)
{
    dst[0] = v->d[0];
    dst[1] = v->d[1];
    dst[2] = v->d[2];
}

/**
 * @brief Store all 4 components of a vector into an array of MyLongDouble values.
 *
 * This function stores all 4 components of a vector from a t_vector structure into an array of MyLongDouble values.
 *
 * @param dst Pointer to the array of MyLongDouble values where the vector components will be stored.
 * @param v Pointer to the t_vector structure containing the vector components.
 */
static inline void STORE_VECTOR4(MyLongDouble *dst, t_vector *v)
{
    dst[0] = v->d[0];
    dst[1] = v->d[1];
    dst[2] = v->d[2];
    dst[3] = v->d[3];
}

/**
 * @brief Initialize the first 3 components of a vector, leaving the 4th component undefined.
 *
 * This function initializes the first 3 components of a vector in a t_vector structure with the specified value,
 * leaving the 4th component undefined.
 *
 * @param a The value to initialize the first 3 components of the vector with.
 * @param v Pointer to the t_vector structure to be initialized.
 */
static inline void SET_VECTOR3(MyLongDouble a, t_vector *v)
{
    v->d[0] = a;
    v->d[1] = a;
    v->d[2] = a;
}

/**
 * @brief Initializes all components of a 4-dimensional vector with the same scalar value.
 *
 * This function sets all components of the specified 4-dimensional vector to the given scalar value.
 *
 * @param a The scalar value to set all components of the vector to.
 * @param v Pointer to the 4-dimensional vector to be initialized.
 */
static inline void SET_VECTOR4(MyLongDouble a, t_vector *v)
{
    v->d[0] = a;
    v->d[1] = a;
    v->d[2] = a;
    v->d[3] = a;
}

// Nitya - adding declare directive
#ifdef ACC_OPENMP
#pragma omp declare target
static inline void INIT_VECTOR3(MyLongDouble a0, MyLongDouble a1, MyLongDouble a2, t_vector *v);
#pragma omp end declare target
#endif // ACC_OPENMP

/**
 * @brief Initializes the first three components of a vector, leaving the fourth component undefined.
 *
 * This function sets the first three components of the specified vector to the provided values.
 * The fourth component of the vector remains unchanged (undefined).
 *
 * @param a0 The value for the first component of the vector.
 * @param a1 The value for the second component of the vector.
 * @param a2 The value for the third component of the vector.
 * @param v Pointer to the vector to be initialized.
 */
static inline void INIT_VECTOR3(MyLongDouble a0, MyLongDouble a1, MyLongDouble a2, t_vector *v)
{
    v->d[0] = a0;
    v->d[1] = a1;
    v->d[2] = a2;
}

/**
 * @brief Adds the first three components of two vectors, leaving the fourth component undefined.
 *
 * This function calculates the sum of the first three components of two vectors and stores the result
 * in the corresponding components of the result vector. The fourth component of the result vector remains undefined.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param result Pointer to the vector where the sum will be stored.
 */
static inline void ADD_VECTOR3(t_vector *a, t_vector *b, t_vector *result)
{
    result->d[0] = a->d[0] + b->d[0];
    result->d[1] = a->d[1] + b->d[1];
    result->d[2] = a->d[2] + b->d[2];
}

/**
 * @brief Multiplies the first three components of two vectors, leaving the fourth component undefined.
 *
 * This function calculates the product of the first three components of two vectors and stores the result
 * in the corresponding components of the result vector. The fourth component of the result vector remains undefined.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param result Pointer to the vector where the product will be stored.
 */
static inline void MUL_VECTOR3(t_vector *a, t_vector *b, t_vector *result)
{
    result->d[0] = a->d[0] * b->d[0];
    result->d[1] = a->d[1] * b->d[1];
    result->d[2] = a->d[2] * b->d[2];
}

// multiplies the first 3 components of 2 vectors (4th component undefined)
#ifdef ACC_OPENMP
#pragma omp declare target
static inline void SCALE_VECTOR3(MyLongDouble a, t_vector *b, t_vector *v);
#pragma omp end declare target
#endif // ACC_OPENMP

/**
 * @brief Scales the first three components of a vector by a scalar value, leaving the fourth component undefined.
 *
 * This function multiplies each of the first three components of the specified vector by the given scalar value
 * and stores the result in the corresponding components of the result vector. The fourth component of the result
 * vector remains undefined.
 *
 * @param a The scalar value by which to scale the components of the vector.
 * @param b Pointer to the vector whose components will be scaled.
 * @param v Pointer to the vector where the scaled components will be stored.
 */
static inline void SCALE_VECTOR3(MyLongDouble a, t_vector *b, t_vector *v)
{
    v->d[0] = a * b->d[0];
    v->d[1] = a * b->d[1];
    v->d[2] = a * b->d[2];
}

/**
 * @brief Checks if any component of one vector is less than the corresponding component of another vector.
 *
 * This function compares each component of two vectors and returns a value greater than zero if any component of
 * vector 'a' is less than the corresponding component of vector 'b' for indices 1 to 3.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param mask Unused parameter (optional).
 * @return A value greater than zero if any component of vector 'a' is less than the corresponding component of vector
 * 'b'.
 */
static inline int ANY_COMP_LT_VECTOR3(t_vector *a, t_vector *b, int mask)
{
    return (a->d[0] < b->d[0]) | (a->d[1] < b->d[1]) | (a->d[2] < b->d[2]);
}

/**
 * @brief Checks if all components of one vector are less than the corresponding components of another vector.
 *
 * This function compares each component of two vectors and returns zero if all components of vector 'a' are less than
 * the corresponding components of vector 'b' for indices 1 to 3.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param mask Unused parameter (optional).
 * @return Zero if all components of vector 'a' are less than the corresponding components of vector 'b'.
 */
static inline int ALL_COMP_LT_VECTOR3(t_vector *a, t_vector *b, int mask)
{
    return (a->d[0] < b->d[0]) & (a->d[1] < b->d[1]) & (a->d[2] < b->d[2]);
}

/**
 * @brief Calculates the L2 norm of the first three components of a vector.
 *
 * This function computes the L2 norm (Euclidean norm) of the first three components of the specified vector.
 *
 * @param v Pointer to the vector.
 * @return The L2 norm of the first three components of the vector.
 */
static inline MyDouble L2NORM_VECTOR3(t_vector *v)
{
    return v->d[0] * v->d[0] + v->d[1] * v->d[1] + v->d[2] * v->d[2];
}
#endif

#ifdef ACC_OPENMP
#pragma omp declare target
inline static int ngb_check_node(struct NODE *cur, t_vector *v2, t_vector *box, t_vector *hbox, MyDouble hsml);
#pragma omp end declare target
#endif // ACC_OPENMP

/**
 * @brief Checks if a node should be opened for further examination based on the distance criteria.
 *
 * This function determines whether a node should be opened for further examination based on the distance
 * criteria involving the provided node's center, the input vector, and certain box dimensions.
 *
 * On older Intel and AMD processors, it seems better to avoid certain computations and branch early.
 * However, it's essential to validate this behavior.
 *
 * @param cur Pointer to the node to be checked.
 * @param v2 Pointer to the vector.
 * @param box Pointer to the box dimensions.
 * @param hbox Pointer to the half-box dimensions.
 * @param hsml The smoothing length.
 * @return The index of the next node to process, either the sibling or the next node depending on the distance
 * criteria.
 */
inline static int ngb_check_node(struct NODE *cur, t_vector *v2, t_vector *box, t_vector *hbox, MyDouble hsml)
{
    MyLongDouble dist = hsml + 0.5 * cur->len;
    MyLongDouble d2, dx, dy, dz;
    int node = cur->u.d.sibling; // in case the node can be discarded

    dx = NGB_PERIODIC_LONG(cur->center[0] - v2->d[0], box->d[0], hbox->d[0]);
    if (dx <= dist)
    {
        dy = NGB_PERIODIC_LONG(cur->center[1] - v2->d[1], box->d[1], hbox->d[1]);
        if (dy <= dist)
        {
            dz = NGB_PERIODIC_LONG(cur->center[2] - v2->d[2], box->d[2], hbox->d[2]);
            if (dz <= dist)
            {
                // now test against the minimal sphere enclosing everything
                dist += FACT1 * cur->len;
                d2 = dx * dx + dy * dy + dz * dz;
                if (d2 <= dist * dist)
                    node = cur->u.d.nextnode; // ok, we need to open the node
            }
        }
    }
    return node;
}

#endif // VECTOR_HPP
