/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This header file contains the declaration for the variable types managing the precision of the code.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef PRECISION_HPP
#define PRECISION_HPP

#include "../CodeBase/switches.hpp"
#include <limits>
#include <mpi.h>

typedef int sort_type;
#define sizesort sizeof(sort_type)

#ifndef LONGIDS
typedef unsigned int MyIDType;
#else // LONGIDS
typedef unsigned long long MyIDType;
#endif // LONGIDS

#if !defined(DOUBLEPRECISION) || (DOUBLEPRECISION + 0) == 0 /* default is single-precision */
typedef float MyFloat;
typedef float MyDouble;
typedef float MyLongDouble;
typedef double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_FLOAT;
#elif (DOUBLEPRECISION + 0) == -3 /* everything shor-precision, only for testing */
typedef __bf16 MyFloat;
typedef __bf16 MyDouble;
typedef float MyLongDouble;
typedef double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_SHORT;
#elif (DOUBLEPRECISION + 0) == -2 /* everything shor-precision, only for testing */
typedef _Float16 MyFloat;
typedef _Float16 MyDouble;
typedef float MyLongDouble;
typedef double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_SHORT;
#elif (DOUBLEPRECISION + 0) == -1 /* everything shor-precision, only for testing */
typedef float MyFloat;
typedef float MyDouble;
typedef float MyLongDouble;
typedef float MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_FLOAT;
#elif (DOUBLEPRECISION + 0) == 1 /* everything double-precision, more memory but faster */
typedef double MyFloat;
typedef double MyDouble;
typedef double MyLongDouble;
typedef double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_DOUBLE;
#elif (DOUBLEPRECISION + 0) == 2 /* everything long double-precision */
typedef long double MyFloat;
typedef long double MyDouble;
typedef long double MyLongDouble;
typedef long double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_LONG_DOUBLE;
#elif (DOUBLEPRECISION + 0) ==                                                                                         \
    3 /* light weighted mixed precision, very low memory, only Pos, accelerations, and local variables are double */
typedef float MyFloat;
typedef float MyDouble;
typedef double MyLongDouble;
typedef double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_FLOAT;
#elif (DOUBLEPRECISION + 0) == 4 /* true mixed precision, moderate memory, all precisions as declared */
typedef float MyFloat;
typedef double MyDouble;
typedef long double MyLongDouble;
typedef double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_FLOAT;
#elif (DOUBLEPRECISION + 0) == 5 /* same as 3, but including long double as best precission */
typedef float MyFloat;
typedef double MyDouble;
typedef long double MyLongDouble;
typedef long double MyAtLeastDouble;
const MPI_Datatype MPI_MyFloat = MPI_FLOAT;
#else /* Not allowd any longer! */
#error DOUBLEPRECISION has to be a value -3, -2, -1 or between 1 and 5 if set !
#endif // DOUBLEPRECISION

#if (((DOUBLEPRECISION + 0) == 1) || ((DOUBLEPRECISION + 0) == 2))
#define FLOAT_IS_DOUBLE
#endif // (((DOUBLEPRECISION+0)==1) || ((DOUBLEPRECISION+0)==2))

static const MyFloat smallest_float = std::numeric_limits<MyFloat>::min();
static const MyFloat small_float = 8 * smallest_float;

#ifdef OUTPUT_IN_DOUBLEPRECISION
typedef double MyOutputFloat;
#else // OUTPUT_IN_DOUBLEPRECISION
typedef float MyOutputFloat;
#endif // OUTPUT_IN_DOUBLEPRECISION

#ifdef INPUT_IN_DOUBLEPRECISION
typedef double MyInputFloat;
#else // INPUT_IN_DOUBLEPRECISION
typedef float MyInputFloat;
#endif // INPUT_IN_DOUBLEPRECISION

#ifdef ENLARGE_DYNAMIC_RANGE_IN_TIME
typedef long long integertime;
#define TIMEBINS 60
#define TIMEBASE                                                                                                       \
    (((long long)1) << TIMEBINS) /*!< The simulated timespan is mapped onto the integer interval [0,TIMESPAN],         \
                                  *   where TIMESPAN needs to be a power of 2.                                         \
                                  */
#else // ENLARGE_DYNAMIC_RANGE_IN_TIME
#define TIMEBINS 29
#define TIMEBASE                                                                                                       \
    (1 << TIMEBINS) /*!< The simulated timespan is mapped onto the integer interval [0,TIMESPAN],                      \
                     *   where TIMESPAN needs to be a power of 2. Note that (1<<28) corresponds                        \
                     *   to 2^29                                                                                       \
                     */
typedef int integertime;
#endif // ENLARGE_DYNAMIC_RANGE_IN_TIME

typedef unsigned long long peanokey;
typedef MyDouble MyBigFloat;

#include <cmath>

namespace mymath
{

#if defined(DOUBLEPRECISION) && (DOUBLEPRECISION + 0) == -3
inline __bf16 abs(__bf16 x)
{
    return x < (__bf16)0 ? -x : x;
}
#endif
#if defined(DOUBLEPRECISION) && (DOUBLEPRECISION + 0) == -2
inline _Float16 abs(_Float16 x)
{
    return x < (_Float16)0 ? -x : x;
}
#endif
using std::abs;

#if defined(DOUBLEPRECISION) && (DOUBLEPRECISION + 0) == -3
inline __bf16 max(__bf16 x, __bf16 y)
{
    return (x > y) ? x : y;
}
#endif
#if defined(DOUBLEPRECISION) && (DOUBLEPRECISION + 0) == -2
inline _Float16 max(_Float16 x, _Float16 y)
{
    return (x > y) ? x : y;
}
#endif
using std::max;

#if defined(DOUBLEPRECISION) && (DOUBLEPRECISION + 0) == -3
inline __bf16 min(__bf16 x, __bf16 y)
{
    return (x < y) ? x : y;
}
#endif
#if defined(DOUBLEPRECISION) && (DOUBLEPRECISION + 0) == -2
inline _Float16 min(_Float16 x, _Float16 y)
{
    return (x < y) ? x : y;
}
#endif
using std::min;

} // namespace mymath

using mymath::abs;
using mymath::max;
using mymath::min;

#endif // PRECISION_HPP
