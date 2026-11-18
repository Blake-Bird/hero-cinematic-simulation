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

/**
 *  @brief definition of an assertation macro that
 *  behaves more coherent with the Gadget-Framework
 *  than the standard definition of the macro
 */
#ifndef ASSERT_HPP
#define ASSERT_HPP

#ifdef DEBUG
#include "../CodeBase/verbose.hpp"
#endif // DEBUG

#if !defined(DEBUG)

// if DEBUG is not active, this function/macro should do nothing, thus also not leading to any overhead.
#define assert(expr)

#else // DEBUG

// with DEBUG, make sure the program terminates in the case of assertion faults.
#ifdef __func__
#define assert(expr)                                                                                                   \
    ENDRUN_IF(!(expr), "ASSERTATION FAULT: File: %s -- %s() -- Line: %i -- Assertation (%s) failed.\n", __FILE__,      \
              __func__, __LINE__, __STRING(expr))
#else // __func__
#define assert(expr)                                                                                                   \
    ENDRUN_IF(!(expr), "ASSERTATION FAULT: File: %s -- Line: %i -- Assertation (%s) failed.\n", __FILE__, __LINE__,    \
              __STRING(expr))
#endif // __func__
#endif // DEBUG

#endif // ASSERT_HPP
