/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros for the VERBOSE and ENDRUN outputs.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Geray S. Karademir and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef VERBOSE_HPP
#define VERBOSE_HPP

#include <cstdio>

// forward declare the function here, so we can use it whenever this file is used, independent of proto.hpp
void Endrun(int do_output, const char *, const char *, const int, const char *, ...);

// Nitya we want output from all ranks during DEBUG, so removing  VERBOSE_DEFAULT_CONDITION from check
#ifdef INTEL_ALL_RANKS
#define VERBOSE_DEFAULT_CONDITION 1
#else // INTEL_ALL_RANKS
#define VERBOSE_DEFAULT_CONDITION ThisTask == 0
#endif // INTEL_ALL_RANKS

#define VERBOSE_OUTPUT(f_, ...)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        printf((f_), ##__VA_ARGS__);                                                                                   \
        printf("\n");                                                                                                  \
        fflush(stdout);                                                                                                \
    } while (0)

#define VERBOSE(level, f_, ...)                                                                                        \
    if constexpr (level <= Verbose_Level)                                                                              \
        if (VERBOSE_DEFAULT_CONDITION)                                                                                 \
    VERBOSE_OUTPUT(f_, ##__VA_ARGS__)

#define VERBOSE_ALL(level, f_, ...)                                                                                    \
    if constexpr (level <= Verbose_Level)                                                                              \
    VERBOSE_OUTPUT(f_, ##__VA_ARGS__)

#define VERBOSE_IF(condition, level, f_, ...)                                                                          \
    if (condition)                                                                                                     \
    VERBOSE(level, f_, ##__VA_ARGS__)

#define VERBOSE_IF_ALL(condition, level, f_, ...)                                                                      \
    if (condition)                                                                                                     \
    VERBOSE_ALL(level, f_, ##__VA_ARGS__)

// Endrun macros
#define ENDRUN_OUTPUT(output_condition, f_, ...)                                                                       \
    Endrun(output_condition, __FUNCTION__, __FILE__, __LINE__, (f_), ##__VA_ARGS__)

#define ENDRUN_MAIN(f_, ...) ENDRUN_OUTPUT((ThisTask == 0), f_, ##__VA_ARGS__)

#define ENDRUN(f_, ...) ENDRUN_OUTPUT(1, f_, ##__VA_ARGS__)

#define ENDRUN_IF(condition, f_, ...)                                                                                  \
    if (condition)                                                                                                     \
    ENDRUN_OUTPUT(1, f_, ##__VA_ARGS__)

#endif // VERBOSE_HPP
