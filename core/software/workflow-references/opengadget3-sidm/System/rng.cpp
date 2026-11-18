/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Random number generator (std::ranlux48)
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "rng.hpp"
#include <sstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>

std::ranlux48 random_generator;

/**
 * @brief Serialize the global RNG state into a fixed-size restart buffer.
 *
 * Writes the textual representation of ::random_generator (as defined by
 * the C++ standard's operator<< for random engines) into @p buf. The
 * remainder of the buffer is zero-padded so the contents are fully
 * deterministic, and the string is NUL-terminated.
 *
 * @param[out] buf  Destination buffer of at least #RNG_STATE_BUFSIZE bytes.
 *
 * @note Aborts the program (std::abort) if the serialized state, including
 *       its NUL terminator, does not fit in #RNG_STATE_BUFSIZE bytes.
 */
void rng_state_to_buffer(char *buf)
{
    std::ostringstream os;
    os << random_generator; /* standard-defined textual state */
    std::string s = os.str();

    if (s.size() + 1 > RNG_STATE_BUFSIZE)
    {
        std::fprintf(stderr, "RNG state too large for restart buffer\n");
        std::abort();
    }
    std::memset(buf, 0, RNG_STATE_BUFSIZE);
    std::memcpy(buf, s.c_str(), s.size() + 1);
}

/**
 * @brief Restore the global RNG state from a restart buffer.
 *
 * Reads the textual engine state previously produced by
 * rng_state_to_buffer() from @p buf and loads it into
 * ::random_generator via the standard operator>>. Parsing stops at the
 * first NUL byte, so the zero padding added during serialization is
 * ignored.
 *
 * @param[in] buf  NUL-terminated buffer of #RNG_STATE_BUFSIZE bytes
 *                 containing a serialized RNG state.
 *
 * @note Aborts the program (std::abort) if the buffer does not contain a
 *       valid state for the engine type (e.g. corrupted restart file or
 *       a state written by an incompatible engine/build).
 */
void rng_state_from_buffer(const char *buf)
{
    std::istringstream is{std::string(buf)}; /* stops at the '\0' padding */
    is >> random_generator;
    if (is.fail())
    {
        std::fprintf(stderr, "failed to restore RNG state from restart file\n");
        std::abort();
    }
}
