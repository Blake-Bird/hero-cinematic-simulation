
/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for random number generator
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef RNG_HPP
#define RNG_HPP

#include <random>
#include <cstddef>

extern std::ranlux48 random_generator; /*!< the random number generator used */

// uniform double in [0, 1), analogous to gsl_rng_uniform().
inline double my_rng_uniform(std::ranlux48 &g)
{
    return static_cast<double>(g() - g.min()) / (static_cast<double>(g.max() - g.min()) + 1.0);
}

// fixed-size state buffer for the restart file
constexpr std::size_t RNG_STATE_BUFSIZE = 8192;

void rng_state_to_buffer(char *buf); /* serialize into buf (RNG_STATE_BUFSIZE bytes) */
void rng_state_from_buffer(const char *buf); /* restore from buf */

#endif // RNG_HPP