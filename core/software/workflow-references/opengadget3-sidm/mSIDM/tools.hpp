/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file with some helpful functions and classes.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_TOOLS_HPP
#define MSIDM_TOOLS_HPP

#include "../CodeBase/precision.hpp"
#include "../CodeBase/allvars.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include "auxiliaries.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Enumeration to specify verbose levels.
 */
enum verbose_levels
{
    error = 0,
    important = 1,
    warning = 1,
    info = 2,
    sendstart = 3,
    receiveend = 4,
    extra = 5
};

using std::endl;
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Basic class for custom cout.
 */
extern class cout_base
{
  public:
    cout_base(short print_mode_in) : print_mode(print_mode_in){};

    const short print_mode;
} cb, cn, ce;
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Ostream operator for cout_base.
 */
template <typename DTYPE> std::ostream &operator<<(cout_base &b, DTYPE d)
{
    if (b.print_mode == 0)
    {
        std::cout.setstate(std::ios_base::badbit);
        return std::cout;
    }
    else if (b.print_mode == 1)
    {
        return std::cerr << d;
    }
    else
    {
        std::cout.clear();
        return std::cout << d;
    }
};
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Main class for custom cout.
 */
extern class cout_verbose
{
  public:
    cout_verbose(const int level_in) : ref_level(level_in)
    {
    }

    cout_base &operator()(const int level) const
    {
        if (level > ref_level)
        {
            return cn;
        }
        else if (level == error)
        {
            return ce;
        }
        else
        {
            return cb;
        }
    };

  private:
    const int ref_level;
} cout;
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Custom cout for root process printing only.
 */
extern class cout_verbose_root : private cout_verbose
{
  public:
    cout_verbose_root(const int level_in) : cout_verbose(level_in)
    {
    }

    cout_base &operator()(const int level) const
    {
        if (ThisTask == 0)
        {
            return cout_verbose::operator()(level);
        }
        else
        {
            return cn;
        }
    };
} coutr;
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Template for maximum of array.
 */
template <typename DTYPE> DTYPE get_max_of_array(DTYPE array[], int length)
{
    DTYPE result = array[0];
    for (int loop = 1; loop < length; ++loop)
    {
        result = std::max(result, array[loop]);
    }
    return result;
}
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Template for sorting array.
 */
template <typename DTYPE> void sort_large_to_small(DTYPE array[], long long unsigned int size)
{
    std::qsort(array, size, sizeof *array, [](const void *a, const void *b) {
        DTYPE arg1 = *static_cast<const DTYPE *>(a);
        DTYPE arg2 = *static_cast<const DTYPE *>(b);
        return (arg1 < arg2) - (arg1 > arg2);
    });
}
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Template for linear interpolation.
 */
template <typename DTYPE> DTYPE interpol(DTYPE x, DTYPE x1, DTYPE x2, DTYPE y1, DTYPE y2)
{
    return (y1 * (x2 - x) + y2 * (x - x1)) / (x2 - x1);
};
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Template for length of vector.
 */
template <typename DTYPE> DTYPE vector_length(DTYPE vec[3])
{
    return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
};
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Template to normalize vector.
 */
template <typename DTYPE> void vector_norm(DTYPE vec[3])
{
    DTYPE vec_len = vector_length(vec);
    if (vec_len != 0)
    {
        DTYPE D_vec_len;
        D_vec_len = 1.0 / vec_len;
        for (int loopc = 0; loopc < 3; ++loopc)
        {
            vec[loopc] *= D_vec_len;
        }
    }
};
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Function for additional checks not related to scattering.
 */
void beyond_scatter_checks();
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_TOOLS_HPP */
