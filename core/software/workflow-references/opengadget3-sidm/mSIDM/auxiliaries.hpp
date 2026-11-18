/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains some auxiliary functions that are used for the SIDM and IDM modules.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_AUXILIARIES_HPP
#define MSIDM_AUXILIARIES_HPP

#include <iostream>

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Function that throws exception.
 */
inline void Expects(bool pre_condition)
{
    if (!pre_condition)
        throw std::exception();
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Function that throw exception and prints a string.
 */
inline void Expects(bool pre_condition, const std::string &text)
{
    if (!pre_condition)
    {
        std::cout << text << std::endl;
        throw std::exception();
    }
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Function template to check for not a number.
 */
template <typename DTYPE> void check_nan(const DTYPE x, const std::string &text)
{
    if (x != x || isinf(x))
    {
        std::cout << text << " " << x << std::endl;
        throw std::exception();
    }
}
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Template to convert strings to another data type.
 */
template <typename DTYPE> DTYPE convert_from_string(const std::string &text)
{
    DTYPE result;
    std::stringstream ss(text);
    ss >> result;
    return result;
}
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_AUXILIARIES_HPP */
