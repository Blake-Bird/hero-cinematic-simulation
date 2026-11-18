/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Basic specifications for the SIDM module.
 *
 * The specifications given here are needed for the execute_pxp template.
 * They specify the data types for MPI communication and search and selection
 * criteria.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_MSIDM_BASE_HPP
#define MSIDM_MSIDM_BASE_HPP

#include "execute_pxp_base.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/** Data structures for MPI communication */
class msidmdata_in;
class msidmdata_out;
class msidmdata_in_ref;
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Functor to select dark matter particle species in greentree.
 */
class msidm_isactive_class
{
  public:
    int operator()(short int &Type) const
    {
        return (Type == type_dm);
    };
};

/**
 * @brief Declaration of msidm_isactive.
 */
extern class msidm_isactive_class msidm_isactive;
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
using namespace pxp_execute;

/**
 * @brief Typedef for base class of execute_pxp used for the main SIDM class (MSIDM).
 */
typedef execute_pxp_base<msidmdata_in, msidmdata_out, msidmdata_in_ref, decltype(msidm_isactive),
                         decltype(treewalk_filter_dm_only)>
    msidm_pxp;
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Base class for the SIDM module with specifications for dark matter.
 *
 * Base class that contains specifications for finding neighbouring dark matter
 * particles and data types for MPI communication.
 *
 */
class msidm_base : public msidm_pxp
{
  public:
    msidm_base(const int module_in, const int neighboursearchcriteria_in)
        : msidm_pxp(module_in, neighboursearchcriteria_in, msidm_isactive, treewalk_filter_dm_only)
    {
        module_name = "mSIDM";
    }
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_MSIDM_BASE_HPP */
