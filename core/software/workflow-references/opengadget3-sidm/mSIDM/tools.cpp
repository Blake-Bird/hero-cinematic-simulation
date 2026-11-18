/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file with some helpful functions and classes.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "tools.hpp"
#include <iostream>

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/** Gobal instances needed for custom cout */
cout_base cb(2), ce(1), cn(0);

cout_verbose cout(VERBOSE_LEVEL);

cout_verbose_root coutr(VERBOSE_LEVEL);
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::tools
{
/**
 * @brief Function for additional checks not related to scattering.
 */
void beyond_scatter_checks()
{
    // Additional checks not diectly related to SIDM
#ifdef PERIODIC
    ENDRUN_IF(All.BoxSize <= 0.0, "BoxSize must be larger than zero.");
#else // PERIODIC
    ENDRUN_IF(All.BoxSize <= 0.0, "BoxSize must be larger than zero. When no periodic boundaries are used, the maximum "
                                  "kernel size is limited by half the BoxSize (in Hydro/find_hsml.c).");
#endif // PERIODIC

#ifdef COOLING
    ENDRUN_IF((All.HubbleParam != 1.0) && (All.ComovingIntegrationOn == 0),
              "HubbleParam must be set to 1.0 for non-comoving simulations using cooling.");
#endif // COOLING
}
} // namespace scatter::tools

//////////////////////////////////////////////////////////////////////////////////////////////////
