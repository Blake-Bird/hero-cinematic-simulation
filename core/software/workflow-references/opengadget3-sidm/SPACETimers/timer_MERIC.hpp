/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros related to the TimerHierarchy class when using MERIC.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

/* How to use MERIC:
- Download MERIC from: https://code.it4i.cz/vys0053/meric
- Install using make.
- Add paths to compilation:
  e.g. MERIC_INCL = path/meric
  XXX_INCL += -I${MERIC_INCL}/include
  XXX_LIBS += -L${MERIC_INCL}/lib -lmericmpi
- run code.
*/

#ifndef MERIC_TIMER_HPP
#define MERIC_TIMER_HPP

#ifdef TIMER_MERIC
#include <meric.h>

#define TIMER_PUSH_MERIC_BASE(label)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (All.NumCurrentTiStep >= 1)                                                                                 \
        {                                                                                                              \
            MERIC_MeasureStart(label);                                                                                 \
        }                                                                                                              \
    } while (0)

#define TIMER_POP_MERIC_BASE(label)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (All.NumCurrentTiStep >= 1)                                                                                 \
        {                                                                                                              \
            MERIC_MeasureStop(label);                                                                                  \
        }                                                                                                              \
    } while (0)

#define TIMER_POPPUSH_MERIC_BASE(label_pop, label_push)                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (All.NumCurrentTiStep >= 1)                                                                                 \
        {                                                                                                              \
            MERIC_MeasureStopStart(label_pop, label_push);                                                             \
        }                                                                                                              \
    } while (0)

#endif // TIMER_MERIC

#endif // MERIC_TIMER_HPP
