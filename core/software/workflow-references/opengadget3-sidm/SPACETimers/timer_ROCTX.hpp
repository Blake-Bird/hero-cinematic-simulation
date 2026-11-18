/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros related to the TimerHierarchy class when using the ROCTX API.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

/*
Note: To use the ROCTX API one has to link against -lroctx64
*/

#ifndef ROCTX_TIMER_HPP
#define ROCTX_TIMER_HPP

#ifdef TIMER_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>

#define TIMER_ALLVARS_HPP()
#define TIMER_ALLVARS_CPP()

#define TIMER_PUSH_ROCTX_BASE(label)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        roctxRangePush(label);                                                                                         \
    } while (0)
#define TIMER_POP_ROCTX_BASE()                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        roctxRangePop();                                                                                               \
    } while (0)

#define TIMER_RESET(TimerClass)                                                                                        \
    {                                                                                                                  \
        TIMER_POP_ROCTX_BASE();                                                                                        \
        TIMER_PUSH_ROCTX_BASE("Timestep");                                                                             \
    }

#define TIMER_PUSH(level, label) TIMER_PUSH_ROCTX_BASE(label);
#define TIMER_POP(level, label) TIMER_POP_ROCTX_BASE();
#define TIMER_POPPUSH(level, label_pop, label_push)                                                                    \
    {                                                                                                                  \
        TIMER_POP_ROCTX_BASE();                                                                                        \
        TIMER_PUSH_ROCTX_BASE(label_push);                                                                             \
    }

// empty macros when using profiler
#define TIMER_WRITE_SYMBOLS_TO_BALANCEFILE()                                                                           \
    {                                                                                                                  \
    }
#define TIMER_WRITE_TIMES_TO_CPUFILES()                                                                                \
    {                                                                                                                  \
    }
#define TIMER_WRITE_INIT_TO_FILE()                                                                                     \
    {                                                                                                                  \
    }
#define TIMER_WRITE_SYMBOLS_TO_BALANCEFILE()                                                                           \
    {                                                                                                                  \
    }
#define TIMER_WRITE_TIMES_TO_CPUFILES()                                                                                \
    {                                                                                                                  \
    }
#define TIMER_PRINT(TimerClass)                                                                                        \
    {                                                                                                                  \
    }
#endif // TIMER_ROCTX

#endif // ROCTX_TIMER_HPP