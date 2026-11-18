/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros related to the TimerHierarchy class when using the ITT API.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef ITT_TIMER_HPP
#define ITT_TIMER_HPP

#ifdef TIMER_VTUNE
#include <ittnotify.h>

#define TIMER_ALLVARS_HPP() extern __itt_domain *ITT_DOMAIN;

#define TIMER_ALLVARS_CPP() __itt_domain *ITT_DOMAIN = __itt_domain_create("OpenGadget3");

#define TIMER_PUSH_VTUNE_BASE(label)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        static __itt_domain *itt_dom = ITT_DOMAIN;                                                                     \
        static __itt_string_handle *__itt_str_handle = __itt_string_handle_create(label);                              \
        __itt_task_begin(itt_dom, __itt_null, __itt_null, __itt_str_handle);                                           \
    } while (0)

// Macro to end the current region
#define TIMER_POP_VTUNE_BASE()                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        static __itt_domain *itt_dom = ITT_DOMAIN;                                                                     \
        __itt_task_end(itt_dom);                                                                                       \
    } while (0)

#define TIMER_PUSH(level, label) TIMER_PUSH_VTUNE_BASE(label);

#define TIMER_POP(level, label) TIMER_POP_VTUNE_BASE();

#define TIMER_POPPUSH(level, label_pop, label_push)                                                                    \
    {                                                                                                                  \
        TIMER_POP_VTUNE_BASE();                                                                                        \
        TIMER_PUSH_VTUNE_BASE(label_push);                                                                             \
    }

#define TIMER_RESET(TimerClass)                                                                                        \
    {                                                                                                                  \
        TIMER_POP_VTUNE_BASE();                                                                                        \
        TIMER_POP_VTUNE_BASE();                                                                                        \
        TIMER_PUSH_VTUNE_BASE("Timestep");                                                                             \
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
#endif // TIMER_VTUNE

#endif // ITT_TIMER_HPP