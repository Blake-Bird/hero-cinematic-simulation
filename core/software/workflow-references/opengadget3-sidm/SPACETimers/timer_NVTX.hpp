/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros related to the TimerHierarchy class when using the NVTX timers.
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
 Note: To use the NVTX API one has to link against -lnvToolsExt
*/

#ifndef NVTX_TIMER_HPP
#define NVTX_TIMER_HPP

#ifdef TIMER_NVTX
#include "nvToolsExt.h"

#define TIMER_ALLVARS_HPP()                                                                                            \
    extern const uint32_t colors[];                                                                                    \
    /* = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,                                                             \
             0x0000ffff, 0x00ff0000, 0x00ffffff }; */                                                                  \
    extern const int num_colors;                                                                                       \
    /* = sizeof(colors) / sizeof(uint32_t); */                                                                         \
    /* We generate a color according to the first letter of the name */                                                \
    /* nvtxEventAttributes_t eventAttrib = {0}; */                                                                     \
    /* #define PUSH_T(name) do { */                                                                                    \
    /* We force the user to add a semicolon after! */

#define TIMER_ALLVARS_CPP()                                                                                            \
    const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00800000,     \
                               0x00000080, 0x00008000, 0x00808080, 0x00808000, 0x00008080, 0x00800080};                \
    const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define TIMER_PUSH_NVTX_BASE(label)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        int color_id = (label[2] * 4 + label[1] * 2 + label[0]) % num_colors;                                          \
        nvtxEventAttributes_t eventAttrib = {0};                                                                       \
        eventAttrib.version = NVTX_VERSION;                                                                            \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                              \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                                                                       \
        eventAttrib.color = colors[color_id];                                                                          \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                                             \
        eventAttrib.message.ascii = label;                                                                             \
        nvtxRangePushEx(&eventAttrib);                                                                                 \
    } while (0)
#define TIMER_POP_NVTX_BASE()                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        nvtxRangePop();                                                                                                \
    } while (0)

#define TIMER_RESET(TimerClass)                                                                                        \
    {                                                                                                                  \
        TIMER_POP_NVTX_BASE();                                                                                         \
        TIMER_POP_NVTX_BASE();                                                                                         \
        TIMER_PUSH_NVTX_BASE("Timestep");                                                                              \
    }

#define TIMER_PUSH(level, label) TIMER_PUSH_NVTX_BASE(label);
#define TIMER_POP(level, label) TIMER_POP_NVTX_BASE();
#define TIMER_POPPUSH(level, label_pop, label_push)                                                                    \
    {                                                                                                                  \
        TIMER_POP_NVTX_BASE();                                                                                         \
        TIMER_PUSH_NVTX_BASE(label_push);                                                                              \
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

#endif // TIMER_NVTX

#endif // NVTX_TIMER_HPP