/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros related to the TimerHierarchy class when using user timers.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef USER_TIMER_HPP
#define USER_TIMER_HPP

// User defined timer classes - for testing and debugging purposes.
// Create new instance of the timer class
#define TIMER_NEW_CLASS(TimerClass, label) TimerHierarchy TimerClass(label);

#define TIMER_PUSH_CLASS(TimerClass, level, label)                                                                     \
    if constexpr (level <= Timer_Level)                                                                                \
    {                                                                                                                  \
        TimerClass.push(label);                                                                                        \
    }
#define TIMER_POP_CLASS(TimerClass, level, label)                                                                      \
    if constexpr (level <= Timer_Level)                                                                                \
    {                                                                                                                  \
        TimerClass.pop(label, __FILE__, __LINE__);                                                                     \
    }
#define TIMER_POPPUSH_CLASS(TimerClass, level, label_pop, label_push)                                                  \
    if constexpr (level <= Timer_Level)                                                                                \
    {                                                                                                                  \
        TimerClass.poppush(label_pop, label_push, __FILE__, __LINE__);                                                 \
    }

// only execute on main rank
#define TIMER_PUSH_CLASS_MAINONLY(TimerClass, level, label)                                                            \
    if (ThisTask == 0)                                                                                                 \
    {                                                                                                                  \
        TIMER_PUSH_CLASS(TimerClass, level, label)                                                                     \
    }
#define TIMER_POP_CLASS_MAINONLY(TimerClass, level, label)                                                             \
    if (ThisTask == 0)                                                                                                 \
    {                                                                                                                  \
        TIMER_POP_CLASS(TimerClass, level, label)                                                                      \
    }
#define TIMER_POPPUSH_CLASS_MAINONLY(TimerClass, level, label_pop, label_push)                                         \
    if (ThisTask == 0)                                                                                                 \
    {                                                                                                                  \
        TIMER_POPPUSH_CLASS(TimerClass, level, label_pop, label_push)                                                  \
    }

// Print timer class to console
#define TIMER_PRINT_CLASS(TimerClass) TimerClass.report(std::cout);

#endif // USER_TIMER_HPP
