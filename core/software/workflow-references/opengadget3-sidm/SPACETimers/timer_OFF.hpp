/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros related to the TimerHierarchy class when switching the timers off.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef NO_TIMER_HPP
#define NO_TIMER_HPP

#ifdef SWITCH_TIMER_OFF
// Empty macros in case timers are not used.
#define TIMER_WRITE_SYMBOLS_TO_BALANCEFILE()
#define TIMER_WRITE_TIMES_TO_CPUFILES()
#define TIMER_PRINT(TimerClass)
#define TIMER_POP(level, label)
#define TIMER_PUSH(level, label)
#define TIMER_POPPUSH(level, label_pop, label_push)
#define TIMER_WRITE_SYMBOLS_TO_BALANCEFILE()
#define TIMER_WRITE_TIMES_TO_CPUFILES()
#define TIMER_WRITE_INIT_TO_FILE()
#define TIMER_ALLVARS_HPP()
#define TIMER_ALLVARS_CPP()
// still do reset as global and timestep times need to be availiable even without using the timers
#define TIMER_RESET(TimerClass) TimerClass.reset();
#endif // SWITCH_TIMER_OFF

#endif // NO_TIMER_HPP
