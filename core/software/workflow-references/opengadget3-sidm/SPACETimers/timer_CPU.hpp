/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the macros related to the TimerHierarchy class when using CPU Timers.
 *
 * @author OpenGadget3-development team, in particular Geray S. Karademir.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef CPU_TIMER_HPP
#define CPU_TIMER_HPP
// ############################ GENERAL TIMER MACROS ############################
// Timer macros
#define TIMER_RESET(TimerClass) TimerClass.reset();

#define TIMER_ALLVARS_HPP()
#define TIMER_ALLVARS_CPP()

// Only do things on thread 0 (if openmp is used)
#ifdef _OPENMP
#define OMP_TASK_CHECK _Pragma("omp master")
#else // _OPENMP
#define OMP_TASK_CHECK                                                                                                 \
    {                                                                                                                  \
    }
#endif // _OPENMP

#ifdef TIMER_MERIC
#include "timer_MERIC.hpp"
#else // TIMER_MERIC
#define TIMER_PUSH_MERIC_BASE(label)                                                                                   \
    {                                                                                                                  \
    }
#define TIMER_POP_MERIC_BASE(label)                                                                                    \
    {                                                                                                                  \
    }
#define TIMER_POPPUSH_MERIC_BASE(label_push, label_pop)                                                                \
    {                                                                                                                  \
    }
#endif // TIMER_MERIC

// activate TIMER_DETAILS to get information about all MPI tasks, otherwise only the main with measure and report
#ifdef TIMER_DETAILS
// Timer makros depending on the TIMER_DETAILS flag
#define TIMER_WRITE_TIMES_TO_CPUFILES()                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        OMP_TASK_CHECK                                                                                                 \
        {                                                                                                              \
            TimestepTimer.report(FdCPUTimestep, true);                                                                 \
            if (ThisTask == 0)                                                                                         \
            {                                                                                                          \
                GlobalTimer.report(FdCPU);                                                                             \
                GlobalTimer.delete_node("Initialisation");                                                             \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define MPI_TASK_CHECK                                                                                                 \
    {                                                                                                                  \
    }

#else // TIMER_DETAILS
#define TIMER_WRITE_TIMES_TO_CPUFILES()                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        OMP_TASK_CHECK                                                                                                 \
        {                                                                                                              \
            if (ThisTask == 0)                                                                                         \
            {                                                                                                          \
                GlobalTimer.report(FdCPU);                                                                             \
                GlobalTimer.delete_node("Initialisation");                                                             \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#define MPI_TASK_CHECK if (ThisTask == 0)
#endif // TIMER_DETAILS

#ifdef TIMER_TIMESTAMP
#define REPORT_TIMESTAMPS(level, label, type)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr (level <= Timer_Level)                                                                            \
        {                                                                                                              \
            if (ThisTask == 0)                                                                                         \
            {                                                                                                          \
                auto now = std::chrono::high_resolution_clock::now();                                                  \
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);                                         \
                std::tm *tm_ptr = std::localtime(&now_c);                                                              \
                auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000; \
                FdTimersTimestamp << type << " , " << level << " , " << label << " , "                                 \
                                  << std::put_time(tm_ptr, "%Y/%m/%d %H:%M:%S") << '.' << std::setfill('0')            \
                                  << std::setw(6) << now_us.count() << std::endl;                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)
#else // TIMER_TIMESTAMP
#define REPORT_TIMESTAMPS(level, label, type)                                                                          \
    {                                                                                                                  \
    }
#endif // TIMER_TIMESTAMP

// General timer makros
#define TIMER_POP_BASE(label)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        GlobalTimer.pop(label, __FILE__, __LINE__);                                                                    \
        TimestepTimer.pop(label, __FILE__, __LINE__);                                                                  \
    } while (0)
#define TIMER_PUSH_BASE(label)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        GlobalTimer.push(label);                                                                                       \
        TimestepTimer.push(label);                                                                                     \
    } while (0)
#define TIMER_POPPUSH_BASE(label_pop, label_push)                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        GlobalTimer.poppush(label_pop, label_push, __FILE__, __LINE__);                                                \
        TimestepTimer.poppush(label_pop, label_push, __FILE__, __LINE__);                                              \
    } while (0)
#define TIMER_WRITE_SYMBOLS_TO_BALANCEFILE()                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        OMP_TASK_CHECK                                                                                                 \
        {                                                                                                              \
            if (ThisTask == 0)                                                                                         \
            {                                                                                                          \
                TimestepTimer.write_symbols(FdBalance);                                                                \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

// print current timings
#define TIMER_PRINT(TimerClass)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        OMP_TASK_CHECK                                                                                                 \
        {                                                                                                              \
            TimerClass.report(std::cout);                                                                              \
        }                                                                                                              \
    } while (0)

// macros to use for poping/pushing and writing output
#ifdef TIMER_DEBUG

#define TIMER_DEBUG_PUSH(label)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        printf("TIMER: PUSH %s in File %s, Line %d \n", label, __FILE__, __LINE__);                                    \
        fflush(stdout);                                                                                                \
    } while (0)
#define TIMER_DEBUG_POP(label)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        printf("TIMER: POP %s in File %s, Line %d \n", label, __FILE__, __LINE__);                                     \
        fflush(stdout);                                                                                                \
    } while (0)
#define TIMER_DEBUG_POPPUSH(label_pop, label_push)                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        printf("TIMER: POPPUSH %s -> %s in File %s, Line %d\n", label_pop, label_push, __FILE__, __LINE__);            \
        fflush(stdout);                                                                                                \
    } while (0)

#else // TIMER_DEBUG

#define TIMER_DEBUG_PUSH(label)                                                                                        \
    {                                                                                                                  \
    }
#define TIMER_DEBUG_POP(label)                                                                                         \
    {                                                                                                                  \
    }
#define TIMER_DEBUG_POPPUSH(label_pop, label_push)                                                                     \
    {                                                                                                                  \
    }

#endif // TIMER_DEBUG

#define TIMER_PUSH(level, label)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr (level <= Timer_Level)                                                                            \
        {                                                                                                              \
            MPI_TASK_CHECK                                                                                             \
            {                                                                                                          \
                OMP_TASK_CHECK                                                                                         \
                {                                                                                                      \
                    REPORT_TIMESTAMPS(level, label, "PUSH");                                                           \
                    TIMER_DEBUG_PUSH(label);                                                                           \
                    TIMER_PUSH_BASE(label);                                                                            \
                }                                                                                                      \
            }                                                                                                          \
            TIMER_PUSH_MERIC_BASE(label);                                                                              \
        }                                                                                                              \
    } while (0)

#define TIMER_POP(level, label)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr (level <= Timer_Level)                                                                            \
        {                                                                                                              \
            MPI_TASK_CHECK                                                                                             \
            {                                                                                                          \
                OMP_TASK_CHECK                                                                                         \
                {                                                                                                      \
                    REPORT_TIMESTAMPS(level, label, "POP");                                                            \
                    TIMER_DEBUG_POP(label);                                                                            \
                    TIMER_POP_BASE(label);                                                                             \
                }                                                                                                      \
            }                                                                                                          \
            TIMER_POP_MERIC_BASE(label);                                                                               \
        }                                                                                                              \
    } while (0)

#define TIMER_POPPUSH(level, label_pop, label_push)                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr (level <= Timer_Level)                                                                            \
        {                                                                                                              \
            MPI_TASK_CHECK                                                                                             \
            {                                                                                                          \
                OMP_TASK_CHECK                                                                                         \
                {                                                                                                      \
                    REPORT_TIMESTAMPS(level, label_pop, "POP");                                                        \
                    REPORT_TIMESTAMPS(level, label_push, "PUSH");                                                      \
                    TIMER_DEBUG_POPPUSH(label_pop, label_push);                                                        \
                    TIMER_POPPUSH_BASE(label_pop, label_push);                                                         \
                }                                                                                                      \
            }                                                                                                          \
            TIMER_POPPUSH_MERIC_BASE(label_pop, label_push);                                                           \
        }                                                                                                              \
    } while (0)

#endif // CPU_TIMER_HPP
