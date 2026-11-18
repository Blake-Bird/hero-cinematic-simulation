/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef DEBUG
#include <fenv.h>

#include <map>
#include <string>
#include <iostream>
#include <iomanip>

/**
 * @brief For testing purposes:Accumulate and print total counts for various events.
 *
 * This function maintains a static map to accumulate counts for different events identified by `counter_name`.
 * When `final_output` is false, it increments the count for the specified event.
 * When `final_output` is true, it prints all accumulated counts in a formatted manner.
 *
 * @param counter_name The name of the event to count (e.g. a function call).
 * @param final_output (default false) If true, prints the accumulated counts; if false, increments the count.
 */
int total_counts(const std::string &counter_name, bool final_output)
{
    static std::map<std::string, long long> counters;
    if (!final_output)
    {
        // Accumulate counts (automatically initializes to 0 if key doesn't exist)
        if (counters.find(counter_name) == counters.end())
        {
            counters[counter_name] = 0;
        }
        counters[counter_name] += 1;
    }
    else
    {
        // Print all accumulated counts
        VERBOSE(0, "\n=== Total Counts ===");
        for (const auto &[name, total_count] : counters)
        {
            VERBOSE(0, "%s: \t %lli", name.c_str(), total_count);
        }
        VERBOSE(0, "====================\n");
        counters.clear(); // Optional: clear after printing
    }
    return counters[counter_name];
}

/**
 * @brief For testing purposes: Accumulate and print total execution times for various functions.
 *
 * This function maintains a static map to accumulate execution times for different functions identified by `func_name`.
 * When `final_output` is false, it adds the provided `time_diff` to the total time for the specified function.
 * When `final_output` is true, it prints all accumulated execution times in a formatted manner.
 *
 * @param time_diff The time difference to add (in seconds).
 * @param func_name The name of the function whose time is being recorded.
 * @param final_output (default false) If true, prints the accumulated times; if false, adds to the total time.
 */
void total_execution_times(double time_diff, const std::string &func_name, bool final_output)
{
    static std::map<std::string, double> execution_times;

    if (!final_output)
    {
        // Accumulate execution time (automatically initializes to 0.0 if key doesn't exist)
        if (execution_times.find(func_name) == execution_times.end())
        {
            execution_times[func_name] = 0.0;
        }
        execution_times[func_name] += time_diff;
    }
    else
    {
        // Print all accumulated times in scientific notation
        std::cout << "\n=== Total Execution Times ===\n";
        for (const auto &[name, total_time] : execution_times)
        {
            std::cout << std::setw(25) << std::left << name << ": " << std::scientific << std::setprecision(3)
                      << total_time << " s\n";
        }
        std::cout << "============================\n";
        execution_times.clear(); // Optional: clear after printing
    }
}
/**
 * @brief Enable core dumps and floating point exceptions
 *
 * This function enables core dumps and floating point exceptions by configuring system settings
 * and signal handlers.
 */
void enable_core_dumps_and_fpu_exceptions(void)
{
    struct rlimit rlim;
    extern int feenableexcept(int __excepts) __THROW;

    /* enable floating point exceptions */

    /*
       feenableexcept(FE_DIVBYZERO | FE_INVALID);
     */

    /* Note: FPU exceptions appear not to work properly
     * when the Intel C-Compiler for Linux is used
     */

    /* set core-dump size to infinity */
    getrlimit(RLIMIT_CORE, &rlim);
    rlim.rlim_cur = RLIM_INFINITY;
    setrlimit(RLIMIT_CORE, &rlim);

    signal(SIGSEGV, SIG_DFL);
    signal(SIGBUS, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
    signal(SIGINT, SIG_DFL);

    /* Establish a handler for SIGABRT signals. */
    signal(SIGABRT, catch_abort);
}

/**
 * @brief Signal handler for abort signals.
 *
 * This function finalizes the MPI environment using MPI_Finalize() and then exits the program
 * with a status code of 0.
 *
 * @param sig The signal number.
 */
void catch_abort(int sig)
{
    MPI_Finalize();
    exit(0);
}

/**
 * @brief Write the PID of each process to a file.
 *
 * This function writes the PID of each process to a file named "PIDs.txt" located in the output directory.
 * Each line in the file contains the hostname and PID of a process.
 */
void write_pid_file(void)
{
    char mode[8], buf[500];
    FILE *fd;

    pid_t my_pid = getpid();

    sprintf(buf, "%s%s", All.OutputDir, "PIDs.txt");

    if (RestartFlag == 0)
    {
        strcpy(mode, "w");
    }
    else
    {
        strcpy(mode, "a");
    }

    for (int i = 0; i < NTask; i++)
    {
        if (ThisTask == i)
        {
            if (ThisTask == 0)
            {
                sprintf(mode, "w");
            }
            else
            {
                sprintf(mode, "a");
            }

            if ((fd = fopen(buf, mode)))
            {
                fprintf(fd, "%s %d\n", getenv("HOST"), (int)my_pid);
                fclose(fd);
            }
        }
        MPI_Barrier(MYMPI_COMM_WORLD);
    }
}
#endif // DEBUG

#ifdef PAUSE_RUN_TO_ATTACH_DEBUGGER
/**
 * @brief Pause the simulation to allow attachment of a debugger.
 *
 * This function pauses the simulation execution to allow for the attachment of a debugger.
 * It retrieves the process ID (PID) and hostname of the current process and prints a message indicating that it is
 * ready for the debugger to attach.
 */
void pause_run_to_attach_debugger()
{
    int continue_run = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    int my_pid = getpid();
#if (PAUSE_RUN_TO_ATTACH_DEBUGGER < 0)
    if (!getenv("GADGET_PAUSE_RUN_TO_ATTACH_DEBUGGER"))
    {
        VERBOSE(2,
                "GADGET_PAUSE_RUN_TO_ATTACH_DEBUGGER environment variable unset: will not wait for debugger to attach. "
                "my_pid=%d, hostname=%s\n",
                my_pid, hostname);
        return;
    }
#endif // ( PAUSE_RUN_TO_ATTACH_DEBUGGER < 0 )
    VERBOSE_ALL(2, "PID %d on %s ready for debugger to attach.", my_pid, hostname);

    int *all_pids = NULL;
    if (ThisTask == 0)
    {
        all_pids = (int *)malloc(sizeof(int) * NTask);
    }

    MPI_Gather(&my_pid, 1, MPI_INT, all_pids, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    if (ThisTask == 0)
    {
        FILE *fd = fopen("pid_list_for_debugger.txt", "w");
        // check if file is open
        if (fd != NULL)
        {
            for (int j = 0; j < NTask; ++j)
            {
                fprintf(fd, "%i \n", all_pids[j]);
            }
            fclose(fd);
        }
        else
        {
            ENDRUN("PID file cant be opened.");
        }
        free(all_pids);
        VERBOSE_ALL(2, "PID file written");
    }

    while (continue_run ==
           0) /* continue_run needsto be set to 1 ("set var continue_run = 1") by debugger to continue */
    {
        sleep(1);
    }
}
#endif // PAUSE_RUN_TO_ATTACH_DEBUGGER

/*
double get_random_number(unsigned int id)
{
  return RndTable[(id % RNDTABLE)];
}
*/

/**
 * @brief Get a random number based on the given ID.
 *
 * This function retrieves a random number from a precomputed table based on the given ID.
 * The ID is used to index into the random number table to retrieve a pseudo-random number.
 *
 * @param id The ID used to generate the random number.
 * @return A pseudo-random number based on the given ID.
 */
double get_random_number(MyIDType id)
{
    return RndTable[(int)(id % RNDTABLE)];
}

/**
 * @brief Set pseudo-random numbers in the RndTable.
 *
 * This function populates the RndTable array with pseudo-random numbers generated using
 * random_generator (see System/rng.hpp).
 */
void set_random_numbers(void)
{
    for (int i = 0; i < RNDTABLE; i++)
    {
        RndTable[i] = my_rng_uniform(random_generator);
    }
}

/**
 * @brief Get the current wall-clock time in seconds.
 *
 * This function returns the current wall-clock time in seconds using MPI_Wtime() function.
 *
 * @return The current wall-clock time in seconds.
 *
 * @note On some systems like AIX and potentially others, the resolution of clock() function is limited to 10ms (0.01
 * seconds).
 * @note For permament time measurements use the availiable timer class.
 */
double second(void)
{
    return MPI_Wtime();
}

/**
 * @brief Report the elapsed time since the last call to this function.
 *
 * This function reports the elapsed time since the last call to this function. It is typically used to measure
 * sub-times of functions.
 *
 * @return The elapsed time in seconds since the last call to this function.
 *
 * @note The function uses the `second` function to obtain the current wall-clock time.
 * @note For permament time measurements use the availiable timer class.
 */
double report_time(void) /* strategy: call this to measure sub-times of functions */
{
    double t = second();
    double dt = t - WallclockTime;
    return dt;
}

/**
 * @brief Calculate the time difference between two measurements obtained with `second()`.
 *
 * This function calculates the time difference between two measurements obtained with the `second()` function. It takes
 * care of the possible overflow of the tick counter on 32-bit systems.
 *
 * @param t0 The start time measurement.
 * @param t1 The end time measurement.
 *
 * @return The time difference between the two measurements in seconds.
 *
 * @note If an overflow has occurred (for systems with a 32-bit tick counter), the function returns 0 to prevent
 * negative time differences.
 * @note For permament time measurements use the availiable timer class.
 */
double timediff(double t0, double t1)
{
    double dt = t1 - t0;
    if (dt < 0) /* overflow has occured (for systems with 32bit tick counter) */
    {
        dt = 0;
    }
    return dt;
}

#ifdef X86FIX

#define _FPU_SETCW(x) asm volatile("fldcw %0" : : "m"(x));
#define _FPU_GETCW(x) asm volatile("fnstcw %0" : "=m"(x));
#define _FPU_EXTENDED 0x0300
#define _FPU_DOUBLE 0x0200

/**
 * @brief Fix for x86 floating-point precision issue.
 *
 * This function applies a fix for x86 floating-point precision issues by adjusting the floating-point control word.
 */
void x86_fix(void)
{
    unsigned short dummy, new_cw;
    unsigned short *old_cw;

    old_cw = &dummy;

    _FPU_GETCW(*old_cw);
    new_cw = (*old_cw & ~_FPU_EXTENDED) | _FPU_DOUBLE;
    _FPU_SETCW(new_cw);
}
#endif // X86FIX

#ifdef ENLARGE_DYNAMIC_RANGE_IN_TIME
/**
 * @brief Find the minimum values of large integers across all MPI processes.
 *
 * This function finds the minimum values of large integers across all MPI processes. It gathers the integers from all
 * processes, compares them element-wise, and stores the minimum value for each element in the resulting array.
 *
 * @param n The number of elements in the array.
 * @param src Pointer to the array of large integers on the calling MPI process.
 * @param res Pointer to the array where the minimum values will be stored.
 */
void minimum_large_ints(int n, long long *src, long long *res)
{
    long long *numlist = (long long *)mymalloc("numlist", NTask * n * sizeof(long long));
    MPI_Allgather(src, n * sizeof(long long), MPI_BYTE, numlist, n * sizeof(long long), MPI_BYTE, MYMPI_COMM_WORLD);

    for (int j = 0; j < n; j++)
    {
        res[j] = src[j];
    }

    for (int i = 0; i < NTask; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (res[j] > numlist[i * n + j])
            {
                res[j] = numlist[i * n + j];
            }
        }
    }
    myfree(numlist);
}
#endif // ENLARGE_DYNAMIC_RANGE_IN_TIME
/**
 * @brief Sum up large integers across all MPI processes.
 *
 * This function sums up large integers across all MPI processes. It gathers the integers from all processes,
 * computes the sum element-wise, and stores the result in the resulting array.
 *
 * @param n The number of elements in the array.
 * @param src Pointer to the array of integers on the calling MPI process.
 * @param res Pointer to the array where the sum values will be stored.
 */
void sumup_large_ints(int n, int *src, long long *res)
{
    int *numlist;

    numlist = (int *)mymalloc("numlist", NTask * n * sizeof(int));
    MPI_Allgather(src, n, MPI_INT, numlist, n, MPI_INT, MYMPI_COMM_WORLD);

    for (int j = 0; j < n; j++)
    {
        res[j] = 0;
    }

    for (int i = 0; i < NTask; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res[j] += numlist[i * n + j];
        }
    }

    myfree(numlist);
}

/**
 * @brief Sum up long integers across all MPI processes.
 *
 * This function sums up long integers across all MPI processes. It gathers the integers from all processes,
 * computes the sum element-wise, and stores the result in the resulting array.
 *
 * @param n The number of elements in the array.
 * @param src Pointer to the array of long long on the calling MPI process.
 * @param res Pointer to the array where the sum values will be stored.
 */
void sumup_longs(int n, long long *src, long long *res)
{
    long long *numlist = (long long *)mymalloc("numlist", NTask * n * sizeof(long long));
    MPI_Allgather(src, n * sizeof(long long), MPI_BYTE, numlist, n * sizeof(long long), MPI_BYTE, MYMPI_COMM_WORLD);

    for (int j = 0; j < n; j++)
    {
        res[j] = 0;
    }

    for (int i = 0; i < NTask; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res[j] += numlist[i * n + j];
        }
    }
    myfree(numlist);
}

/**
 * @brief Returns the maximum of two size_t values.
 *
 * This function compares two size_t values and returns the larger of the two.
 *
 * @param a The first size_t value to compare.
 * @param b The second size_t value to compare.
 * @return The maximum of the two size_t values.
 */
size_t sizemax(size_t a, size_t b)
{
    if (a < b)
    {
        return b;
    }
    else
    {
        return a;
    }
}

/**
 * @brief Reports the amount of committable memory available.
 *
 * This function reads system memory information from the "/proc/meminfo" file
 * to determine the total memory, committed memory, total swap, and free swap.
 * It calculates and returns the amount of committable memory, which is the
 * difference between the total memory and the committed memory.
 *
 * @param MemTotal Pointer to a variable to store the total system memory.
 * @param Committed_AS Pointer to a variable to store the amount of committed memory.
 * @param SwapTotal Pointer to a variable to store the total swap memory.
 * @param SwapFree Pointer to a variable to store the free swap memory.
 * @return The amount of committable memory (total memory - committed memory).
 */
long long report_comittable_memory(long long *MemTotal, long long *Committed_AS, long long *SwapTotal,
                                   long long *SwapFree)
{
    FILE *fd;
    char buf[1024];

    if ((fd = fopen("/proc/meminfo", "r")))
    {
        while (1)
        {
            if (fgets(buf, 500, fd) != buf)
            {
                break;
            }

            if (bcmp(buf, "MemTotal", 8) == 0)
            {
                *MemTotal = atoll(buf + 10);
            }
            if (strncmp(buf, "Committed_AS", 12) == 0)
            {
                *Committed_AS = atoll(buf + 14);
            }
            if (strncmp(buf, "SwapTotal", 9) == 0)
            {
                *SwapTotal = atoll(buf + 11);
            }
            if (strncmp(buf, "SwapFree", 8) == 0)
            {
                *SwapFree = atoll(buf + 10);
            }
        }
        fclose(fd);
    }

    return (*MemTotal - *Committed_AS);
}

int FullestTask, EmptiestTask;
/**
 * @brief Reports various memory statistics across MPI tasks.
 *
 * This function calculates and reports various memory statistics, including
 * available memory, total memory, committed memory, total swap, and free swap,
 * across MPI tasks. It uses MPI_Allgather to gather memory information from
 * all MPI tasks and then calculates the largest, smallest, and average memory
 * sizes for each statistic.
 */
void mpi_report_comittable_memory(void)
{
    long long *sizelist, maxsize[6], minsize[6];
    double avgsize[6];
    int mintask[6], maxtask[6];
    long long Mem[6];
    char label[512];

    Mem[0] = report_comittable_memory(&Mem[1], &Mem[2], &Mem[3], &Mem[4]);
    Mem[5] = Mem[1] - Mem[0];

    for (int imem = 0; imem < 6; imem++)
    {
        sizelist = (long long *)malloc(NTask * sizeof(long long));
        MPI_Allgather(&Mem[imem], sizeof(long long), MPI_BYTE, sizelist, sizeof(long long), MPI_BYTE, MPI_COMM_WORLD);
        int i;
        for (i = 1, mintask[imem] = 0, maxtask[imem] = 0, maxsize[imem] = minsize[imem] = sizelist[0],
            avgsize[imem] = sizelist[0];
             i < NTask; i++)
        {
            if (sizelist[i] > maxsize[imem])
            {
                maxsize[imem] = sizelist[i];
                maxtask[imem] = i;
            }
            if (sizelist[i] < minsize[imem])
            {
                minsize[imem] = sizelist[i];
                mintask[imem] = i;
            }
            avgsize[imem] += sizelist[i];
        }

        free(sizelist);
    }

    if (ThisTask == 0)
    {
        VERBOSE(2, "___________________________________________________________________________________________________"
                   "________________");
        for (int imem = 0; imem < 6; imem++)
        {
            switch (imem)
            {
            case 0:
                sprintf(label, "AvailMem");
                break;
            case 1:
                sprintf(label, "Total Mem");
                break;
            case 2:
                sprintf(label, "Committed_AS");
                break;
            case 3:
                sprintf(label, "SwapTotal");
                break;
            case 4:
                sprintf(label, "SwapFree");
                break;
            case 5:
                sprintf(label, "AllocMem");
                break;
            }
            VERBOSE(2, "%s:\t Largest = %10.2f Mb (on task=%d), Smallest = %10.2f Mb (on task=%d), Average = %10.2f Mb",
                    label, maxsize[imem] / (1024.0), maxtask[imem], minsize[imem] / (1024.0), mintask[imem],
                    avgsize[imem] / (1024.0 * NTask));
        }
        VERBOSE(2, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                   "~~~~~~~~~~~~~~~~");
    }

    FullestTask = maxtask[2];
    EmptiestTask = mintask[2];

    if (ThisTask == maxtask[2])
    {
        char name[MPI_MAX_PROCESSOR_NAME];
        int len;
        MPI_Get_processor_name(name, &len);
        VERBOSE_ALL(2, "Task=%d has the maximum commited memory and is host: %s", ThisTask, name);
        VERBOSE_ALL(2, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                       "~~~~~~~~~~~~~~~~~~~~");
    }
}

/**
 * @brief Reports the memory usage of the current process.
 *
 * This function reports the memory usage of the current process by reading the "/proc/meminfo" file.
 */
void report_task_list(void)
{
    if (ThisTask == FullestTask)
    {
        char name[MPI_MAX_PROCESSOR_NAME];
        int len;
        MPI_Get_processor_name(name, &len);

        char command[10000];
        sprintf(command, "top -b -n 1 -o RES > processes_largest_memuse_%d_%s.txt", ThisTask, name);
        (void)system(command);
        sprintf(command, "free > free_largest_memuse_%d_%s.txt", ThisTask, name);
        (void)system(command);
        sprintf(command, " ps -eo pid,pmem,rss,vsize,comm --sort vsize > ps_largest_memuse_%d_%s.txt", ThisTask, name);
        (void)system(command);
    }

    if (ThisTask == EmptiestTask)
    {
        char name[MPI_MAX_PROCESSOR_NAME];
        int len;
        MPI_Get_processor_name(name, &len);

        char command[10000];
        sprintf(command, "top -b -n 1 -o RES > processes_smallest_memuse_%d_%s.txt", ThisTask, name);
        (void)system(command);
        sprintf(command, "free > free_smallest_memuse_%d_%s.txt", ThisTask, name);
        (void)system(command);
        sprintf(command, " ps -eo pid,pmem,rss,vsize,comm --sort vsize > ps_smallest_memuse_%d_%s.txt", ThisTask, name);
        (void)system(command);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);
}

/**
 * @brief Write the process ID (PID) of each process to a file.
 *
 * This function writes the process ID (PID) of each process to a file named "PIDs.txt".
 * It also retrieves memory information for each process and writes it to a separate file.
 */
void write_ps_files(void)
{
    long long pid;
    FILE *fd, *fd2, *ofd;
    char buf[1024], buf2[1024];

    char command[10000], filename[200], memfilename[200], outfile[200];

    if (ThisTask == FullestTask || ThisTask == EmptiestTask)
    {
        sprintf(filename, "ps_file_%i.txt", ThisTask);
        sprintf(command, "ps -eo pid > %s", filename);
        (void)system(command);

        if (ThisTask == FullestTask)
        {
            sprintf(outfile, "memory_largest_%i.txt", ThisTask);
        }

        if (ThisTask == EmptiestTask)
        {
            sprintf(outfile, "memory_smallest_%i.txt", ThisTask);
        }
        if (!(ofd = fopen(outfile, "w")))
        {
            ENDRUN("Cant open file.");
        }

        if ((fd = fopen(filename, "r")))
        {
            fgets(buf, 500, fd);

            while (1)
            {
                if (fgets(buf, 500, fd) != buf)
                {
                    break;
                }
                pid = atoll(buf);

                sprintf(memfilename, "/proc/%lld/status", pid);

                if ((fd2 = fopen(memfilename, "r")))
                {
                    while (1)
                    {
                        if (fgets(buf2, 500, fd2) != buf2)
                        {
                            break;
                        }

                        if (strncmp(buf2, "VmRSS", 5) == 0)
                        {
                            fprintf(ofd, "PID %lld: %s \n", pid, buf2);
                        }
                        if (strncmp(buf2, "VmSize", 6) == 0)
                        {
                            fprintf(ofd, "PID %lld: %s \n", pid, buf2);
                        }
                    }
                    fclose(fd2);
                }
            }
            fclose(fd);
        }
        fclose(ofd);
    }
}
