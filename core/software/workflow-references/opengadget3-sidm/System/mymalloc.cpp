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

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#define MAXBLOCKS 500
#define MAXCHARS 16

static size_t TotBytes;
#ifndef USE_CALLOC
static void *Base;
static void ***BasePointers;
#endif // USE_CALLOC
static unsigned long Nblocks;

static void **Table;
static size_t *BlockSize;
static char *MovableFlag;

static char *VarName;
static char *FunctionName;
static char *FileName;
static int *LineNumber;

/**
 * @brief Initializes memory allocation for custom memory management.
 *
 * This function initializes memory allocation for custom memory management.
 * It allocates memory for internal data structures
 * and sets up the initial state for memory allocation.
 */
void mymalloc_init(void)
{
    size_t n;

    BlockSize = (size_t *)malloc(MAXBLOCKS * sizeof(size_t));
    Table = (void **)malloc(MAXBLOCKS * sizeof(void *));
    MovableFlag = (char *)malloc(MAXBLOCKS * sizeof(char));
#ifndef USE_CALLOC
    BasePointers = (void ***)malloc(MAXBLOCKS * sizeof(void **));
#endif // USE_CALLOC
    VarName = (char *)malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
    FunctionName = (char *)malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
    FileName = (char *)malloc(MAXBLOCKS * MAXCHARS * sizeof(char));
    LineNumber = (int *)malloc(MAXBLOCKS * sizeof(int));

    memset(VarName, 0, MAXBLOCKS * MAXCHARS);
    memset(FunctionName, 0, MAXBLOCKS * MAXCHARS);
    memset(FileName, 0, MAXBLOCKS * MAXCHARS);

    n = All.MaxMemSize * ((size_t)1024 * 1024);
    VERBOSE_ALL(5, "Task %d: dofftw=%d", ThisTask, dofftw);

    n -= MAXBLOCKS *
         (sizeof(size_t) + sizeof(void *) + sizeof(char) + sizeof(void **) + 3 * MAXCHARS * sizeof(char) + sizeof(int));

#ifndef USE_CALLOC
    while (!(Base = malloc(n)) && n > All.MaxMemSize * ((size_t)1024 * 1024 / 2))
    {
        VERBOSE_ALL(0, "Task %d: Failed to allocate memory for `Base' (%d Mbytes).", ThisTask,
                    (int)(n / ((size_t)1024 * 1024)));
        n = n - ((size_t)50 * 1024 * 1024);
    }

    if (n <= All.MaxMemSize * ((size_t)1024 * 1024 / 2))
    {
        ENDRUN("Task %d: Gave up to allocate memory for `Base' (%d Mbytes).", ThisTask,
               (int)(n / ((size_t)1024 * 1024)));
    }
#endif // USE_CALLOC

    TotBytes = FreeBytes = n;

    AllocatedBytes = 0;
    Nblocks = 0;
    HighMarkBytes = 0;
}

/**
 * @brief Print memory table.
 *
 * This function prints a formatted memory table showing allocated memory blocks for each task. The table includes
 * information such as task number, block number, variable name, memory size in megabytes, cumulative memory size, and
 * the corresponding function, file, and line number where the memory was allocated. The memory table provides a
 * detailed overview of allocated memory blocks across all tasks, helping to track memory usage and identify potential
 * memory leaks or inefficiencies.
 */
void dump_memory_table()
{
    size_t totBlocksize = 0;

    VERBOSE_ALL(0, "------------------------ Allocated Memory Blocks----------------------------------------");
    VERBOSE_ALL(0, "Task   Nr F          Variable      MBytes   Cumulative         Function/File/Linenumber");
    VERBOSE_ALL(0, "----------------------------------------------------------------------------------------");
    for (unsigned int i = 0; i < Nblocks; i++)
    {
        totBlocksize += BlockSize[i];

        VERBOSE_ALL(0, "%4d %4d %d  %16s  %10.4f   %10.4f  %s()/%s/%d", ThisTask, i, MovableFlag[i],
                    VarName + i * MAXCHARS, BlockSize[i] / (1024.0 * 1024.0), totBlocksize / (1024.0 * 1024.0),
                    FunctionName + i * MAXCHARS, FileName + i * MAXCHARS, LineNumber[i]);
    }
    VERBOSE_ALL(0, "----------------------------------------------------------------------------------------");
}

#ifndef DISABLE_MEMORY_MANAGER
/**
 * @brief Reports detailed memory usage of the largest task in the MPI communicator.
 *
 * This function reports detailed memory usage of the largest task in the MPI communicator. It gathers memory usage
 * information from all tasks, identifies the task with the largest memory allocation, and compares it with the previous
 * high mark. If the memory allocation of the largest task exceeds a certain threshold relative to the previous high
 * mark, detailed memory usage statistics are printed for that task, including the largest allocation size, smallest
 * allocation size, and average allocation size across all tasks. Additionally, the function optionally dumps the memory
 * table to provide further insights into memory usage. This function is typically called periodically to monitor memory
 * usage and identify potential memory issues.
 *
 * @param OldHighMarkBytes Pointer to the previous high mark of memory allocation.
 * @param label            A label indicating the context or stage where memory usage is being reported.
 * @param func             The name of the function where memory usage is being reported.
 * @param file             The name of the source file where memory usage is being reported.
 * @param line             The line number in the source file where memory usage is being reported.
 */
void report_detailed_memory_usage_of_largest_task(size_t *OldHighMarkBytes, const char *label, const char *func,
                                                  const char *file, int line)
{
    size_t *sizelist, maxsize, minsize;
    double avgsize;
    int i, task;

    sizelist = (size_t *)mymalloc("sizelist", NTask * sizeof(size_t));
    MPI_Allgather(&AllocatedBytes, sizeof(size_t), MPI_BYTE, sizelist, sizeof(size_t), MPI_BYTE, MYMPI_COMM_WORLD);

    for (i = 1, task = 0, maxsize = minsize = sizelist[0], avgsize = sizelist[0]; i < NTask; i++)
    {
        if (sizelist[i] > maxsize)
        {
            maxsize = sizelist[i];
            task = i;
        }
        if (sizelist[i] < minsize)
        {
            minsize = sizelist[i];
        }
        avgsize += sizelist[i];
    }

    myfree(sizelist);

    if (maxsize > 1.1 * (*OldHighMarkBytes))
    {
        *OldHighMarkBytes = maxsize;
        avgsize /= NTask;

        if (ThisTask == task)
        {
            VERBOSE_ALL(2,
                        "\nAt '%s', %s()/%s/%d: Largest Allocation = %g Mbyte (on task=%d), Smallest = %g Mbyte, "
                        "Average = %g Mbyte",
                        label, func, file, line, maxsize / (1024.0 * 1024.0), task, minsize / (1024.0 * 1024.0),
                        avgsize / (1024.0 * 1024.0));
            if constexpr (Verbose_Level <= 2)
            {
                dump_memory_table();
            }
        }
    }
    MPI_Barrier(MYMPI_COMM_WORLD);
}

/**
 * @brief Allocate memory with full information.
 *
 * This function allocates memory of size 'n' bytes and records detailed information such as variable name, function
 * name, file name, and line number where the allocation was made. It also performs checks for available memory and
 * ensures alignment to a multiple of 64 bytes. If there is not enough memory available or the maximum number of blocks
 * has been reached, the function prints an error message and terminates the program. After successful allocation, the
 * function updates the memory table and returns a pointer to the allocated memory block.
 *
 * @param varname Name of the variable being allocated.
 * @param n Size of memory to allocate (in bytes).
 * @param func Name of the function where allocation is requested.
 * @param file Name of the file where allocation is requested.
 * @param line Line number in the file where allocation is requested.
 * @return Pointer to the allocated memory block.
 */
void *mymalloc_fullinfo(const char *varname, size_t n, const char *func, const char *file, int line)
{
    // VERBOSE(-1,"%s %d %s %s %d", varname, n, func, file, line);
    if ((n % 64) > 0)
    {
        n = (n / 64 + 1) * 64;
    }

    if (n < 64)
    {
        n = 64;
    }

    if (Nblocks >= MAXBLOCKS)
    {
        ENDRUN("Task=%d: No blocks left in mymalloc_fullinfo() at %s()/%s/line %d. MAXBLOCKS=%d", ThisTask, func, file,
               line, MAXBLOCKS);
    }

    if (n > FreeBytes)
    {
        dump_memory_table();
        ENDRUN("\nTask=%d: Not enough memory in mymalloc_fullinfo() to allocate %g MB for variable '%s' at "
               "%s()/%s/line %d (FreeBytes=%g MB).",
               ThisTask, n / (1024.0 * 1024.0), varname, func, file, line, FreeBytes / (1024.0 * 1024.0));
    }
#ifndef USE_CALLOC
    Table[Nblocks] = (char *)Base + (TotBytes - FreeBytes);
#else // USE_CALLOC
    Table[Nblocks] = (char *)calloc(n, 1);
#endif // USE_CALLOC
    FreeBytes -= n;

    strncpy(VarName + Nblocks * MAXCHARS, varname, MAXCHARS - 1);
    strncpy(FunctionName + Nblocks * MAXCHARS, func, MAXCHARS - 1);
    strncpy(FileName + Nblocks * MAXCHARS, file, MAXCHARS - 1);
    LineNumber[Nblocks] = line;

    AllocatedBytes += n;
    BlockSize[Nblocks] = n;
#ifndef USE_CALLOC
    MovableFlag[Nblocks] = 0;
#else // USE_CALLOC
    MovableFlag[Nblocks] = 1;
#endif // USE_CALLOC
    Nblocks += 1;

    if (AllocatedBytes > HighMarkBytes)
    {
        HighMarkBytes = AllocatedBytes;
    }

    return Table[Nblocks - 1];
}

/**
 * @brief Allocate movable memory with full information.
 *
 * This function allocates movable memory of size 'n' bytes and records detailed information such as variable name,
 * function name, file name, and line number where the allocation was made. It also performs checks for available memory
 * and ensures alignment to a multiple of 64 bytes. If there is not enough memory available or the maximum number of
 * blocks has been reached, the function prints an error message and terminates the program. After successful
 * allocation, the function updates the memory table, sets the movable flag, and returns a pointer to the allocated
 * memory block.
 *
 * @param ptr Pointer to the memory block that can be moved.
 * @param varname Name of the variable being allocated.
 * @param n Size of memory to allocate (in bytes).
 * @param func Name of the function where allocation is requested.
 * @param file Name of the file where allocation is requested.
 * @param line Line number in the file where allocation is requested.
 * @return Pointer to the allocated memory block.
 */
void *mymalloc_movable_fullinfo(void *ptr, const char *varname, size_t n, const char *func, const char *file, int line)
{
    if ((n % 64) > 0)
    {
        n = (n / 64 + 1) * 64;
    }

    if (n < 64)
    {
        n = 64;
    }

    if (Nblocks >= MAXBLOCKS)
    {
        ENDRUN("Task=%d: No blocks left in mymalloc_fullinfo() at %s()/%s/line %d. MAXBLOCKS=%d", ThisTask, func, file,
               line, MAXBLOCKS);
    }

    if (n > FreeBytes)
    {
        dump_memory_table();
        ENDRUN("\nTask=%d: Not enough memory in mymalloc_fullinfo() to allocate %g MB for variable '%s' at "
               "%s()/%s/line %d (FreeBytes=%g MB).",
               ThisTask, n / (1024.0 * 1024.0), varname, func, file, line, FreeBytes / (1024.0 * 1024.0));
    }
#ifndef USE_CALLOC
    Table[Nblocks] = (char *)Base + (TotBytes - FreeBytes);
#else // USE_CALLOC
    Table[Nblocks] = (char *)calloc(n, 1);
#endif // USE_CALLOC
    FreeBytes -= n;

    strncpy(VarName + Nblocks * MAXCHARS, varname, MAXCHARS - 1);
    strncpy(FunctionName + Nblocks * MAXCHARS, func, MAXCHARS - 1);
    strncpy(FileName + Nblocks * MAXCHARS, file, MAXCHARS - 1);
    LineNumber[Nblocks] = line;

    AllocatedBytes += n;
    BlockSize[Nblocks] = n;
    MovableFlag[Nblocks] = 1;
#ifndef USE_CALLOC
    BasePointers[Nblocks] = (void **)ptr;
#endif // USE_CALLOC
    Nblocks += 1;

    if (AllocatedBytes > HighMarkBytes)
    {
        HighMarkBytes = AllocatedBytes;
    }

    return Table[Nblocks - 1];
}

/**
 * @brief Free movable memory with full information.
 *
 * This function frees movable memory allocated using mymalloc_movable_fullinfo. It checks if the specified pointer 'p'
 * corresponds to the last allocated memory block in the memory table. If not, it prints an error message and terminates
 * the program. After successful deallocation, it updates memory allocation statistics and frees the memory block.
 *
 * @param p Pointer to the memory block to be freed.
 * @param func Name of the function where deallocation is requested.
 * @param file Name of the file where deallocation is requested.
 * @param line Line number in the file where deallocation is requested.
 */
void myfree_fullinfo(void *p, const char *func, const char *file, int line)
{
    // VERBOSE(-1, "%s %s %d", func, file , line);
    ENDRUN_IF(Nblocks == 0, "Number of blocks is 0");

    if (p != Table[Nblocks - 1])
    {
        dump_memory_table();
        ENDRUN("Task=%d: Wrong call of myfree() at %s()/%s/line %d: not the last allocated block!", ThisTask, func,
               file, line);
    }

    Nblocks -= 1;
    AllocatedBytes -= BlockSize[Nblocks];
    FreeBytes += BlockSize[Nblocks];
#ifdef USE_CALLOC
    free(p);
#endif // USE_CALLOC
}

/**
 * @brief Free movable memory with full information.
 *
 * This function frees movable memory allocated using mymalloc_movable_fullinfo. It first finds the block corresponding
 * to the specified pointer 'p'. Then, it checks if the block is not the last allocated block and whether all subsequent
 * blocks are movable. If not, it prints an error message and terminates the program. After that, it updates memory
 * allocation statistics, adjusts memory pointers, and moves memory blocks if necessary. Finally, it updates the memory
 * table and decreases the number of allocated blocks.
 *
 * @param p Pointer to the memory block to be freed.
 * @param func Name of the function where deallocation is requested.
 * @param file Name of the file where deallocation is requested.
 * @param line Line number in the file where deallocation is requested.
 */
void myfree_movable_fullinfo(void *p, const char *func, const char *file, int line)
{
    ENDRUN_IF(Nblocks == 0, "Number of blocks is 0.");

    if (p == Table[Nblocks - 1])
    {
        Nblocks -= 1;
        AllocatedBytes -= BlockSize[Nblocks];
        FreeBytes += BlockSize[Nblocks];
#ifdef USE_CALLOC
        free(p);
#endif // USE_CALLOC
        return;
    }
    else
    {

        /* first, let's find the block */
        unsigned int nr;
        for (nr = Nblocks - 1; nr >= 0; nr--)
        {
            if (p == Table[nr])
            {
                break;
            }
        }

        if (nr < Nblocks - 1) /* the block is not the last allocated block */
        {
            /* check that all subsequent blocks are actually movable */
            for (unsigned int i = nr + 1; i < Nblocks; i++)
            {
                if (MovableFlag[i] == 0)
                {
                    dump_memory_table();
                    ENDRUN("Task=%d: Wrong call of myfree_movable() from %s()/%s/line %d - behind block=%d there are "
                           "subsequent non-movable allocated blocks",
                           ThisTask, func, file, line, nr);
                }
            }
        }

        AllocatedBytes -= BlockSize[nr];
        FreeBytes += BlockSize[nr];

#ifndef USE_CALLOC
        size_t offset = -BlockSize[nr];
        size_t length = 0;

        for (unsigned int i = nr + 1; i < Nblocks; i++)
        {
            length += (size_t)BlockSize[i];
        }

        if (nr < Nblocks - 1)
        {
            memmove((char *)Table[nr + 1] + offset, Table[nr + 1], length);
        }

        for (unsigned int i = nr + 1; i < Nblocks; i++)
        {
            Table[i] = (char *)Table[i] + offset;
            *BasePointers[i] = (char *)*BasePointers[i] + offset;
        }
#endif // USE_CALLOC

        for (unsigned int i = nr + 1; i < Nblocks; i++)
        {
            Table[i - 1] = Table[i];
#ifndef USE_CALLOC
            BasePointers[i - 1] = BasePointers[i];
#endif // USE_CALLOC
            BlockSize[i - 1] = BlockSize[i];
            MovableFlag[i - 1] = MovableFlag[i];

            memcpy(VarName + (i - 1) * MAXCHARS, VarName + i * MAXCHARS, MAXCHARS - 1);
            memcpy(FunctionName + (i - 1) * MAXCHARS, FunctionName + i * MAXCHARS, MAXCHARS - 1);
            memcpy(FileName + (i - 1) * MAXCHARS, FileName + i * MAXCHARS, MAXCHARS - 1);
            LineNumber[i - 1] = LineNumber[i];
        }

        Nblocks -= 1;
#ifdef USE_CALLOC
        free(p);
#endif // USE_CALLOC
    }
}

/**
 * @brief Reallocate memory with full information.
 *
 * This function reallocates memory previously allocated using mymalloc_fullinfo or mymalloc_movable_fullinfo. It first
 * checks if the specified pointer 'p' points to the last allocated block. If not, it prints an error message and
 * terminates the program. Then, it updates memory allocation statistics, adjusts memory pointers, and checks if there
 * is enough free memory to perform reallocation. If not, it prints an error message and terminates the program.
 * Finally, it updates the memory table and returns the pointer to the reallocated memory block.
 *
 * @param p Pointer to the memory block to be reallocated.
 * @param n Size of the new memory block.
 * @param func Name of the function where reallocation is requested.
 * @param file Name of the file where reallocation is requested.
 * @param line Line number in the file where reallocation is requested.
 * @return Pointer to the reallocated memory block.
 */
void *myrealloc_fullinfo(void *p, size_t n, const char *func, const char *file, int line)
{
    if ((n % 64) > 0)
    {
        n = (n / 64 + 1) * 64;
    }

    if (n < 64)
    {
        n = 64;
    }

    ENDRUN_IF(Nblocks == 0, "Number of blocks is 0");

    if (p != Table[Nblocks - 1])
    {
        dump_memory_table();
        ENDRUN("Task=%d: Wrong call of myrealloc() at %s()/%s/line %d - not the last allocated block!", ThisTask, func,
               file, line);
    }

    AllocatedBytes -= BlockSize[Nblocks - 1];
    FreeBytes += BlockSize[Nblocks - 1];

    if (n > FreeBytes)
    {
        dump_memory_table();
        ENDRUN("Task=%d: Not enough memory in myremalloc(n=%g MB) at %s()/%s/line %d. previous=%g FreeBytes=%g MB",
               ThisTask, n / (1024.0 * 1024.0), func, file, line, BlockSize[Nblocks - 1] / (1024.0 * 1024.0),
               FreeBytes / (1024.0 * 1024.0));
    }
#ifndef USE_CALLOC
    Table[Nblocks - 1] = (char *)Base + (TotBytes - FreeBytes);
#else // USE_CALLOC
    free(p);
    Table[Nblocks - 1] = (char *)calloc(n, 1);
#endif // USE_CALLOC
    FreeBytes -= n;

    AllocatedBytes += n;
    BlockSize[Nblocks - 1] = n;

    if (AllocatedBytes > HighMarkBytes)
    {
        HighMarkBytes = AllocatedBytes;
    }

    return Table[Nblocks - 1];
}

/**
 * @brief Reallocate movable memory with full information.
 *
 * This function reallocates movable memory previously allocated using mymalloc_movable_fullinfo. It first finds the
 * block corresponding to the specified pointer 'p'. Then, it checks if the subsequent blocks are movable. If not, it
 * prints an error message and terminates the program. Next, it updates memory allocation statistics, adjusts memory
 * pointers, and checks if there is enough free memory to perform reallocation. If not, it prints an error message and
 * terminates the program. Finally, it updates the memory table and returns the pointer to the reallocated memory block.
 *
 * @param p Pointer to the memory block to be reallocated.
 * @param n Size of the new memory block.
 * @param func Name of the function where reallocation is requested.
 * @param file Name of the file where reallocation is requested.
 * @param line Line number in the file where reallocation is requested.
 * @return Pointer to the reallocated memory block.
 */
void *myrealloc_movable_fullinfo(void *p, size_t n, const char *func, const char *file, int line)
{
    if ((n % 64) > 0)
    {
        n = (n / 64 + 1) * 64;
    }

    if (n < 64)
    {
        n = 64;
    }

    ENDRUN_IF(Nblocks == 0, "Number of blocks is 0");

    /* first, let's find the block */
    int nr;

    for (nr = Nblocks - 1; nr >= 0; nr--)
    {
        if (p == Table[nr])
        {
            break;
        }
    }

    if (nr < 0)
    {
        dump_memory_table();
        ENDRUN("Task=%d: Wrong call of myrealloc_movable() from %s()/%s/line %d - this block has not been allocated!",
               ThisTask, func, file, line);
    }

    if (nr < Nblocks - 1) /* the block is not the last allocated block */
    {
        /* check that all subsequent blocks are actually movable */
        for (unsigned int i = nr + 1; i < Nblocks; i++)
        {
            if (MovableFlag[i] == 0)
            {
                dump_memory_table();
                ENDRUN("Task=%d: Wrong call of myrealloc_movable() from %s()/%s/line %d - behind block=%d there "
                       "are subsequent non-movable allocated blocks",
                       ThisTask, func, file, line, nr);
            }
        }
    }

    AllocatedBytes -= BlockSize[nr];
    FreeBytes += BlockSize[nr];

    if (n > FreeBytes)
    {
        dump_memory_table();
        ENDRUN("Task=%d: at %s()/%s/line %d: Not enough memory in myremalloc_movable(n=%g MB). previous=%g "
               "FreeBytes=%g MB",
               ThisTask, func, file, line, n / (1024.0 * 1024.0), BlockSize[nr] / (1024.0 * 1024.0),
               FreeBytes / (1024.0 * 1024.0));
    }

#ifndef USE_CALLOC
    size_t offset = n - BlockSize[nr];
    size_t length = 0;

    for (unsigned int i = nr + 1; i < Nblocks; i++)
    {
        length += BlockSize[i];
    }

    if (nr < Nblocks - 1)
    {
        memmove((char *)Table[nr + 1] + offset, Table[nr + 1], length);
    }

    for (unsigned int i = nr + 1; i < Nblocks; i++)
    {
        Table[i] = (char *)Table[i] + offset;
        *BasePointers[i] = (char *)*BasePointers[i] + offset;
    }

    FreeBytes -= n;
    AllocatedBytes += n;
    BlockSize[nr] = n;
#else // USE_CALLOC
    ENDRUN("Task=%d: at %s()/%s/line %d: ReallocMovable with CALLOC not yet implemented", ThisTask, func, file, line);
#endif // USE_CALLOC

    if (AllocatedBytes > HighMarkBytes)
    {
        HighMarkBytes = AllocatedBytes;
    }

    return Table[nr];
}

#endif // DISABLE_MEMORY_MANAGER
