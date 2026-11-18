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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>
#include <algorithm>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#define TRANSFER_SIZE_LIMIT 1000000000

#define TAG_TRANSFER 100

/* #define CHECK_LOCAL_RANK
 */
static void msort_serial_with_tmp(sort_type *base, size_t n, size_t s, int (*compar)(const void *, const void *),
                                  sort_type *t, int level, int maxlevel);

static void get_local_rank(char *element, size_t tie_braking_rank, char *base, size_t nmemb, size_t size,
                           size_t noffs_thistask, long long left, long long right, size_t *loc,
                           int (*compar)(const void *, const void *));

#ifdef CHECK_LOCAL_RANK
static void check_local_rank(char *element, size_t tie_braking_rank, char *base, size_t nmemb, size_t size,
                             size_t noffs_thistask, long long left, long long right, size_t loc,
                             int (*compar)(const void *, const void *));
#endif // CHECK_LOCAL_RANK

/**
 * @brief Perform parallel sorting using MPI and OpenMP parallelism.
 *
 * This function performs parallel sorting using MPI and OpenMP parallelism. It divides the sorting process
 * among multiple MPI tasks, each potentially running on multiple threads. The sorting is based on a comparison
 * function provided by the user. The function first determines a communicator that contains only tasks with
 * non-zero data, making it easier to handle CPUs without any data. It then performs a serial sort if the
 * number of elements is non-zero, and splits the data among tasks for parallel sorting if necessary.
 * The sorting process involves finding split points for data distribution, exchanging data among tasks,
 * and performing a final serial sort on the merged data.
 *
 * @param base Pointer to the base address of the array to be sorted.
 * @param nmemb Number of elements in the array.
 * @param size Size of each element in the array.
 * @param compar Pointer to the comparison function used for sorting.
 */
void parallel_sort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
    int max_task, ranks_not_found, Local_ThisTask, Local_NTask, Local_PTask, Color;
    MPI_Comm MPI_CommLocal;

    /* we create a communicator that contains just those tasks with nmemb > 0. This makes
     *  it easier to deal with CPUs that do not hold any data.
     */

    if (nmemb)
    {
        Color = 1;
        serial_sort((char *)base, nmemb, size, compar);
    }
    else
    {
        Color = 0;
    }

    MPI_Comm_split(MYMPI_COMM_WORLD, Color, 0, &MPI_CommLocal);
    MPI_Comm_rank(MPI_CommLocal, &Local_ThisTask);
    MPI_Comm_size(MPI_CommLocal, &Local_NTask);

    if (Local_NTask > 1 && Color == 1)
    {
        for (Local_PTask = 0; Local_NTask > (1 << Local_PTask); Local_PTask++)
            ;

        size_t *nlist = (size_t *)mymalloc("nlist", Local_NTask * sizeof(size_t));
        size_t *noffs = (size_t *)mymalloc("noffs", Local_NTask * sizeof(size_t));

        MPI_Allgather(&nmemb, sizeof(size_t), MPI_BYTE, nlist, sizeof(size_t), MPI_BYTE, MPI_CommLocal);

        noffs[0] = 0;
        for (int i = 1; i < Local_NTask; i++)
        {
            noffs[i] = noffs[i - 1] + nlist[i - 1];
        }

        char *element_guess = (char *)mymalloc("element_guess", (Local_NTask - 1) * size);
        size_t *element_tie_braking_rank =
            (size_t *)mymalloc("element_tie_braking_rank", (Local_NTask - 1) * sizeof(size_t));
        size_t *desired_glob_rank = (size_t *)mymalloc("desired_glob_rank", (Local_NTask - 1) * sizeof(size_t));
        size_t *current_glob_rank = (size_t *)mymalloc("current_glob_rank", (Local_NTask - 1) * sizeof(size_t));
        size_t *current_loc_rank = (size_t *)mymalloc("current_loc_rank", (Local_NTask - 1) * sizeof(size_t));
        long long *range_left = (long long *)mymalloc("range_left", (Local_NTask - 1) * sizeof(long long));
        long long *range_right = (long long *)mymalloc("range_right", (Local_NTask - 1) * sizeof(long long));
        int *max_loc = (int *)mymalloc("max_loc", (Local_NTask - 1) * sizeof(int));
        size_t max_nmemb;

        /* find the largest nmemb value and initialize the guesses with the median element found there */
        max_task = max_nmemb = 0;
        for (int i = 0; i < Local_NTask; i++)
        {
            if (max_nmemb < nlist[i])
            {
                max_nmemb = nlist[i];
                max_task = i;
            }
        }

        if (Local_ThisTask == max_task)
        {
            memcpy(element_guess, (char *)base + size * (max_nmemb / 2), size);
        }
        MPI_Bcast(element_guess, size, MPI_BYTE, max_task, MPI_CommLocal);

        for (int i = 1; i < Local_NTask - 1; i++)
        {
            memcpy(element_guess + i * size, element_guess, size);
        }
        for (int i = 0; i < Local_NTask - 1; i++)
        {
            desired_glob_rank[i] = noffs[i + 1];
            element_tie_braking_rank[i] = (max_nmemb / 2) + noffs[max_task];
            max_loc[i] = max_task;

            range_left[i] = 0; /* first element that it can be */
            range_right[i] = nmemb; /* first element that it can not be */

            current_glob_rank[i] = 0;
        }
        int iter = 0;

        do
        {
            for (int i = 0; i < Local_NTask - 1; i++)
            {
                if (current_glob_rank[i] != desired_glob_rank[i])
                {
                    get_local_rank(element_guess + i * size, element_tie_braking_rank[i], (char *)base, nmemb, size,
                                   noffs[Local_ThisTask], range_left[i], range_right[i], &current_loc_rank[i], compar);

#ifdef CHECK_LOCAL_RANK
                    check_local_rank(element_guess + i * size, element_tie_braking_rank[i], (char *)base, nmemb, size,
                                     noffs[Local_ThisTask], range_left[i], range_right[i], current_loc_rank[i], compar);
#endif // CHECK_LOCAL_RANK
                }
            }

            /* now compute the global ranks by summing the local ranks */
            size_t *list = (size_t *)mymalloc("list", Local_NTask * sizeof(size_t));
            for (int i = 0; i < Local_NTask - 1; i++)
            {
                if (current_glob_rank[i] != desired_glob_rank[i])
                {
                    MPI_Allgather(&current_loc_rank[i], sizeof(size_t), MPI_BYTE, list, sizeof(size_t), MPI_BYTE,
                                  MPI_CommLocal);

                    current_glob_rank[i] = 0;
                    for (int j = 0; j < Local_NTask; j++)
                    {
                        current_glob_rank[i] += list[j];
                    }
                }
            }
            myfree(list);

            ranks_not_found = 0;
            for (int i = 0; i < Local_NTask - 1; i++)
            {
                if (current_glob_rank[i] != desired_glob_rank[i]) /* here we're not yet done */
                {
                    ranks_not_found++;

                    if (current_glob_rank[i] < desired_glob_rank[i])
                    {
                        range_left[i] = current_loc_rank[i];

                        if (Local_ThisTask == max_loc[i]) // && range_right[i] == range_left[i]+1)
                        {
                            range_left[i]++;
                        }
                    }

                    if (current_glob_rank[i] > desired_glob_rank[i])
                    {
                        range_right[i] = current_loc_rank[i];
                    }
                }
            }

            /* now we need to determine new element guesses */

            long long *range_len_list = (long long *)mymalloc("range_len_list", Local_NTask * sizeof(long long));

            for (int i = 0; i < Local_NTask - 1; i++)
            {
                if (current_glob_rank[i] != desired_glob_rank[i]) /* here we're not yet done */
                {
                    long long max_range_len, range_len = range_right[i] - range_left[i] + 1;
                    MPI_Allgather(&range_len, sizeof(long long), MPI_BYTE, range_len_list, sizeof(long long), MPI_BYTE,
                                  MPI_CommLocal);

                    /* find the largest one, and the cpu that holds it */
                    max_loc[i] = max_range_len = 0;
                    for (int j = 0; j < Local_NTask; j++)
                    {
                        if (max_range_len < range_len_list[j])
                        {
                            max_range_len = range_len_list[j];
                            max_loc[i] = j;
                        }
                    }

                    if (Local_ThisTask == max_loc[i])
                    {
                        long long mid = (range_left[i] + range_right[i]) / 2;

                        memcpy(element_guess + i * size, (char *)base + mid * size, size);
                        element_tie_braking_rank[i] = mid + noffs[Local_ThisTask];
                    }

                    MPI_Bcast(element_guess + i * size, size, MPI_BYTE, max_loc[i], MPI_CommLocal);
                    MPI_Bcast(&element_tie_braking_rank[i], sizeof(size_t), MPI_BYTE, max_loc[i], MPI_CommLocal);
                }
            }

            myfree(range_len_list);

            iter++;

            if (iter > 400 + NTask && Local_ThisTask == 0)
            {
                VERBOSE_ALL(5, "iter=%d: ranks_not_found=%d  Local_NTask=%d ", iter, ranks_not_found, Local_NTask);
                if (iter > 500 + NTask)
                {
                    ENDRUN_MAIN("can't find the split points. That's odd");
                }
            }
        } while (ranks_not_found);

        /* At this point we have found all the elements corresponding to the desired split points */
        /* we can now go ahead and determine how many elements of the local CPU have to go to each other CPU */

        size_t *Send_count = (size_t *)mymalloc("Send_count", Local_NTask * sizeof(size_t));
        size_t *Recv_count = (size_t *)mymalloc("Recv_count", Local_NTask * sizeof(size_t));
        size_t *Send_offset = (size_t *)mymalloc("Send_offset", Local_NTask * sizeof(size_t));
        size_t *Recv_offset = (size_t *)mymalloc("Recv_offset", Local_NTask * sizeof(size_t));

        for (int i = 0; i < Local_NTask; ++i)
        {
            Send_count[i] = 0;
        }
        int target = 0;

        for (unsigned int i = 0; i < nmemb; ++i)
        {
            while (target < Local_NTask - 1)
            {
                int cmp = compar((char *)base + i * size, element_guess + target * size);
                if (cmp == 0)
                {
                    if (i + noffs[Local_ThisTask] < element_tie_braking_rank[target])
                    {
                        cmp = -1;
                    }
                    else if (i + noffs[Local_ThisTask] > element_tie_braking_rank[target])
                    {
                        cmp = +1;
                    }
                }
                if (cmp >= 0)
                {
                    target++;
                }
                else
                {
                    break;
                }
            }
            Send_count[target]++;
        }

        MPI_Alltoall(Send_count, sizeof(size_t), MPI_BYTE, Recv_count, sizeof(size_t), MPI_BYTE, MPI_CommLocal);

        size_t nimport;
        nimport = 0;
        Recv_offset[0] = 0;
        Send_offset[0] = 0;
        for (int j = 0; j < Local_NTask; j++)
        {
            nimport += Recv_count[j];

            if (j > 0)
            {
                Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
            }
        }

        if (nimport != nmemb)
        {
            ENDRUN_MAIN("nimport != nmemb");
        }

        char *basetmp = (char *)mymalloc("basetmp", nmemb * size);

#ifndef NO_USE_ISEND_IRECV_PSORT
        MPI_Request *requests;
        int nrequests = 0;

        requests = (MPI_Request *)mymalloc("requests", 2 * Local_NTask * sizeof(MPI_Request));

        for (int recvTask = 0; recvTask < Local_NTask; recvTask++)
        {
            MPI_Isend((char *)base + Send_offset[recvTask] * size, Send_count[recvTask] * size, MPI_BYTE, recvTask,
                      TAG_TRANSFER, MPI_CommLocal, &requests[nrequests++]);

            MPI_Irecv(basetmp + Recv_offset[recvTask] * size, Recv_count[recvTask] * size, MPI_BYTE, recvTask,
                      TAG_TRANSFER, MPI_CommLocal, &requests[nrequests++]);
        }

        MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
        myfree(requests);
#else // USE_ISEND_IRECV

        /* exchange the data */
        for (int ngrp = 0; ngrp < (1 << Local_PTask); ngrp++)
        {
            int recvTask = Local_ThisTask ^ ngrp;

            if (recvTask < Local_NTask)
            {
                if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
#ifndef MPISENDRECV_SIZELIMIT
                    if (Send_count[recvTask] > TRANSFER_SIZE_LIMIT || Recv_count[recvTask] > TRANSFER_SIZE_LIMIT)
                    {
                        ENDRUN_MAIN("we are above TRANSFER_SIZE_LIMIT");
                    }
#endif // MPISENDRECV_SIZELIMIT
                    MPI_Sendrecv((char *)base + Send_offset[recvTask] * size, Send_count[recvTask] * size, MPI_BYTE,
                                 recvTask, TAG_TRANSFER, basetmp + Recv_offset[recvTask] * size,
                                 Recv_count[recvTask] * size, MPI_BYTE, recvTask, TAG_TRANSFER, MPI_CommLocal,
                                 MPI_STATUS_IGNORE);
                }
            }
        }
#endif // USE_ISEND_IRECV

        /* instead of doing another serial sort at the end, we could also exploit that the different incoming pieces
           are already individually sorted. This can be used in a variant of merge sort to do the final step a bit
           quicker */

        memcpy(base, basetmp, nmemb * size);
        myfree(basetmp);
        serial_sort((char *)base, nmemb, size, compar);

        myfree(Recv_offset);
        myfree(Send_offset);
        myfree(Recv_count);
        myfree(Send_count);

        myfree(max_loc);
        myfree(range_right);
        myfree(range_left);
        myfree(current_loc_rank);
        myfree(current_glob_rank);
        myfree(desired_glob_rank);
        myfree(element_tie_braking_rank);
        myfree(element_guess);
        myfree(noffs);
        myfree(nlist);
    }

    MPI_Comm_free(&MPI_CommLocal);
}

/**
 * @brief Get the local rank of an element within a range of elements on a local task.
 *
 * This function determines the local rank of a specified element within a given range of elements on a local task.
 * It compares the specified element with each element in the local data array using the user-specified comparison
 * function. The local rank is calculated based on the number of elements smaller than the specified element within the
 * given range. Optionally, it considers tie-breaking by taking into account the initial global rank of the element. The
 * calculated local rank is returned as an output parameter.
 *
 * @param element The element of which the rank is to be determined.
 * @param tie_braking_rank The initial global rank of the element, needed for breaking ties.
 * @param base The base address of the local data array.
 * @param nmemb The number of elements in the local data array.
 * @param size The size of each element in the local data array.
 * @param noffs_thistask The cumulative length of data on lower tasks.
 * @param left The left boundary of the range of elements on the local task that may hold the specified element.
 * @param right The right boundary of the range of elements on the local task that may hold the specified element.
 * @param loc Pointer to the variable where the calculated local rank will be stored.
 * @param compar The user-specified comparison function used to compare elements.
 *
 * @note The function assumes that the right boundary is greater than or equal to the left boundary.
 */
static void get_local_rank(char *element, size_t tie_braking_rank, char *base, size_t nmemb, size_t size,
                           size_t noffs_thistask, long long left, long long right, size_t *loc,
                           int (*compar)(const void *, const void *))
{
    if (right < left)
    {
        ENDRUN_MAIN("right < left");
    }

    if (left == 0 && right == (long long int)(nmemb + 1))
    {
        if (compar(base + (nmemb - 1) * size, element) < 0)
        {
            *loc = nmemb;
            return;
        }
        else if (compar(base, element) > 0)
        {
            *loc = 0;
            return;
        }
    }

    if (right == left) /* looks like we already converged to the proper rank */
    {
        *loc = left;
    }
    else
    {
        if (compar(base + (right - 1) * size, element) <
            0) /* the last element is smaller, hence all elements are on the left */
        {
            *loc = (right - 1) + 1;
        }
        else if (compar(base + left * size, element) >
                 0) /* the first element is already larger, hence no element is on the left */
        {
            *loc = left;
        }
        else
        {
            while (right > left)
            {
                long long mid = ((right - 1) + left) / 2;

                int cmp = compar(base + mid * size, element);
                if (cmp == 0)
                {
                    if (mid + noffs_thistask < tie_braking_rank)
                    {
                        cmp = -1;
                    }
                    else if (mid + noffs_thistask > tie_braking_rank)
                    {
                        cmp = +1;
                    }
                }

                if (cmp == 0) /* element has exactly been found */
                {
                    *loc = mid;
                    break;
                }

                if ((right - 1) == left) /* elements is not on this CPU */
                {
                    if (cmp < 0)
                    {
                        *loc = mid + 1;
                    }
                    else
                    {
                        *loc = mid;
                    }
                    break;
                }

                if (cmp < 0)
                {
                    left = mid + 1;
                }
                else
                {
                    if ((right - 1) == left + 1)
                    {
                        if (mid != left)
                        {
                            VERBOSE_ALL(0, "-->left=%lld  right=%lld ", left, right);
                            ENDRUN_MAIN("can't be");
                        }
                        *loc = left;
                        break;
                    }

                    right = mid;
                }
            }
        }
    }
}

#ifdef CHECK_LOCAL_RANK
/**
 * @brief Check the local rank of an element within a range of elements on a local task.
 *
 * This function determines the local rank of a specified element within a given range of elements on a local task.
 * It compares the specified element with each element in the local data array using the user-specified comparison
 * function. The local rank is calculated based on the number of elements smaller than the specified element within the
 * given range. Optionally, it considers tie-breaking by taking into account the initial global rank of the element.
 *
 * @param element The element of which the rank is to be determined.
 * @param tie_braking_rank The initial global rank of the element, needed for breaking ties.
 * @param base The base address of the local data array.
 * @param nmemb The number of elements in the local data array.
 * @param size The size of each element in the local data array.
 * @param noffs_thistask The cumulative length of data on lower tasks.
 * @param left The left boundary of the range of elements on the local task that may hold the specified element.
 * @param right The right boundary of the range of elements on the local task that may hold the specified element.
 * @param loc The expected local rank of the specified element.
 * @param compar The user-specified comparison function used to compare elements.
 */
static void check_local_rank(char *element, size_t tie_braking_rank, char *base, size_t nmemb, size_t size,
                             size_t noffs_thistask, long long left, long long right, size_t loc,
                             int (*compar)(const void *, const void *))
{
    long long count = 0;

    for (int i = 0; i < nmemb; i++)
    {
        int cmp = compar(base + i * size, element);

        if (cmp == 0)
        {
            if (noffs_thistask + i < tie_braking_rank)
            {
                cmp = -1;
            }
        }

        if (cmp < 0)
        {
            count++;
        }
    }

    if (count != loc)
    {
        VERBOSE_ALL(0, "Task=%d: loc=%lld count=%lld  left=%lld right=%lld  nmemb=%lld ", ThisTask, (long long)loc,
                    count, left, right, (long long)nmemb);
        ENDRUN_MAIN("inconsisteny");
    }
}
#endif // CHECK_LOCAL_RANK

#ifdef _OPENMP
/**
 * @brief Perform serial sorting using OpenMP parallelism with multiple threads and a temporary buffer.
 *
 * This function performs serial sorting using OpenMP parallelism with multiple threads and a temporary buffer.
 * It divides the sorting process into multiple levels, each with a different number of threads, and applies
 * a multi-threaded merge sort algorithm to each level. It ensures that the size of the structure is a multiple
 * of the size of the sort_type.
 *
 * @param base Pointer to the base address of the array to be sorted.
 * @param nmemb Number of elements in the array.
 * @param size Size of each element in the array.
 * @param compar Pointer to the comparison function used for sorting.
 */
void serial_sort_omp(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
    size_t *ln, *lstart;
    sort_type *tmp;
    int nthreads, maxlevel;
    sort_type *b2, *local_tmp;

    maxlevel = floorLog2(maxThreads);
    if ((size % sizesort) != 0)
    {
        ENDRUN_MAIN("size of structure must be a multiple of sizeof(sort_type)");
    }

    ln = (size_t *)mymalloc("ln", sizeof(size_t) * maxThreads * (maxlevel + 1));
    lstart = (size_t *)mymalloc("lstart", sizeof(size_t) * maxThreads * (maxlevel + 1));
    tmp = (sort_type *)mymalloc("tmp", size * nmemb);

    for (int lev = maxlevel; lev >= 0; lev--)
    {
        nthreads = 1 << (maxlevel - lev);
        for (int i = 0; i < nthreads; i += 2)
        {
            if (lev == maxlevel)
            {
                ln[lev * maxThreads + i] = nmemb;
                lstart[lev * maxThreads + i] = 0;
            }
            else
            {
                ln[lev * maxThreads + i] = ln[(lev + 1) * maxThreads + i / 2] / 2;
                lstart[lev * maxThreads + i] = lstart[(lev + 1) * maxThreads + i / 2];
                ln[lev * maxThreads + i + 1] = ln[(lev + 1) * maxThreads + i / 2] - ln[lev * maxThreads + i];
                lstart[lev * maxThreads + i + 1] = lstart[(lev + 1) * maxThreads + i / 2] + ln[lev * maxThreads + i];
            }
        }
    }

    for (int lev = 0; lev <= maxlevel; lev++)
    {
        nthreads = 1 << (maxlevel - lev);

        if (ThisTask == 0)
        {
            VERBOSE(3, "Task:%d using %d threads in level %d ", ThisTask, nthreads, lev);
        }

#pragma omp parallel for private(b2, local_tmp)
        for (int tid = nthreads - 1; tid >= 0; tid--)
        {
            local_tmp = tmp + lstart[lev * maxThreads + tid] * size / sizesort;
            b2 = (sort_type *)base + lstart[lev * maxThreads + tid] * size / sizesort;

            msort_serial_with_tmp(b2, ln[lev * maxThreads + tid], size, compar, local_tmp, 0, (lev == 0) ? 10000 : 1);
        }
    }

    myfree(tmp);
    myfree(lstart);
    myfree(ln);
}

/**
 * @brief Sorts an array using either parallel or serial sorting algorithms based on the number of elements.
 *
 * This function sorts an array using parallel sorting algorithms if the number of elements exceeds a threshold.
 * Otherwise, it uses a serial sorting algorithm. It ensures that it's not called from within an OpenMP parallel region.
 *
 * @param base Pointer to the base address of the array to be sorted.
 * @param nmemb Number of elements in the array.
 * @param size Size of each element in the array.
 * @param compar Pointer to the comparison function used for sorting.
 *
 * @note This function checks if it's called from an OpenMP parallel region and raises a panic if so.
 */
void omp_mysort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
    ENDRUN_IF(THREAD_ID_OR_ZERO() > 0, "Call to omp_myqsort from an already OpenMP parallel reagion !");

    if (nmemb > OMP_SORT_THRESH)
    {
        VERBOSE(3, "Parallel mysort");
        serial_sort_omp(base, nmemb, size, compar);
    }
    else
    {
        mysort_dataindex(base, nmemb, size, compar);
    }
}

/**
 * @brief Calculate the floor of the base-2 logarithm of an unsigned integer.
 *
 * This function calculates the floor of the base-2 logarithm of an unsigned integer.
 * It efficiently finds the position of the most significant bit that is set to 1 in the binary representation of the
 * number. The result represents the highest power of 2 that is less than or equal to the given number.
 *
 * @param n The unsigned integer for which the floor of the base-2 logarithm is to be calculated.
 * @return The floor of the base-2 logarithm of the given unsigned integer. Returns -1 if the given integer is 0.
 */
int floorLog2(unsigned int n)
{
    int pos = 0;
    if (n >= 1 << 16)
    {
        n >>= 16;
        pos += 16;
    }
    if (n >= 1 << 8)
    {
        n >>= 8;
        pos += 8;
    }
    if (n >= 1 << 4)
    {
        n >>= 4;
        pos += 4;
    }
    if (n >= 1 << 2)
    {
        n >>= 2;
        pos += 2;
    }
    if (n >= 1 << 1)
    {
        pos += 1;
    }
    return ((n == 0) ? (-1) : pos);
}
#endif // _OPENMP

/**
 * @brief Perform serial sorting of an array.
 *
 * This function performs serial sorting of an array using a merge sort algorithm with temporary storage.
 * It checks if the size of the structure is a multiple of the size of the sorting type and then allocates
 * temporary storage for sorting. The sorting process involves recursively splitting the array into smaller
 * sub-arrays, sorting them individually, and merging them back together. Once sorting is complete, the
 * temporary storage is freed.
 *
 * @param base Pointer to the base address of the array to be sorted.
 * @param nmemb Number of elements in the array.
 * @param size Size of each element in the array.
 * @param compar Pointer to the comparison function used for sorting.
 */
void serial_sort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
    const size_t storage = nmemb * size;

    if ((size % sizesort) != 0)
    {
        ENDRUN_MAIN("size of structure must be a multiple of sizeof(sort_type)");
    }
    sort_type *tmp = (sort_type *)mymalloc("int *tmp", storage);

    msort_serial_with_tmp((sort_type *)base, nmemb, size, compar, tmp, 0, 10000);

    myfree(tmp);
}

/**
 * @brief Perform merge sort with temporary storage in a serial manner.
 *
 * This function implements the merge sort algorithm in a serial manner with the use of temporary storage.
 * It recursively divides the array into smaller sub-arrays until each sub-array contains only one element,
 * then merges the sub-arrays back together in sorted order. The process continues until the entire array
 * is sorted. Temporary storage is used to hold intermediate results during the merging process.
 *
 * @param base Pointer to the base address of the array to be sorted.
 * @param n Number of elements in the array.
 * @param s Size of each element in the array.
 * @param compar Pointer to the comparison function used for sorting.
 * @param t Pointer to the temporary storage array used during sorting.
 * @param level Current level of recursion in the merge sort algorithm.
 * @param maxlevel Maximum level of recursion allowed in the merge sort algorithm.
 */
void msort_serial_with_tmp(sort_type *base, size_t n, size_t s, int (*compar)(const void *, const void *), sort_type *t,
                           int level, int maxlevel)
{
    sort_type *tmp;
    sort_type *b1, *b2;

    if (n <= 1 || level >= maxlevel)
    {
        return;
    }

    size_t n1 = n / 2;
    size_t n2 = n - n1;
    b1 = base;
    b2 = base + n1 * s / sizesort;

    msort_serial_with_tmp(b1, n1, s, compar, t, level + 1, maxlevel);
    msort_serial_with_tmp(b2, n2, s, compar, t, level + 1, maxlevel);

    tmp = t;
    size_t ss, st;
    st = s / sizesort;
    while (n1 > 0 && n2 > 0)
    {
        ss = st;
        if (compar(b1, b2) < 0)
        {
            --n1;
            while (ss--)
                *tmp++ = *b1++;
        }
        else
        {
            --n2;
            while (ss--)
                *tmp++ = *b2++;
        }
    }

    if (n1 > 0)
    {
        memcpy(tmp, b1, n1 * s);
    }
    memcpy(base, t, (n - n2) * s);
}

typedef long WORD;
constexpr size_t W = sizeof(WORD); /* must be a power of 2 */
#define SWAPINIT(a, es) swaptype = ((a - (char *)0) | (es)) % W ? 2 : es > W ? 1 : 0

#define exch(a, b, t) (t = a, a = b, b = t);
#define swap(a, b) swaptype != 0 ? swapfunc(a, b, es, swaptype) : (void)exch(*(WORD *)(a), *(WORD *)(b), t)

#define vecswap(a, b, n)                                                                                               \
    if (n > 0)                                                                                                         \
    swapfunc(a, b, n, swaptype)

#include <stddef.h>
/**
 * @brief Swap the contents of two memory blocks.
 *
 * This function swaps the contents of two memory blocks 'a' and 'b' of size 'n' bytes.
 * It can perform the swap either on a word-by-word basis or byte-by-byte basis, depending
 * on the value of the 'swaptype' parameter.
 *
 * @param a Pointer to the first memory block.
 * @param b Pointer to the second memory block.
 * @param n Size of each memory block in bytes.
 * @param swaptype Type of swap operation:
 *                  - 0 or 1: Perform the swap on a word-by-word basis.
 *                  - Other values: Perform the swap on a byte-by-byte basis.
 */
static void swapfunc(char *a, char *b, size_t n, int swaptype)
{
    if (swaptype <= 1)
    {
        WORD t;
        for (; n > 0; a += W, b += W, n -= W)
        {
            exch(*(WORD *)a, *(WORD *)b, t);
        }
    }
    else
    {
        char t;
        for (; n > 0; a += 1, b += 1, n -= 1)
        {
            exch(*a, *b, t);
        }
    }
}

#define PVINIT(pv, pm)                                                                                                 \
    if (swaptype != 0)                                                                                                 \
    {                                                                                                                  \
        pv = a;                                                                                                        \
        swap(pv, pm);                                                                                                  \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        pv = (char *)&v;                                                                                               \
        v = *(WORD *)pm;                                                                                               \
    }

/**
 * @brief Return the median value among three elements.
 *
 * This function returns the median value among three elements 'a', 'b', and 'c',
 * based on the comparison function 'cmp'.
 *
 * @param a Pointer to the first element.
 * @param b Pointer to the second element.
 * @param c Pointer to the third element.
 * @param cmp Comparison function pointer taking two const void* arguments and returning an int.
 *            This function is used to compare the elements 'a', 'b', and 'c'.
 *
 * @return Pointer to the median element among 'a', 'b', and 'c'.
 */
static char *med3(char *a, char *b, char *c, int (*cmp)(const void *, const void *))
{
    return cmp(a, b) < 0 ? (cmp(b, c) < 0 ? b : cmp(a, c) < 0 ? c : a) : (cmp(b, c) > 0 ? b : cmp(a, c) > 0 ? c : a);
}

/**
 * @brief Sorts an array using the Quicksort algorithm.
 *
 * This function sorts the elements of the array pointed to by 'av' in ascending order
 * using the Quicksort algorithm.
 *
 * @param av Pointer to the array to be sorted.
 * @param n Number of elements in the array.
 * @param es Size of each element in bytes.
 * @param cmp Pointer to the comparison function used to compare elements.
 *            The comparison function must return an integer less than, equal to,
 *            or greater than zero if the first argument is considered to be
 *            respectively less than, equal to, or greater than the second.
 */
void quicksort(void *av, size_t n, size_t es, int (*cmp)(const void *, const void *))
{
    char *a, *pa, *pb, *pc, *pd, *pl, *pm, *pn, *pv;
    int r, swaptype;
    WORD t, v;
    size_t s;

    a = (char *)av;

    SWAPINIT(a, es);
    if (n < 7)
    { /* Insertion sort on smallest arrays */
        for (pm = a + es; pm < a + n * es; pm += es)
        {
            for (pl = pm; pl > a && cmp(pl - es, pl) > 0; pl -= es)
            {
                swap(pl, pl - es);
            }
        }
        return;
    }
    pm = a + (n / 2) * es; /* Small arrays, middle element */
    if (n > 7)
    {
        pl = a;
        pn = a + (n - 1) * es;
        if (n > 40)
        { /* Big arrays, pseudomedian of 9 */
            s = (n / 8) * es;
            pl = med3(pl, pl + s, pl + 2 * s, cmp);
            pm = med3(pm - s, pm, pm + s, cmp);
            pn = med3(pn - 2 * s, pn - s, pn, cmp);
        }
        pm = med3(pl, pm, pn, cmp); /* Mid-size, med of 3 */
    }
    PVINIT(pv, pm); /* pv points to partition value */
    pa = pb = a;
    pc = pd = a + (n - 1) * es;
    for (;;)
    {
        while (pb <= pc && (r = cmp(pb, pv)) <= 0)
        {
            if (r == 0)
            {
                swap(pa, pb);
                pa += es;
            }
            pb += es;
        }
        while (pc >= pb && (r = cmp(pc, pv)) >= 0)
        {
            if (r == 0)
            {
                swap(pc, pd);
                pd -= es;
            }
            pc -= es;
        }
        if (pb > pc)
            break;
        swap(pb, pc);
        pb += es;
        pc -= es;
    }
    pn = a + n * es;
    s = std::min(pa - a, pb - pa);
    vecswap(a, pb - s, s);
    s = std::min(size_t(pd - pc), pn - pd - es);
    vecswap(pb, pn - s, s);

#pragma omp task
    if ((s = pb - pa) > es)
    {
        quicksort(a, s / es, es, cmp);
    }
    if ((s = pd - pc) > es)
    {
        quicksort(pn - s, s / es, es, cmp);
    }
#pragma omp taskwait
}

#ifdef _OPENMP
/**
 * @brief Sorts an array using either parallel or serial quicksort algorithm.
 *
 * This function sorts the elements of the array pointed to by 'base' in ascending order
 * using either parallel quicksort or serial quicksort algorithm based on the size of the array.
 * If the number of elements in the array is greater than OMP_SORT_THRESH, parallel quicksort
 * is used; otherwise, serial quicksort is used.
 *
 * @param base Pointer to the array to be sorted.
 * @param nmemb Number of elements in the array.
 * @param size Size of each element in bytes.
 * @param compar Pointer to the comparison function used to compare elements.
 *               The comparison function must return an integer less than, equal to,
 *               or greater than zero if the first argument is considered to be
 *               respectively less than, equal to, or greater than the second.
 */
void omp_qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *))
{
    ENDRUN_IF(THREAD_ID_OR_ZERO() > 0, "Call to omp_qsort from an already OpenMP parallel reagion !");

    if (nmemb > OMP_SORT_THRESH)
    {
        VERBOSE(3, "Parallel quicksort");

#pragma omp parallel
#pragma omp single
        {
            quicksort(base, nmemb, size, compar);
        }
    }
    else
    {
        qsort(base, nmemb, size, compar);
    }
}
#endif // _OPENMP
