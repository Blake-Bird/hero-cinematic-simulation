/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the routines for the Peano-Hilbert space-filling curve.
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

#include "allvars.hpp"
#include "proto.hpp"

#ifdef THRUST_SORT
#include "../OpenACC/thrust_sort.hpp"
#endif // THRUST_SORT

/**
 * @brief Structure representing Peano-Hilbert data.
 */
static struct peano_hilbert_data
{
    peanokey key;
    int index;
} *__restrict__ mp;

static int *__restrict__ Id;

/**
 * @brief Orders particles using the Peano-Hilbert method.
 *
 * This function orders particles using the Peano-Hilbert method.
 */
void peano_hilbert_order(void)
{
    TIMER_PUSH(1, "PEANO");
    VERBOSE(1, "PEANO: Begin Peano-Hilbert order...");

    if (N_gas)
    {
        TIMER_PUSH(3, "sorting_gas_keys");
        mp = (struct peano_hilbert_data *)mymalloc("mp", sizeof(struct peano_hilbert_data) * N_gas);
        Id = (int *)mymalloc("Id", sizeof(int) * N_gas);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int ii = 0; ii < N_gas; ii++)
        {
            mp[ii].index = ii;
            mp[ii].key = Key[ii];
        }

#ifdef _OPENMP
        if (N_gas > OMP_SORT_THRESH)
        {
            serial_sort_omp(mp, N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);
        }
        else
        {
            mysort_peano(mp, N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);
        }
#else // _OPENMP
        mysort_peano(mp, N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);
#endif // _OPENMP

        TIMER_POPPUSH(3, "sorting_gas_keys", "reodering_gas_particles");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int ii = 0; ii < N_gas; ii++)
        {
            Id[mp[ii].index] = ii;
        }

#ifdef _OPENMP
        kd_reorder_gas2();
#else // _OPENMP
        reorder_gas();
#endif // _OPENMP

        myfree(Id);
        myfree(mp);

        TIMER_POP(3, "reodering_gas_particles");
    }

    if (NumPart - N_gas > 0)
    {
        TIMER_PUSH(3, "sorting_non_gas_particles_keys");
        mp = (struct peano_hilbert_data *)mymalloc("mp", sizeof(struct peano_hilbert_data) * (NumPart - N_gas));
        mp -= (N_gas);

        Id = (int *)mymalloc("Id", sizeof(int) * (NumPart - N_gas));
        Id -= (N_gas);

        int NumPart_4 = N_gas + ((NumPart - N_gas) / 4) * 4;
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int ii = N_gas; ii < NumPart_4; ii += 4)
        {
            mp[ii].index = ii;
            mp[ii + 1].index = ii + 1;
            mp[ii + 2].index = ii + 2;
            mp[ii + 3].index = ii + 3;
            mp[ii].key = Key[ii];
            mp[ii + 1].key = Key[ii + 1];
            mp[ii + 2].key = Key[ii + 2];
            mp[ii + 3].key = Key[ii + 3];
        }
        for (int ii = NumPart_4; ii < NumPart; ii++)
        {
            mp[ii].index = ii;
            mp[ii].key = Key[ii];
        }

#ifdef THRUST_SORT
        int *indexes = (int *)mymalloc("indexes", NumPart * sizeof(int));
        unsigned long long *keys = (unsigned long long *)mymalloc("keys", NumPart * sizeof(unsigned long long));

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            indexes[i] = i;
            keys[i] = Key[i];
        }
        thrust_device_keys_sort(keys, indexes, NumPart);

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            int j = indexes[i];
            mp[i].index = j;
            mp[i].key = Key[j];
        }

        myfree(keys);
        myfree(indexes);
#else // THRUST_SORT
#ifdef _OPENMP
        if (NumPart - N_gas > OMP_SORT_THRESH)
        {
            serial_sort_omp(mp + N_gas, NumPart - N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);
        }
        else
        {
            mysort_peano(mp + N_gas, NumPart - N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);
        }
#else // _OPENMP
        mysort_peano(mp + N_gas, NumPart - N_gas, sizeof(struct peano_hilbert_data), peano_compare_key);
#endif // _OPENMP
#endif // THRUST_SORT

        TIMER_POPPUSH(3, "sorting_non_gas_particles_keys", "reodering_non_gas_particles");

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int ii = N_gas; ii < NumPart_4; ii += 4)
        {
            Id[mp[ii].index] = ii;
            Id[mp[ii + 1].index] = ii + 1;
            Id[mp[ii + 2].index] = ii + 2;
            Id[mp[ii + 3].index] = ii + 3;
        }
        for (int ii = NumPart_4; ii < NumPart; ii++)
        {
            Id[mp[ii].index] = ii;
        }

#ifdef _OPENMP
        kd_reorder_particles2();
#else // _OPENMP
        reorder_particles();
#endif // _OPENMP

        Id += N_gas;
        myfree(Id);
        mp += N_gas;
        myfree(mp);

        TIMER_POP(3, "reodering_non_gas_particles");
    }

    VERBOSE(1, "PEANO: done!");
    TIMER_POP(1, "PEANO");
}

/**
 * @brief Compares two Peano-Hilbert keys.
 *
 * This function compares two Peano-Hilbert keys.
 *
 * @param a Pointer to the first Peano-Hilbert data.
 * @param b Pointer to the second Peano-Hilbert data.
 * @return -1 if the key of 'a' is less than the key of 'b',
 *          1 if the key of 'a' is greater than the key of 'b',
 *          0 if the keys are equal.
 */
int peano_compare_key(const void *a, const void *b)
{
    if (((struct peano_hilbert_data *)a)->key < (((struct peano_hilbert_data *)b)->key))
    {
        return -1;
    }

    if (((struct peano_hilbert_data *)a)->key > (((struct peano_hilbert_data *)b)->key))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Reorders gas particles.
 *
 * This function reorders gas particles based on their IDs.
 *
 * The function iterates over all gas particles and checks if their IDs match their positions.
 * If the IDs don't match, the function swaps the particle data until the IDs match their positions.
 */
void reorder_gas(void)
{
    struct particle_data Psave, Psource;
    struct sph_particle_data SphPsave, SphPsource;
    int idsource, idsave, dest;
#ifdef WRITE_KEY_FILES
    peanokey Ksource, Ksave;
#endif // WRITE_KEY_FILES

    for (int i = 0; i < N_gas; i++)
    {
        if (Id[i] != i)
        {
#ifdef WRITE_KEY_FILES
            Ksource = Key[i];
#endif // WRITE_KEY_FILES
            Psource = P[i];
            SphPsource = SphP[i];

            idsource = Id[i];
            dest = Id[i];

            do
            {
#ifdef WRITE_KEY_FILES
                Ksave = Key[dest];
#endif // WRITE_KEY_FILES
                Psave = P[dest];
                SphPsave = SphP[dest];
                idsave = Id[dest];

#ifdef WRITE_KEY_FILES
                Key[dest] = Ksource;
#endif // WRITE_KEY_FILES
                P[dest] = Psource;
                SphP[dest] = SphPsource;
                Id[dest] = idsource;

                if (dest == i)
                {
                    break;
                }

                Psource = Psave;
                SphPsource = SphPsave;
                idsource = idsave;
#ifdef WRITE_KEY_FILES
                Ksource = Ksave;
#endif // WRITE_KEY_FILES

                dest = idsource;
            } while (1);
        }
    }
}

/**
 * @brief Reorders particles, except gas particles.
 *
 * This function reorders particles based on their IDs.
 *
 * The function iterates over all particles and checks if their IDs match their positions.
 * If the IDs don't match, the function swaps the particle data until the IDs match their positions.
 */
void reorder_particles(void)
{
    struct particle_data Psave, Psource;
    int idsource, idsave, dest;
#ifdef WRITE_KEY_FILES
    peanokey Ksource, Ksave;
#endif // WRITE_KEY_FILES

    for (int i = N_gas; i < NumPart; i++)
    {
        if (Id[i] != i)
        {
#ifdef WRITE_KEY_FILES
            Ksource = Key[i];
#endif // WRITE_KEY_FILES
            Psource = P[i];
            idsource = Id[i];
            dest = Id[i];

            do
            {
#ifdef WRITE_KEY_FILES
                Ksave = Key[dest];
#endif // WRITE_KEY_FILES
                Psave = P[dest];
                idsave = Id[dest];

#ifdef WRITE_KEY_FILES
                Key[dest] = Ksource;
#endif // WRITE_KEY_FILES
                P[dest] = Psource;
                Id[dest] = idsource;
#ifdef SFR
                if (P[dest].Type == 4)
                {
                    MetP[P[dest].pt.MetID].PID = dest;
                }
#endif // SFR
#ifdef BLACK_HOLES
                if (P[dest].Type == 5)
                {
                    BHP[P[dest].pt.BHID].PID = dest;
                }
#endif // BLACK_HOLES

#ifdef AXION_DM
                if (P[dest].Type == AXION_TYPE)
                {
                    APP(dest).PID = dest;
                }
#endif // AXION_DM

                if (dest == i)
                {
                    break;
                }

                Psource = Psave;
                idsource = idsave;
#ifdef WRITE_KEY_FILES
                Ksource = Ksave;
#endif // WRITE_KEY_FILES

                dest = idsource;
            } while (1);
        }
    }
}

#ifdef _OPENMP
/**
 * @brief Reorders gas particles using OpenMP parallelization.
 *
 * This function reorders gas particles based on their IDs using OpenMP parallelization.
 *
 * It first checks if there's enough memory available to perform the reordering. If not, it falls back to serial
 * reordering. Otherwise, it performs the reordering in parallel using OpenMP.
 */
void kd_reorder_gas2(void)
{
    struct particle_data *__restrict__ Psave;
    struct sph_particle_data *__restrict__ SphPsave;
#ifdef WRITE_KEY_FILES
    peanokey *__restrict__ Ksave;
#endif // WRITE_KEY_FILES

    if (FreeBytes < N_gas * (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(peanokey)))
    {
        reorder_gas();
    }
    else
    {
        TIMER_PUSH(3, "reodering_particles_stage1");
        Psave = (struct particle_data *)mymalloc("Psave", N_gas * sizeof(struct particle_data));
        SphPsave = (struct sph_particle_data *)mymalloc("Psave", N_gas * sizeof(struct sph_particle_data));
#ifdef WRITE_KEY_FILES
        Ksave = (peanokey *)mymalloc("Ksave", N_gas * sizeof(peanokey));
#endif // WRITE_KEY_FILES

        VERBOSE(2, "Peano: using %g MB of remaining %g MB for non inline sorting of sph particles and keys",
                N_gas * (sizeof(struct particle_data) + sizeof(struct sph_particle_data) + sizeof(peanokey)) /
                    (1024.0 * 1024.0),
                FreeBytes / (1024.0 * 1024.0));

        int N_gas_4 = (N_gas / 4) * 4;

#pragma omp parallel for
        for (int i0 = 0; i0 < N_gas_4; i0 += 4)
        {
            Psave[Id[i0]] = P[i0];
            Psave[Id[i0 + 1]] = P[i0 + 1];
            Psave[Id[i0 + 2]] = P[i0 + 2];
            Psave[Id[i0 + 3]] = P[i0 + 3];

            memcpy(&SphPsave[Id[i0]], &SphP[i0], sizeof(sph_particle_data));
            memcpy(&SphPsave[Id[i0 + 1]], &SphP[i0 + 1], sizeof(sph_particle_data));
            memcpy(&SphPsave[Id[i0 + 2]], &SphP[i0 + 2], sizeof(sph_particle_data));
            memcpy(&SphPsave[Id[i0 + 3]], &SphP[i0 + 3], sizeof(sph_particle_data));

#ifdef WRITE_KEY_FILES
            Ksave[Id[i0]] = Key[i0];
            Ksave[Id[i0 + 1]] = Key[i0 + 1];
            Ksave[Id[i0 + 2]] = Key[i0 + 2];
            Ksave[Id[i0 + 3]] = Key[i0 + 3];
#endif // WRITE_KEY_FILES
        }
        for (int i0 = N_gas_4; i0 < N_gas; i0++)
        {
            Psave[Id[i0]] = P[i0];
            memcpy(&SphPsave[Id[i0]], &SphP[i0], sizeof(sph_particle_data));
#ifdef WRITE_KEY_FILES
            Ksave[Id[i0]] = Key[i0];
#endif // WRITE_KEY_FILES
        }

        TIMER_POPPUSH(3, "reodering_particles_stage1", "reodering_particles_stage2");

#pragma omp parallel for
        for (int i0 = 0; i0 < N_gas_4; i0 += 4)
        {
            P[i0] = Psave[i0];
            P[i0 + 1] = Psave[i0 + 1];
            P[i0 + 2] = Psave[i0 + 2];
            P[i0 + 3] = Psave[i0 + 3];

            SphP[i0] = SphPsave[i0];
            SphP[i0 + 1] = SphPsave[i0 + 1];
            SphP[i0 + 2] = SphPsave[i0 + 2];
            SphP[i0 + 3] = SphPsave[i0 + 3];

#ifdef WRITE_KEY_FILES
            Key[i0] = Ksave[i0];
            Key[i0 + 1] = Ksave[i0 + 1];
            Key[i0 + 2] = Ksave[i0 + 2];
            Key[i0 + 3] = Ksave[i0 + 3];
#endif // WRITE_KEY_FILES
        }
        for (int i0 = N_gas_4; i0 < N_gas; i0++)
        {
            P[i0] = Psave[i0];
            SphP[i0] = SphPsave[i0];
#ifdef WRITE_KEY_FILES
            Key[i0] = Ksave[i0];
#endif // WRITE_KEY_FILES
        }
        // #else
        //       memcpy(P, Psave, N_gas * sizeof(struct particle_data));
        //       memcpy(SphP, SphPsave, N_gas * sizeof(struct sph_particle_data));
        // #ifdef WRITE_KEY_FILES
        //       memcpy(Key, Ksave, N_gas * sizeof(peanokey));
        // #endif // WRITE_KEY_FILES
        // #endif // _OPENMP

        TIMER_POP(3, "reodering_particles_stage2");

#ifdef WRITE_KEY_FILES
        myfree(Ksave);
#endif // WRITE_KEY_FILES
        myfree(SphPsave);
        myfree(Psave);
    }
}

/**
 * @brief Reorders particles, except gas, using a parallelized approach.
 *
 * This function reorders particles using a parallelized approach.
 *
 * It first checks if there's enough memory available to perform the reordering.
 * If not, it falls back to serial reordering.
 * Otherwise, it performs the reordering in parallel.
 */
void kd_reorder_particles2(void)
{
    struct particle_data *__restrict__ Psave;
#ifdef WRITE_KEY_FILES
    peanokey *__restrict__ Ksave;
#endif // WRITE_KEY_FILES
    if (FreeBytes < (NumPart - N_gas) * (sizeof(struct particle_data) + sizeof(peanokey)))
    {
        reorder_particles();
    }
    else
    {
        TIMER_PUSH(3, "reodering_particles_stage1");
        Psave = (struct particle_data *)mymalloc("Psave", (NumPart - N_gas) * sizeof(struct particle_data));
#ifdef WRITE_KEY_FILES
        Ksave = (peanokey *)mymalloc("Ksave", (NumPart - N_gas) * sizeof(peanokey));
#endif // WRITE_KEY_FILES

        VERBOSE(2, "Peano: using %g MB of remaining %g MB for non inline sorting of particles and keys",
                (NumPart - N_gas) * (sizeof(struct particle_data) + sizeof(peanokey)) / (1024.0 * 1024.0),
                FreeBytes / (1024.0 * 1024.0));

#pragma omp parallel for
        for (int i1 = N_gas; i1 < NumPart; i1++)
        {
            int i0 = Id[i1] - N_gas;
            Psave[i0] = P[i1];
#ifdef WRITE_KEY_FILES
            Ksave[i0] = Key[i1];
#endif // WRITE_KEY_FILES

#ifdef SFR
            if (Psave[i0].Type == 4)
            {
                MetP[Psave[i0].pt.MetID].PID = Id[i1];
            }
#endif // SFR
#ifdef BLACK_HOLES
            if (Psave[i0].Type == 5)
            {
                BHP[Psave[i0].pt.BHID].PID = Id[i1];
            }
#endif // BLACK_HOLES

#ifdef AXION_DM
            if (Psave[i0].Type == AXION_TYPE)
            {
                APP(Id[i1]).PID = Id[i1];
            }
#endif // AXION_DM
        }

        TIMER_POPPUSH(3, "reodering_particles_stage1", "reodering_particles_stage2");

#pragma omp parallel for
        for (int i0 = 0; i0 < NumPart - N_gas; i0++)
        {
            P[i0 + N_gas] = Psave[i0];
#ifdef WRITE_KEY_FILES
            Key[i0 + N_gas] = Ksave[i0];
#endif // WRITE_KEY_FILES
        }
        // #else
        //       memcpy(&P[N_gas], Psave, (NumPart - N_gas) * sizeof(struct particle_data));
        // #ifdef WRITE_KEY_FILES
        //       memcpy(&Key[N_gas], Ksave, (NumPart - N_gas) * sizeof(peanokey));
        // #endif // WRITE_KEY_FILES
        // #endif

#ifdef WRITE_KEY_FILES
        myfree(Ksave);
#endif // WRITE_KEY_FILES
        myfree(Psave);
        TIMER_POP(3, "reodering_particles_stage2");
    }
}
#endif // _OPENMP

/**
 * @brief Peano-Hilbert key generation function.
 *
 * This function generates Peano-Hilbert keys with improved performance compared to the original function.
 * It was rewritten by MARTIN REINECKE and is approximately 2.3 - 2.5 times faster than the old routine.
 */
const unsigned char rottable3[48][8] = {
    {36, 28, 25, 27, 10, 10, 25, 27}, {29, 11, 24, 24, 37, 11, 26, 26}, {8, 8, 25, 27, 30, 38, 25, 27},
    {9, 39, 24, 24, 9, 31, 26, 26},   {40, 24, 44, 32, 40, 6, 44, 6},   {25, 7, 33, 7, 41, 41, 45, 45},
    {4, 42, 4, 46, 26, 42, 34, 46},   {43, 43, 47, 47, 5, 27, 5, 35},   {33, 35, 36, 28, 33, 35, 2, 2},
    {32, 32, 29, 3, 34, 34, 37, 3},   {33, 35, 0, 0, 33, 35, 30, 38},   {32, 32, 1, 39, 34, 34, 1, 31},
    {24, 42, 32, 46, 14, 42, 14, 46}, {43, 43, 47, 47, 25, 15, 33, 15}, {40, 12, 44, 12, 40, 26, 44, 34},
    {13, 27, 13, 35, 41, 41, 45, 45}, {28, 41, 28, 22, 38, 43, 38, 22}, {42, 40, 23, 23, 29, 39, 29, 39},
    {41, 36, 20, 36, 43, 30, 20, 30}, {37, 31, 37, 31, 42, 40, 21, 21}, {28, 18, 28, 45, 38, 18, 38, 47},
    {19, 19, 46, 44, 29, 39, 29, 39}, {16, 36, 45, 36, 16, 30, 47, 30}, {37, 31, 37, 31, 17, 17, 46, 44},
    {12, 4, 1, 3, 34, 34, 1, 3},      {5, 35, 0, 0, 13, 35, 2, 2},      {32, 32, 1, 3, 6, 14, 1, 3},
    {33, 15, 0, 0, 33, 7, 2, 2},      {16, 0, 20, 8, 16, 30, 20, 30},   {1, 31, 9, 31, 17, 17, 21, 21},
    {28, 18, 28, 22, 2, 18, 10, 22},  {19, 19, 23, 23, 29, 3, 29, 11},  {9, 11, 12, 4, 9, 11, 26, 26},
    {8, 8, 5, 27, 10, 10, 13, 27},    {9, 11, 24, 24, 9, 11, 6, 14},    {8, 8, 25, 15, 10, 10, 25, 7},
    {0, 18, 8, 22, 38, 18, 38, 22},   {19, 19, 23, 23, 1, 39, 9, 39},   {16, 36, 20, 36, 16, 2, 20, 10},
    {37, 3, 37, 11, 17, 17, 21, 21},  {4, 17, 4, 46, 14, 19, 14, 46},   {18, 16, 47, 47, 5, 15, 5, 15},
    {17, 12, 44, 12, 19, 6, 44, 6},   {13, 7, 13, 7, 18, 16, 45, 45},   {4, 42, 4, 21, 14, 42, 14, 23},
    {43, 43, 22, 20, 5, 15, 5, 15},   {40, 12, 21, 12, 40, 6, 23, 6},   {13, 7, 13, 7, 41, 41, 22, 20}};

/**
 * @brief Subpixel reordering table.
 *
 * This table defines the subpixel reordering pattern for Peano-Hilbert curve generation.
 * Each row represents a specific orientation of the subpixels.
 */
const unsigned char subpix3[48][8] = {
    {0, 7, 1, 6, 3, 4, 2, 5}, {7, 4, 6, 5, 0, 3, 1, 2}, {4, 3, 5, 2, 7, 0, 6, 1}, {3, 0, 2, 1, 4, 7, 5, 6},
    {1, 0, 6, 7, 2, 3, 5, 4}, {0, 3, 7, 4, 1, 2, 6, 5}, {3, 2, 4, 5, 0, 1, 7, 6}, {2, 1, 5, 6, 3, 0, 4, 7},
    {6, 1, 7, 0, 5, 2, 4, 3}, {1, 2, 0, 3, 6, 5, 7, 4}, {2, 5, 3, 4, 1, 6, 0, 7}, {5, 6, 4, 7, 2, 1, 3, 0},
    {7, 6, 0, 1, 4, 5, 3, 2}, {6, 5, 1, 2, 7, 4, 0, 3}, {5, 4, 2, 3, 6, 7, 1, 0}, {4, 7, 3, 0, 5, 6, 2, 1},
    {6, 7, 5, 4, 1, 0, 2, 3}, {7, 0, 4, 3, 6, 1, 5, 2}, {0, 1, 3, 2, 7, 6, 4, 5}, {1, 6, 2, 5, 0, 7, 3, 4},
    {2, 3, 1, 0, 5, 4, 6, 7}, {3, 4, 0, 7, 2, 5, 1, 6}, {4, 5, 7, 6, 3, 2, 0, 1}, {5, 2, 6, 1, 4, 3, 7, 0},
    {7, 0, 6, 1, 4, 3, 5, 2}, {0, 3, 1, 2, 7, 4, 6, 5}, {3, 4, 2, 5, 0, 7, 1, 6}, {4, 7, 5, 6, 3, 0, 2, 1},
    {6, 7, 1, 0, 5, 4, 2, 3}, {7, 4, 0, 3, 6, 5, 1, 2}, {4, 5, 3, 2, 7, 6, 0, 1}, {5, 6, 2, 1, 4, 7, 3, 0},
    {1, 6, 0, 7, 2, 5, 3, 4}, {6, 5, 7, 4, 1, 2, 0, 3}, {5, 2, 4, 3, 6, 1, 7, 0}, {2, 1, 3, 0, 5, 6, 4, 7},
    {0, 1, 7, 6, 3, 2, 4, 5}, {1, 2, 6, 5, 0, 3, 7, 4}, {2, 3, 5, 4, 1, 0, 6, 7}, {3, 0, 4, 7, 2, 1, 5, 6},
    {1, 0, 2, 3, 6, 7, 5, 4}, {0, 7, 3, 4, 1, 6, 2, 5}, {7, 6, 4, 5, 0, 1, 3, 2}, {6, 1, 5, 2, 7, 0, 4, 3},
    {5, 4, 6, 7, 2, 3, 1, 0}, {4, 3, 7, 0, 5, 2, 6, 1}, {3, 2, 0, 1, 4, 5, 7, 6}, {2, 5, 1, 6, 3, 4, 0, 7}};

/**
 * @brief Computes a Peano-Hilbert key for an integer triplet (x,y,z).
 *
 * This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
 * with x,y,z in the range between 0 and 2^bits-1.
 *
 * @param x The x-coordinate of the triplet.
 * @param y The y-coordinate of the triplet.
 * @param z The z-coordinate of the triplet.
 * @param bits The number of bits used to represent each coordinate.
 * @return The computed Peano-Hilbert key.
 */
peanokey peano_hilbert_key(int x, int y, int z, int bits)
{
    unsigned char rotation = 0;
    peanokey key = 0;

    for (int mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
        unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

        key <<= 3;
        key |= subpix3[rotation][pix];
        rotation = rottable3[rotation][pix];
    }

    return key;
}

/**
 * @brief Computes a Morton key (also known as Z-order or Morton code) for an integer triplet (x, y, z).
 *
 * This function computes a Morton key for an integer triplet (x, y, z),
 * with x, y, z in the range between 0 and 2^bits-1.
 *
 * @param x The x-coordinate of the triplet.
 * @param y The y-coordinate of the triplet.
 * @param z The z-coordinate of the triplet.
 * @param bits The number of bits used to represent each coordinate.
 * @return The computed Morton key.
 */
peanokey morton_key(int x, int y, int z, int bits)
{
    peanokey morton = 0;

    for (int mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
        morton <<= 3;
        morton += ((z & mask) ? 4 : 0) + ((y & mask) ? 2 : 0) + ((x & mask) ? 1 : 0);
    }

    return morton;
}

/**
 * @brief Computes both a Peano-Hilbert key and a Morton key for an integer triplet (x, y, z).
 *
 * This function computes both a Peano-Hilbert key and a Morton key for an integer triplet (x, y, z),
 * with x, y, z in the range between 0 and 2^bits-1.
 *
 * @param x The x-coordinate of the triplet.
 * @param y The y-coordinate of the triplet.
 * @param z The z-coordinate of the triplet.
 * @param bits The number of bits used to represent each coordinate.
 * @param morton_key A pointer to store the computed Morton key.
 * @return The computed Peano-Hilbert key.
 */
peanokey peano_and_morton_key(int x, int y, int z, int bits, peanokey *morton_key)
{
    unsigned char rotation = 0;
    peanokey key = 0;
    peanokey morton = 0;

    for (int mask = 1 << (bits - 1); mask > 0; mask >>= 1)
    {
        unsigned char pix = ((x & mask) ? 4 : 0) | ((y & mask) ? 2 : 0) | ((z & mask) ? 1 : 0);

        key <<= 3;
        key |= subpix3[rotation][pix];
        rotation = rottable3[rotation][pix];

        morton <<= 3;
        morton += ((z & mask) ? 4 : 0) + ((y & mask) ? 2 : 0) + ((x & mask) ? 1 : 0);
    }

    *morton_key = morton;

    return key;
}

/**
 * @brief Lookup table for quadrants.
 *
 * This static array defines the quadrants for rotations in a specific pattern.
 * It is used in various calculations involving rotations.
 */
static int quadrants[24][2][2][2] = {
    /* rotx=0, roty=0-3 */
    {{{0, 7}, {1, 6}}, {{3, 4}, {2, 5}}},
    {{{7, 4}, {6, 5}}, {{0, 3}, {1, 2}}},
    {{{4, 3}, {5, 2}}, {{7, 0}, {6, 1}}},
    {{{3, 0}, {2, 1}}, {{4, 7}, {5, 6}}},
    /* rotx=1, roty=0-3 */
    {{{1, 0}, {6, 7}}, {{2, 3}, {5, 4}}},
    {{{0, 3}, {7, 4}}, {{1, 2}, {6, 5}}},
    {{{3, 2}, {4, 5}}, {{0, 1}, {7, 6}}},
    {{{2, 1}, {5, 6}}, {{3, 0}, {4, 7}}},
    /* rotx=2, roty=0-3 */
    {{{6, 1}, {7, 0}}, {{5, 2}, {4, 3}}},
    {{{1, 2}, {0, 3}}, {{6, 5}, {7, 4}}},
    {{{2, 5}, {3, 4}}, {{1, 6}, {0, 7}}},
    {{{5, 6}, {4, 7}}, {{2, 1}, {3, 0}}},
    /* rotx=3, roty=0-3 */
    {{{7, 6}, {0, 1}}, {{4, 5}, {3, 2}}},
    {{{6, 5}, {1, 2}}, {{7, 4}, {0, 3}}},
    {{{5, 4}, {2, 3}}, {{6, 7}, {1, 0}}},
    {{{4, 7}, {3, 0}}, {{5, 6}, {2, 1}}},
    /* rotx=4, roty=0-3 */
    {{{6, 7}, {5, 4}}, {{1, 0}, {2, 3}}},
    {{{7, 0}, {4, 3}}, {{6, 1}, {5, 2}}},
    {{{0, 1}, {3, 2}}, {{7, 6}, {4, 5}}},
    {{{1, 6}, {2, 5}}, {{0, 7}, {3, 4}}},
    /* rotx=5, roty=0-3 */
    {{{2, 3}, {1, 0}}, {{5, 4}, {6, 7}}},
    {{{3, 4}, {0, 7}}, {{2, 5}, {1, 6}}},
    {{{4, 5}, {7, 6}}, {{3, 2}, {0, 1}}},
    {{{5, 2}, {6, 1}}, {{4, 3}, {7, 0}}}};

/**
 * @brief Lookup table for x-axis rotations.
 *
 * This static array defines the mapping of x-axis rotations for a specific pattern.
 * It is used in various calculations involving rotations.
 */
static int rotxmap_table[24] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 17, 18, 19, 16, 23, 20, 21, 22};

/**
 * @brief Lookup table for y-axis rotations.
 *
 * This static array defines the mapping of y-axis rotations for a specific pattern.
 * It is used in various calculations involving rotations.
 */
static int rotymap_table[24] = {1, 2, 3, 0, 16, 17, 18, 19, 11, 8, 9, 10, 22, 23, 20, 21, 14, 15, 12, 13, 4, 5, 6, 7};

static int rotx_table[8] = {3, 0, 0, 2, 2, 0, 0, 1};
static int roty_table[8] = {0, 1, 1, 2, 2, 3, 3, 0};
static int sense_table[8] = {-1, -1, -1, +1, +1, -1, -1, -1};

/**
 * @brief Sorts an array of peano_hilbert_data structures based on their key field using merge sort.
 *
 * This function sorts an array of peano_hilbert_data structures based on their key field using a merge sort algorithm.
 *
 * @param b Pointer to the array of peano_hilbert_data structures to be sorted.
 * @param n The number of elements in the array.
 * @param t Temporary storage for merging.
 */
static void msort_peano_with_tmp(struct peano_hilbert_data *b, size_t n, struct peano_hilbert_data *t)
{
    struct peano_hilbert_data *tmp;
    struct peano_hilbert_data *b1, *b2;
    size_t n1, n2;

    if (n <= 1)
    {
        return;
    }

    n1 = n / 2;
    n2 = n - n1;
    b1 = b;
    b2 = b + n1;

    msort_peano_with_tmp(b1, n1, t);
    msort_peano_with_tmp(b2, n2, t);

    tmp = t;

    while (n1 > 0 && n2 > 0)
    {
        if (b1->key <= b2->key)
        {
            --n1;
            *tmp++ = *b1++;
        }
        else
        {
            --n2;
            *tmp++ = *b2++;
        }
    }

    if (n1 > 0)
    {
        memcpy(tmp, b1, n1 * sizeof(struct peano_hilbert_data));
    }
    memcpy(b, t, (n - n2) * sizeof(struct peano_hilbert_data));
}

/**
 * @brief Sorts an array of elements using Peano-Hilbert key comparison.
 *
 * This function sorts an array of elements using the Peano-Hilbert key comparison method.
 *
 * @param b Pointer to the array of elements to be sorted.
 * @param n Number of elements in the array.
 * @param s Size of each element in bytes.
 * @param cmp Pointer to the comparison function which compares two elements.
 *            It should return an integer less than, equal to, or greater than zero
 *            if the first argument is considered to be respectively less than, equal to, or greater than the second.
 */
void mysort_peano(void *b, size_t n, size_t s, int (*cmp)(const void *, const void *))
{
    const size_t size = n * s;

    struct peano_hilbert_data *tmp = (struct peano_hilbert_data *)mymalloc("struct peano_hilbert_data *tmp", size);

    msort_peano_with_tmp((struct peano_hilbert_data *)b, n, tmp);

    myfree(tmp);
}
