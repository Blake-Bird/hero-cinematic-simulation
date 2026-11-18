/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the functions for the io.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Geray S. Karademir and Volker Springel.
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
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <algorithm>
#include <cstdint>

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif // HAVE_HDF5

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "io.hpp"
#include "io_matrix.hpp"

#ifdef COOLING
double GetCoolingTimeSTD(double u_old, double rho, double *ne_guess);
#endif // COOLING

#if defined(LT_METAL_COOLING_WAL) || defined(LDF_HYDROGEN_FRACTIONS)
#ifdef ACC
#define ACC_IO_C_INCLUDES_LT
#endif // ACC
// #undef ACC
#include "../CoolingSfr/Sfr_LT/lt_wal_cooling.hpp"
#endif // LT_METAL_COOLING_WAL


static int n_type[6];
static long long ntot_type_all[6];

#ifdef LT_SEvDbg
double metals[2 * LT_NMetP + 3], tot_metals[2 * LT_NMetP + 3];
#endif // LT_SEvDbg

#ifdef LT_SEv_INFO_DETAILS
double DetailsWSum[LT_NMetP], DetailsWoSum[LT_NMetP];
double my_weight_sum_details, My_weight_sum_details;
unsigned int *my_weight_sum_details_N;
#endif // LT_SEv_INFO_DETAILS

#if (defined(LT_STELLAREVOLUTION) || defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
#ifdef LT_LOG_SMOOTH
double metalmass;
void Spy_Z(int);
#endif // LT_LOG_SMOOTH
#endif //  (defined(LT_STELLAREVOLUTION) || defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING) ||
       //  defined(LT_METAL_COOLING_WAL))

#ifdef LT_METAL_COOLING_WAL
double Redshift, DZ;
#endif // LT_METAL_COOLING_WAL

#ifdef WRITE_KEY_FILES
struct key_file_index
{
    peanokey key;
    int filenr;
} *MyKeyData;

struct key_file_list
{
    peanokey low_list, high_list;
    int file_list;
} *KeyFileList, *KeyFileListAll;
#endif // WRITE_KEY_FILES

static int n_info;

/*!
 * @brief Writes a snapshot of the particle distribution to one or several files.
 *
 * This function outputs a snapshot of the particle distribution in Gadget's default
 * file format. Depending on the configuration, the snapshot can be split across
 * multiple files. If `NumFilesPerSnapshot` is greater than 1, the snapshot is divided
 * among several files, with each file containing data from a subset of processors
 * (approximately `NTask/NumFilesPerSnapshot` processors per file).
 *
 * @param num The snapshot file number, which determines the specific file being written.
 * @param flag An integer flag that can influence the file writing process. If `flag` is
 *             `-1`, the function writes the file regardless of the current `DumpFlag`
 *             value. Otherwise, the function only writes the file if `DumpFlag` is `1`
 *             or `2`.
 */
void savepositions(int num, int flag)
{
    char buf[500];
    int filenr, masterTask, lastTask;

#if defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)
    rearrange_particle_sequence();
#endif // if defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

#ifdef PHIDOT
    phidot();
#endif // PHIDOT

#ifdef LMB_CR_OUTPUT_SYNCHROTRON
    synchrotron_emissivity();
#endif // LMB_CR_OUTPUT_SYNCHROTRON

#if defined(JD_DPP) && defined(JD_DPPONSNAPSHOTONLY)
    compute_Dpp(0); /* Particle loop already inside, just add water */
#endif // if defined(JD_DPP) && defined(JD_DPPONSNAPSHOTONLY)

    // Check if it's necessary to write file
    if ((DumpFlag == 1 || DumpFlag == 2) || (flag == -1))
    {
        TIMER_PUSH(0, "IO");
        TIMER_PUSH(0, "SNAPSHOT");

        VERBOSE(2, "\nwriting snapshot file #%d...  ", num);
#ifdef ORDER_SNAPSHOTS_BY_ID
        double t0, t1;

        if (All.TotN_gas > 0)
        {
            ENDRUN_MAIN("The option ORDER_SNAPSHOTS_BY_ID does not work yet with gas particles, only simulations with "
                        "collisionless particles are allowed.");
        }

        t0 = second();
        for (int n = 0; n < NumPart; n++)
        {
            P[n].GrNr = ThisTask;
            P[n].SubNr = n;
        }
        parallel_sort(P, NumPart, sizeof(struct particle_data), io_compare_P_ID);
        t1 = second();
        VERBOSE(2, "Reordering of particle-data in ID-sequence took = %g sec ", timediff(t0, t1));
#endif // ORDER_SNAPSHOTS_BY_ID

// Write key files
#ifdef WRITE_KEY_FILES
        for (int n = 0; n < 6; n++)
        {
            NKeys[n] = NKeysTotal[n] = 0;
        }
        domain_Decomposition(0, 1);
#endif // WRITE_KEY_FILES

        CommBuffer = mymalloc("CommBuffer", All.BufferSize * 1024.0 * 1024.0);

        VERBOSE(3, "IO: using %g MB for buffering, %g MB remaining for other buffers ", All.BufferSize,
                FreeBytes / (1024.0 * 1024.0));

        // Check number of processors
        if (NTask < All.NumFilesPerSnapshot)
        {
            ENDRUN_MAIN("Fatal error.\nNumber of processors must be larger or equal than All.NumFilesPerSnapshot. ");
        }
        // Check snapshot file format
        if (All.SnapFormat < 1 || All.SnapFormat > 3)
        {
            ENDRUN_MAIN("Unsupported File-Format.");
        }
#ifndef HAVE_HDF5
        if (All.SnapFormat == 3)
        {
            ENDRUN_MAIN("Code wasn't compiled with HDF5 support enabled! ");
        }
#endif // HAVE_HDF5

        // Determine global and local particle numbers
        for (int n = 0; n < 6; n++)
        {
            n_type[n] = 0;
        }

        for (int n = 0; n < NumPart; n++)
        {
            n_type[P[n].Type]++;
        }

        sumup_large_ints(6, n_type, ntot_type_all);

// Determine global values across all processors
#ifdef WRITE_KEY_FILES
        sumup_large_ints(6, NKeys, NKeysTotal);
#endif // WRITE_KEY_FILES

#ifdef LT_SEvDbg
        for (int i = 0; i < 2 * LT_NMetP + 3; i++)
        {
            metals[i] = tot_metals[i] = 0;
        }
#endif // LT_SEvDbg

#ifdef LT_SEv_INFO_DETAILS
        MPI_Reduce(DetailsW, DetailsWSum, LT_NMetP, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
        MPI_Reduce(DetailsWo, DetailsWoSum, LT_NMetP, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);

        if (ThisTask == 0)
        {
            for (int i = 0; i < LT_NMetP; i++)
            {
                VERBOSE(5, "%g DETAILS: %4s :: %12.8g  *  %12.8g ", All.Time, MetNames[i], DetailsWSum[i],
                        DetailsWoSum[i]);
            }
        }
#endif // LT_SEv_INFO_DETAILS

#ifdef LT_METAL_COOLING_WAL
        if (All.ComovingIntegrationOn)
        {
            Redshift = 1.0 / All.Time - 1; /* note that ComovingIntegration is alway on with wal cooling */
        }
        else
        {
            Redshift = 0.0;
        }
        get_cool_redshift(Redshift, &DZ);
#endif // LT_METAL_COOLING_WAL

        /* assign processors to output files */
        distribute_file(All.NumFilesPerSnapshot, 0, 0, NTask - 1, &filenr, &masterTask, &lastTask);

        if (All.NumFilesPerSnapshot > 1)
        {
            if (ThisTask == 0)
            {
#ifdef LONG_OUTPUT_NUMBERS
                sprintf(buf, "%s/snapdir_%04d", All.OutputDir, num);
#else // LONG_OUTPUT_NUMBERS
                sprintf(buf, "%s/snapdir_%03d", All.OutputDir, num);
#endif // LONG_OUTPUT_NUMBERS
                mkdir(buf, 02755);
            }
            MPI_Barrier(MYMPI_COMM_WORLD);
        }

#ifdef LONG_OUTPUT_NUMBERS
        if (All.NumFilesPerSnapshot > 1)
        {
            sprintf(buf, "%s/snapdir_%04d/%s_%04d.%d", All.OutputDir, num, All.SnapshotFileBase, num, filenr);
        }
        else
        {
            sprintf(buf, "%s%s_%04d", All.OutputDir, All.SnapshotFileBase, num);
        }
#else // LONG_OUTPUT_NUMBERS
        if (All.NumFilesPerSnapshot > 1)
        {
            sprintf(buf, "%s/snapdir_%03d/%s_%03d.%d", All.OutputDir, num, All.SnapshotFileBase, num, filenr);
        }
        else
        {
            sprintf(buf, "%s%s_%03d", All.OutputDir, All.SnapshotFileBase, num);
        }
#endif // LONG_OUTPUT_NUMBERS

#if (defined(LT_STELLAREVOLUTION) || defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING) || defined(LT_METAL_COOLING_WAL))
        XH = HYDROGEN_MASSFRAC;
        yhelium = (1 - XH) / (4 * XH);

#ifdef LT_LOG_SMOOTH
        VERBOSE(5, "logging z-smooth effect.. ");
        Spy_Z(num);
#endif // LT_LOG_SMOOTH
#endif // if (defined(LT_STELLAREVOLUTION) || defined(LT_SMOOTH_Z) || defined(LT_METAL_COOLING) ||
       // defined(LT_METAL_COOLING_WAL))

        int ngroups = All.NumFilesPerSnapshot / All.NumFilesWrittenInParallel;
        if ((All.NumFilesPerSnapshot % All.NumFilesWrittenInParallel))
        {
            ngroups++;
        }

        for (int gr = 0; gr < ngroups; gr++)
        {
            if ((filenr / All.NumFilesWrittenInParallel) == gr) /* ok, it's this processor's turn */
            {
                write_file(buf, masterTask, lastTask);
            }
            MPI_Barrier(MYMPI_COMM_WORLD);
        }

        myfree(CommBuffer);

#ifdef ORDER_SNAPSHOTS_BY_ID
        TIMER_PUSH(2, "restoring_order_of_particle_data");
        parallel_sort(P, NumPart, sizeof(struct particle_data), io_compare_P_GrNr_SubNr);
        TIMER_POP(2, "restoring_order_of_particle_data");
#endif // ORDER_SNAPSHOTS_BY_ID

#ifdef JD_RELAX_CLUSTERS
        VERBOSE(3, "Setting vel=0  ");
        for (size_t ipart = 0; ipart < N_gas; ipart++)
        {
            P[0].Vel[0] = P[0].Vel[1] = P[0].Vel[2] = 0;
        }
#endif // JD_RELAX_CLUSTERS

#ifdef WRITE_KEY_FILES
        write_key_index_file(num, filenr);
#endif // WRITE_KEY_FILES

        VERBOSE(2, "Done with snapshot.");

        All.Ti_lastoutput = All.Ti_Current;

        TIMER_POP(0, "SNAPSHOT");
        TIMER_POP(0, "IO");
#ifdef SUBFIND_RESHUFFLE_CATALOGUE
        ENDRUN_MAIN("Ending run due to SUBFIND_RESHUFFLE_CATALOGUE ");
#endif // SUBFIND_RESHUFFLE_CATALOGUE
    }

    /*
    #ifdef WRITE_KEY_FILES
         domain_Decomposition(0, 0);
    #endif // WRITE_KEY_FILES
    */

#if defined(FOF) && !defined(FOF_LIGHTCONE)
    if (RestartFlag != 4)
    {
        VERBOSE(3, "\ncomputing group catalogue... ");

        fof_fof(num);

        VERBOSE(4, "done with group catalogue. ");
    }
#endif // defined(FOF) && !defined(FOF_LIGHTCONE)

#ifdef POWERSPEC_ON_OUTPUT
    if (RestartFlag != 4)
    {
        VERBOSE(3, "\ncomputing power spectra... ");

        calculate_power_spectra(num, &ntot_type_all[0]);

        VERBOSE(4, "done with power spectra. ");
    }
#endif // POWERSPEC_ON_OUTPUT

#ifdef LT_SEvDbg
    MPI_Reduce(&metals[0], &tot_metals[0], 2 * LT_NMetP + 3, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    if (ThisTask == 0)
    {
        fprintf(FdMetSumCheck, "@.@ %g ", All.Time);
        for (int i = 0; i < LT_NMetP * 2 + 3; i++)
        {
            fprintf(FdMetSumCheck, " %g ", tot_metals[i] / All.HubbleParam);
        }
        fprintf(FdMetSumCheck, " \n ");
    }
#endif // LT_SEvDbg
}

#ifdef OUTPUT_IN_FP16
/* Encodes log10(f) into a float16.
 * Input:  f  > 0  (positive finite float; zero/negative/NaN -> encodes -Inf or NaN)
 * Output: float16 bit pattern representing log10(f)
 *
 * Range:  log10 of the input is computed in float32 first, then packed into float16.
 *         float16 can hold values up to ~65504, so this covers inputs up to 10^65504.
 *         The practical lower bound is ~10^-65504, far below any real sensor floor.
 */
static uint16_t float_to_log10_half(float f)
{
    // compute log10 in float32
    float log_val = log10f(f);
    /*
     * Edge cases from log10f:
     *   f <= 0  -> log10f returns -Inf (f==0) or NaN (f<0); both propagate correctly
     *   f = NaN -> log10f returns NaN;
     *   f = +Inf-> log10f returns +Inf;
     * All these will propagte through the rest of the function.
     */

    // convert float32 log value into float16
    uint32_t x;
    memcpy(&x, &log_val, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) /* Inf or NaN */
    {
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp <= 0)
    {
        if (exp < -10) /* too small even for subnormal half -> zero */
        {
            return (uint16_t)sign;
        }
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        uint32_t remainder = mant & ((1u << shift) - 1u);
        if (remainder > (1u << (shift - 1)) || (remainder == (1u << (shift - 1)) && (half_mant & 1u)))
        {
            half_mant++;
        }
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) /* overflow -> Inf */
    {
        return (uint16_t)(sign | 0x7C00u);
    }
    uint32_t half_mant = mant >> 13;
    uint32_t remainder = mant & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mant & 1u)))
    {
        half_mant++;
        if (half_mant == 0x400u) /* mantissa carry into exponent */
        {
            half_mant = 0;
            exp++;
            if (exp >= 31)
            {
                return (uint16_t)(sign | 0x7C00u);
            }
        }
    }
    return (uint16_t)(sign | (uint32_t)(exp << 10) | half_mant);
}

/**
 * @brief Decodes an IEEE-754 binary16 bit pattern into a float32.
 * @param h Half-precision bit pattern (as packed by float_to_log10_half()).
 * @return Decoded float32 value.
 */
static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0)
    {
        if (mant == 0)
        {
            bits = sign; /* zero */
        }
        else
        {
            /* subnormal half -> normalize into a float32 exponent/mantissa */
            int shift = 0;
            while (!(mant & 0x400u))
            {
                mant <<= 1;
                shift++;
            }
            mant &= 0x3FFu;
            uint32_t fexp = (uint32_t)(127 - 15 - shift);
            bits = sign | (fexp << 23) | (mant << 13);
        }
    }
    else if (exp == 0x1Fu)
    {
        bits = sign | 0x7F800000u | (mant << 13); /* Inf or NaN */
    }
    else
    {
        uint32_t fexp = exp - 15 + 127;
        bits = sign | (fexp << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/**
 * @brief Inverts float_to_log10_half(): decodes a half-precision log10 value and exponentiates it.
 * @param h Half-precision bit pattern holding log10(value).
 * @return The value in float, 10^(decoded half).
 */
float float_from_log10_half(uint16_t h)
{
    return powf(10.0f, half_to_float(h));
}
#endif // OUTPUT_IN_FP16

/* Helper macros for the simple cases in fill_write_buffer_block_switch(): scan the next `pc` particles
 * of type `type` starting at `pindex`, writing one scalar/vector value per particle into `buf`. */
#define IO_WRITE_SCALAR(buf, expr)                                                                                     \
    for (int n = 0; n < pc; pindex++)                                                                                  \
    {                                                                                                                  \
        if (P[pindex].Type == type)                                                                                    \
        {                                                                                                              \
            *(buf)++ = (expr);                                                                                         \
            n++;                                                                                                       \
        }                                                                                                              \
    }

#define IO_WRITE_VEC(buf, ncomp, expr)                                                                                 \
    for (int n = 0; n < pc; pindex++)                                                                                  \
    {                                                                                                                  \
        if (P[pindex].Type == type)                                                                                    \
        {                                                                                                              \
            for (int k = 0; k < (ncomp); k++)                                                                          \
            {                                                                                                          \
                *(buf)++ = (expr);                                                                                     \
            }                                                                                                          \
            n++;                                                                                                       \
        }                                                                                                              \
    }

#if defined(COOLING) && defined(LDF_HYDROGEN_FRACTIONS)
/**
 * @brief Compute hydrogen fractions for a given particle based on its internal energy,
 *        density, and metallicity interpolating LDF cooling tables.
 *
 * @param pindex   Index of the particle
 * @param a3inv    Cosmological scale factor to the power -3 (1/a^3)
 * @param Redshift Current redshift.
 * @param DZ       Interpolating coefficient for redshift.
 *
 * @return An Hfraction struct containing the computed fractions for the
 *         particle (e.g. HI_frac, HII_frac, H2_frac).
 *
 * @warning Only compiled when COOLING + LT_METAL_COOLING_WAL and LDF_HYDROGEN_FRACTIONS are
 *          defined
 */
static inline Hfraction compute_hydrogen_fractions(int pindex, double a3inv, double Redshift, double DZ)
{
    double Metallicities[LT_NMet];
    double rho = SphP[pindex].Density * a3inv;
    double fac_entr_to_u = pow(rho, GAMMA_MINUS1) / GAMMA_MINUS1;
    double u = max(All.MinEgySpec, (double)SphP[pindex].Entropy * fac_entr_to_u);
    set_metallicities(pindex, &Metallicities[0], a3inv);
    u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
    double temp = convert_u_to_tempMET(u, Redshift, DZ, &Metallicities[0]);

    return AbundanceRatios_fromCoolingTables(temp, Redshift, DZ, &Metallicities[0]);
}
#endif // defined(COOLING) && defined(LDF_HYDROGEN_FRACTIONS)

/**
 * @brief Executes the per-block-type logic for fill_write_buffer().
 *
 * Contains the switch over `blocknr` that fills CommBuffer with particle
 * data of type `type`, scanning particle indices starting at `pindex`.
 *
 * @param blocknr The block number specifying which type of data to write.
 * @param pindex The particle index to start scanning from.
 * @param pc The number of particles of type `type` to write.
 * @param type The particle type being processed.
 * @param a3inv Precomputed comoving volume scale factor 1/a^3.
 * @return The particle index following the last particle scanned.
 */
static int fill_write_buffer_block(enum iofields blocknr, int pindex, int pc, int type, double a3inv
#ifdef OUTPUTACCELERATION
                                   ,
                                   double fac1, double fac2
#endif // OUTPUTACCELERATION
#ifdef MAGNETIC
                                   ,
                                   double a2_inv, double gadget2gauss
#endif // MAGNETIC
)
{
    MyOutputFloat *fp = (MyOutputFloat *)CommBuffer;
    MyIDType *ip = (MyIDType *)CommBuffer;
    int *ip_int = (int *)CommBuffer;
#ifdef OUTPUT_IN_FP16
    uint16_t *fp16 = (uint16_t *)CommBuffer;
#endif // OUTPUT_IN_FP16

#ifdef PERIODIC
    MyLongDouble boxSize;
#endif // PERIODIC

#if defined(OUTPUTCOOLRATE) || defined(OUTPUTCOOLTIME)
    double tcool, u;
#endif // OUTPUTCOOLRATE || OUTPUTCOOLTIME

#if defined(LT_STELLAREVOLUTION) && !defined(GM_MUPPI)
    float Temperature, xclouds;
#endif // defined(LT_STELLAREVOLUTION) && !defined(GM_MUPPI)

#ifdef LT_STELLAREVOLUTION
    int control[LT_NMetP];
#endif // LT_STELLAREVOLUTION

#ifdef LT_TRACK_CONTRIBUTES
    Contrib *contribp = (Contrib *)CommBuffer;
#endif // LT_TRACK_CONTRIBUTES

#if defined(LT_SMOOTH_ALLMETALS) || (defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)) ||                         \
    defined(LT_TRACK_CONTRIBUTES) || (defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND))
    float *fp_single = (float *)CommBuffer;
#endif // defined(LT_SMOOTH_ALLMETALS) || (defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)) ||
       // defined(LT_TRACK_CONTRIBUTES) || (defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND))

    switch (blocknr)
    {
    case IO_POS: /* positions */
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int k = 0; k < 3; k++)
                {
                    MyLongDouble mypos = P[pindex].Pos[k];
#ifdef PERIODIC
                    boxSize = All.BoxSize;
#ifdef LONG_X
                    if (k == 0)
                    {
                        boxSize = All.BoxSize * LONG_X;
                    }
#endif // LONG_X
#ifdef LONG_Y
                    if (k == 1)
                    {
                        boxSize = All.BoxSize * LONG_Y;
                    }
#endif // LONG_Y
#ifdef LONG_Z
                    if (k == 2)
                    {
                        boxSize = All.BoxSize * LONG_Z;
                    }
#endif // LONG_Z
                    while (mypos < 0)
                    {
                        mypos += boxSize;
                    }
                    while (mypos >= boxSize)
                    {
                        mypos -= boxSize;
                    }
#ifdef DS_SHIFT_BOX
                    mypos -= boxSize / 2;
#endif // DS_SHIFT_BOX
#endif // PERIODIC
                    fp[k] = mypos;
                }
                n++;
                fp += 3;
            }
        }
        break;

    case IO_VEL: /* velocities */
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
#ifdef SYNCRONIZ_OUTPUT
                for (int k = 0; k < 3; k++)
                {
                    fp[k] = P[pindex].Vel[k];
                }
#else // SYNCRONIZ_OUTPUT
                integertime dt_step = (P[pindex].TimeBin ? (((integertime)1) << P[pindex].TimeBin) : 0);
                double dt_gravkick = 0, dt_hydrokick = 0;
                if (All.ComovingIntegrationOn)
                {
                    dt_gravkick = get_gravkick_factor(P[pindex].Ti_begstep, All.Ti_Current) -
                                  get_gravkick_factor(P[pindex].Ti_begstep, P[pindex].Ti_begstep + dt_step / 2);
                    dt_hydrokick = get_hydrokick_factor(P[pindex].Ti_begstep, All.Ti_Current) -
                                   get_hydrokick_factor(P[pindex].Ti_begstep, P[pindex].Ti_begstep + dt_step / 2);
                }
                else
                {
                    dt_gravkick = dt_hydrokick =
                        (All.Ti_Current - (P[pindex].Ti_begstep + dt_step / 2)) * All.Timebase_interval;
                }

                for (int k = 0; k < 3; k++)
                {
                    fp[k] = P[pindex].Vel[k];

#ifdef SPH_BND_PARTICLES
                    if (P[pindex].ID != 0)
#endif // SPH_BND_PARTICLES
                    {
                        fp[k] += P[pindex].GravAccel[k] * dt_gravkick;
#ifdef PMGRID
                        fp[k] += P[pindex].GravPM[k] * dt_gravkick;
#endif // PMGRID
                        if (P[pindex].Type == type_gas)
                        {
                            fp[k] += SphP[pindex].HydroAccel[k] * dt_hydrokick;
                        }
                    }
                }

#endif // SYNCRONIZ_OUTPUT
                for (int k = 0; k < 3; k++)
                {
                    fp[k] *= sqrt(a3inv);
                }
                n++;
                fp += 3;
            }
        }
        break;

    case IO_ID: /* particle ID */
        IO_WRITE_SCALAR(ip, P[pindex].ID)
        break;

    case IO_MASS: /* particle mass */
#ifdef OUTPUT_IN_FP16
        IO_WRITE_SCALAR(fp16, float_to_log10_half((float)P[pindex].Mass))
#else // OUTPUT_IN_FP16
        IO_WRITE_SCALAR(fp, P[pindex].Mass)
#endif // OUTPUT_IN_FP16
        break;

    case IO_U: /* internal energy */
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                *fp++ = max(All.MinEgySpec, (double)(SphP[pindex].Entropy / GAMMA_MINUS1 *
                                                     pow(SphP[pindex].Density * a3inv, GAMMA_MINUS1)
#ifdef GT_DUST
                                                     * pow(1. - SphP[pindex].TotalDustFrac, GAMMA_MINUS1)
#endif // GT_DUST
                                                         ));
                n++;
            }
        }
        break;

    case IO_RHO: /* density */
        IO_WRITE_SCALAR(fp, SphP[pindex].Density)
        break;

    case IO_NE: /* electron abundance */
#if defined(COOLING) || defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        for (int n = 0; n < pc; pindex++)
            if (P[pindex].Type == type)
            {
#if defined(LT_METAL_COOLING_WAL) && defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)
                /* XXXX GM FIX */
                double ne = 0, nh0 = 0, nHeII = 0;
                double Metallicities[LT_NMet];

                double rho = SphP[pindex].Density * a3inv;
                double fac_entr_to_u = pow(rho, GAMMA_MINUS1) / GAMMA_MINUS1;
                double u = max(All.MinEgySpec, (double)SphP[pindex].Entropy * fac_entr_to_u);

                if (SphP[pindex].elec >= 0 && SphP[pindex].elec <= 2)
                {
                    ne = SphP[pindex].elec;
                }
                else
                {
                    ne = 0;
                    VERBOSE_ALL(1, "Task=%d: WARNING, SphP[%d].elec = %g ", ThisTask, pindex, SphP[pindex].elec);
                }

                set_metallicities(pindex, &Metallicities[0], a3inv);
                u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
                double temp = convert_u_to_tempMET(u, Redshift, DZ, &Metallicities[0]);

                if (SphP[pindex].Density < 1e-20 || SphP[pindex].Density > 1e20 || temp < 10 || temp > 1e10)
                {
                    VERBOSE_ALL(1, "Task %d: WARNING !!! Something odd when writing Ne into snapfile rho,temp %g,%g ",
                                ThisTask, SphP[pindex].Density, temp);
                    nh0 = ne = 0;
                }
                else
                {
                    nh0 = AbundanceRatios(temp, rho, &ne, &nh0, &nHeII);
                }
                SphP[pindex].elec = ne;

#endif //  defined(LT_METAL_COOLING_WAL) && defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)
                *fp++ = SphP[pindex].elec;
                n++;
            }
#endif // defined(COOLING) || defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_NH: /* neutral hydrogen fraction */
#if defined(COOLING) || defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
                *fp++ = SphP[pindex].HI;
#elif defined(LDF_HYDROGEN_FRACTIONS)
                Hfraction Hfrac = compute_hydrogen_fractions(pindex, a3inv, Redshift, DZ);
                *fp++ = (MyOutputFloat)Hfrac.HI_frac;
#else // COOLING only
                double nh0 = 0;

#if !defined(LT_METAL_COOLING_WAL) || defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)
                double ne = 0;
#endif // !defined(LT_METAL_COOLING_WAL) || defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)

#if !defined(LT_METAL_COOLING_WAL) || defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)
                double fac_entr_to_u = pow(SphP[pindex].Density * a3inv, GAMMA_MINUS1) / GAMMA_MINUS1;
                double u = max(All.MinEgySpec, (double)SphP[pindex].Entropy * fac_entr_to_u);
#endif // !defined( LT_METAL_COOLING_WAL) || defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)

                if (SphP[pindex].elec >= 0 && SphP[pindex].elec <= 2)
                {
#if !defined(LT_METAL_COOLING_WAL) || defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)
                    ne = SphP[pindex].elec;
#endif // !defined(LT_METAL_COOLING_WAL) || defined(FORCE_COMPUTE_NETWORK_ON_OUTPUT)
                }
                else
                {
                    VERBOSE_ALL(1, "Task=%d: WARNING, SphP[%d].elec = %g ", ThisTask, pindex, SphP[pindex].elec);
                }

#ifndef LT_METAL_COOLING_WAL
                double nHeII = 0;
                AbundanceRatios(u, SphP[pindex].Density * a3inv, &ne, &nh0, &nHeII);
#else // LT_METAL_COOLING_WAL
#ifdef FORCE_COMPUTE_NETWORK_ON_OUTPUT
                double nHeII = 0;
                double Metallicities[LT_NMet];
                double rho = SphP[pindex].Density * a3inv;

                set_metallicities(pindex, &Metallicities[0], a3inv);
                u *= All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
                double temp = convert_u_to_tempMET(u, Redshift, DZ, &Metallicities[0]);

                if (SphP[pindex].Density < 1e-20 || SphP[pindex].Density > 1e20 || temp < 10 || temp > 1e10)
                {
                    VERBOSE_ALL(1, "Task %d: WARNING !!! Something odd when writing Ne into snapfile rho,temp %g,%g ",
                                ThisTask, SphP[pindex].Density, temp);
                    nh0 = ne = 0;
                }
                else
                {
                    nh0 = AbundanceRatios(temp, rho, &ne, &nh0, &nHeII);
                }
#endif // FORCE_COMPUTE_NETWORK_ON_OUTPUT
#endif // LT_METAL_COOLING_WAL
                *fp++ = (MyOutputFloat)nh0;
#endif // COOLING only
                n++;
            }
        }
#endif
        break;

    case IO_HII: /* ionized hydrogen abundance */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        IO_WRITE_SCALAR(fp, SphP[pindex].HII)
#elif defined(COOLING) && defined(LDF_HYDROGEN_FRACTIONS)
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                Hfraction Hfrac = compute_hydrogen_fractions(pindex, a3inv, Redshift, DZ);
                *fp++ = (MyOutputFloat)Hfrac.HII_frac;
                n++;
            }
        }
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_HeI: /* neutral Helium */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        IO_WRITE_SCALAR(fp, SphP[pindex].HeI)
#endif //  defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_HeII: /* ionized Heluum */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        IO_WRITE_SCALAR(fp, SphP[pindex].HeII)
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_HeIII: /* double ionised Helium */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        IO_WRITE_SCALAR(fp, SphP[pindex].HeIII)
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_H2I: /* H2 molecule */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        IO_WRITE_SCALAR(fp, SphP[pindex].H2I)
#elif defined(COOLING) && defined(LDF_HYDROGEN_FRACTIONS)
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                Hfraction Hfrac = compute_hydrogen_fractions(pindex, a3inv, Redshift, DZ);
                *fp++ = (MyOutputFloat)Hfrac.H2_frac;
                n++;
            }
        }
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_H2II: /* ionised H2 molecule */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        IO_WRITE_SCALAR(fp, SphP[pindex].H2II)
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_HM: /* H minus */
#if defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        IO_WRITE_SCALAR(fp, SphP[pindex].HM)
#endif // defined(CHEMISTRY) || defined(UM_CHEMISTRY)
        break;

    case IO_HD: /* HD */
#ifdef UM_CHEMISTRY
        IO_WRITE_SCALAR(fp, SphP[pindex].HD)
#endif // UM_CHEMISTRY
        break;

    case IO_DI: /* deuterium */
#ifdef UM_CHEMISTRY
        IO_WRITE_SCALAR(fp, SphP[pindex].DI)
#endif // UM_CHEMISTRY
        break;

    case IO_DII: /* deuteriumII */
#ifdef UM_CHEMISTRY
        IO_WRITE_SCALAR(fp, SphP[pindex].DII)
#endif // UM_CHEMISTRY
        break;

    case IO_HeHII: /* HeH+ */
#ifdef UM_CHEMISTRY
        IO_WRITE_SCALAR(fp, SphP[pindex].HeHII)
#endif // UM_CHEMISTRY
        break;

    case IO_HSML: /* SPH smoothing length */
        IO_WRITE_SCALAR(fp, P[pindex].Hsml)
        break;

    case IO_SFR: /* star formation rate */
#ifdef SFR
#if !defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)
        IO_WRITE_SCALAR(fp, get_starformation_rate(pindex))
#else // !defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)
        IO_WRITE_SCALAR(fp, get_starformation_rate(pindex, &Temperature, &xclouds))
#endif // !defined(LT_STELLAREVOLUTION) || defined(GM_MUPPI)
#endif // SFR
        break;

    case IO_AGE: /* stellar formation time */
#ifdef SFR
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                if (P[pindex].Type == type_star)
                {
                    *fp++ = MPP(pindex).StellarAge;
                }

#ifdef BLACK_HOLES
                if (P[pindex].Type == type_bh)
                {
                    *fp++ = BPP(pindex).StellarAge;
                }
#endif // BLACK_HOLES
                n++;
            }
        }
#endif // SFR
        break;

    case IO_HSMS: /* smoothing length for star particles */
#ifdef SUBFIND
        IO_WRITE_SCALAR(fp, P[pindex].DM_Hsml)
#endif // SUBFIND
        break;

    case IO_ACRS: /* accreation length for star particles */
#ifdef LT_STELLAREVOLUTION
        IO_WRITE_SCALAR(fp, P[pindex].Hsml)
#endif // LT_STELLAREVOLUTION
        break;

    case IO_Z: /* gas and star metallicity */
#if defined(SFR) && !defined(LT_STELLAREVOLUTION)
#ifdef OUTPUT_IN_FP16
        IO_WRITE_SCALAR(fp16, float_to_log10_half((float)P[MPP(pindex)].Metallicity))
#else // OUTPUT_IN_FP16
        IO_WRITE_SCALAR(fp, MPP(pindex).Metallicity)
#endif // OUTPUT_IN_FP16
#endif // SFR && !LT_STELLAREVOLUTION
        break;

    case IO_POT: /* gravitational potential */
#if defined(OUTPUTPOTENTIAL) || defined(SUBFIND_RESHUFFLE_AND_POTENTIAL)
        IO_WRITE_SCALAR(fp, P[pindex].Potential)
#endif // defined(OUTPUTPOTENTIAL)  || defined(SUBFIND_RESHUFFLE_AND_POTENTIAL)
        break;

    case IO_PHIDOT: /* gravitational potential dt */
#ifdef PHIDOT
        IO_WRITE_SCALAR(fp, SphP[pindex].PhiDot)
#endif // PHIDOT
        break;

    case IO_ACCEL: /* acceleration */
#ifdef OUTPUTACCELERATION
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int k = 0; k < 3; k++)
                {
                    fp[k] = fac1 * P[pindex].GravAccel[k];
                }
#ifdef PMGRID
                for (int k = 0; k < 3; k++)
                {
                    fp[k] += fac1 * P[pindex].GravPM[k];
                }
#endif // PMGRID

                if (P[pindex].Type == type_gas)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        fp[k] += fac2 * SphP[pindex].HydroAccel[k];
                    }
                }
                fp += 3;
                n++;
            }
        }
#endif // OUTPUTACCELERATION
        break;

    case IO_DTENTR: /* rate of change of entropy */
#ifdef OUTPUTCHANGEOFENTROPY
        IO_WRITE_SCALAR(fp, SphP[pindex].DtEntropy)
#endif // OUTPUTCHANGEOFENTROPY
        break;

    case IO_STRESSDIAG: /* Diagonal components of viscous shear tensor */
#ifdef OUTPUTSTRESS
        IO_WRITE_VEC(fp, 3, SphP[pindex].StressDiag[k])
#endif // OUTPUTSTRESS
        break;

    case IO_STRESSOFFDIAG: /* Offdiagonal components of viscous shear tensor */
#ifdef OUTPUTSTRESS
        IO_WRITE_VEC(fp, 3, SphP[pindex].StressOffDiag[k])
#endif // OUTPUTSTRESS
        break;

    case IO_STRESSBULK: /* Viscous bulk tensor */
#ifdef OUTPUTBULKSTRESS
        IO_WRITE_SCALAR(fp, SphP[pindex].StressBulk)
#endif // OUTPUTBULKSTRESS
        break;

    case IO_DELAYTIME:
#ifdef WINDS
        IO_WRITE_SCALAR(fp, SphP[pindex].DelayTime)
#endif // WINDS
        break;

    case IO_DECOUPLE_TIME_AGN:
#ifdef PB_BH_KIN_FEED_DECOUPLE_WIND
        IO_WRITE_SCALAR(fp, SphP[pindex].DecoupleTime_AGN)
#endif // PB_BH_KIN_FEED_DECOUPLE_WIND
        break;

    case IO_SHEARCOEFF: /* Shear viscosity coefficient */
#ifdef OUTPUTSHEARCOEFF
        IO_WRITE_SCALAR(fp, SphP[pindex].shear_visc_coeff)
#endif // OUTPUTSHEARCOEFF
        break;

    case IO_PRESSURE_ANISO: /* Pressure anisotropy */
#ifdef OUTPUT_PRESSURE_ANISO
        IO_WRITE_SCALAR(fp, SphP[pindex].P_aniso)
#endif // PRESSURE_ANISO
        break;

    case IO_ENERGY_VISC: /* Viscous heating */
#ifdef OUTPUT_ENERGY_VISC
        IO_WRITE_SCALAR(fp, SphP[pindex].visc_E)
#endif // ENERGY_VISC
        break;

    case IO_TSTP: /* timestep  */
#ifdef OUTPUTTIMESTEP
        IO_WRITE_SCALAR(fp, (P[pindex].TimeBin ? (((integertime)1) << P[pindex].TimeBin) : 0) * All.Timebase_interval)
#endif // OUTPUTTIMESTEP
        break;

    case IO_BFLD: /* magnetic field  */
#ifdef MAGNETIC
        IO_WRITE_VEC(fp, 3, SphP[pindex].BPred[k] * a2_inv * gadget2gauss)
#endif // MAGNETIC
        break;

    case IO_BCRV: /* magnetic field curvature */
#if defined(MAGNETIC) && defined(OUTPUT_MAGNETIC_CURVATURE)
        IO_WRITE_VEC(fp, 3, SphP[pindex].MagCurv[k] * a2_inv * gadget2gauss)
#endif // MAGNETIC
        break;

    case IO_GRADU: /* gradU from TIME_DEP_ART_COND  */
#ifdef TIME_DEP_ART_COND
        IO_WRITE_VEC(fp, 3, SphP[pindex].GradA[k])
#endif // TIME_DEP_ART_COND
        break;

    case IO_DBDT: /* rate of change of magnetic field  */
#ifdef DBOUTPUT
        IO_WRITE_VEC(fp, 3, SphP[pindex].DtB[k] * a2_inv * gadget2gauss)
#endif // DBOUTPUT
        break;

    case IO_VRMS: /* turbulent velocity around mean */
#ifdef JD_VTURB
        IO_WRITE_SCALAR(fp, SphP[pindex].Vrms * sqrt(a3inv))
#endif // JD_VTURB
        break;

    case IO_VBULK: /* mean velocity in kernel */
#ifdef JD_VTURB
        IO_WRITE_VEC(fp, 3, SphP[pindex].Vbulk[k] * sqrt(a3inv))
#endif // JD_VTURB
        break;

    case IO_VTAN: /* turbulent velocity around mean, tangential part */
#ifdef JD_DECOMPOSE_VTURB
        IO_WRITE_SCALAR(fp, SphP[pindex].Vtan * sqrt(a3inv))
#endif // JD_DECOMPOSE_VTURB
        break;

    case IO_VRAD: /* turbulent velocity around mean, radial part */
#ifdef JD_DECOMPOSE_VTURB
        IO_WRITE_SCALAR(fp, SphP[pindex].Vrad * sqrt(a3inv))
#endif // JD_DECOMPOSE_VTURB
        break;

    case IO_MAINHALO: /* Subhalo Level  */
#if defined(KD_MAIN_HALO) && defined(SUBFIND)
        IO_WRITE_SCALAR(ip_int, P[pindex].SubLevel)
#endif // KD_MAIN_HALO && SUBFIND
        break;

    case IO_TRUENGB: /* True Number of Neighbours  */
        IO_WRITE_SCALAR(ip_int, P[pindex].TrueNGB)
        break;

    case IO_WNGB: /* True Number of Neighbours  */
        IO_WRITE_SCALAR(fp, P[pindex].NumNgb)
        break;

    case IO_DPP: /* Reacceleration Coefficient */
#if (defined(JD_DPP) || defined(LMB_DPP))
        IO_WRITE_SCALAR(fp, SphP[pindex].Dpp)
#endif // (defined(JD_DPP) || defined(LMB_DPP))
        break;

    case IO_VDIV: /* Divergence of Vel */
#ifdef OUTPUT_DIV_CURL
        IO_WRITE_SCALAR(fp, SphP[pindex].DivVel)
#endif // OUTPUT_DIV_CURL
        break;

    case IO_VROT: /* Velocity Curl */
#ifdef OUTPUT_DIV_CURL
        IO_WRITE_SCALAR(fp, SphP[pindex].CurlVel)
#endif // OUTPUT_DIV_CURL
        break;

    case IO_VORT: /* Vorticity */
#ifdef OUTPUT_VORTICITY
        IO_WRITE_VEC(fp, 3, SphP[pindex].Rot[k])
#endif // OUTPUT_VORTICITY
        break;

    case IO_GRADIENT_ORDER: /* Velocity Curl */
#ifdef OUTPUT_GRADIENT_ORDER
        IO_WRITE_SCALAR(fp, SphP[pindex].GradientOrder)
#endif // OUTPUT_GRADIENT_ORDER
        break;

    case IO_GRADIENT_DISTANCEMATRIX:
#ifdef OUTPUT_GRADIENT_MATRICES
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int xdim = 0; xdim < 3; xdim++)
                {
                    for (int ydim = 0; ydim < 3; ydim++)
                    {
                        *fp++ = SphP[pindex].DistanceMatrix[xdim][ydim];
                    }
                }
                n++;
            }
        }
#endif // OUTPUT_GRADIENT_MATRICES
        break;

    case IO_GRADIENT_VELOCITYMATRIX:
#ifdef OUTPUT_GRADIENT_MATRICES
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int xdim = 0; xdim < 3; xdim++)
                {
                    for (int ydim = 0; ydim < 3; ydim++)
                    {
                        *fp++ = SphP[pindex].dVelMatrix[xdim][ydim];
                    }
                }
                n++;
            }
        }
#endif // OUTPUT_GRADIENT_MATRICES
        break;

    case IO_DIVB: /* divergence of magnetic field  */
#ifdef MAGNETIC
#if GADGET_HYDRO == HYDRO_MFM
        /* In MFM, divB holds the accumulated face flux int(B.dA) = div(B) * V_cell, not a divergence
         * density. Multiply by NumDens (= 1/V_cell, the same factor the cleaning source terms apply)
         * so the snapshot DIVB is a div(B) density, consistent with the SPH branch below. */
        IO_WRITE_SCALAR(fp, SphP[pindex].divB * SphP[pindex].NumDens * a2_inv * gadget2gauss)
#else // GADGET_HYDRO == HYDRO_MFM
        IO_WRITE_SCALAR(fp, SphP[pindex].divB * a2_inv * gadget2gauss)
#endif // GADGET_HYDRO == HYDRO_MFM
#endif // MAGNETIC
        break;

    case IO_ABVC: /* artificial viscosity of particle  */
        IO_WRITE_SCALAR(fp, SphP[pindex].alpha)
        break;

    case IO_ABVR: /* artificial viscosity of particle  */
#if (TIME_DEP_ART_VISC > 0) && defined(ART_VISC_DEBUG)
        IO_WRITE_SCALAR(fp, SphP[pindex].visc_R)
#endif // (TIME_DEP_ART_VISC > 0 && defined)(ART_VISC_DEBUG)
        break;

    case IO_ABVH: /* artificial viscosity of particle  */
#if (TIME_DEP_ART_VISC > 0) && defined(ART_VISC_DEBUG)
        IO_WRITE_SCALAR(fp, SphP[pindex].h2A)
#endif // TIME_DEP_ART_VISC && defined(ART_VISC_DEBUG)
        break;

    case IO_ACVC: /* artificial conductivity of particle  */
#ifdef TIME_DEP_ART_COND
        IO_WRITE_SCALAR(fp, SphP[pindex].Calpha)
#endif // TIME_DEP_ART_COND
        break;

    case IO_AMDC: /* artificial magnetic dissipation of particle  */
#ifdef TIME_DEP_MAGN_DISP
        IO_WRITE_SCALAR(fp, SphP[pindex].Balpha)
#endif // TIME_DEP_MAGN_DISP
        break;

    case IO_PHI: /* divBcleaning fuction of particle  */
#ifdef DIVBCLEANING_TRICCO
        IO_WRITE_SCALAR(fp, SphP[pindex].PsiPred)
#endif // DIVBCLEANING_TRICCO
#ifdef OUTPUTDEDNER
        IO_WRITE_SCALAR(fp, SphP[pindex].PhiPred * a2_inv * gadget2gauss)
#endif // OUTPUTDEDNER
        break;

    case IO_XPHI: /* Cold fraction in SF  */
#if defined(MAGNETIC) && defined(SFR)
        IO_WRITE_SCALAR(fp, SphP[pindex].XColdCloud)
#endif // defined(MAGNETIC) && defined(SFR)
        break;

    case IO_GRADPSI: /* divBcleaning function of particle  */
#ifdef DIVBCLEANING_TRICCO
#if GADGET_HYDRO == HYDRO_MFM
        IO_WRITE_VEC(fp, 3, SphP[pindex].GradPsi[k] * a3inv * SphP[pindex].NumDens * sqrt(mu0) * gadget2gauss)
#else // GADGET_HYDRO == HYDRO_MFM
        IO_WRITE_VEC(fp, 3, SphP[pindex].GradPsi[k] * a2_inv * gadget2gauss)
#endif // GADGET_HYDRO == HYDRO_MFM
#endif // DIVBCLEANING_TRICCO
        break;

    case IO_FACLO: /* face-closure residual of the MFM cell  */
#if defined(MFM_FACE_CLOSURE_RESIDUAL)
        /* sum_j A_ij should vanish for a cell whose effective faces close; this is the L1 norm of
         * the residual over the nominal cell surface, the dimensionless quantity the DtB and
         * internal-energy limiters gate on (Mfm*ClosureMin) and build their tolerance from
         * (Mfm*ClosureWeight). */
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                const double area_norm = 4.0 * M_PI * P[pindex].Hsml * P[pindex].Hsml;
                const double area_sum = fabs(SphP[pindex].FaceAreaSum[0]) + fabs(SphP[pindex].FaceAreaSum[1]) +
                                        fabs(SphP[pindex].FaceAreaSum[2]);
                *fp++ = (float)(area_norm > 0.0 ? area_sum / area_norm : 0.0);
                n++;
            }
        }
#endif // defined(MFM_FACE_CLOSURE_RESIDUAL)
        break;

    case IO_GRADPHI: /* divBcleaning fuction of particle  */
#ifdef OUTPUTDEDNER
        IO_WRITE_VEC(fp, 3, SphP[pindex].GradPhi[k] * a2_inv * gadget2gauss)
#endif // OUTPUTDEDNER
        break;

    case IO_ROTB: /* rot of magnetic field  */
#ifdef OUTPUT_ROTB
        IO_WRITE_VEC(fp, 3, SphP[pindex].RotB[k] * a2_inv * gadget2gauss)
#endif // OUTPUT_ROTB
        break;

    case IO_SROTB: /* smoothed rot of magnetic field  */
#ifdef OUTPUT_SROTB
        IO_WRITE_VEC(fp, 3, SphP[pindex].SmoothedRotB[k] * a2_inv * gadget2gauss)
#endif // OUTPUT_SROTB
        break;

    case IO_COOLRATE: /* current cooling rate of particle  */
#ifdef OUTPUTCOOLRATE
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                coolingdata_in in;
                coolingdata_out out;

                particle2in_cooling(&in, pindex);
                double tcool = cl_GetCoolingTimeAll(in, &out);

                *fp++ = (float)out.LambdaNet;
                n++;
            }
        }
#endif // OUTPUTCOOLRATE
        break;

    case IO_COOLTIME: /* current cooling time of the particle  */
#ifdef OUTPUTCOOLTIME
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                coolingdata_in in;
                coolingdata_out out;

                particle2in_cooling(&in, pindex);
                double tcool = cl_GetCoolingTimeAll(in, &out);

                *fp++ = (float)tcool;
                n++;
            }
        }
#endif // OUTPUTCOOLTIME
        break;

    case IO_CONDRATE: /* current heating/cooling due to thermal conduction  */
        break;

    case IO_DENN: /* density normalization factor */
#ifdef OUTPUTDENSNORM
        IO_WRITE_SCALAR(fp, SphP[pindex].Density)
#endif // OUTPUTDENSNORM
        break;

    case IO_LMBCR_pNORM:
#ifdef LMB_CR_PROTONS
        // caution! IO for norm as log10(norm) to avoid overflow!
        IO_WRITE_VEC(fp, LMB_CR_PROTONS, log10(SphP[pindex].CRpNorm[k]))
#endif
        break;

    case IO_LMBCR_eNORM:
#ifdef LMB_CR_ELECTRONS
        // caution! IO for norm as log10(norm) to avoid overflow!
        IO_WRITE_VEC(fp, LMB_CR_ELECTRONS, log10(SphP[pindex].CReNorm[k]))
#endif // LMB_CR_ELECTRONS
        break;

    case IO_LMBCR_pSLOPE:
#ifdef LMB_CR_PROTONS
        IO_WRITE_VEC(fp, LMB_CR_PROTONS, SphP[pindex].CRpSlope[k])
#endif // LMB_CR_PROTONS
        break;

    case IO_LMBCR_eSLOPE:
#ifdef LMB_CR_ELECTRONS
        IO_WRITE_VEC(fp, LMB_CR_ELECTRONS, SphP[pindex].CReSlope[k])
#endif // LMB_CR_ELECTRONS
        break;

    case IO_LMBCR_pCUT:
#ifdef LMB_CR_PROTONS
        IO_WRITE_SCALAR(fp, SphP[pindex].CRpCut)
#endif // LMB_CR_PROTONS
        break;

    case IO_LMBCR_eCUT:
#ifdef LMB_CR_ELECTRONS
        IO_WRITE_SCALAR(fp, SphP[pindex].CReCut)
#endif // LMB_CR_ELECTRONS
        break;

    case IO_LMBCR_pPRESSURE:
        break;

    case IO_LMBCR_ePRESSURE:
        break;

    case IO_LMBCR_eSYNCHROTRON:
#ifdef LMB_CR_OUTPUT_SYNCHROTRON
        IO_WRITE_VEC(fp, 8, log10(SphP[pindex].SynchEmissivity[k]))
#endif // LMB_CR_OUTPUT_SYNCHROTRON
        break;

    case IO_CRP_pmax: /* density Old */
#ifdef LMB_CRs_CALCULATE_P_MAX
        IO_WRITE_SCALAR(fp, SphP[pindex].CRPpmax)
#endif // LMB_CRs_CALCULATE_P_MAX
        break;

    case IO_RHO_OLD: /* density Old */
#ifdef READ_RHO_OLD
        IO_WRITE_SCALAR(fp, SphP[pindex].DensityOld)
#endif // READ_RHO_OLD
        break;

    case IO_BHMASS:
#ifdef BLACK_HOLES
        IO_WRITE_SCALAR(fp, BPP(pindex).BH_Mass)
#endif // BLACK_HOLES
        break;

    case IO_BHMDOT:
#ifdef BLACK_HOLES
        IO_WRITE_SCALAR(fp, BPP(pindex).BH_Mdot)
#endif // BLACK_HOLES
        break;

    case IO_BH_Kin_ConeAxis:
#if (defined(BLACK_HOLES) && defined(PB_BH_KIN_EJECT_PART_INSIDE_CONE))
        IO_WRITE_VEC(fp, 3, BPP(pindex).UnitVector_AGN_Wind[k])
#endif // BLACK_HOLES && PB_BH_KIN_EJECT_PART_INSIDE_CONE
        break;

    case IO_BHPROGS:
#ifdef BLACK_HOLES
        IO_WRITE_SCALAR(ip_int, BPP(pindex).BH_CountProgs)
#endif // BLACK_HOLES
        break;

    case IO_BHMRAD:
        break;

    case IO_ACRB:
#ifdef BLACK_HOLES
        IO_WRITE_SCALAR(fp, P[pindex].Hsml)
#endif // BLACK_HOLES
        break;

    case IO_BHSPIN:
#if defined(BLACK_HOLES) && defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
        IO_WRITE_VEC(fp, 3, BPP(pindex).BH_Spin[k])
#endif // defined(BLACK_HOLES) && defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
        break;

    case IO_BHETADISK:
#if defined(BLACK_HOLES) && defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
        IO_WRITE_SCALAR(fp, BPP(pindex).BH_EtaDisk)
#endif // defined(BLACK_HOLES) && defined(LS_BLACK_HOLES_SPIN_EVOLUTION)
        break;

    case IO_MAXSIGNALVEL:
#ifdef OUTPUTMAXSIGVEL
        IO_WRITE_SCALAR(fp, SphP[pindex].MaxSignalVel)
#endif // OUTPUTMAXSIGVEL
        break;

    case IO_MACH:
#ifdef SHOCKFINDER
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Mach)
#endif // SHOCKFINDER
        break;

    case IO_SHSPEED:
#ifdef SHOCKFINDER
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Speed)
#endif // SHOCKFINDER
        break;

    case IO_SHCOMPRESS:
#ifdef SHOCKFINDER
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Compress)
#endif // SHOCKFINDER
        break;

    case IO_SHNORMAL:
#ifdef SHOCKFINDER
        IO_WRITE_VEC(fp, 3, SphP[pindex].Shock_N1[k])
#endif // SHOCKFINDER
        break;

    case IO_SHRHOUP:
#ifdef SHOCKFINDER_RECONSTRUCT
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Up_Rho)
#endif
        break;

    case IO_SHPRESUP:
#ifdef SHOCKFINDER_RECONSTRUCT
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Up_Pressure)
#endif // SHOCKFINDER_RECONSTRUCT
        break;

    case IO_SHPRESDOWN:
#ifdef SHOCKFINDER_RECONSTRUCT
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Down_Pressure)
#endif // SHOCKFINDER_RECONSTRUCT
        break;

    case IO_SHVELUP:
#ifdef SHOCKFINDER_RECONSTRUCT
        IO_WRITE_SCALAR(fp, sqrt(SphP[pindex].Shock_Up_V1[0] * SphP[pindex].Shock_Up_V1[0] +
                                 SphP[pindex].Shock_Up_V1[1] * SphP[pindex].Shock_Up_V1[1] +
                                 SphP[pindex].Shock_Up_V1[2] * SphP[pindex].Shock_Up_V1[2]))
#endif // SHOCKFINDER_RECONSTRUCT
        break;

    case IO_SHVELDOWN:
#ifdef SHOCKFINDER_RECONSTRUCT
        IO_WRITE_SCALAR(fp, sqrt(SphP[pindex].Shock_Down_V1[0] * SphP[pindex].Shock_Down_V1[0] +
                                 SphP[pindex].Shock_Down_V1[1] * SphP[pindex].Shock_Down_V1[1] +
                                 SphP[pindex].Shock_Down_V1[2] * SphP[pindex].Shock_Down_V1[2]))
#endif // SHOCKFINDER_RECONSTRUCT
        break;

    case IO_MACH_ALFVEN:
#if defined(SHOCKFINDER) && defined(MAGNETIC)
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Mach_Alfven)
#endif // defined(SHOCKFINDER) && defined(MAGNETIC)
        break;

    case IO_SHALFVENDOWN:
        break;

    case IO_SHALFVENUP:
#if defined(MAGNETIC) && defined(SHOCKFINDER_RECONSTRUCT)
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Up_Alfven)
#endif // defined(SHOCKFINDER_MAGNETIC) && defined(SHOCKFINDER_RECONSTRUCT)
        break;

    case IO_SHOBLIQUITY:
#if defined(SHOCKFINDER) && defined(MAGNETIC)
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Obliquity)
#endif // defined(SHOCKFINDER) && defined(MAGNETIC)
        break;

    case IO_SHEDISS:
        break;

    case IO_SHCAND:
        break;

    case IO_SHPRIM:
        break;

    case IO_SHSIGNALUP:
#if defined(SHOCKFINDER) && defined(SHOCKFINDER_RECONSTRUCT)
        IO_WRITE_SCALAR(fp, SphP[pindex].Shock_Up_Signal)
#endif // defined(SHOCKFINDER) && defined(SHOCKFINDER_RECONSTRUCT)
        break;

    case IO_DMHSML:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        IO_WRITE_SCALAR(fp_single, P[pindex].DM_Hsml)
#endif // defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        break;

    case IO_DMDENSITY:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        IO_WRITE_SCALAR(fp_single, P[pindex].u.DM_Density)
#endif // defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        break;

    case IO_DMVELDISP:
#if defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        IO_WRITE_SCALAR(fp_single, P[pindex].v.DM_VelDisp)
#endif // defined(SUBFIND_RESHUFFLE_CATALOGUE) && defined(SUBFIND)
        break;


#if defined(GM_DMVELS) && defined(GM_DMVELS_OUTPUT)
    case IO_DM_AVE_VEL:
        IO_WRITE_SCALAR(fp, SphP[pindex].DmAveVel / All.Time)
        break;
    case IO_DM_AVE_SIG:
        for (int n = 0; n < pc; pindex++)
            if (P[pindex].Type == type)
            {
                MyAtLeastDouble sigma;
                sigma = SphP[pindex].DmAveVel2 - SphP[pindex].DmAveVel * SphP[pindex].DmAveVel;
                if (sigma > 0.)
                    sigma = sqrt(sigma) / All.Time;
                else
                    sigma = 0.;
                *fp++ = sigma;
                n++;
            }

        break;

    case IO_DM_NPART:
        IO_WRITE_SCALAR(ip, SphP[pindex].Ndm)
        break;

#endif // defined(GM_DMVELS) && defined(GM_DMVELS_OUTPUT)

    case IO_Zs:
#ifdef LT_STELLAREVOLUTION
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int k = 0; k < LT_NMetP; k++)
                    control[k] = 0;
                for (int k = 0; k < LT_NMetP; k++)
                {
                    if (type == 4)
                    {
                        *fp++ = (MyOutputFloat)MetP[P[pindex].pt.MetID].Metals[k];
                    }
                    else
                    {
                        *fp++ = (MyOutputFloat)SphP[pindex].Metals[k];
                    }
                    if (*(fp - 1) > 1)
                    {
                        control[k]++;
                    }
#ifdef LT_SEvDbg
                    metals[k + LT_NMetP * (type == 4)] += *(fp - 1);
                    metals[2 * LT_NMetP + 2] += *(fp - 1);
                    metals[2 * LT_NMetP + (type == 4)] += *(fp - 1);
#endif // LT_SEvDbg
                }
                if (*(fp - 1) > 1)
                {
                    VERBOSE_ALL(2, " ------- [%d][%d] ", ThisTask, pindex);
                    for (int k = 0; k < LT_NMetP; k++)
                    {
                        VERBOSE_ALL(2, " %d ", control[k]);
                    }
                    VERBOSE_ALL(2, "\n");
                }
                n++;
            }
        }
#endif // LT_STELLAREVOLUTION
        break;

    case IO_ContribII:
#ifdef LT_TRACK_CONTRIBUTES
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                if (type == 4)
                {
                    unpack_contrib(MetP[P[pindex].pt.MetID].contrib, fractionII, fractionIa, fractionAGB);
                }
                else
                {
                    unpack_contrib(SphP[pindex].contrib, fractionII, fractionIa, fractionAGB);
                }
                for (int k = 0; k < LT_NMetP; k++)
                {
                    *fp_single++ = fractionII[k];
                }
                n++;
            }
        }
#endif // LT_TRACK_CONTRIBUTES
        break;

    case IO_ContribIa:
#ifdef LT_TRACK_CONTRIBUTES
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                if (type == 4)
                {
                    unpack_contrib(MetP[P[pindex].pt.MetID].contrib, fractionII, fractionIa, fractionAGB);
                }
                else
                {
                    unpack_contrib(SphP[pindex].contrib, fractionII, fractionIa, fractionAGB);
                }
                for (int k = 0; k < LT_NMetP; k++)
                {
                    *fp_single++ = fractionIa[k];
                }
                n++;
            }
        }
#endif // LT_TRACK_CONTRIBUTES
        break;

    case IO_ContribAGB:
#ifdef LT_TRACK_CONTRIBUTES
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                if (type == 4)
                {
                    unpack_contrib(MetP[P[pindex].pt.MetID].contrib, fractionII, fractionIa, fractionAGB);
                }
                else
                {
                    unpack_contrib(SphP[pindex].contrib, fractionII, fractionIa, fractionAGB);
                }
                for (int k = 0; k < LT_NMetP; k++)
                {
                    *fp_single++ = fractionAGB[k];
                }
                n++;
            }
        }
#endif // LT_TRACK_CONTRIBUTES
        break;

    case IO_iMass:
#ifdef LT_STELLAREVOLUTION
        IO_WRITE_SCALAR(fp, (MyOutputFloat)MetP[P[pindex].pt.MetID].iMass)
#endif // LT_STELLAREVOLUTION
        break;

    case IO_CLDX:
#ifdef LT_STELLAREVOLUTION
        IO_WRITE_SCALAR(fp, SphP[pindex].XColdCloud)
#endif // LT_STELLAREVOLUTION
        break;

    case IO_HTEMP:
#ifdef LT_STELLAREVOLUTION
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
#ifdef LT_METAL_COOLING_WAL
                double Metallicities[LT_NMet];
#endif // LT_METAL_COOLING_WAL
                double u = max(All.MinEgySpec, (double)(SphP[pindex].Entropy / GAMMA_MINUS1 *
                                                        pow(SphP[pindex].Density * a3inv, GAMMA_MINUS1) *
                                                        All.UnitPressure_in_cgs / All.UnitDensity_in_cgs));
#ifndef LT_METAL_COOLING_WAL
                double Ne = (double)SphP[pindex].elec;
                // double mu = (1 + 4 * yhelium) / (1 + yhelium + SphP[pindex].elec);
                double rho = SphP[pindex].Density * a3inv * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
#endif // LT_METAL_COOLING_WAL
                if (SphP[pindex].XColdCloud > 0)
                {
                    u -= SphP[pindex].XColdCloud * All.EgySpecCold; /* obtain the energy density of the hot phase */
                    u /= (1 - SphP[pindex].XColdCloud);
#ifndef LT_METAL_COOLING_WAL
                    rho *= (1 - SphP[pindex].XColdCloud);
#endif // LT_METAL_COOLING_WAL
                }

#ifndef LT_METAL_COOLING_WAL
                *fp++ = (MyFloat)convert_u_to_tempSTD(u, rho, &Ne);
#else // LT_METAL_COOLING_WAL
                set_metallicities(pindex, &Metallicities[0], a3inv);
                Metallicities[HydPos] *= (1 - SphP[pindex].XColdCloud);

                *fp++ = (MyFloat)convert_u_to_tempMET(u, Redshift, DZ, &Metallicities[0]);
#endif // LT_METAL_COOLING_WAL
                n++;
            }
        }
#endif // LT_STELLAREVOLUTION
        break;

    case IO_TEMP:
#ifdef COOLING
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
#ifdef LT_METAL_COOLING_WAL
                double Metallicities[LT_NMet];
#endif // LT_METAL_COOLING_WAL
                double u = max(All.MinEgySpec, (double)(SphP[pindex].Entropy / GAMMA_MINUS1 *
                                                        pow(SphP[pindex].Density * a3inv, GAMMA_MINUS1) *
                                                        All.UnitPressure_in_cgs / All.UnitDensity_in_cgs));
#ifndef LT_METAL_COOLING_WAL
                double Ne = (double)SphP[pindex].elec;
                // double mu = (1 + 4 * yhelium) / (1 + yhelium + SphP[pindex].elec);
                double rho = SphP[pindex].Density * a3inv * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;
                *fp++ = (MyFloat)convert_u_to_tempSTD(u, rho, &Ne);
#else // LT_METAL_COOLING_WAL
                set_metallicities(pindex, &Metallicities[0], a3inv);
                *fp++ = (MyFloat)convert_u_to_tempMET(u, Redshift, DZ, &Metallicities[0]);
#endif // LT_METAL_COOLING_WAL
                n++;
            }
        }
#endif
        break;

    case IO_ZAGE:
#ifdef LT_ZAGE
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                if (type == 0)
                {
                    if (SphP[pindex].ZAgeW > 0)
                    {
#ifndef LT_LOGZAGE
                        *fp++ = SphP[pindex].ZAge / SphP[pindex].ZAgeW;
#else // LT_LOGZAGE
                        *fp++ = exp(10, SphP[pindex].ZAge) / SphP[pindex].ZAgeW;
#endif // LT_LOGZAGE
                    }
                    else
                    {
                        *fp++ = 0;
                    }
                }
                else
                {
                    *fp++ = MetP[P[pindex].pt.MetID].ZAge;
                }
                n++;
            }
        }
#endif // LT_ZAGE
        break;

    case IO_ZAGE_LLV:
#ifdef LT_ZAGE_LLV
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                if (type == 0)
                {
                    if (SphP[pindex].ZAgeW > 0)
                    {
#ifndef LT_LOGZAGE
                        *fp++ = SphP[pindex].ZAge_llv / SphP[pindex].ZAgeW_llv;
#else // LT_LOGZAGE
                        *fp++ = exp(10, SphP[pindex].ZAge_llv) / SphP[pindex].ZAgeW_llv;
#endif // LT_LOGZAGE
                    }
                    else
                    {
                        *fp++ = 0;
                    }
                }
                else
                {
                    *fp++ = MetP[P[pindex].pt.MetID].ZAge_llv;
                }
                n++;
            }
        }
#endif // LT_ZAGE_LLV
        break;

    case IO_CONTRIB:
#ifdef LT_TRACK_CONTRIBUTES
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                if (type == 0)
                {
                    *contribp++ = SphP[pindex].contrib;
                }
                else if (type == 4)
                {
                    *contribp++ = MetP[P[pindex].pt.MetID].contrib;
                }
                n++;
            }
        }
#endif // LT_TRACK_CONTRIBUTES
        break;

    case IO_ZSMOOTH:
#if defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)
        IO_WRITE_SCALAR(fp_single, SphP[pindex].Zsmooth)
#endif // defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)
        break;

    case IO_allZSMOOTH:
#ifdef LT_SMOOTH_ALLMETALS
        IO_WRITE_VEC(fp_single, LT_NMet, SphP[pindex].Zsmooth[k])
#endif // LT_SMOOTH_ALLMETALS
        break;

    case IO_AGS_SOFT: /* Adaptive Gravitational Softening: softening */
#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVSOFT)
        IO_WRITE_SCALAR(fp, P[pindex].AGS_Hsml)
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVSOFT)
        break;

    case IO_AGS_DENS: /* Adaptive Gravitational Softening: (number) density */
#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVNUMDENS)
        IO_WRITE_SCALAR(fp, P[pindex].AGS_Density)
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTGRAVNUMDENS)
        break;

    case IO_AGS_ZETA: /* Adaptive Gravitational Softening: zeta */
#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTZETA)
        IO_WRITE_SCALAR(fp, P[pindex].AGS_zeta)
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTZETA)
        break;

    case IO_AGS_OMEGA: /* Adaptive Gravitational Softening: omega */
#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTOMEGA)
        IO_WRITE_SCALAR(fp, P[pindex].AGS_omega)
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTOMEGA)
        break;

    case IO_AGS_NGBS: /* Adaptive Gravitational Softening: neighbours */
#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTNGBS)
        IO_WRITE_SCALAR(fp, P[pindex].AGS_REALNumNgb)
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTNGBS)
        break;

    case IO_AGS_CORR: /* Adaptive Gravitational Softening: correction */
#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
        IO_WRITE_SCALAR(fp, P[pindex].AGS_corr)
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
        break;

    case IO_GRADENERGY:
#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
        IO_WRITE_VEC(fp, 3, SphP[pindex].GradEnergy[k])
#endif // defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
        break;

    case IO_GRADPRESSURE:
#ifdef AA_OUTPUT_GRADS
        IO_WRITE_VEC(fp, 3, SphP[pindex].GradPressure[k])
#endif // AA_OUTPUT_GRADS
        break;

    case IO_MFM_GRADIENTS_VX:
#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int k = 0; k < NUMDIMS; k++)
                {
                    fp[k] = SphP[pindex].gradW[0][k];
                }
                for (int k = NUMDIMS; k < mfm_numdims; k++)
                {
                    fp[k] = 0.0;
                }
                fp += mfm_numdims;
                n++;
            }
        }
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        break;

    case IO_MFM_GRADIENTS_VY:
#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        if constexpr (mfm_numdims > 1)
        {
            for (int n = 0; n < pc; pindex++)
            {
                if (P[pindex].Type == type)
                {
                    for (int k = 0; k < NUMDIMS; k++)
                    {
                        fp[k] = SphP[pindex].gradW[1][k];
                    }
                    for (int k = NUMDIMS; k < mfm_numdims; k++)
                    {
                        fp[k] = 0.0;
                    }
                    fp += mfm_numdims;
                    n++;
                }
            }
        }
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        break;

    case IO_MFM_GRADIENTS_VZ:
#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        if constexpr (mfm_numdims == 3)
        {
            for (int n = 0; n < pc; pindex++)
            {
                if (P[pindex].Type == type)
                {
                    for (int k = 0; k < NUMDIMS; k++)
                    {
                        fp[k] = SphP[pindex].gradW[2][k];
                    }
                    for (int k = NUMDIMS; k < mfm_numdims; k++)
                    {
                        fp[k] = 0.0;
                    }
                    fp += mfm_numdims;
                    n++;
                }
            }
        }
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS) && (mfm_numdims == 3)
        break;

    case IO_MFM_GRADIENTS_RHO:
#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int k = 0; k < NUMDIMS; k++)
                {
                    fp[k] = SphP[pindex].gradW[mfm_numdims][k];
                }
                for (int k = NUMDIMS; k < mfm_numdims; k++)
                {
                    fp[k] = 0.0;
                }
                fp += mfm_numdims;
                n++;
            }
        }
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        break;

    case IO_MFM_GRADIENTS_U:
#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int k = 0; k < NUMDIMS; k++)
                {
                    fp[k] = SphP[pindex].gradW[mfm_numdims + 1][k];
                }
                for (int k = NUMDIMS; k < mfm_numdims; k++)
                {
                    fp[k] = 0.0;
                }
                fp += mfm_numdims;
                n++;
            }
        }
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS)
        break;

    case IO_MFM_GRADIENTS_BX:
        break;

    case IO_MFM_GRADIENTS_BY:
        break;

    case IO_MFM_GRADIENTS_BZ:
        break;

    case IO_MFM_GRADIENTS_PHI:
#if (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS) && defined(MFM_MAGNETIC) &&                           \
    defined(DIVBCLEANING_DEDNER)
        for (int n = 0; n < pc; pindex++)
        {
            if (P[pindex].Type == type)
            {
                for (int k = 0; k < NUMDIMS; k++)
                {
                    fp[k] = SphP[pindex].gradW[mfm_numdims + 5][k];
                }
                for (int k = NUMDIMS; k < mfm_numdims; k++)
                {
                    fp[k] = 0.0;
                }
                fp += mfm_numdims;
                n++;
            }
        }
#endif // (GADGET_HYDRO == HYDRO_MFM) && defined(MFM_OUTPUT_GRADIENTS) && defined(MFM_MAGNETIC) &&
       // defined(DIVBCLEANING_DEDNER)
        break;

    case IO_RHOSTAR:
#ifdef GM_STARDENSITY
        IO_WRITE_SCALAR(fp, P[pindex].StarDensity)
#endif // GM_STARDENSITY
        break;

    case IO_HSMLSTAR:
#ifdef GM_STARDENSITY
        IO_WRITE_SCALAR(fp, P[pindex].StarHsml)
#endif // GM_STARDENSITY
        break;

    case IO_NEIGHSTAR:
#ifdef GM_STARDENSITY
        IO_WRITE_SCALAR(fp, P[pindex].StarNumNgb)
#endif // GM_STARDENSITY
        break;

    case IO_DUSTL:
#if defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        IO_WRITE_VEC(fp, LT_NMetP, (MyOutputFloat)SphP[pindex].DustL[k])
#endif // defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        break;

    case IO_DUSTS:
#if defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        IO_WRITE_VEC(fp, LT_NMetP, (MyOutputFloat)SphP[pindex].DustS[k])
#endif // defined(LT_STELLAREVOLUTION) && defined(GL_CR_DUST)
        break;

#ifdef GT_DUST
    case IO_DVEL: /* relative velocity */
        for (int n = 0; n < pc; pindex++)
            if (P[pindex].Type == type)
            {
                for (int idust = 0; idust < GT_NUM_DUST; idust++)
                {
                    for (int k = 0; k < 3; k++)
                        fp[idust * 3 + k] = SphP[pindex].DeltaVel[idust][k];
                }
                n++;
                fp += 3 * GT_NUM_DUST;
            }
        break;

    case IO_DFRC: /* dust fraction */
        IO_WRITE_VEC(fp, GT_NUM_DUST, dustfrac(SphP[pindex].DustFrac[k]))
        break;

    case IO_DRAG: /* dust fraction */
        IO_WRITE_VEC(fp, GT_NUM_DUST, SphP[pindex].DragCoefficient[k])
        break;

#endif // GT_DUST

    case IO_LASTENTRY:
        ENDRUN_MAIN("Last entry in output list reached.");
        break;
    default:
        VERBOSE(1, "Warning in filling write buffer: case unknown (blocknr = %d)", blocknr);
    }

    return pindex;
}

#undef IO_WRITE_SCALAR
#undef IO_WRITE_VEC

/**
 * @brief Fills the write buffer with particle data.
 *
 * This function populates the write buffer with data related to particles.
 * It prepares the data based on the specified block number and other parameters.
 * New output blocks can be integrated into the function, allowing for extension
 * of the data writing capabilities.
 *
 * @param blocknr The block number specifying which type of data to write. This
 *                is an enumeration value of type `enum iofields`.
 * @param startindex Pointer to an integer indicating the starting index for
 *                   the data to be written.
 * @param pc The particle count, indicating the number of particles to be
 *           processed.
 * @param type An integer representing the type of particles or data being
 *             handled.
 */
void fill_write_buffer(enum iofields blocknr, int *startindex, int pc, int type)
{
    double a3inv = 1;

#ifdef MAGNETIC
    double a2_inv, gadget2gauss = 1.0;
#endif // MAGNETIC

#ifdef OUTPUTACCELERATION
    double fac1, fac2;
#endif // OUTPUTACCELERATION

    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
#ifdef OUTPUTACCELERATION
        fac1 = 1 / (All.Time * All.Time);
        fac2 = 1 / pow(All.Time, 3 * GAMMA - 2);
#endif // OUTPUTACCELERATION

#ifdef MAGNETIC
#ifndef MU0_UNITY
        gadget2gauss /= sqrt(All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g /
                             (All.HubbleParam * All.HubbleParam));
#endif // MU0_UNITY
        a2_inv = 1.0 / (All.Time * All.Time);
#endif // MAGNETIC
    }
    else
    {
#ifdef OUTPUTACCELERATION
        fac1 = fac2 = 1;
#endif // OUTPUTACCELERATION
        a3inv = 1;
#ifdef MAGNETIC
#ifndef MU0_UNITY
        gadget2gauss /= sqrt(All.UnitTime_in_s * All.UnitTime_in_s * All.UnitLength_in_cm / All.UnitMass_in_g);
#endif // MU0_UNITY
        a2_inv = 1;
#endif // MAGNETIC
    }

    *startindex = fill_write_buffer_block(blocknr, *startindex, pc, type, a3inv
#ifdef OUTPUTACCELERATION
                                          ,
                                          fac1, fac2
#endif // OUTPUTACCELERATION
#ifdef MAGNETIC
                                          ,
                                          a2_inv, gadget2gauss
#endif // MAGNETIC
    );
}

/**
 * @brief Writes an integer value to a file stream.
 *
 * This function writes the provided block size value to the given file
 * descriptor using a custom write routine.
 *
 * @param blksize Pointer to an integer value to be written to the file.
 * @param fd Pointer to the file stream to write to.
 */
inline void skip_write(unsigned int *blksize, FILE *fd)
{
    my_fwrite(blksize, sizeof(int), 1, fd);
}

/**
 * @brief Compute global particle counts per type across a range of MPI tasks.
 *
 * This routine aggregates the number of particles of each type (0–5) from all
 * MPI ranks in the interval [writeTask, lastTask]. The designated @p writeTask
 * acts as the root: it receives local counts from other tasks, accumulates the
 * totals, and redistributes the global sums back to all participating ranks.
 *
 * When WRITE_KEY_FILES is enabled, the function additionally:
 * - Computes per-type write offsets for each task.
 * - Aggregates the total number of keys per type across tasks.
 * - Computes the total number of keys in the file.
 *
 * Communication pattern:
 * - Non-root tasks send local particle counts to @p writeTask.
 * - Root task gathers and accumulates counts, then broadcasts totals.
 * - Optional key-related data follows a similar gather/broadcast pattern.
 *
 * @param[in]  writeTask   MPI rank designated as the root (collector) task.
 * @param[in]  lastTask    Last MPI rank (inclusive) participating in the aggregation.
 * @param[in]  n_type      Local number of particles per type on this task (array of size 6).
 * @param[out] ntot_type   Global total number of particles per type (array of size 6).
 *
 * @param[out] write_offset   (WRITE_KEY_FILES only) Per-type offsets for writing data.
 * @param[out] NKeys_file     (WRITE_KEY_FILES only) Global number of keys per type.
 * @param[out] TotNumKeysFile (WRITE_KEY_FILES only) Total number of keys across all types.
 */
inline void compute_total_particle_number(int writeTask, int lastTask, int n_type[6], int ntot_type[6]
#ifdef WRITE_KEY_FILES
                                          ,
                                          int write_offset[6], int NKeys_file[6], unsigned int &TotNumKeysFile
#endif // WRITE_KEY_FILES
)
{
    MPI_Status status;
    int nn[6];
    /* determine particle numbers of each type in file */
    if (ThisTask == writeTask)
    {
        for (int n = 0; n < 6; n++)
        {
            ntot_type[n] = n_type[n];
        }
#ifdef WRITE_KEY_FILES
        for (int n = 0; n < 6; n++)
        {
            write_offset[n] = 0;
        }
#endif // WRITE_KEY_FILES

        for (int task = writeTask + 1; task <= lastTask; task++)
        {
            MPI_Recv(&nn[0], 6, MPI_INT, task, TAG_LOCALN, MYMPI_COMM_WORLD, &status);
#ifdef WRITE_KEY_FILES
            MPI_Send(&ntot_type[0], 6, MPI_INT, task, TAG_WRT_OFF, MYMPI_COMM_WORLD);
#endif // WRITE_KEY_FILES
            for (int n = 0; n < 6; n++)
            {
                ntot_type[n] += nn[n];
            }
        }

        for (int task = writeTask + 1; task <= lastTask; task++)
        {
            MPI_Send(&ntot_type[0], 6, MPI_INT, task, TAG_N, MYMPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Send(&n_type[0], 6, MPI_INT, writeTask, TAG_LOCALN, MYMPI_COMM_WORLD);
#ifdef WRITE_KEY_FILES
        MPI_Recv(&write_offset[0], 6, MPI_INT, writeTask, TAG_WRT_OFF, MYMPI_COMM_WORLD, &status);
#endif // WRITE_KEY_FILES
        MPI_Recv(&ntot_type[0], 6, MPI_INT, writeTask, TAG_N, MYMPI_COMM_WORLD, &status);
    }

#ifdef WRITE_KEY_FILES
    if (ThisTask == writeTask)
    {
        for (int n = 0; n < 6; n++)
        {
            NKeys_file[n] = NKeys[n];
        }

        for (int task = writeTask + 1; task <= lastTask; task++)
        {
            MPI_Recv(&nn[0], 6, MPI_INT, task, TAG_LOCALN, MYMPI_COMM_WORLD, &status);
            for (int n = 0; n < 6; n++)
            {
                NKeys_file[n] += nn[n];
            }
        }

        for (int task = writeTask + 1; task <= lastTask; task++)
        {
            MPI_Send(&NKeys_file[0], 6, MPI_INT, task, TAG_N, MYMPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Send(&NKeys[0], 6, MPI_INT, writeTask, TAG_LOCALN, MYMPI_COMM_WORLD);
        MPI_Recv(&NKeys_file[0], 6, MPI_INT, writeTask, TAG_N, MYMPI_COMM_WORLD, &status);
    }

    TotNumKeysFile = 0;
    for (int n = 0; n < 6; n++)
    {
        TotNumKeysFile += NKeys_file[n];
    }
#endif // WRITE_KEY_FILES
}

/**
 * @brief Build the INFO block that lists every data block that will be
 * written (label, type string, dimensionality, particle-type presence mask).
 * Optionally writes the block immediately to the file for SnapFormat==2.
 * Only the writeTask executes this.
 *
 * @param fd File handle
 * @param n_info Output parameter returning the number of INFO entries.
 * @param InfoBlock Output pointer to the allocated array of INFO block entries.
 */
static void write_info_block(FILE *fd, int &n_info, struct info_block *&InfoBlock)
{
    n_info = 0;
    InfoBlock = (struct info_block *)mymalloc("InfoBlock", sizeof(struct info_block) * 10000);

    static const char *type_names[][2] = {
        /* scalar          vector     */
        {"LONG    ", "LONGN   "}, // case 0
#ifdef OUTPUT_IN_DOUBLEPRECISION
        {"DOUBLE  ", "DOUBLEN "}, // case 1
#else // OUTPUT_IN_DOUBLEPRECISION
        {"FLOAT   ", "FLOATN  "}, // case 1
#endif // OUTPUT_IN_DOUBLEPRECISION
        {"LLONG   ", "LLONGN  "}, // case 2
        {"FLOAT16 ", "FLOAT16N"}, // case 3
    };

    for (int bnr = 0; bnr < IO_LASTENTRY; bnr++)
    {
        enum iofields blocknr = (enum iofields)bnr;
        if (!io_matrix::check_if_read_or_write_block(blocknr, 4))
        {
            continue;
        }

        unsigned int typelist[6];
        int npart = io_matrix::get_particles_in_block(blocknr, typelist);
        if (npart == 0)
        {
            continue;
        }

        for (int type = 0; type < 6; type++)
        {
            InfoBlock[n_info].is_present[type] = typelist[type];
        }

        InfoBlock[n_info].ndim = io_matrix::get_values_per_blockelement(blocknr);

        char label[8];
        io_matrix::get_Tab_IO_Label(blocknr, label);
        for (int i = 0; i < 4; i++)
        {
            InfoBlock[n_info].label[i] = label[i];
        }

        // Determine type string from datatype + dimensionality
        int ndim = InfoBlock[n_info].ndim;
        bool is_vec = (ndim > 1);

        // write data type to info block
        int dtype = io_matrix::get_datatype_in_block(blocknr);
        // check that type is within range
        assert(dtype >= 0 && dtype < 4);
        // copy type to info block
        memcpy(InfoBlock[n_info].type, type_names[dtype][is_vec], 8);

        n_info++;
    }

#ifndef DONT_WRITE_INFO_BLOCK
    // Write INFO block at the top of the file (before data blocks)
    unsigned int blksize;
    if (All.SnapFormat == 2)
    {
        blksize = sizeof(int) + 4 * sizeof(char);
        skip_write(&blksize, fd);
        my_fwrite((void *)"INFO", sizeof(char), 4, fd);
        int nextblock = n_info * sizeof(struct info_block) + 2 * sizeof(int);
        my_fwrite(&nextblock, sizeof(int), 1, fd);
        skip_write(&blksize, fd);
    }
    blksize = n_info * sizeof(struct info_block);
    skip_write(&blksize, fd);
    my_fwrite(InfoBlock, sizeof(struct info_block), n_info, fd);
    skip_write(&blksize, fd);
#endif // DONT_WRITE_INFO_BLOCK
}

/**
 * @brief Fill the snapshot header struct with simulation metadata,
 * IC-flag updates, cosmological parameters, and compile-time feature flags.
 */
static void fill_snapshot_header(int ntot_type[6])
{
    // Particle counts (already stored in ntot_type / ntot_type_all by caller)
    for (int n = 0; n < 6; n++)
    {
        header.npart[n] = ntot_type[n];
        header.npartTotal[n] = (unsigned int)ntot_type_all[n];
        header.npartTotalHighWord[n] = (unsigned int)(ntot_type_all[n] >> 32);
    }

#if defined(NEUTRINOS) && defined(OMIT_NEUTRINOS_IN_SNAPS)
    if (DumpFlag != 2)
    {
        header.npart[2] = 0;
        header.npartTotal[2] = 0;
        header.npartTotalHighWord[2] = 0;
    }
#endif // NEUTRINOS && OMIT_NEUTRINOS_IN_SNAPS

    // IC-flag evolution tracking
    if (header.flag_ic_info == FLAG_SECOND_ORDER_ICS || header.flag_ic_info == FLAG_NORMALICS_2LPT)
    {
        header.flag_ic_info = FLAG_EVOLVED_2LPT;
    }

    if (header.flag_ic_info == FLAG_ZELDOVICH_ICS)
    {
        header.flag_ic_info = FLAG_EVOLVED_ZELDOVICH;
    }

    if (header.flag_ic_info == 0 && All.ComovingIntegrationOn != 0)
    {
        header.flag_ic_info = FLAG_EVOLVED_ZELDOVICH;
    }

    // Mass table, time, redshift
    for (int n = 0; n < 6; n++)
    {
        header.mass[n] = All.MassTable[n];
    }

    header.time = All.Time;
    header.redshift = All.ComovingIntegrationOn ? (1.0 / All.Time - 1) : 0;

    // Feature flags (reset first, then set according to compile options)
    header.flag_sfr = 0;
    header.flag_feedback = 0;
    header.flag_cooling = 0;
    header.flag_stellarage = 0;
    header.flag_metals = 0;
    header.flag_entropy_instead_u = 0;

#ifdef COOLING
    header.flag_cooling = 1;
#endif // COOLING
#ifdef SFR
    header.flag_sfr = 1;
    header.flag_feedback = 1;
    header.flag_stellarage = 1;
#if !defined(LT_STELLAREVOLUTION)
    header.flag_metals = 1;
#endif // !LT_STELLAREVOLUTION
#endif // SFR

    // Cosmological / box parameters
    header.num_files = All.NumFilesPerSnapshot;
    header.BoxSize = All.BoxSize;
    header.Omega0 = All.Omega0;
    header.OmegaLambda = All.OmegaLambda;
    header.HubbleParam = All.HubbleParam;

#ifdef OUTPUT_IN_DOUBLEPRECISION
    header.flag_doubleprecision = 1;
#else // OUTPUT_IN_DOUBLEPRECISION
    header.flag_doubleprecision = 0;
#endif // OUTPUT_IN_DOUBLEPRECISION

#ifdef LT_STELLAREVOLUTION
    for (int n = 0; n < std::min(15, LT_NMet); n++)
    {
        header.names[n][0] = MetNames[n][0];
        header.names[n][1] = MetNames[n][1];
    }
    for (int n = LT_NMet - 1; n < 15; n++)
    {
        header.names[n][0] = ' ';
        header.names[n][1] = ' ';
    }
#endif // LT_STELLAREVOLUTION
}

/**
 * @brief Open all output file handles required for this snapshot
 * (main file or HDF5, plus optional files: .key, .dbg, .trck,
 * .zsmooth, .muppi, snap_potential).  Only the writeTask executes I/O.
 *
 * @param writeTask Task supposed to write snapshot
 * @param fname Snapshot filename
 * @param ... Various file handles
 */
static void open_output_files(int writeTask, const char *fname,
#ifdef WRITE_KEY_FILES
                              struct io_header &key_header, FILE *&FdKey, char *buf_key, int NKeys_file[6],
#endif // WRITE_KEY_FILES
#ifdef HAVE_HDF5
                              hid_t &hdf5_file, hid_t hdf5_grp[6], hid_t &hdf5_headergrp,
#endif // HAVE_HDF5
                              FILE *&fd
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
                              ,
                              FILE *&fd_pot
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
#ifdef LT_SEvDbg
                              ,
                              FILE *&fd_dbg
#endif // LT_SEvDbg
#ifdef LT_TRACK_CONTRIBUTES
                              ,
                              FILE *&FdTrck
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
                              ,
                              FILE *&FdZsmooth
#endif // LT_SMOOTH_Z
)
{
    if (ThisTask != writeTask)
    {
        return;
    }

    if (All.SnapFormat == 3)
    {
#ifdef HAVE_HDF5
        char buf[500];
        sprintf(buf, "%s.hdf5", fname);
        hdf5_file = H5Fcreate(buf, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hdf5_headergrp = H5Gcreate2(hdf5_file, "/Header", 0, H5P_DEFAULT, H5P_DEFAULT);

        for (int type = 0; type < 6; type++)
        {
            if (header.npart[type] > 0)
            {
                sprintf(buf, "/PartType%d", type);
                hdf5_grp[type] = H5Gcreate2(hdf5_file, buf, 0, H5P_DEFAULT, H5P_DEFAULT);
            }
        }
        write_header_attributes_in_hdf5(hdf5_headergrp);
#endif // HAVE_HDF5
    }
    else
    {
        // ---- main snapshot file ----
        if (!(fd = fopen(fname, "w")))
        {
            ENDRUN("can't open file `%s' for writing snapshot.", fname);
        }

        unsigned int blksize;
        if (All.SnapFormat == 2)
        {
            blksize = sizeof(int) + 4 * sizeof(char);
            skip_write(&blksize, fd);
            my_fwrite((void *)"HEAD", sizeof(char), 4, fd);
            int nextblock = sizeof(header) + 2 * sizeof(int);
            my_fwrite(&nextblock, sizeof(int), 1, fd);
            skip_write(&blksize, fd);
        }
        blksize = sizeof(header);
        skip_write(&blksize, fd);
        my_fwrite(&header, sizeof(header), 1, fd);
        skip_write(&blksize, fd);

#ifdef WRITE_KEY_FILES
        sprintf(buf_key, "%s.key", fname);
        FdKey = fopen(buf_key, "w");

        // Build key_header from main header, overriding spatial metadata
        key_header = header;
        for (int n = 0; n < 6; n++)
        {
            key_header.npart[n] = NKeys_file[n];
            key_header.npartTotal[n] = (unsigned int)NKeysTotal[n];
            key_header.npartTotalHighWord[n] = (unsigned int)(NKeysTotal[n] >> 32);
        }
        for (int n = 0; n < 3; n++)
        {
            key_header.mass[n] = DomainCorner[n];
        }
        key_header.mass[3] = DomainFac;
        key_header.mass[4] = 0;
        key_header.mass[5] = 0;
        key_header.flag_feedback = BITS_PER_DIMENSION_SAVE_KEYS;

        if (All.SnapFormat == 2)
        {
            blksize = sizeof(int) + 4 * sizeof(char);
            skip_write(&blksize, FdKey);
            my_fwrite((void *)"HEAD", sizeof(char), 4, FdKey);
            int nextblock = sizeof(header) + 2 * sizeof(int);
            my_fwrite(&nextblock, sizeof(int), 1, FdKey);
            skip_write(&blksize, FdKey);
        }
        blksize = sizeof(key_header);
        skip_write(&blksize, FdKey);
        my_fwrite(&key_header, sizeof(header), 1, FdKey);
        skip_write(&blksize, FdKey);
#endif // WRITE_KEY_FILES
    }

    // ---- optional sidecar files ----

#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
    char fname_pot[1000];
    sprintf(fname_pot, "snap_potential");
    if (!(fd_pot = fopen(fname_pot, "w")))
    {
        ENDRUN("can't open file `%s' for writing pot_snapshot.", fname_pot);
    }
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL

#ifdef LT_SEvDbg
    {
        char buf[300];
        sprintf(buf, "%s.dbg", fname);
        fd_dbg = fopen(buf, "w");
    }
#endif // LT_SEvDbg

#ifdef LT_TRACK_CONTRIBUTES
    {
        char buf[300];
        sprintf(buf, "%s.trck", fname);
        if ((FdTrck = fopen(buf, "w")) == nullptr)
        {
            ENDRUN("Task %d : it was impossible to open file %s", ThisTask, buf);
        }

        unsigned int blksize = 6 * sizeof(int);
        skip_write(&blksize, FdTrck);
        my_fwrite(&header.npart[0], sizeof(int), 1, FdTrck);
        my_fwrite(&header.npart[4], sizeof(int), 1, FdTrck);
        my_fwrite(&PowerBase, sizeof(int), 1, FdTrck);
        blksize = LT_Nbits;
        my_fwrite(&blksize, sizeof(int), 1, FdTrck);
        blksize = LT_power10_Nbits;
        my_fwrite(&blksize, sizeof(int), 1, FdTrck);
        blksize = SFs_dim;
        my_fwrite(&blksize, sizeof(int), 1, FdTrck);
        blksize = 6 * sizeof(int);
        skip_write(&blksize, FdTrck);
    }
#endif // LT_TRACK_CONTRIBUTES

#ifdef LT_SMOOTH_Z
    {
        char buf[300];
        sprintf(buf, "%s.zsmooth", fname);
        if ((FdZsmooth = fopen(buf, "w")) == nullptr)
        {
            ENDRUN("Task %d : it was impossible to open file %s", ThisTask, buf);
        }

        unsigned int blksize;
#ifndef LT_SMOOTHZ_IN_IMF_SWITCH
        blksize = 4;
        skip_write(&blksize, FdZsmooth);
        my_fwrite(&header.npart[0], sizeof(int), 1, FdZsmooth);
        skip_write(&blksize, FdZsmooth);
#else // LT_SMOOTHZ_IN_IMF_SWITCH
        blksize = 8;
        skip_write(&blksize, FdZsmooth);
        my_fwrite(&header.npart[0], sizeof(int), 1, FdZsmooth);
        my_fwrite(&header.npart[4], sizeof(int), 1, FdZsmooth);
        skip_write(&blksize, FdZsmooth);
#endif // LT_SMOOTHZ_IN_IMF_SWITCH
    }
#endif // LT_SMOOTH_Z

}

#ifdef HAVE_HDF5
/**
 * @brief Map an internal data type identifier to an HDF5 datatype.
 *
 * Creates and returns a copy of the corresponding HDF5 native datatype
 * based on the provided integer code. Supports unsigned integers, floating
 * point types (single/double precision depending on compile-time flags),
 * 64-bit unsigned integers, and a manually configured 16-bit float.
 *
 * @param dtype Integer identifier of the desired datatype:
 *              - 0: unsigned int
 *              - 1: float or double (controlled by OUTPUT_IN_DOUBLEPRECISION)
 *              - 2: uint64
 *              - 3: float16 (custom HDF5 configuration)
 *
 * @return hid_t HDF5 datatype handle (must be closed by caller).
 *
 * @note For dtype = 3, a half-precision floating-point format is constructed
 *       manually using HDF5 low-level type configuration.
 *
 * @throws Terminates execution if an unknown dtype is provided.
 */
hid_t GetHDF5Datatype(int dtype)
{
    hid_t hdf5_datatype;
    switch (dtype)
    {
    case 0:
        hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT);
        break;
    case 1:
#ifdef OUTPUT_IN_DOUBLEPRECISION
        hdf5_datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
#else // OUTPUT_IN_DOUBLEPRECISION
        hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
#endif // OUTPUT_IN_DOUBLEPRECISION
        break;
    case 2:
        hdf5_datatype = H5Tcopy(H5T_NATIVE_UINT64);
        break;
    case 3: // float16
        hdf5_datatype = H5Tcopy(H5T_NATIVE_FLOAT);
        H5Tset_size(hdf5_datatype, 2);
        H5Tset_fields(hdf5_datatype, 15, 10, 5, 0, 10);
        H5Tset_ebias(hdf5_datatype, 15);
        H5Tset_norm(hdf5_datatype, H5T_NORM_IMPLIED);
        H5Tset_inpad(hdf5_datatype, H5T_PAD_ZERO);
        H5Tset_precision(hdf5_datatype, 16);
        break;
    default:
        ENDRUN("ERROR: Unknown dataype in GetHDF5Datatype!");
    }
    return hdf5_datatype;
}
#endif // HAVE_HDF5

/**
 * @brief Write all data blocks. Each block is distributed across MPI ranks;
 * the writeTask collects data and writes it to the appropriate file handle
 * (main, HDF5, or one of the sidecar files).
 *
 * @param writeTask Current task id
 * @param lastTask Id of number of tasks
 * @param ... Various file handles
 */
static void write_data_blocks(int writeTask, int lastTask, FILE *&fd,
#ifdef HAVE_HDF5
                              hid_t hdf5_file, hid_t hdf5_grp[6],
#endif // HAVE_HDF5
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
                              FILE *fd_pot,
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
#ifdef LT_TRACK_CONTRIBUTES
                              FILE *FdTrck, FILE *&fd_trck_back,
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
                              FILE *FdZsmooth, FILE *&fd_zsmooth_back,
#endif // LT_SMOOTH_Z
                              int &n_info, struct info_block *InfoBlock)
{
    MPI_Status status;

    for (int bnr = 0; bnr < IO_LASTENTRY; bnr++)
    {
        enum iofields blocknr = (enum iofields)bnr;
        if (!io_matrix::check_if_read_or_write_block(blocknr, 4))
        {
            continue;
        }

        unsigned int typelist[6];
        int bytes_per_blockelement = io_matrix::get_bytes_per_blockelement(blocknr, 0);
        size_t blockmaxlen = (size_t)((All.BufferSize * 1024 * 1024) / bytes_per_blockelement);
        int npart = io_matrix::get_particles_in_block(blocknr, typelist);
        if (npart == 0)
        {
            continue;
        }

        if (ThisTask == 0)
        {
            char buf[1000];
            io_matrix::get_dataset_name(blocknr, buf);
            VERBOSE_ALL(3, "writing block %d (%s)... ", bnr, buf);
        }

        // save pointer for fortran block markers
        FILE *fd_before_block = fd;

        // ---- redirect fd to sidecar if needed (writeTask only) ----
        if (ThisTask == writeTask && (All.SnapFormat == 1 || All.SnapFormat == 2))
        {
#ifdef LT_TRACK_CONTRIBUTES
            if (blocknr == IO_CONTRIB)
            {
                fd_trck_back = fd;
                fd = FdTrck;
            }
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
#ifndef LT_SMOOTH_ALLMETALS
            if (blocknr == IO_ZSMOOTH)
            {
                fd_zsmooth_back = fd;
                fd = FdZsmooth;
            }
#else // LT_SMOOTH_ALLMETALS
            if (blocknr == IO_allZSMOOTH)
            {
                fd_zsmooth_back = fd;
                fd = FdZsmooth;
            }
#endif // LT_SMOOTH_ALLMETALS
#endif // LT_SMOOTH_Z
        }

        // ---- write SnapFormat-2 block label descriptor ----
        unsigned int blksize = 0;
        if (ThisTask == writeTask && (All.SnapFormat == 2))
        {
            char label[8];
            io_matrix::get_Tab_IO_Label(blocknr, label);
            blksize = sizeof(int) + 4 * sizeof(char);

#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
            FILE *target = (blocknr == IO_POT) ? fd_pot : fd;
#else // SUBFIND_RESHUFFLE_AND_POTENTIAL
            FILE *target = fd;
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
            skip_write(&blksize, target);
            my_fwrite(label, sizeof(char), 4, target);
            int nextblock = npart * bytes_per_blockelement + 2 * sizeof(int);
            my_fwrite(&nextblock, sizeof(int), 1, target);
            skip_write(&blksize, target);
        }

        // ---- opening fortran-style block marker ----
        if (ThisTask == writeTask && (All.SnapFormat == 1 || All.SnapFormat == 2))
        {
            blksize = npart * bytes_per_blockelement;
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
            skip_write(&blksize, (blocknr == IO_POT) ? fd_pot : fd_before_block);
#else // SUBFIND_RESHUFFLE_AND_POTENTIAL
            skip_write(&blksize, fd_before_block);
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
        }

        // ---- per-particle-type MPI gather and write ----
        for (int type = 0; type < 6; type++)
        {
            if (!typelist[type])
            {
                continue;
            }

#ifdef HAVE_HDF5
            hid_t hdf5_datatype = 0, hdf5_dataspace_in_file = 0, hdf5_dataset = 0;
            hid_t hdf5_dataspace_memory;
            int rank = 0, pcsum = 0;
            hsize_t dims[2], count[2], start[2];
            char hdf5_buf[500];

            if (ThisTask == writeTask && All.SnapFormat == 3 && header.npart[type] > 0)
            {
                hdf5_datatype = GetHDF5Datatype(io_matrix::get_datatype_in_block(blocknr));
                dims[0] = header.npart[type];
                dims[1] = io_matrix::get_values_per_blockelement(blocknr);
                rank = (dims[1] == 1) ? 1 : 2;

                io_matrix::get_dataset_name(blocknr, hdf5_buf);
                hdf5_dataspace_in_file = H5Screate_simple(rank, dims, NULL);
                hdf5_dataset = H5Dcreate2(hdf5_grp[type], hdf5_buf, hdf5_datatype, hdf5_dataspace_in_file, H5P_DEFAULT,
                                          H5P_DEFAULT, H5P_DEFAULT);
                pcsum = 0;
            }
#endif // HAVE_HDF5

            for (int task = writeTask, offset = 0; task <= lastTask; task++)
            {
                int n_for_this_task;
                if (task == ThisTask)
                {
                    n_for_this_task = n_type[type];
                    for (int p = writeTask; p <= lastTask; p++)
                    {
                        if (p != ThisTask)
                        {
                            MPI_Send(&n_for_this_task, 1, MPI_INT, p, TAG_NFORTHISTASK, MYMPI_COMM_WORLD);
                        }
                    }
                }
                else
                {
                    MPI_Recv(&n_for_this_task, 1, MPI_INT, task, TAG_NFORTHISTASK, MYMPI_COMM_WORLD, &status);
                }

                while (n_for_this_task > 0)
                {
                    int pc = std::min(n_for_this_task, (int)blockmaxlen);

                    if (ThisTask == task)
                    {
                        fill_write_buffer(blocknr, &offset, pc, type);
                    }

                    if (ThisTask == writeTask && task != writeTask)
                    {
                        MPI_Recv(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, task, TAG_PDATA, MYMPI_COMM_WORLD,
                                 &status);
                    }

                    if (ThisTask != writeTask && task == ThisTask)
                    {
                        MPI_Ssend(CommBuffer, bytes_per_blockelement * pc, MPI_BYTE, writeTask, TAG_PDATA,
                                  MYMPI_COMM_WORLD);
                    }

                    if (ThisTask == writeTask)
                    {
                        if (All.SnapFormat == 3)
                        {
#ifdef HAVE_HDF5
#if defined(LT_TRACK_CONTRIBUTES) || defined(LT_SMOOTH_Z)
                            if (blocknr != IO_CONTRIB && blocknr != IO_ZSMOOTH && blocknr != IO_allZSMOOTH)
#endif // LT_TRACK_CONTRIBUTES || LT_SMOOTH_Z
                            {
                                start[0] = pcsum;
                                start[1] = 0;
                                count[0] = pc;
                                count[1] = io_matrix::get_values_per_blockelement(blocknr);
                                pcsum += pc;

                                H5Sselect_hyperslab(hdf5_dataspace_in_file, H5S_SELECT_SET, start, NULL, count, NULL);

                                dims[0] = pc;
                                dims[1] = io_matrix::get_values_per_blockelement(blocknr);
                                hdf5_dataspace_memory = H5Screate_simple(rank, dims, NULL);

                                herr_t hdf5_status = H5Dwrite(hdf5_dataset, hdf5_datatype, hdf5_dataspace_memory,
                                                              hdf5_dataspace_in_file, H5P_DEFAULT, CommBuffer);
                                VERBOSE_IF(hdf5_status < 0, 0, "Writing hdf5 file failed");
                                H5Sclose(hdf5_dataspace_memory);
                            }
#endif // HAVE_HDF5
                        }
                        else
                        {
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
                            my_fwrite(CommBuffer, bytes_per_blockelement, pc, (blocknr == IO_POT) ? fd_pot : fd);
#else // SUBFIND_RESHUFFLE_AND_POTENTIAL
                            my_fwrite(CommBuffer, bytes_per_blockelement, pc, fd);
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
                        }
                    }
                    n_for_this_task -= pc;
                }
            } // for task

#ifdef HAVE_HDF5
            if (ThisTask == writeTask && All.SnapFormat == 3 && header.npart[type] > 0)
            {
                H5Dclose(hdf5_dataset);
                H5Sclose(hdf5_dataspace_in_file);
                H5Tclose(hdf5_datatype);
            }
#endif // HAVE_HDF5
        } // for type

        // ---- closing fortran-style block marker ----
        if (ThisTask == writeTask && (All.SnapFormat == 1 || All.SnapFormat == 2))
        {
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
            skip_write(&blksize, (blocknr == IO_POT) ? fd_pot : fd_before_block);
#else // SUBFIND_RESHUFFLE_AND_POTENTIAL
            skip_write(&blksize, fd_before_block);
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
        }

        // ---- restore fd if it was redirected to a sidecar ----
        if (ThisTask == writeTask)
        {
#ifdef LT_TRACK_CONTRIBUTES
            if (blocknr == IO_CONTRIB)
            {
                fd = fd_trck_back;
            }
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
#ifndef LT_SMOOTH_ALLMETALS
            if (blocknr == IO_ZSMOOTH)
            {
                fd = fd_zsmooth_back;
            }
#else // LT_SMOOTH_ALLMETALS
            if (blocknr == IO_allZSMOOTH)
            {
                fd = fd_zsmooth_back;
            }
#endif // LT_SMOOTH_ALLMETALS
#endif
        }
    } // for bnr

    // ---- write trailing INFO block (SnapFormat 2, if not written upfront) ----
    if ((All.SnapFormat == 1 || All.SnapFormat == 2) && ThisTask == writeTask)
    {
#ifdef DONT_WRITE_INFO_BLOCK
        unsigned int blksize;
        if (All.SnapFormat == 2)
        {
            blksize = sizeof(int) + 4 * sizeof(char);
            skip_write(&blksize, fd);
            my_fwrite((void *)"INFO", sizeof(char), 4, fd);
            int nextblock = n_info * sizeof(struct info_block) + 2 * sizeof(int);
            my_fwrite(&nextblock, sizeof(int), 1, fd);
            skip_write(&blksize, fd);
        }
        blksize = n_info * sizeof(struct info_block);
        skip_write(&blksize, fd);
        my_fwrite(InfoBlock, sizeof(struct info_block), n_info, fd);
        skip_write(&blksize, fd);
#endif // DONT_WRITE_INFO_BLOCK
        myfree(InfoBlock);
    }
}

#ifdef WRITE_KEY_FILES
// ------------------------------------------------------------------
// Step 6 – write_key_files  (writeTask gathers from all tasks)
// ------------------------------------------------------------------
static void write_key_files(int writeTask, int lastTask, FILE *FdKey, int /*npart*/, struct info_block *KeyInfoBlock,
                            unsigned int TotNumKeysFile, int NKeys_file[6], int write_offset[6], size_t blockmaxlen)
{
    MPI_Status status;
    char label[8];

    // Build KeyInfoBlock (KEY / NKEY / OKEY descriptors)
    if (ThisTask == writeTask)
    {
        const char *labels[3] = {"KEY ", "NKEY", "OKEY"};
        for (int bnr = 0; bnr < 3; bnr++)
        {
            strncpy(KeyInfoBlock[bnr].label, labels[bnr], 4);

            if (bnr == 0)
            {
                strncpy(KeyInfoBlock[bnr].type, (BITS_PER_DIMENSION_SAVE_KEYS > 10) ? "LLONG   " : "LONG    ", 8);
            }
            else
            {
                strncpy(KeyInfoBlock[bnr].type, "LONG    ", 8);
            }

            for (int type = 0; type < 6; type++)
            {
                KeyInfoBlock[bnr].is_present[type] = (NKeys_file[type] > 0) ? 1 : 0;
            }

            KeyInfoBlock[bnr].ndim = 1;
        }

        // Write INFO block for key file
        if (All.SnapFormat == 2)
        {
            unsigned int blksize = sizeof(int) + 4 * sizeof(char);
            skip_write(&blksize, FdKey);
            my_fwrite((void *)"INFO", sizeof(char), 4, FdKey);
            int nextblock = 3 * sizeof(struct info_block) + 2 * sizeof(int);
            my_fwrite(&nextblock, sizeof(int), 1, FdKey);
            skip_write(&blksize, FdKey);
        }
        unsigned int blksize = 3 * sizeof(struct info_block);
        skip_write(&blksize, FdKey);
        my_fwrite(KeyInfoBlock, sizeof(struct info_block), 3, FdKey);
        skip_write(&blksize, FdKey);
    }

    // Write KEY / NKEY / OKEY data blocks
    for (int bnr = 0; bnr < 3; bnr++)
    {
        int *data_point_int = nullptr;
        peanokey *data_point_key = nullptr;
        int size_of_element;

        switch (bnr)
        {
        case 0:
            data_point_key = KeyIndex;
            size_of_element = (BITS_PER_DIMENSION_SAVE_KEYS > 10) ? sizeof(peanokey) : sizeof(int);
            break;
        case 1:
            data_point_int = NPartPerKey;
            size_of_element = sizeof(int);
            break;
        case 2:
            data_point_int = PartKeyOffset;
            size_of_element = sizeof(int);
            break;
        default:
            ENDRUN("Undefined block to write to KEY file... (bnr = %d)", bnr);
        }

        VERBOSE(3, "writing block (%c%c%c%c) ... ", KeyInfoBlock[bnr].label[0], KeyInfoBlock[bnr].label[1],
                KeyInfoBlock[bnr].label[2], KeyInfoBlock[bnr].label[3]);

        if (ThisTask == writeTask)
        {
            strncpy(label, KeyInfoBlock[bnr].label, 4);
            unsigned int blksize = TotNumKeysFile * size_of_element;

            if (All.SnapFormat == 2)
            {
                unsigned int hdr = sizeof(int) + 4 * sizeof(char);
                skip_write(&hdr, FdKey);
                my_fwrite(label, sizeof(char), 4, FdKey);
                int nextblock = blksize + 2 * sizeof(int);
                my_fwrite(&nextblock, sizeof(int), 1, FdKey);
                skip_write(&hdr, FdKey);
            }
            skip_write(&blksize, FdKey);
        }

        for (int type = 0; type < 6; type++)
        {
            for (int task = writeTask; task <= lastTask; task++)
            {
                int local_offset = 0;
                for (int n = 0; n < type; n++)
                {
                    local_offset += NKeys[n];
                }

                int n_for_this_task;
                if (task == ThisTask)
                {
                    n_for_this_task = NKeys[type];
                    for (int p = writeTask; p <= lastTask; p++)
                    {
                        if (p != ThisTask)
                        {
                            MPI_Send(&n_for_this_task, 1, MPI_INT, p, TAG_NFORTHISTASK, MYMPI_COMM_WORLD);
                        }
                    }
                }
                else
                {
                    MPI_Recv(&n_for_this_task, 1, MPI_INT, task, TAG_NFORTHISTASK, MYMPI_COMM_WORLD, &status);
                }

                while (n_for_this_task > 0)
                {
                    int pc = std::min(n_for_this_task, (int)blockmaxlen);

                    if (ThisTask == task)
                    {
                        int *ci = (int *)CommBuffer;
                        peanokey *ck = (peanokey *)CommBuffer;
                        switch (bnr)
                        {
                        case 0:
                            for (int n = 0; n < pc; n++)
                            {
                                if (BITS_PER_DIMENSION_SAVE_KEYS > 10)
                                {
                                    *ck++ = data_point_key[local_offset++];
                                }
                                else
                                {
                                    *ci++ = (int)data_point_key[local_offset++];
                                }
                            }
                            break;
                        case 1:
                            for (int n = 0; n < pc; n++)
                            {
                                *ci++ = data_point_int[local_offset++];
                            }
                            break;
                        case 2:
                            for (int n = 0; n < pc; n++)
                            {
                                *ci++ = data_point_int[local_offset++] + write_offset[type];
                            }
                            break;
                        }
                    }

                    if (ThisTask == writeTask && task != writeTask)
                    {
                        MPI_Recv(CommBuffer, size_of_element * pc, MPI_BYTE, task, TAG_PDATA, MYMPI_COMM_WORLD,
                                 &status);
                    }

                    if (ThisTask != writeTask && task == ThisTask)
                    {
                        MPI_Ssend(CommBuffer, size_of_element * pc, MPI_BYTE, writeTask, TAG_PDATA, MYMPI_COMM_WORLD);
                    }

                    if (ThisTask == writeTask)
                    {
                        my_fwrite(CommBuffer, size_of_element, pc, FdKey);
                    }

                    n_for_this_task -= pc;
                }
            } // for task
        } // for type

        if (ThisTask == writeTask)
        {
            unsigned int blksize = TotNumKeysFile * size_of_element;
            skip_write(&blksize, FdKey);
        }
    } // for bnr

    if (ThisTask == writeTask)
    {
        myfree(KeyInfoBlock);
        fclose(FdKey);
    }
}
#endif // WRITE_KEY_FILES

/**
 * @brief Flush and close every file handle opened in open_output_files().
 */
static void close_output_files(int writeTask,
#ifdef HAVE_HDF5
                               hid_t hdf5_file, hid_t hdf5_grp[6], hid_t hdf5_headergrp,
#endif
                               FILE *fd
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
                               ,
                               FILE *fd_pot
#endif
#ifdef LT_SEvDbg
                               ,
                               FILE *fd_dbg
#endif
#ifdef LT_TRACK_CONTRIBUTES
                               ,
                               FILE *FdTrck
#endif
#ifdef LT_SMOOTH_Z
                               ,
                               FILE *FdZsmooth
#endif
#ifdef WRITE_KEY_FILES
                               ,
                               FILE * /*FdKey*/ // already closed in write_key_files
                               ,
                               struct info_block * /*KeyInfoBlock*/ // already freed
#endif
)
{
    if (ThisTask != writeTask)
    {
        return;
    }

    if (All.SnapFormat == 3)
    {
#ifdef HAVE_HDF5
        for (int type = 5; type >= 0; type--)
        {
            if (header.npart[type] > 0)
            {
                H5Gclose(hdf5_grp[type]);
            }
        }
        H5Gclose(hdf5_headergrp);
        H5Fclose(hdf5_file);
#endif // HAVE_HDF5
#ifdef LT_SEvDbg
        if (fd_dbg)
        {
            fclose(fd_dbg);
        }
#endif // LT_SEvDbg
#ifdef LT_TRACK_CONTRIBUTES
        if (FdTrck)
        {
            fclose(FdTrck);
        }
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
        if (FdZsmooth)
        {
            fclose(FdZsmooth);
        }
#endif // LT_SMOOTH_Z
    }
    else
    {
        fclose(fd);
    }

#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
    if (fd_pot)
    {
        fclose(fd_pot);
    }
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
}

/**
 * @brief Writes a snapshot file containing data from specified processors.
 *
 * This function handles the creation and writing of a snapshot file that includes data from processors ranging from
 * `writeTask` to `lastTask`. The processor specified by `writeTask` is responsible for the actual writing operation.
 * The snapshot file is organized into several sections:
 *
 * - Header: Contains metadata about the snapshot.
 * - Particle Data:** Includes particle positions, velocities, and IDs.
 * - Masses: Writes particle masses for types with zero entries in `MassTable`.
 * - Internal Energies and Density: For SPH particles, it writes internal energies and density.
 * - Cooling Information: If cooling is enabled, it writes mean molecular weight and neutral hydrogen abundance for gas
 * particles.
 * - Additional Data: Includes SPH smoothing length and other blocks of information based on included physics and
 * compile-time flags.
 *
 * @param fname File name for the snapshot output.
 * @param writeTask The processor responsible for writing the snapshot file.
 * @param lastTask The last processor involved in the snapshot data gathering.
 */
void write_file(char *fname, int writeTask, int lastTask)
{
    int ntot_type[6];
    FILE *fd = nullptr;

#ifdef WRITE_KEY_FILES
    int NKeys_file[6];
    unsigned int TotNumKeysFile;
    struct io_header key_header;
    FILE *FdKey;
    char buf_key[300];
    struct info_block *KeyInfoBlock = nullptr;
    int write_offset[6];
#endif // WRITE_KEY_FILES

#ifdef HAVE_HDF5
    hid_t hdf5_file = 0, hdf5_grp[6] = {}, hdf5_headergrp = 0;
#endif // HAVE_HDF5

#if defined(LT_SEvDbg) || defined(LT_TRACK_CONTRIBUTES) || defined(LT_POPIII_FLAGS) || defined(LT_SMOOTH_Z)

#ifdef LT_SEvDbg
    FILE *fd_dbg = nullptr;
#endif // LT_SEvDbg

#ifdef LT_TRACK_CONTRIBUTES
    FILE *FdTrck = nullptr, *fd_trck_back = nullptr;
#endif // LT_TRACK_CONTRIBUTES

#ifdef LT_SMOOTH_Z
    FILE *FdZsmooth = nullptr, *fd_zsmooth_back = nullptr;
#endif // LT_SMOOTH_Z
#endif // defined(LT_SEvDbg) || defined(LT_TRACK_CONTRIBUTES) || defined(LT_POPIII_FLAGS) || defined(LT_SMOOTH_Z)


#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
    FILE *fd_pot = nullptr;
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL

#ifdef WRITE_KEY_FILES
    compute_total_particle_number(writeTask, lastTask, n_type, ntot_type, write_offset, NKeys_file, TotNumKeysFile);
#else // WRITE_KEY_FILES
    compute_total_particle_number(writeTask, lastTask, n_type, ntot_type);
#endif // WRITE_KEY_FILES

    // ------------------------------------------------------------------
    // populate the global header struct
    // ------------------------------------------------------------------
    fill_snapshot_header(ntot_type);

    // ------------------------------------------------------------------
    // open all output files (writeTask only for actual I/O)
    // ------------------------------------------------------------------
    open_output_files(writeTask, fname,
#ifdef WRITE_KEY_FILES
                      key_header, FdKey, buf_key, NKeys_file,
#endif // WRITE_KEY_FILES
#ifdef HAVE_HDF5
                      hdf5_file, hdf5_grp, hdf5_headergrp,
#endif // HAVE_HDF5
                      fd
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
                      ,
                      fd_pot
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
#ifdef LT_SEvDbg
                      ,
                      fd_dbg
#endif // LT_SEvDbg
#ifdef LT_TRACK_CONTRIBUTES
                      ,
                      FdTrck
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
                      ,
                      FdZsmooth
#endif // LT_SMOOTH_Z
    );

    if ((All.SnapFormat == 1 || All.SnapFormat == 2) && ThisTask == writeTask)
    {
        write_info_block(fd, n_info, InfoBlock);
    }

    // ------------------------------------------------------------------
    // Write all data blocks to file
    // ------------------------------------------------------------------
    write_data_blocks(writeTask, lastTask, fd,
#ifdef HAVE_HDF5
                      hdf5_file, hdf5_grp,
#endif // HAVE_HDF5
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
                      fd_pot,
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
#ifdef LT_TRACK_CONTRIBUTES
                      FdTrck, fd_trck_back,
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
                      FdZsmooth, fd_zsmooth_back,
#endif // LT_SMOOTH_Z
                      n_info, InfoBlock);

#ifdef WRITE_KEY_FILES
    // ------------------------------------------------------------------
    // Write Peano-Hilbert key files
    // ------------------------------------------------------------------
    if (ThisTask == writeTask)
    {
        KeyInfoBlock = (struct info_block *)mymalloc("KeyInfoBlock", sizeof(struct info_block) * 3);
    }
    {
        int npart_dummy = 0; // recomputed inside; passed for block-size guard
        size_t blockmaxlen = (size_t)((All.BufferSize * 1024 * 1024) / sizeof(int));
        write_key_files(writeTask, lastTask, FdKey, npart_dummy, KeyInfoBlock, TotNumKeysFile, NKeys_file, write_offset,
                        blockmaxlen);
    }
#endif // WRITE_KEY_FILES

    // ------------------------------------------------------------------
    // Close and flush all output files
    // ------------------------------------------------------------------
    close_output_files(writeTask,
#ifdef HAVE_HDF5
                       hdf5_file, hdf5_grp, hdf5_headergrp,
#endif // HAVE_HDF5
                       fd
#ifdef SUBFIND_RESHUFFLE_AND_POTENTIAL
                       ,
                       fd_pot
#endif // SUBFIND_RESHUFFLE_AND_POTENTIAL
#ifdef LT_SEvDbg
                       ,
                       fd_dbg
#endif // LT_SEvDbg
#ifdef LT_TRACK_CONTRIBUTES
                       ,
                       FdTrck
#endif // LT_TRACK_CONTRIBUTES
#ifdef LT_SMOOTH_Z
                       ,
                       FdZsmooth
#endif // LT_SMOOTH_Z
#ifdef WRITE_KEY_FILES
                       ,
                       FdKey, KeyInfoBlock
#endif // WRITE_KEY_FILES
    );
}

#ifdef WRITE_KEY_FILES
/**
 * @brief Writes key index data to a file and handles MPI communication for parallel processing.
 *
 * @param num Snapshot number used in the output file name.
 * @param filenr File number associated with the keys.
 */
void write_key_index_file(int num, int filenr)
{
    int MaxKeyData = 0;
    unsigned int TotNumKeys = 0;
    int icount1 = 0, n = 0, lend = 0;
    int icount2 = 1;

    // Calculate the total number of keys
    for (int i = 0; i < 6; i++)
    {
        TotNumKeys += NKeys[i];
    }

    // Allocate memory for KeyFileList and MyKeyData
    KeyFileList = (struct key_file_list *)mymalloc("MyKeyData", 1000 * sizeof(struct key_file_list));
    MyKeyData = (struct key_file_index *)mymalloc("MyKeyData", TotNumKeys * sizeof(struct key_file_index));

    // Populate MyKeyData with key and filenr values
    for (int i = 0; i < TotNumKeys; i++)
    {
        MyKeyData[i].key = KeyIndex[i];
        MyKeyData[i].filenr = filenr;
    }
    // Sort MyKeyData based on key values
    parallel_sort(MyKeyData, TotNumKeys, sizeof(struct key_file_index), key_compare);

    // Generate KeyFileList based on MyKeyData
    do
    {
        if (MyKeyData[icount1].filenr != MyKeyData[icount2].filenr)
        {
            KeyFileList[n].low_list = MyKeyData[icount1].key;
            KeyFileList[n].high_list = MyKeyData[icount2 - 1].key;
            KeyFileList[n].file_list = MyKeyData[icount1].filenr;
            n++;
            icount1 = icount2;
            icount2++;
        }
        else
        {
            icount2++;
        }
        if (icount2 >= TotNumKeys)
        {
            lend = 1;
        }
    } while (lend == 0);

    KeyFileList[n].low_list = MyKeyData[icount1].key;
    KeyFileList[n].high_list = MyKeyData[icount2 - 1].key;
    KeyFileList[n].file_list = MyKeyData[icount1].filenr;
    n++;

    // Free MyKeyData
    myfree(MyKeyData);

    // Perform MPI operations
    MPI_Allreduce(&n, &MaxKeyData, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    KeyFileListAll = (struct key_file_list *)mymalloc("MyKeyDataAll", MaxKeyData * sizeof(struct key_file_list));

    int *count = (int *)mymalloc("count", NTask * sizeof(int));
    MPI_Allgather(&n, 1, MPI_INT, count, 1, MPI_INT, MYMPI_COMM_WORLD);

    for (int task = 0, j = 0; task < NTask; task++)
    {
        if (ThisTask == 0 && task != 0)
        {
            MPI_Recv(KeyFileList, count[task] * sizeof(struct key_file_list), MPI_BYTE, task, TAG_WRT_OFF,
                     MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (ThisTask != 0 && task == ThisTask)
        {
            MPI_Ssend(KeyFileList, count[task] * sizeof(struct key_file_list), MPI_BYTE, 0, TAG_WRT_OFF,
                      MYMPI_COMM_WORLD);
        }

        if (ThisTask == 0)
        {
            int i = 0;
            if (task > 0 && KeyFileListAll[j - 1].file_list == KeyFileList[i].file_list)
            {
                KeyFileListAll[j - 1].high_list = KeyFileList[i].high_list;
                i++;
            }

            for (; i < count[task]; i++)
            {
                KeyFileListAll[j].low_list = KeyFileList[i].low_list;
                KeyFileListAll[j].high_list = KeyFileList[i].high_list;
                KeyFileListAll[j].file_list = KeyFileList[i].file_list;
                j++;
            }

            TotNumKeys = j;
        }
    }

    // Allocate memory for outbuffer
    void *outbuffer = mymalloc("outbuffer", MaxKeyData * sizeof(unsigned long long));

    // Assign keylist and filelist pointers to outbuffer
    unsigned long long *keylist = (unsigned long long *)outbuffer;
    int *filelist = (int *)outbuffer;

    if (ThisTask == 0)
    {
        // Generate the output key index file name
        char buf_key[300];

#ifdef LONG_OUTPUT_NUMBERS
        if (All.NumFilesPerSnapshot > 1)
        {
            sprintf(buf_key, "%s/snapdir_%04d/%s_%04d.key.index", All.OutputDir, num, All.SnapshotFileBase, num);
        }
        else
        {
            sprintf(buf_key, "%s%s_%04d.key.index", All.OutputDir, All.SnapshotFileBase, num);
        }
#else // LONG_OUTPUT_NUMBERS
        if (All.NumFilesPerSnapshot > 1)
        {
            sprintf(buf_key, "%s/snapdir_%03d/%s_%03d.key.index", All.OutputDir, num, All.SnapshotFileBase, num);
        }
        else
        {
            sprintf(buf_key, "%s%s_%03d.key.index", All.OutputDir, All.SnapshotFileBase, num);
        }
#endif // LONG_OUTPUT_NUMBERS

        // Open the output file
        FILE *FdKey = fopen(buf_key, "w");

        // Sort KeyFileListAll based on low_list values
        qsort(KeyFileListAll, TotNumKeys, sizeof(struct key_file_list), key_list_compare);

        // Write the total number of keys to the file
        my_fwrite(&TotNumKeys, sizeof(int), 1, FdKey);

        // Populate keylist with low_list values and write to file
        for (int i = 0; i < TotNumKeys; i++)
        {
            keylist[i] = (unsigned long long)KeyFileListAll[i].low_list;
        }
        my_fwrite(outbuffer, sizeof(peanokey), TotNumKeys, FdKey);

        // Populate keylist with high_list values and write to file
        for (int i = 0; i < TotNumKeys; i++)
        {
            keylist[i] = (unsigned long long)KeyFileListAll[i].high_list;
        }
        my_fwrite(outbuffer, sizeof(peanokey), TotNumKeys, FdKey);

        // Populate filelist with file_list values and write to file
        for (int i = 0; i < TotNumKeys; i++)
        {
            filelist[i] = (int)KeyFileListAll[i].file_list;
        }
        my_fwrite(outbuffer, sizeof(int), TotNumKeys, FdKey);

        // Close the output file
        fclose(FdKey);
    }

    // Free allocated memory
    myfree(outbuffer);
    myfree(count);
    myfree(KeyFileListAll);
    myfree(KeyFileList);
}

/**
 * @brief Compares two `key_file_index` elements based on their `key` values.
 *
 * This function compares two elements of type `key_file_index` by their `key` values, typically used for sorting or
 * ordering. It returns an integer based on the comparison of the `key` values of the two elements.
 *
 * @param a Pointer to the first `key_file_index` element to compare.
 * @param b Pointer to the second `key_file_index` element to compare.
 *
 * @return -1 if the `key` value of the first element is less than the `key` value of the second element,
 *         +1 if the `key` value of the first element is greater than the `key` value of the second element,
 *         0 if the `key` values of both elements are equal.
 */
int key_compare(const void *a, const void *b)
{
    // Create structs of pointers related to key_file_index
    const struct key_file_index *keyA = (const struct key_file_index *)a;
    const struct key_file_index *keyB = (const struct key_file_index *)b;

    return (keyA->key < keyB->key) ? -1 : (keyA->key > keyB->key);
}

/**
 * @brief Compares two `key_file_list` elements based on their `low_list` values.
 *
 * This function is used to compare two elements of type `key_file_list` by their `low_list` values for sorting or
 * ordering purposes. It returns an integer based on the comparison of the `low_list` values of the two elements.
 *
 * @param a Pointer to the first `key_file_list` element to compare.
 * @param b Pointer to the second `key_file_list` element to compare.
 *
 * @return -1 if the `low_list` value of the first element is less than the `low_list` value of the second element,
 *         +1 if the `low_list` value of the first element is greater than the `low_list` value of the second element,
 *         0 if the `low_list` values of both elements are equal.
 */
int key_list_compare(const void *a, const void *b)
{
    // Create structs of pointers related to key_file_list
    const struct key_file_list *listA = (const struct key_file_list *)a;
    const struct key_file_list *listB = (const struct key_file_list *)b;

    return (listA->low_list < listB->low_list) ? -1 : (listA->low_list > listB->low_list);
}
#endif // WRITE_KEY_FILES

#ifdef HAVE_HDF5
/**
 * @brief Creates and writes an attribute to an HDF5 file.
 *
 * This function creates an attribute in the HDF5 file or group specified by `handle`, with the given name and data
 * type. It writes the provided data to this attribute. The function supports both scalar and array attributes, with
 * dimensions specified by `adim` if provided.
 *
 * @param name Name of the attribute to be created.
 * @param datatype HDF5 data type identifier for the attribute's data.
 * @param data Pointer to the data to be written to the attribute.
 * @param handle The HDF5 file or group identifier where the attribute is to be created.
 * @param dataspace_type Type of dataspace to be used for the attribute (e.g., `H5S_SIMPLE` for simple dataspace).
 * @param adim Optional array specifying the dimensions of the attribute if `dataspace_type` is `H5S_SIMPLE`. If
 * `nullptr`, the attribute is assumed to be scalar.
 */
void attribute(const char *name, hid_t datatype, void *data, hid_t handle, H5S_class_t dataspace_type,
               hsize_t adim[1] = nullptr)
{
    hid_t hdf5_dataspace, hdf5_attribute;
    hdf5_dataspace = H5Screate(dataspace_type);
    if (adim != nullptr)
    {
        H5Sset_extent_simple(hdf5_dataspace, 1, adim, NULL);
    }

    hdf5_attribute = H5Acreate2(handle, name, datatype, hdf5_dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(hdf5_attribute, datatype, data);
    H5Aclose(hdf5_attribute);
    H5Sclose(hdf5_dataspace);
};

/**
 * @brief Writes header attributes to an HDF5 file.
 *
 * This function writes various attributes from the `header` structure to an HDF5 file using the provided handle.
 *
 * @param handle The HDF5 file identifier or group identifier where the attributes are to be written.
 */
void write_header_attributes_in_hdf5(hid_t handle)
{
    hsize_t adim[1] = {6};
    attribute("NumPart_ThisFile", H5T_NATIVE_INT, header.npart, handle, H5S_SIMPLE, adim);
    attribute("NumPart_Total", H5T_NATIVE_UINT, header.npartTotal, handle, H5S_SIMPLE, adim);
    attribute("NumPart_Total_HighWord", H5T_NATIVE_UINT, header.npartTotalHighWord, handle, H5S_SIMPLE, adim);
    attribute("MassTable", H5T_NATIVE_DOUBLE, header.mass, handle, H5S_SIMPLE, adim);
    attribute("Time", H5T_NATIVE_DOUBLE, &header.time, handle, H5S_SCALAR);
    attribute("Redshift", H5T_NATIVE_DOUBLE, &header.redshift, handle, H5S_SCALAR);
    attribute("BoxSize", H5T_NATIVE_DOUBLE, &header.BoxSize, handle, H5S_SCALAR);
    attribute("NumFilesPerSnapshot", H5T_NATIVE_INT, &header.num_files, handle, H5S_SCALAR);
    attribute("Omega0", H5T_NATIVE_DOUBLE, &header.Omega0, handle, H5S_SCALAR);
    attribute("OmegaLambda", H5T_NATIVE_DOUBLE, &header.OmegaLambda, handle, H5S_SCALAR);
    attribute("HubbleParam", H5T_NATIVE_DOUBLE, &header.HubbleParam, handle, H5S_SCALAR);
    attribute("Flag_Sfr", H5T_NATIVE_INT, &header.flag_sfr, handle, H5S_SCALAR);
    attribute("Flag_Cooling", H5T_NATIVE_INT, &header.flag_cooling, handle, H5S_SCALAR);
    attribute("Flag_StellarAge", H5T_NATIVE_INT, &header.flag_stellarage, handle, H5S_SCALAR);
    attribute("Flag_Metals", H5T_NATIVE_INT, &header.flag_metals, handle, H5S_SCALAR);
    attribute("Flag_Feedback", H5T_NATIVE_INT, &header.flag_feedback, handle, H5S_SCALAR);
    attribute("Flag_DoublePrecision", H5T_NATIVE_INT, &header.flag_doubleprecision, handle, H5S_SCALAR);
    attribute("Flag_IC_Info", H5T_NATIVE_INT, &header.flag_ic_info, handle, H5S_SCALAR);
}
#endif // HAVE_HDF5

/**
 * @brief Writes data to a file stream and handles potential I/O errors.
 *
 * This function wraps around `fwrite` to handle errors and ensure that the correct number of items are written. If the
 * number of bytes to be written is greater than zero, it checks if the number of items written matches the expected
 * count. If not, it reports an error and halts the program. If the number of bytes to write is zero, the function
 * simply returns zero.
 *
 * @param ptr Pointer to the data to be written to the file.
 * @param size Size of each item to be written, in bytes.
 * @param nmemb Number of items to be written.
 * @param stream Pointer to the `FILE` object that specifies the output stream.
 *
 * @return The number of items successfully written. If an error occurs, an appropriate error message is printed, and
 * the program is halted.
 */
size_t my_fwrite(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    size_t nwritten;

    if (size * nmemb > 0)
    {
        if ((nwritten = fwrite(ptr, size, nmemb, stream)) != nmemb)
        {
            ENDRUN_MAIN("I/O error (fwrite) on task=%d has occured: %s ", ThisTask, strerror(errno));
        }
    }
    else
    {
        nwritten = 0;
    }
    return nwritten;
}

/**
 * @brief Reads data from a file stream and handles potential I/O errors.
 *
 * This function wraps around `fread` to handle errors and provide verbose error messages. It checks if the number of
 * items read matches the expected count and reports errors if not. If the number of bytes to read is zero, it simply
 * returns zero.
 *
 * @param ptr Pointer to a buffer where the read data will be stored.
 * @param size Size of each item to be read, in bytes.
 * @param nmemb Number of items to be read.
 * @param stream Pointer to the `FILE` object that specifies the input stream.
 *
 * @return The number of items successfully read. If an error occurs, an appropriate error message is printed, and the
 * program is halted.
 */
size_t my_fread_impl(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *caller_func,
                     const char *caller_file, int caller_line)
{
    // Check if number of bytes to be read is zero
    if (size == 0 || nmemb == 0)
    {
        return 0;
    }

    errno = 0; // reset before call
    size_t nread = fread(ptr, size, nmemb, stream);
    int saved_errno = errno;
    // Check if the number of items read is different to the expected count
    if (nread != nmemb)
    {
        size_t bytes_expected = size * nmemb;
        size_t bytes_read = size * nread;

        long pos = ftell(stream); // may be -1 if unsupported

        if (feof(stream))
        {
            ENDRUN("fread EOF: requested=%zu elements (%zu bytes), read=%zu elements (%zu bytes)\n"
                   "stream=%p, file_pos=%ld\n"
                   "caller=%s (%s:%d), task=%d",
                   nmemb, bytes_expected, nread, bytes_read, (void *)stream, pos, caller_func, caller_file, caller_line,
                   ThisTask);
        }
        else if (ferror(stream))
        {
            ENDRUN("fread ERROR: %s\n"
                   "requested=%zu elements (%zu bytes), read=%zu elements (%zu bytes)\n"
                   "stream=%p, file_pos=%ld, errno=%d\n"
                   "caller=%s (%s:%d), task=%d",
                   strerror(saved_errno), nmemb, bytes_expected, nread, bytes_read, (void *)stream, pos, saved_errno,
                   caller_func, caller_file, caller_line, ThisTask);
        }
        else
        {
            // Rare but possible: neither feof nor ferror set
            ENDRUN("fread UNKNOWN SHORT READ\n"
                   "requested=%zu elements, read=%zu elements\n"
                   "stream=%p, file_pos=%ld\n"
                   "caller=%s (%s:%d), task=%d",
                   nmemb, nread, (void *)stream, pos, caller_func, caller_file, caller_line, ThisTask);
        }
    }

    return nread;
}

#if defined(ORDER_SNAPSHOTS_BY_ID) || defined(SUBFIND_READ_FOF) || defined(SUBFIND_RESHUFFLE_CATALOGUE)
/**
 * @brief Compares two `particle_data` structures based on their ID field.
 *
 * This function is used for sorting or ordering `particle_data` structures by their ID field.
 * It returns a comparison result based on the ID values of the two `particle_data` elements.
 *
 * @param a Pointer to the first `particle_data` structure to compare.
 * @param b Pointer to the second `particle_data` structure to compare.
 *
 * @return -1 if the ID of the first element is smaller than the ID of the second element,
 *         +1 if the ID of the first element is larger,
 *         0 if both IDs are equal.
 */
int io_compare_P_ID(const void *a, const void *b)
{
    // Cast the input pointers to a data structure type
    const struct particle_data *particle_a = (const struct particle_data *)a;
    const struct particle_data *particle_b = (const struct particle_data *)b;

    // Compare the IDs
    if (particle_a->ID < particle_b->ID)
    {
        return -1;
    }
    else if (particle_a->ID > particle_b->ID)
    {
        return +1;
    }
    return 0;
}

/**
 * @brief Compares two `particle_data` structures based on their Group Number (GrNr) and Subgroup Number (SubNr) fields.
 *
 * This function is used for sorting or ordering `particle_data` structures. It first compares the Group Numbers (GrNr)
 * of the two elements. If the Group Numbers are equal, it then compares the Subgroup Numbers (SubNr).
 *
 * @param a Pointer to the first `particle_data` structure to compare.
 * @param b Pointer to the second `particle_data` structure to compare.
 *
 * @return -1 if the Group Number (GrNr) of the first element is smaller than that of the second element,
 *         +1 if the Group Number (GrNr) of the first element is larger,
 *         0 if both Group Numbers (GrNr) are equal and the Subgroup Number (SubNr) of the first element is smaller,
 *         +1 if the Subgroup Number (SubNr) of the first element is larger,
 *         or 0 if both Group Numbers and Subgroup Numbers are equal.
 */
int io_compare_P_GrNr_SubNr(const void *a, const void *b)
{
    // Cast the input pointers to the data structure type
    const struct particle_data *particle_a = (const struct particle_data *)a;
    const struct particle_data *particle_b = (const struct particle_data *)b;

    // Compare the GrNr
    if (particle_a->GrNr < particle_b->GrNr)
    {
        return -1;
    }
    else if (particle_a->GrNr > particle_b->GrNr)
    {
        return +1;
    }

    // If equal GrNr, compare the IDs
    if (particle_a->SubNr < particle_b->SubNr)
    {
        return -1;
    }
    else if (particle_a->SubNr > particle_b->SubNr)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Compares two particles based on their group number (GrNr) and ID attributes.
 *
 * This function is designed to be used as a comparison function in sorting algorithms, such as `qsort`. It compares two
 * particles based on their `GrNr` (group number) first, and then by their `ID` if the `GrNr` values are equal.
 *
 * The function performs the following comparisons:
 * - If `GrNr` of the first particle is less than the `GrNr` of the second particle, the function returns -1.
 * - If `GrNr` of the first particle is greater than the `GrNr` of the second particle, the function returns 1.
 * - If the `GrNr` values are equal, the function compares the `ID` values:
 *   - If the `ID` of the first particle is less than the `ID` of the second particle, it returns -1.
 *   - If the `ID` of the first particle is greater than the `ID` of the second particle, it returns 1.
 *   - If both `GrNr` and `ID` are equal, the function returns 0.
 *
 * @param a Pointer to the first particle to compare.
 * @param b Pointer to the second particle to compare.
 * @return int Returns -1, 0, or 1 based on the comparison result.
 */
int io_compare_P_GrNr_ID(const void *a, const void *b)
{
    // Cast the input pointers to the data structure type
    const struct particle_data *particle_a = (const struct particle_data *)a;
    const struct particle_data *particle_b = (const struct particle_data *)b;

    // Compare the GrNr
    if (particle_a->GrNr < particle_b->GrNr)
    {
        return -1;
    }

    if (particle_a->GrNr > particle_b->GrNr)
    {
        return +1;
    }

    // If equal GrNr, compare the IDs
    if (particle_a->ID < particle_b->ID)
    {
        return -1;
    }

    if (particle_a->ID > particle_b->ID)
    {
        return +1;
    }

    return 0;
}
#endif // if defined(ORDER_SNAPSHOTS_BY_ID) || defined(SUBFIND_READ_FOF) || defined(SUBFIND_RESHUFFLE_CATALOGUE)

#ifdef LT_SEv_INFO_DETAILS
/**
 * @brief Writes data related to particles of type 4 to separate files named "IDZ.datX", where X represents the task
 * number.
 *
 * This function outputs the particle IDs, masses, and metallicities for all particles of type 4 to a binary file.
 * The output is structured as follows:
 * - The file is named "IDZ.datX", where `X` is the MPI task number (`ThisTask`).
 * - The first entry in the file is the total number of particles of type 4 across all tasks (`ntot_type_all[4]`).
 * - For each particle of type 4, the function writes the particle ID, mass, and metallicity to the file.
 *
 * @return void
 */
void write_IDZ(void)
{
    char buf[500];
    FILE *myFile;
    int totn_written = 0;

#ifdef LONG_OUTPUT_NUMBERS
    sprintf(buf, "%s%s%04d", All.OutputDir, "IDZ.dat", ThisTask);
#else // LONG_OUTPUT_NUMBERS
    sprintf(buf, "%s%s%03d", All.OutputDir, "IDZ.dat", ThisTask);
#endif // LONG_OUTPUT_NUMBERS

    for (int j = 0; j < NTask; j++)
    {
        if (ThisTask == j && n_type[4] > 0)
        {
            if ((myFile = fopen(buf, "wb")) == 0x0)
            {
                ENDRUN_MAIN("Error in opening file '%s' ", buf);
            }
            if (totn_written == 0)
            {
                fwrite(&ntot_type_all[4], sizeof(long long), 1, myFile);
                totn_written = 1;
            }
            for (unsigned long long int i = 0; i < NumPart; i++)
            {
                if (P[i].Type == 4)
                {
                    float myZ = (float)get_metallicity(i, -1);
#ifndef LONGIDS
                    fwrite(&P[i].ID, sizeof(int), 1, myFile);
#else // LONGIDS
                    fwrite(&P[i].ID, sizeof(long long), 1, myFile);
#endif // LONGIDS
                    fwrite(&MetP[P[i].pt.MetID].iMass, sizeof(float), 1, myFile);
                    fwrite(&myZ, sizeof(float), 1, myFile);
                }
            }
            // close file
            fclose(myFile);
        }
        MPI_Barrier(MYMPI_COMM_WORLD);
    }
    return;
}
#endif // LT_SEv_INFO_DETAILS

#if ((defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)) || defined(LT_METAL_COOLING)) && defined(LT_LOG_SMOOTH)
/**
 * @brief Calculates and writes data related to the metallicity of particles to output files.
 *
 * This function processes particle data to compute properties related to metallicity,
 * temperature, and cooling rates, then writes this data to output files.
 *
 * @param num An integer used in naming the output file to distinguish different runs or datasets.
 * @return void This function does not return a value.
 */
void Spy_Z(int num)
{
    unsigned int Ndone, *Ndone_all;
    double a3inv;
    float *buffer;

    int bunch = 500000;

    FILE *file;
    char fname[500];
    char head[1000];

    ignore_failure_in_convert_u = 1;

    if (All.ComovingIntegrationOn)
    {
        a3inv = 1 / (All.Time * All.Time * All.Time);
    }
    else
    {
        a3inv = 1;
    }

    XH = HYDROGEN_MASSFRACTION;
    yhelium = (1 - XH) / (4 * XH);

    int nwrite = (NTask / All.NumFilesWrittenInParallel) + (NTask % All.NumFilesWrittenInParallel);

    if ((buffer = (float *)malloc(bunch * sizeof(float))) != NULL) // Allocate memory for the buffer
    {
        for (int k = 0; k < nwrite; k++)
        {
            int TaskStart = k * All.NumFilesWrittenInParallel;
            int TaskStop = (k + 1) * All.NumFilesWrittenInParallel;
            if (TaskStop > NTask)
            {
                TaskStop = NTask;
            }

            if (ThisTask >= TaskStart && ThisTask < TaskStop)
            {
                double Ls, Ls_a, Ls_b, Lt, L0, Zs, Zt, metal_mass;
                Ndone = 0;
                for (int i = 0; i < NumPart; i++)
                {
                    if (P[i].Type == 0 && SphP[i].a2.Density > 0)
                    {
                        metal_mass = get_metalmass(SphP[i].Metals);
#ifndef LT_SMOOTH_Z
                        if (metal_mass > 0)
                        {
                            Ndone++;
                        }
#else // LT_SMOOTH_Z
                        if ((metal_mass > 0) || (SphP[i].Zsmooth > 0))
                        {
                            Ndone++;
                        }
#endif // LT_SMOOTH_Z
                    }
                }

                if (Ndone > 0)
                {
                    continue;
                }

                sprintf(fname, "zsmooth.%d.%d.txt", num, ThisTask);
                if ((file = fopen(fname, "w")) == NULL)
                {
                    free(buffer);
                    ignore_failure_in_convert_u = 0;
                    return;
                }

                sprintf(head, "temp dens Z zeroZ_cool ");
                int ndata = 4;
#ifdef LT_METAL_COOLING
                sprintf(&head[strlen(head)], "Zcool ");
                ndata++;
#endif // LT_METAL_COOLING

#ifdef LT_SMOOTH_Z
                sprintf(&head[strlen(head)], "smoothZ ");
                ndata++;
#ifdef LT_SMOOTH_SIZE
                sprintf(&head[strlen(head)], "sZ_Nngb ");
                ndata++;
#endif // LT_SMOOTH_SIZE
#ifdef LT_METAL_COOLING
                sprintf(&head[strlen(head)], "sZ_cool ");
                ndata++;
#endif // LT_METAL_COOLING
#endif // LT_SMOOTH_Z
                int i = strlen(head);
                fwrite(&i, sizeof(int), 1, file);
                fprintf(file, "%s", head);

                int b = 0;
                for (int i = 0; i < NumPart; i++)
                {
                    if (P[i].Type == 0 && SphP[i].a2.Density > 0)
                    {
                        metal_mass = get_metalmass(SphP[i].Metals);
#ifndef LT_SMOOTH_Z
                        if (metal_mass == 0)
                        {
                            continue;
                        }
#else // LT_SMOOTH_Z
                        if ((metal_mass == 0) && (SphP[i].Zsmooth == 0))
                        {
                            continue;
                        }
#endif // LT_SMOOTH_Z

                        double Ne = (double)SphP[i].elec;

                        double mu = (1 + 4 * yhelium) / (1 + yhelium + SphP[i].elec);

                        double u = SphP[i].Entropy / GAMMA_MINUS1 * pow(SphP[i].a2.Density * a3inv, GAMMA_MINUS1) *
                                   All.UnitPressure_in_cgs / All.UnitDensity_in_cgs;
                        double rho =
                            SphP[i].a2.Density * a3inv * All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;

                        double T = convert_u_to_tempSTD(u, rho, &Ne);

                        if ((Zt = get_metallicity(i, -1)) / 0.02 < 1e-8)
                        {
                            Zt = 1.77e-3 * 1e-8;
                        }
#ifdef LT_SMOOTH_Z
                        Zs = SphP[i].Zsmooth;
#endif // LT_SMOOTH_Z

#ifdef LT_METAL_COOLING
                        Lt = CoolingRateFromU(u, rho, &Ne, get_metallicity_solarunits(get_metallicity(i, Iron)));
#endif // LT_METAL_COOLING
#ifdef LT_SMOOTH_Z
                        if (metal_mass > 0)
                        {
                            Ls = CoolingRateFromU(u, rho, &Ne,
                                                  get_metallicity_solarunits(Zs * SphP[i].Metals[Iron] / metal_mass));
                        }
                        else
                        {
                            Ls = Ls_a = Ls_b = 0;
                        }
#endif // LT_SMOOTH_Z
                        L0 = CoolingRateFromU(u, rho, &Ne, NO_METAL);

                        /*
                           Ls = CoolingRateGetMetalLambda(T, get_metallicity_solarunits(SphP[i].Zsmooth));
                           Lt = GetMetalLambda(T, get_metallicity(i, -1));
                         */

                        buffer[b++] = (float)T;
                        buffer[b++] = (float)SphP[i].a2.Density * a3inv / SFs[0].PhysDensThresh[0];
                        buffer[b++] = (float)Zt;
                        buffer[b++] = (float)L0;
#ifdef LT_METAL_COOLING
                        buffer[b++] = (float)Lt;
#endif // LT_METAL_COOLING
#ifdef LT_SMOOTH_Z
                        buffer[b++] = (float)Zs;
#ifdef LT_SMOOTH_SIZE
                        buffer[b++] = (float)SphP[i].SmoothNgb;
#endif // LT_SMOOTH_SIZE
#ifdef LT_METAL_COOLING
                        buffer[b++] = (float)Ls;
#endif // LT_METAL_COOLING
#endif // LT_SMOOTH_Z

                        if (b > bunch - ndata) // If the buffer is full, write the data to the file
                        {
                            fwrite(buffer, sizeof(float), b, file);
                            b = 0;
                        }
                    }
                }

                if (b > 0) // Write remaining data in the buffer to the file
                {
                    fwrite(buffer, sizeof(float), b, file);
                    b = 0;
                }
                fclose(file); // Close the file
            }
            MPI_Barrier(MYMPI_COMM_WORLD); // Wait for all tasks to synchronize
        }
    }

    free(buffer);
    if (ThisTask == 0)
    {
        Ndone_all = (unsigned int *)calloc(NTask, sizeof(unsigned));
    }
    MPI_Barrier(MYMPI_COMM_WORLD);

    MPI_Gather(&Ndone, 1, MPI_INT, Ndone_all, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

    if (ThisTask == 0)
    {
        unsigned long long int NdoneTot = Ndone;
        for (int i = 1; i < NTask; i++)
        {
            NdoneTot += Ndone_all[i];
        }
        free(Ndone_all);
        VERBOSE(3, "done for %llu particles ", NdoneTot);
    }

    ignore_failure_in_convert_u = 0;
    return;
}
#endif // if ( (defined(LT_SMOOTH_Z) && !defined(LT_SMOOTH_ALLMETALS)) || defined(LT_METAL_COOLING)) &&
       // defined(LT_LOG_SMOOTH)
