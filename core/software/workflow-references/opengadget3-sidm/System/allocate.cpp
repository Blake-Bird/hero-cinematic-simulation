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
#ifdef ACC
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // ACC

/**
 * @brief Allocates memory for particle storage and other data structures.
 *
 * This function allocates memory for particle storage, including both collisionless and SPH particles.
 * It also allocates memory for other data structures like the ordered binary tree of the timeline.
 */
void allocate_memory(void)
{
    size_t bytes;
    double bytes_tot = 0;
    int NTaskTimesThreads;

    NTaskTimesThreads = maxThreads * NTask;

    Exportflag = (int *)mymalloc("Exportflag", NTaskTimesThreads * sizeof(int));
    Exportindex = (int *)mymalloc("Exportindex", NTaskTimesThreads * sizeof(int));
    Exportnodecount = (int *)mymalloc("Exportnodecount", NTaskTimesThreads * sizeof(int));

    Send_count = (int *)mymalloc("Send_count", sizeof(int) * NTask);
    Send_offset = (int *)mymalloc("Send_offset", sizeof(int) * NTask);
    Recv_count = (int *)mymalloc("Recv_count", sizeof(int) * NTask);
    Recv_offset = (int *)mymalloc("Recv_offset", sizeof(int) * NTask);

    VERBOSE(2, "Max number of particles per task: %llu\n", (unsigned long long)All.MaxPart);

    ProcessedFlag = (unsigned char *)mymalloc("ProcessedFlag", bytes = All.MaxPart * sizeof(unsigned char));
    bytes_tot += bytes;

    NextActiveParticle = (int *)mymalloc("NextActiveParticle", bytes = All.MaxPart * sizeof(int));
    bytes_tot += bytes;

#ifdef _OPENMP
    ActiveParticleList = (int *)mymalloc("ActiveParticleList", bytes = All.MaxPart * sizeof(int));
    bytes_tot += bytes;
#endif // _OPENMP

    NextInTimeBin = (int *)mymalloc("NextInTimeBin", bytes = All.MaxPart * sizeof(int));
    bytes_tot += bytes;

    PrevInTimeBin = (int *)mymalloc("PrevInTimeBin", bytes = All.MaxPart * sizeof(int));
    bytes_tot += bytes;

    if (All.MaxPart > 0)
    {
        if (!(P = (struct particle_data *)mymalloc("P", bytes = All.MaxPart * sizeof(struct particle_data))))
        {
            ENDRUN("failed to allocate memory for `P' (%g MB).\n", bytes / (1024.0 * 1024.0));
        }
        bytes_tot += bytes;

        VERBOSE(2, "\nAllocated %g MByte for particle storage.\n\n", bytes_tot / (1024.0 * 1024.0));
    }

    if (All.MaxPartSph > 0)
    {
        bytes_tot = 0;

        if (!(SphP = (struct sph_particle_data *)mymalloc("SphP",
                                                          bytes = All.MaxPartSph * sizeof(struct sph_particle_data))))
        {
            ENDRUN("failed to allocate memory for `SphP' (%g MB).\n", bytes / (1024.0 * 1024.0));
        }
        bytes_tot += bytes;

#ifdef AR_SOA_LITE // with dictionary this will became automatic
        AR_ALLOC(SphPSoa.wakeup, short int, All.MaxPartSph, bytes_tot);
#endif // AR_SOA_LITE

        N_ToWakeUp = 0;
        ToWakeupList = (int *)mymalloc("WakeUpList", All.MaxPartSph * sizeof(int));

        VERBOSE(0, "Allocated %g MByte for storage of SPH data.\n\n", bytes_tot / (1024.0 * 1024.0));
    }

#ifdef AXION_DM
    if (All.MaxPartAx > 0)
    {
        bytes_tot = 0;

        if (!(AxP =
                  (struct ax_particle_data *)mymalloc("AxP", bytes = All.MaxPartAx * sizeof(struct ax_particle_data))))
        {
            ENDRUN("failed to allocate memory for `AxP' (%g MB).\n", bytes / (1024.0 * 1024.0));
        }
        bytes_tot += bytes;
        VERBOSE(2, "Allocated %g MByte for storage of AX data.\n\n", bytes_tot / (1024.0 * 1024.0));
    }
#endif // AXION_DM

#ifdef SFR
    if (All.MaxPartMet > 0)
    {
        bytes_tot = 0;

        if (!(MetP = (struct met_particle_data *)mymalloc("MetP",
                                                          bytes = All.MaxPartMet * sizeof(struct met_particle_data))))
        {
            ENDRUN("failed to allocate memory for `MetP' (%g MB).\n", bytes / (1024.0 * 1024.0));
        }
        bytes_tot += bytes;

        VERBOSE(3, "Allocated %g MByte for storage of MET data.\n\n", bytes_tot / (1024.0 * 1024.0));
    }
#endif // SFR

#ifdef BLACK_HOLES
    if (All.MaxPartBH > 0)
    {
        bytes_tot = 0;

        if (!(BHP =
                  (struct bh_particle_data *)mymalloc("BHP", bytes = All.MaxPartBH * sizeof(struct bh_particle_data))))
        {
            ENDRUN("failed to allocate memory for `BHP' (%g MB).\n", bytes / (1024.0 * 1024.0));
        }
        bytes_tot += bytes;
        VERBOSE(2, "Allocated %g MByte for storage of BH data.\n\n", bytes_tot / (1024.0 * 1024.0));
    }
#endif // BLACK_HOLES

#ifdef WRITE_KEY_FILES
    if (All.MaxPart > 0)
    {
        bytes_tot = 0;
        KeyIndex = (peanokey *)mymalloc("KeyIndex", bytes = All.MaxPart * sizeof(peanokey));
        bytes_tot += bytes;
        NPartPerKey = (int *)mymalloc("KeyNpart", bytes = All.MaxPart * sizeof(int));
        bytes_tot += bytes;
        PartKeyOffset = (int *)mymalloc("KeyOffset", bytes = All.MaxPart * sizeof(int));
        bytes_tot += bytes;
        VERBOSE(2, "Allocated %g MByte for storage of KEY data.\n\n", bytes_tot / (1024.0 * 1024.0));
    }
#endif // WRITE_KEY_FILES

    // TODO : (David) Find convenient location to destroy objects (i.e. deallocate memory)
#if GADGET_HYDRO == HYDRO_MFM
    eos = new AdiabaticEos<mfm_numdims>(MyFloat(GAMMA));
    fluxSolver = new FluxSolver<mfm_numdims>(eos);

#if MFM_SLOPE_LIMITER == LIMITER_NULL
    slopeLimiter = new SlopeLimiter_Null<mfm_numdims>();
#elif MFM_SLOPE_LIMITER == LIMITER_ZERO_SLOPES
    slopeLimiter = new SlopeLimiter_ZeroSlopes<mfm_numdims>();
#elif MFM_SLOPE_LIMITER == LIMITER_TVD_SCALAR
    slopeLimiter = new SlopeLimiter_TVDScalar<mfm_numdims>();
#elif MFM_SLOPE_LIMITER == LIMITER_SCALAR
    slopeLimiter = new SlopeLimiter_Scalar<mfm_numdims>();
#elif MFM_SLOPE_LIMITER == LIMITER_AREPO
    slopeLimiter = new SlopeLimiter_Arepo<mfm_numdims>();
#elif MFM_SLOPE_LIMITER == LIMITER_GIZMO
    slopeLimiter = new SlopeLimiter_Gizmo<mfm_numdims, SLOPE_LIMITER_TOLERANCE>();
#endif // MFM_SLOPE_LIMITER == ...

#if !defined(NO_PAIRWISE_LIMITER) || defined(PAIRWISE_SLOPE_LIMITER_GIZMO_CODE)
#ifdef PAIRWISE_SLOPE_LIMITER_GIZMO_CODE
#define USE_PAPER_VERSION 0
#else // PAIRWISE_SLOPE_LIMITER_GIZMO_CODE
#define USE_PAPER_VERSION 1
#endif // PAIRWISE_SLOPE_LIMITER_GIZMO_CODE
    pairwiseSlopeLimiter = new PairwiseSlopeLimiter_Gizmo<mfm_numdims, SLOPE_LIMITER_TOLERANCE, USE_PAPER_VERSION>;
#else // !defined(NO_PAIRWISE_LIMITER) || defined(PAIRWISE_SLOPE_LIMITER_GIZMO_CODE)
    pairwiseSlopeLimiter = new PairwiseSlopeLimiter_Null<mfm_numdims>;
#endif // !defined(NO_PAIRWISE_LIMITER) || defined(PAIRWISE_SLOPE_LIMITER_GIZMO_CODE)

#if !defined(DONT_BIN_MFM_FLUXES) && defined(_OPENMP)
    if (!(MfmNgbLockList = (omp_lock_t *)mymalloc("MfmNgbLockList", bytes = All.MaxPartSph * sizeof(omp_lock_t))))
    {
        ENDRUN("failed to allocate memory for %d mfm neighbor flux locks (%g MB).\n", All.MaxPartSph,
               bytes / (1024.0 * 1024.0));
    }
    bytes_tot += bytes;

    for (int i = 0; i < All.MaxPartSph; i++)
    { // note loop gets worse if parallelised
        omp_init_lock(&MfmNgbLockList[i]);
    }
#endif // !defined(DONT_BIN_MFM_FLUXES) && defined(_OPENMP)
#endif // GADGET_HYDRO == HYDRO_MFM

#ifdef AR_TREEBUILD_PARALLEL
    TIMER_PUSH(7, "initialising_openmp_locks");
    double tree_alloc_factor = max(All.TreeAllocFactor, 1.);
    int _MaxNodes = (int)(All.MaxPart * tree_alloc_factor) + MaxTopNodes;
    if (!(NodesLockList = (omp_lock_t *)mymalloc("NodesLockList", bytes = (_MaxNodes + 1) * sizeof(omp_lock_t))))
    {
        ENDRUN("failed to allocate memory for %d tree-nodes locks (%g MB).\n", _MaxNodes, bytes / (1024.0 * 1024.0));
    }
    bytes_tot += bytes;

    for (int i = 0; i < _MaxNodes + 1; i++)
    { // note loop gets worse if parallelised
        omp_init_lock(&NodesLockList[i]);
    }

    VERBOSE(3, "Allocated %d locks: my TreeAllocFactor=%g; All.MaxPart=%d; NTopnodes=%d", _MaxNodes + 1,
            tree_alloc_factor, All.MaxPart, NTopnodes);
    NumberOfAllocatedNodeLocks = _MaxNodes;

    TIMER_POP(7, "initialising_openmp_locks");
#endif // AR_TREEBUILD_PARALLEL

#ifdef PXP_OPENMP
    if (!(pxpPartLockList = (omp_lock_t *)mymalloc("pxpPartLockList", bytes = All.MaxPart * sizeof(omp_lock_t))))
    {
        ENDRUN_MAIN("failed to allocate memory for %d PXP particle locks (%g MB).\n", All.MaxPart,
                    bytes / (1024.0 * 1024.0));
    }
    bytes_tot += bytes;

    for (int i = 0; i < All.MaxPart; ++i)
    { // note loop gets worse if parallelised
        omp_init_lock(&pxpPartLockList[i]);
    }
#endif // PXP_OPENMP
}
