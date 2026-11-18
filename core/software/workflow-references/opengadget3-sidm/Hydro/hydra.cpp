/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the implementation of the hydrodynamic routines.
 *
 * @author OpenGadget3-development team, in particular Ludwig Böss, Klaus Dolag, Frederick Groth and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/communication.hpp"
#include "kernel.hpp"

#ifndef ACC_INCLUDE_HYDRA
#include "hydra_structs.hpp"
#ifdef AR_FIX_TIME_DEP_ART_COND
struct hydro_store_out *HydroStoreOut;
#endif // AR_FIX_TIME_DEP_ART_COND
struct hydrodata_in *HydroDataIn, *HydroDataGet;
struct hydrodata_out *HydroDataResult, *HydroDataOut;
#endif // ACC_INCLUDE_HYDRA

#include "hydra_extra.hpp"

#ifdef SHOCKFINDER_SPH
#include "shockfinder.hpp"
#endif // SHOCKFINDER_SPH

// #ifdef TIME_DEP_ART_VISC
// #include "viscosity.hpp"
// #endif // TIME_DEP_ART_VISC

#include "../CodeBase/greentree.hpp"

#ifndef ACC_INCLUDE_HYDRA

#ifdef ACC_HYDRA
static inline void particle2in_hydra(struct hydrodata_in *in, int i);
#else // ACC_HYDRA
static inline void particle2in_hydra(struct hydrodata_in *in, int i);
#endif // ACC_HYDRA
static inline void out2particle_hydra(struct hydrodata_out *out, int i, int mode);

#include "hydra.hpp"

#if !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA) // we reinclude this same file if ACC is active
#include "../OpenACC/hydra.hpp"
#endif // if !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA)

/**
 * @brief Compute the hydrodynamical forces.
 *
 * This function is the driver routine for the calculation of hydrodynamical
 * force and rate of change of entropy due to shock heating for all active
 * particles.
 */
void hydro_force(void)
{
    TIMER_PUSH(2, "Initializing");
#ifdef AR_FIX_TIME_DEP_ART_COND
    HydroStoreOut = (hydro_store_out *)mymalloc("HydroStoreOut", N_gas * sizeof(hydro_store_out));
#endif // AR_FIX_TIME_DEP_ART_COND

    VERBOSE(6, "ThisTask%d hydro_force.", ThisTask);
#ifdef ACC_FIX_ASSIGN
    if (DoPrimaryOnGPU)
#endif // ACC_FIX_ASSIGN
#ifdef AR_FIX_ASSIGN
#ifdef _OPENMP
#pragma omp parallel for
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
#else // _OPENMP
        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
#endif // _OPENMP
            if (IS_SPH_ALIVE(P[i]))
            {
                P[i].assigned = 0;
            }
        }
#endif // AR_FIX_ASSIGN

    if (All.ComovingIntegrationOn)
    {
        /* Factors for comoving integration of hydro */
        cosmo_hydra.hubble_a = hubble_function(All.Time);
        cosmo_hydra.hubble_a2 = All.Time * All.Time * cosmo_hydra.hubble_a;

        cosmo_hydra.fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
        cosmo_hydra.fac_vsic_fix = cosmo_hydra.hubble_a * pow(All.Time, 3 * GAMMA_MINUS1);

        cosmo_hydra.a3inv = 1 / (All.Time * All.Time * All.Time);
        cosmo_hydra.atime = All.Time;
    }
    else
    {
        cosmo_hydra.hubble_a = cosmo_hydra.hubble_a2 = cosmo_hydra.atime = cosmo_hydra.fac_mu =
            cosmo_hydra.fac_vsic_fix = cosmo_hydra.a3inv = 1.0;
    }

    /* allocate buffers to arrange communication */
    int NTaskTimesNumPart = (static_cast<int>(maxThreads)) * (static_cast<int>(NumPart / DENSITY_LESS_NGB_FACTOR));

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer;", NTaskTimesNumPart * sizeof(int));

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize = static_cast<int>((MyBufferSize * 1024 * 1024) /
                                     (sizeof(struct data_index) + 3 * sizeof(struct data_nodelist) +
                                      sizeof(struct hydrodata_in) + sizeof(struct hydrodata_out) +
                                      sizemax(sizeof(struct hydrodata_in), sizeof(struct hydrodata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeListIn", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "Hydra: using %g MB for buffering, %g MB remaining for other buffers\n", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    NextParticle = FirstActiveParticle; /* beginn with this index */

    int ndone, group_particles = 1;

    TIMER_POP(2, "Initializing");

#ifdef ACC_HYDRA
    struct hydrodata_in *LiteHIn;
    struct hydrodata_out *LiteHOut;

    int *HydraActiveParticleList;
    int NHydraActivePart;
    int should_drift = 0;
    int DoPrimaryOnGPU = 0;
    TIMER_PUSH(1, "ACC_INIT");
    NHydraActivePart = openacc_hydra_init(should_drift, DoPrimaryOnGPU, HydraActiveParticleList, LiteHOut, LiteHIn);
    // note: LiteHIn is in/out property because we need the wakeup properties
    TIMER_POPPUSH(1, "ACC_INIT", "ACC_UPLOAD_AND_I");
    if (DoPrimaryOnGPU)
    {
#ifdef ACC_OPENACC
#pragma acc enter data create(LiteHIn [0:N_gas], HydraActiveParticleList [0:NHydraActivePart],                         \
                              LiteHOut [0:NHydraActivePart], ToWakeupList [0:N_gas])

#pragma acc update device(LiteHIn [0:N_gas], HydraActiveParticleList [0:NHydraActivePart], WalkLiteP [0:N_gas],        \
                          N_ToWakeUp, cosmo_hydra) ACC_ASYNC_COPY1
#elif defined(ACC_OPENMP)
#pragma omp target enter data map(alloc                                                                                \
                                  : LiteHIn [0:N_gas]) map(alloc                                                       \
                                                           : HydraActiveParticleList [0:NHydraActivePart])             \
    map(alloc                                                                                                          \
        : LiteHOut [0:NHydraActivePart]) map(alloc                                                                     \
                                             : ToWakeupList [0:N_gas]) map(to                                          \
                                                                           : cosmo_hydra)

#pragma omp target update to(LiteHIn [0:N_gas], HydraActiveParticleList [0:NHydraActivePart], WalkLiteP [0:N_gas],     \
                             N_ToWakeUp, cosmo_hydra)
#endif // ACC_OPENACC || ACC_OPENMP
    }
    {
        openacc_hydra_primary_phase(DoPrimaryOnGPU, NHydraActivePart, HydraActiveParticleList, LiteHOut, LiteHIn);

        TIMER_POP(1, "ACC_UPLOAD_AND_I");

#else // ACC_HYDRA
    int DoPrimaryOnGPU = 0;
    int should_drift = 1;
#endif // ACC_HYDRA
        do
        {
            TIMER_PUSH(2, "Primary");
            BufferFullFlag = 0;
            Nexport = 0;
            int save_NextParticle = NextParticle;

            for (int j = 0; j < NTask; j++)
            {
                Exportflag[j] = -1;
            }

            /* do local particles and prepare export list */
            TIMER_PUSH(1, "HYD_COMPUTE");
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else // _OPENMP
            int mainthreadid = 0;
#endif // _OPENMP
                ar_green_evaluate_primary(
                    hydro_isactive, hydro_evaluate, Module_Hydro,
#if defined(CS_DECOUPLING) && !defined(CS_DO_NOT_DECOUPLE_HYDRO)
                    TreeWalkType_Gadget3TreeWalk,
#else // CS_DECOUPLING && CS_DO_NOT_DECOUPLE_HYDRO
                TreeWalkType_GreenTreeWalk,
#endif // CS_DECOUPLING && CS_DO_NOT_DECOUPLE_HYDRO
                    NextParticleLoop_ActiveParticleList, NeighbourSearchCriteria_Hydro,
#if defined(CS_DECOUPLING) && !defined(CS_DO_NOT_DECOUPLE_HYDRO)
                    cs_group_criteria<particle_data *>, // actually not called as TreeWalkType_Gadget3TreeWalk is used
                    cs_treewalk_filter_pairs_decoupling_criteria_IPhase<particle_data *>,
#endif // CS_DECOUPLING && CS_DO_NOT_DECOUPLE_HYDRO
                    &mainthreadid, DoPrimaryOnGPU, should_drift, group_particles);
            }
#ifdef ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
            TIMER_POPPUSH(1, "HYD_COMPUTE", "HYD_WAIT");
            MPI_Barrier(MYMPI_COMM_WORLD); // Wait for all tasks to synchronize
            TIMER_POP(1, "HYD_WAIT");
#else // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
        TIMER_POP(1, "HYD_COMPUTE");
#endif // ADD_EXTRA_BARRIER_FOR_EXACT_INBALANCE_MEASURE
            TIMER_POPPUSH(2, "Primary", "Exchange");
            TIMER_PUSH(1, "HYD_COMM_PREP");

            /* if we are not grouping particles and SendRecv fails, then we panic as there is no memory to process even
             * a single particles */
            /*and  if green tree fails, we retry without it */
            int panic_on_error = !group_particles;
            int is_send_receive_ok = prepare_sendrec_offset_send(save_NextParticle, panic_on_error);

            TIMER_POP(1, "HYD_COMM_PREP");

            if (!is_send_receive_ok)
            {
                TIMER_POP(2, "Exchange");
                group_particles = 0;
                continue;
            }

            TIMER_PUSH(1, "HYD_COPY");

            HydroDataGet = (struct hydrodata_in *)mymalloc("HydroDataGet", Nimport * sizeof(struct hydrodata_in));
            DataNodeListGet =
                (struct data_nodelist *)mymalloc("DataNodeListGet", Nimport * sizeof(struct data_nodelist));
            HydroDataIn = (struct hydrodata_in *)mymalloc("HydroDataIn", Nexport * sizeof(struct hydrodata_in));
            DataNodeListIn = (struct data_nodelist *)mymalloc("DataNodeListIn", Nexport * sizeof(struct data_nodelist));

            /* prepare particle data for export */

            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                particle2in_hydra(&HydroDataIn[j], place);
            }
            copy_data_nodelist();

            /* exchange particle data */

            TIMER_POPPUSH(1, "HYD_COPY", "HYD_COMM_EXC");
            templated_exchange_data_in_get(HydroDataIn, HydroDataGet, 1, TAG_HYDRO_A);
            TIMER_POP(1, "HYD_COMM_EXC");

            TIMER_POPPUSH(2, "Exchange", "Secondary");

            TIMER_PUSH(1, "HYD_COMPUTE");

            myfree(DataNodeListIn);
            myfree(HydroDataIn);
            HydroDataResult =
                (struct hydrodata_out *)mymalloc("HydroDataResult", Nimport * sizeof(struct hydrodata_out));
            HydroDataOut = (struct hydrodata_out *)mymalloc("HydroDataOut", Nexport * sizeof(struct hydrodata_out));

            report_memory_usage(&HighMark_sphhydro, "SPH_HYDRO");

            /* now do the particles that were sent to us */
            NextJ = 0;

#ifdef ACC_HYDRA
            openacc_hydra_secondary_phase(HydroDataGet, HydroDataResult, LiteHIn, DoPrimaryOnGPU, should_drift);
#else // ACC_HYDRA
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
#ifdef _OPENMP
            int mainthreadid = omp_get_thread_num();
#else // _OPENMP
            int mainthreadid = 0;
#endif // _OPENMP
            ar_green_evaluate_secondary(
                HydroDataGet, hydro_evaluate, Module_Hydro,
#if defined(CS_DECOUPLING) && !defined(CS_DO_NOT_DECOUPLE_HYDRO)
                TreeWalkType_Gadget3TreeWalk,
#else // CS_DECOUPLING && CS_DO_NOT_DECOUPLE_HYDRO
                TreeWalkType_GreenTreeWalk,
#endif // CS_DECOUPLING && CS_DO_NOT_DECOUPLE_HYDRO
                NeighbourSearchCriteria_Hydro,
#if defined(CS_DECOUPLING) && !defined(CS_DO_NOT_DECOUPLE_HYDRO)
                cs_group_criteria<hydrodata_in *>, // actually not called as TreeWalkType_Gadget3TreeWalk is used
                cs_treewalk_filter_pairs_decoupling_criteria_IIPhase<hydrodata_in *>,
#endif // CS_DECOUPLING && CS_DO_NOT_DECOUPLE_HYDRO
                &mainthreadid, should_drift);
        }
#endif // ACC_HYDRA

            //      ENDRUN_MAIN("addio");
            TIMER_POPPUSH(1, "HYD_COMPUTE", "HYD_WAIT");

            int ndone_flag;
            if (NextParticle < 0)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }

            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

            TIMER_POP(1, "HYD_WAIT");

            TIMER_POPPUSH(2, "Secondary", "Send_Results");

            TIMER_PUSH(1, "HYD_COMM_EXC");
            templated_exchange_data_Sendrecv(HydroDataResult, HydroDataOut, TAG_HYDRO_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POPPUSH(1, "HYD_COMM_EXC", "HYD_COPY");

            /* add the result to the local particles */
            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                out2particle_hydra(&HydroDataOut[j], place, 1);
            }
            TIMER_POP(1, "HYD_COPY");
            TIMER_POP(2, "Send_Results");

            myfree(HydroDataOut);
            myfree(HydroDataResult);
            myfree(DataNodeListGet);
            myfree(HydroDataGet);
        } while (ndone < NTask);

#ifdef ACC_HYDRA
        // note: here below we download the wakeup list from GPU and append the CPU one after it.
        // I wonder if this algorithm will fail in a unified memory run where the array is shared between
        // GPU and CPU, and copied back and forth on each GPU or CPU access?
        // Or on the worst case maybe we just get duplicate entries? AR
        if (DoPrimaryOnGPU)
        {
            TIMER_PUSH(1, "ACC_WAKEUP_CPU");
            int n_ToWakeUp_cpu = N_ToWakeUp;
            int *to_wakeup_list_cpu = NULL;
            // save the CPU wakeup list before the download
            if (n_ToWakeUp_cpu > 0)
            { // not sure how it works with mymalloc of zero size
                to_wakeup_list_cpu = (int *)mymalloc("to wakeup list", sizeof(int) * n_ToWakeUp_cpu);
                memcpy(to_wakeup_list_cpu, ToWakeupList, n_ToWakeUp_cpu * sizeof(int));
            }
            // download the GPU wakeup list
            TIMER_POPPUSH(1, "ACC_WAKEUP_CPU", "ACC_WAIT");
            if (DoPrimaryOnGPU)
            {
#ifdef ACC_OPENACC
#pragma acc wait(1)
#elif defined(ACC_OPENMP)
#pragma omp taskwait
#endif

                TIMER_POPPUSH(1, "ACC_WAIT", "ACC_DOWNLOAD");
#ifdef ACC_OPENACC
#pragma acc update host(LiteHOut [0:NHydraActivePart], N_ToWakeUp)
#elif defined(ACC_OPENMP)
#pragma omp target update from(LiteHOut [0:NHydraActivePart], N_ToWakeUp)
#endif
                if (N_ToWakeUp > 0)
                {
#ifdef ACC_OPENACC
#pragma acc update host(ToWakeupList [0:N_ToWakeUp])
#elif defined(ACC_OPENMP)
#pragma omp target update from(ToWakeupList [0:N_ToWakeUp])
#endif
                }
                //	  printf("DEBUG l%d ThisTask%d wakeup CPU = %d, GPU = %d, N_gas = %d, doIPGPU= %d\n",
                //	 __LINE__, ThisTask, n_ToWakeUp_cpu, N_ToWakeUp, N_gas, DoPrimaryOnGPU);
                assert((n_ToWakeUp_cpu + N_ToWakeUp) < N_gas);

                TIMER_POPPUSH(1, "ACC_DOWNLOAD", "ACC_MEMCPY");
                if (n_ToWakeUp_cpu > 0)
                { // append CPU array to the one uploaded from GPU
                    memcpy(ToWakeupList + N_ToWakeUp, to_wakeup_list_cpu, n_ToWakeUp_cpu * sizeof(int));
                }
                // mask to -1 particles that were updated on the GPU and CPU
                // this loop may be slow: if there are no problems of having duplicates
                // in the wakeup list, then we may remove it.
                TIMER_POPPUSH(1, "ACC_MEMCPY", "ACC_WAKEUP_DOWNLOAD");
                if (N_ToWakeUp > 0 && n_ToWakeUp_cpu > 0)
                {
#ifdef ACC_OPENACC
#pragma acc update host(LiteHIn [0:N_gas])
#elif defined(ACC_OPENMP)
#pragma omp target update from(LiteHIn [0:N_gas])
#endif
                }
                TIMER_POPPUSH(1, "ACC_WAKEUP_DOWNLOAD", "ACC_WAKEUP_MINUS1");

                if (N_ToWakeUp > 0 && n_ToWakeUp_cpu > 0)
                {
#pragma omp parallel for
                    for (int i = 0; i < n_ToWakeUp_cpu; i++)
                    {
                        int j = to_wakeup_list_cpu[i];
                        // if wakeup is != TIMEBINS then the particle has already been woken up by GPU
                        if (LiteHIn[j].wakeup != TIMEBINS)
                        { // GPU woke up that particle too
                            ToWakeupList[N_ToWakeUp + i] = -1;
                        }
                    }
                }

                N_ToWakeUp = N_ToWakeUp + n_ToWakeUp_cpu; // set size as the some of CPU and GPU one

                if (n_ToWakeUp_cpu > 0)
                { // not sure how it works with mymalloc of zero size
                    myfree(to_wakeup_list_cpu);
                }

                TIMER_POPPUSH(1, "ACC_WAKEUP_MINUS1", "ACC_OUT");
#pragma omp parallel for
                for (int counter = 0; counter < NHydraActivePart; counter++)
                {
                    int i = HydraActiveParticleList[counter];
                    out2particle_hydra(&LiteHOut[counter], i, 1);
                }

                TIMER_POPPUSH(1, "ACC_OUT", "ACC_WAKEUP_MERGE");
                if (WAKEUP > 0 && N_ToWakeUp > 0 && n_ToWakeUp_cpu > 0)
                {
                    // LiteHIn is actually an in-out property
                    // just becaouse of the wake up stuff
#pragma omp parallel for
                    for (int i = 0; i < N_gas; i++)
                    {
                        if (P[i].Type == type_gas)
                        {
                            // particle got wakeup in GPU?

                            // Nitya adding checks around variable
#if !defined(AR_SOA_LITE) || defined(AR_SOA_LITE_FROM_RESTART_AOS)

                            if (LiteHIn[i].wakeup != TIMEBINS)
                            {
                                // is the GPU wakeup smaller timebin than CPU?
                                if (SPHP(i, wakeup) == TIMEBINS || SPHP(i, wakeup) > LiteHIn[i].wakeup)
                                {

                                    SPHP(i, wakeup) = LiteHIn[i].wakeup;
                                }
                            }
#endif // !defined(AR_SOA_LITE) || defined(AR_SOA_LITE_FROM_RESTART_AOS)
                        }
                    }
                }
                TIMER_POP(1, "ACC_WAKEUP_MERGE");
            }
        }
    } // and data
    if (DoPrimaryOnGPU)
    {
#ifdef ACC_OPENACC
#pragma acc exit data delete (ToWakeupList [0:N_gas], HydraActiveParticleList [0:NHydraActivePart], LiteHIn [0:N_gas], \
                              LiteHOut [0:NHydraActivePart])
#elif defined(ACC_OPENMP)
#pragma omp target exit data map(delete                                                                                \
                                 : ToWakeupList [0:N_gas], HydraActiveParticleList [0:NHydraActivePart],               \
                                   LiteHIn [0:N_gas], LiteHOut [0:NHydraActivePart])
#endif
    }
    myfree(LiteHOut);
    myfree(HydraActiveParticleList);
    myfree(LiteHIn);
#endif // ACC_HYDRA

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(RecyclingBuffer);
    myfree(Ngblist);

    TIMER_PUSH(2, "Final_Operations"); // may be wrong?
    TIMER_PUSH(1, "HYD_COPY");
    TIMER_POPPUSH(1, "HYD_COPY", "HYD_COMPUTE_FINAL");

/* do final operations on results */
#ifdef _OPENMP
#pragma omp parallel for if (NActivePart > 500)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP

        if (IS_SPH_ALIVE(P[i]))
        {
#ifdef AR_FIX_TIME_DEP_ART_COND
            SphP[i].MaxSignalVel = HydroStoreOut[i].MaxSignalVelOut;
#endif // AR_FIX_TIME_DEP_ART_COND

            scale_dt_entropy(i, cosmo_hydra.hubble_a2);
#if defined(LT_MAX_TEMP_FEEDBACK) && defined(DEBUG)
            // As we want to imply a maximum temperature cut, we can savely assume full ionization
            // and ignore all effects of additional physics
            {
                MyAtLeastDouble u_final = SphP[i].Entropy *
                                          pow(SphP[i].Density * cosmo_hydra.a3inv, get_gamma_minus1(i)) /
                                          get_gamma_minus1(i);
                MyAtLeastDouble temp_test = U_TO_TEMP_FAC * u_final;
                if (!(temp_test > 0 && temp_test < LT_MAX_TEMP_FEEDBACK))
                {
                    VERBOSE_ALL(2,
                                "task=%d: Particle id=%llu,m=%e Detected too large temperature when computing hydro "
                                "forces %e>%e, correcting !",
                                ThisTask, static_cast<unsigned long long>(P[i].ID), static_cast<float>(P[i].Mass),
                                static_cast<double>(temp_test), static_cast<double>(LT_MAX_TEMP_FEEDBACK));
                    SphP[i].EntropyPred = SphP[i].Entropy = LT_MAX_TEMP_FEEDBACK / U_TO_TEMP_FAC * GAMMA_MINUS1 /
                                                            pow(SphP[i].Density * cosmo_hydra.a3inv, GAMMA_MINUS1);
                }
                if (!(SphP[i].DtEntropy > -1e20 && SphP[i].DtEntropy < 1e20))
                {
                    VERBOSE_ALL(2, "task=%d: Particle id=%llu,m=%e Detected strange DtEntropy (%e), correcting !",
                                ThisTask, (unsigned long long)P[i].ID, static_cast<float>(P[i].Mass),
                                static_cast<double>(SphP[i].DtEntropy));
                    SphP[i].DtEntropy = 0;
                }
                if (!(SphP[i].HydroAccel[0] > -1e20 && SphP[i].HydroAccel[0] < 1e20 && SphP[i].HydroAccel[1] > -1e20 &&
                      SphP[i].HydroAccel[1] < 1e20 && SphP[i].HydroAccel[2] > -1e20 && SphP[i].HydroAccel[2] < 1e20))
                {
                    VERBOSE_ALL(2,
                                "task=%d: Particle id=%llu,m=%e Detected strange hydro_accel (%e,%e,%e), correcting !",
                                ThisTask, static_cast<unsigned long long>(P[i].ID), static_cast<float>(P[i].Mass),
                                static_cast<double>(SphP[i].HydroAccel[0]), static_cast<double>(SphP[i].HydroAccel[1]),
                                static_cast<double>(SphP[i].HydroAccel[2]));
                    SphP[i].HydroAccel[0] = SphP[i].HydroAccel[1] = SphP[i].HydroAccel[2] = 0;
#if GADGET_HYDRO == HYDRO_MFM
                    SphP[i].dQdt.GetMomentum[k] = 0;
                    SphP[i].HydroAccelExtra[k] = 0;
#endif // GADGET_HYDRO == HYDRO_MFM
                }
            }
#endif // LT_MAX_TEMP_FEEDBACK && DEBUG

#if defined(AA_ATMOSPHERE_FORCE) && !defined(NOACCEL)
            add_aa_atmosphere_force(i);
#endif // if defined(AA_ATMOSPHERE_FORCE) && !defined(NOACCEL)

#ifdef NAVIERSTOKES
            navierstokes_viscosity_final(i, cosmo_hydra.hubble_a2, cosmo_hydra.a3inv);
#endif // NAVIERSTOKES

#ifdef WINDS
            do_wind_final(i, cosmo_hydra.fac_mu, cosmo_hydra.a3inv);
#endif // WINDS

#ifdef DIVBCLEANING_TRICCO
            tricco_update_signal_velocity(i);
#endif // DIVBCLEANING_TRICCO

#ifdef TIME_DEP_ART_COND
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0)
            eval_art_cond(i);
#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0)
#endif // TIME_DEP_ART_COND

            // #ifdef TIME_DEP_ART_VISC
            //             eval_art_visc(i, cosmo_hydra.hubble_a, cosmo_hydra.hubble_a2);
            // #endif // TIME_DEP_ART_VISC

#ifdef MAGNETIC
#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
            scale_energy_gradient(i);
#endif // if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifdef AA_OUTPUT_GRADS
            scale_pressure_gradient(i);
#endif // AA_OUTPUT_GRADS

#if defined(TIME_DEP_MAGN_DISP) && ((TIME_DEP_MAGN_DISP == 0) || (TIME_DEP_MAGN_DISP == 1))
            calculate_magn_disp(i, cosmo_hydra.fac_mu);
#endif // defined(TIME_DEP_MAGN_DISP) && ((TIME_DEP_MAGN_DISP==0) || (TIME_DEP_MAGN_DISP==1))

            // subtract maxwell tensor from lorentz force
            lorentz_force_final(i);

#ifdef DIVBCLEANING_TRICCO
            perform_tricco_cleaning(i);
#endif // DIVBCLEANING_TRICCO
#ifdef DIVBCLEANING_DEDNER
            perform_dedner_cleaning(i, cosmo_hydra.atime, cosmo_hydra.hubble_a2);
#endif // DEDNER
#endif // MAGNETIC

#if (defined(SHOCKFINDER_SPH) && !defined(READ_MACH))
            // compute Mach number, shock speed and shock compression
            shockfinder_calculate_shock_properties(i);
#endif // SHOCKFINDER_SPH


#ifdef RT_RAD_PRESSURE
            calculate_rad_pressure(i);
#endif // RT_RAD_PRESSURE

#ifdef AA_NO_DTENTROPY
            if (P[i].ID == 0)
            {
                SphP[i].DtEntropy = 0.0;
            }
#endif // AA_NO_DTENTROPY

#ifdef SPH_BND_PARTICLES
            set_bnd_particles(i);
#endif // SPH_BND_PARTICLES
#ifdef FOLLOW_ID
            if (P[i].ID == FOLLOW_ID)
            {
                printf("FOLLOW_ID %llu %s %d, Acc = [%g %g %g]\n", (unsigned long long)P[i].ID, __FILE__, __LINE__,
                       SphP[i].HydroAccel[0], SphP[i].HydroAccel[1], SphP[i].HydroAccel[2]);
            }
#endif
        }
    }

    TIMER_POP(1, "HYD_COMPUTE_FINAL");
    TIMER_POP(2, "Final_Operations");
#ifdef AR_FIX_TIME_DEP_ART_COND
    myfree(HydroStoreOut);
#endif // AR_FIX_TIME_DEP_ART_COND
}
#endif // #if !defined(ACC_INCLUDE_HYDRA) that trick

#ifdef ACC_INCLUDE_HYDRA
#warning "next warnings and errors are for the gpu version"
ACC_DECLARE_ROUTINE_BEFORE
/**
 * @brief Evaluates hydrodynamic properties locally for a target particle.
 *
 * This function specializes in evaluating hydrodynamic properties locally for a target particle, optimized for GPU
 * execution.
 *
 * @param active_target Index of the active target particle for which hydrodynamic properties are evaluated.
 * @param p_target Index of the passive target particle for which hydrodynamic properties are evaluated.
 * @param mode Evaluation mode, determining the specific actions taken by the function.
 * @param ngblist List of neighboring particles within the interaction range.
 * @param numngb Number of neighboring particles in the interaction range.
 * @param HydroDataGet Array of input hydrodynamic data for neighboring particles.
 * @param HydroDataResult Array to store the computed hydrodynamic results.
 * @param P Array of gravity walk data for particles.
 * @param SphP Array of SPH particle data.
 * @param i_node Index of the target particle within its oct-tree node.
 * @param node Index of the oct-tree node containing the target particle.
 *
 * @return Returns 0 upon successful completion of the function. Returns -1 if an error occurs during computation.
 */
int hydro_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                         struct hydrodata_in *HydroDataGet, struct hydrodata_out *HydroDataResult,
                         struct gravity_walkdata *P, struct hydrodata_in *SphP, int i_node, int node)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Evaluates hydrodynamic properties for particles in the simulation.
 *
 * This function evaluates hydrodynamic properties for particles in the simulation, including computation of SPH
 * quantities and interactions with surrounding particles.
 *
 * @param target Index of the target particle for which hydrodynamic properties are evaluated.
 * @param mode Evaluation mode, determining the specific actions taken by the function.
 * @param ngblist List of neighboring particles within the interaction range.
 * @param extended_numngb_inbox Extended number of neighbors in the interaction range.
 * @param i_startnode Starting index of the oct-tree node for the current particle.
 * @param node Index of the oct-tree node containing the target particle.
 * @param i_node Index of the target particle within its oct-tree node.
 * @param data Additional data needed for the computation.
 *
 * @return Returns 0 upon successful completion of the function. Returns -1 if an error occurs during computation.
 */
int hydro_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node,
                   int i_node, void *data)
#endif // ACC_INCLUDE_HYDRA
{
#ifdef ACC_INCLUDE_HYDRA
    int target;
    int extended_numngb_inbox = numngb;
#else // ACC_INCLUDE_HYDRA
    int numngb;
#endif // ACC_INCLUDE_HYDRA

    struct kernel_hydra kernel;

    struct hydrodata_in local;

    hydrodata_out out{};
#ifdef ACC_INCLUDE_HYDRA
    memset_gpu((unsigned char *)&out, 0, sizeof(struct hydrodata_out));
#endif // ACC_INCLUDE_HYDRA

#if ACC_INCLUDE_HYDRA
    if (mode == 2)
    {
        target = p_target;
    }
    if (mode == 1)
    {
        target = active_target;
    }
#endif // ACC_INCLUDE_HYDRA

    if (mode == 0 || mode == 2 || mode == 3)
    {
#ifdef ACC_INCLUDE_HYDRA
        memcpy_gpu(&local, &SphP[target], sizeof(struct hydrodata_in));
#else // ACC_INCLUDE_HYDRA
        particle2in_hydra(&local, target);
#endif // ACC_INCLUDE_HYDRA
    }
    else
    {
#ifdef ACC_INCLUDE_HYDRA
        memcpy_gpu(&local, &HydroDataGet[target], sizeof(struct hydrodata_in));
#else // ACC_HYDRA
        local = HydroDataGet[target];
#endif // ACC_HYDRA
    }

    struct hydrodata_additional additional_variables;
#ifdef ACC_INCLUDE_HYDRA
    set_additional_variables_and_kernel_i_openacc(&local, &out, P, SphP, cosmo_hydra.atime, &additional_variables,
                                                  &kernel);
#else // ACC_INCLUDE_HYDRA
    set_additional_variables_and_kernel_i(&local, &out, cosmo_hydra.atime, &additional_variables, &kernel);
#endif // ACC_INCLUDE_HYDRA

    /* Now start the actual SPH computation for this particle */
    int startnode = node;
    if (i_node > 0)
    {
#ifdef ACC_INCLUDE_HYDRA
        memcpy_gpu(&out, &HydroDataResult[active_target], sizeof(struct hydrodata_out));
#else // ACC_INCLUDE_HYDRA
        out = HydroDataResult[target];
#endif // ACC_INCLUDE_HYDRA
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            if (extended_numngb_inbox == -1)
            {
                return -1;
            }
            numngb = extended_numngb_inbox;
#ifndef ACC_INCLUDE_HYDRA
            startnode = *i_startnode;
            ENDRUN_IF(
                numngb > NumPart / DENSITY_LESS_NGB_FACTOR,
                "hydra: Number of neighbours found (%d) exceeds allowed number (%d) derived from "
                "NumPart/DENSITY_LESS_NGB_FACTOR, to continue, decreas DENSITY_LESS_NGB_FACTOR (currently %f, min "
                "possible val would be 1) !!\n",
                numngb, static_cast<int>(NumPart / DENSITY_LESS_NGB_FACTOR), DENSITY_LESS_NGB_FACTOR);
#else // ACC_INCLUDE_HYDRA
            startnode = -1;
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_HYDRA
            if (mode == 3)
            {
                return numngb;
            }
#endif // ACC_HYDRA
            if (numngb < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb; n++)
            {
                int j = ngblist[n];
#ifdef BLACK_HOLES
                if (P[j].Mass == 0)
                {
                    continue;
                }
#endif // BLACK_HOLES

#ifdef ACC_INCLUDE_HYDRA
                set_kernel_distances_ij_openacc(&local, j, P, &kernel);
#else // ACC_INCLUDE_HYDRA
                set_kernel_distances_ij(&local, j, &kernel);
#endif // ACC_INCLUDE_HYDRA

                if (kernel.r2 < kernel.h_i * kernel.h_i || kernel.r2 < kernel.h_j * kernel.h_j)
                {
                    kernel.r = sqrt(kernel.r2);
                    if (kernel.r > 0)
                    {
#ifdef ACC_INCLUDE_HYDRA
                        set_additional_variables_and_kernel_ij_openacc(&local, j, P, SphP, cosmo_hydra.hubble_a2,
                                                                       &additional_variables, &kernel);
#else // ACC_INCLUDE_HYDRA
                        set_additional_variables_and_kernel_ij(&local, j, cosmo_hydra.hubble_a2, &additional_variables,
                                                               &kernel);
#endif // ACC_INCLUDE_HYDRA

#if defined(SHOCKFINDER_SPH) && !defined(READ_MACH)
#ifdef ACC_INCLUDE_HYDRA
                        shockfinder_calculate_up_and_downstream_quantities_openacc(&local, &out, &kernel, j, P, SphP);
#else
                        shockfinder_calculate_up_and_downstream_quantities(&local, &out, &kernel, j);
#endif
#endif // if defined(SHOCKFINDER_SPH) && !defined(READ_MACH)


#ifdef MAGNETIC
#ifdef WINDS
#ifdef ACC_INCLUDE_HYDRA
                        do_magnetic_wind_openacc(&local, &kernel, j, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        do_magnetic_wind(&local, &kernel, j, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // WINDS

#ifdef MAGNETIC_SN_SEEDING
#ifdef ACC_INCLUDE_HYDRA
                        do_magnetic_sn_seeding_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        do_magnetic_sn_seeding(&local, &kernel, j, &additional_variables, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // MAGNETIC_SN_SEEDING

#ifdef ACC_INCLUDE_HYDRA
                        do_magnetic_induction_openacc(&local, &kernel, j, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        do_magnetic_induction(&local, &kernel, j, &out);
#endif // ACC_INCLUDE_HYDRA

#ifdef DIVBCLEANING_TRICCO
#ifdef ACC_INCLUDE_HYDRA
                        tricco_calculate_gradpsi_openacc(&local, &kernel, j, cosmo_hydra.atime, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        tricco_calculate_gradpsi(&local, &kernel, j, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // DIVBCLEANING_TRICCO

#ifdef DIVBCLEANING_DEDNER
#ifdef ACC_INCLUDE_HYDRA
                        dedner_calculate_gradphi_openacc(&local, &kernel, j, cosmo_hydra.atime, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        dedner_calculate_gradphi(&local, &kernel, j, cosmo_hydra.atime, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // DIVBCLEANING_DEDNER

#ifndef NOMAGFORCE
#ifdef ACC_INCLUDE_HYDRA
                        lorentz_force_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        lorentz_force(&local, &kernel, j, &additional_variables, &out);
#endif // ACC_INCLUDE_HYDRA

#endif // NOMAGFORCE
#endif // MAGNETIC

#ifdef ACC_INCLUDE_HYDRA
                        calculate_vsig_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        calculate_vsig(&local, &kernel, j, &additional_variables, &out);
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
                        calculate_art_visc_and_ahydro_openacc(&local, &kernel, j, &additional_variables,
                                                              cosmo_hydra.fac_mu, cosmo_hydra.fac_vsic_fix, P, SphP,
                                                              &out);
#else // ACC_INCLUDE_HYDRA
                        calculate_art_visc_and_ahydro(&local, &kernel, j, &additional_variables, cosmo_hydra.fac_mu,
                                                      cosmo_hydra.fac_vsic_fix, &out);
#endif // ACC_INCLUDE_HYDRA

#ifdef NAVIERSTOKES
#ifdef ACC_INCLUDE_HYDRA
                        navierstokes_viscosity_evaluate_openacc(&local, &kernel, j, cosmo_hydra.a3inv, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        navierstokes_viscosity_evaluate(&local, &kernel, j, cosmo_hydra.a3inv, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // NAVIERSTOKES
#ifdef ARTIFICIAL_CONDUCTIVITY
#ifdef ACC_INCLUDE_HYDRA
                        artificial_conductivity_evaluate_openacc(&local, &kernel, j, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        artificial_conductivity_evaluate(&local, &kernel, j, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // ARTIFICIAL_CONDUCTIVITY


#ifdef MAGNETIC
#ifdef MAGNETIC_DIFFUSION
#ifdef ACC_INCLUDE_HYDRA
                        calculate_magnetic_diffusion_openacc(&local, &kernel, j, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                        calculate_magnetic_diffusion(&local, &kernel, j, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // MAGNETIC_DIFFUSION
#ifdef ARTIFICIAL_MAGNETIC_DISSIPATION
#ifdef ACC_INCLUDE_HYDRA
                        calculate_magnetic_dissipation_openacc(&local, &kernel, j, &additional_variables, P, SphP,
                                                               &out);
#else // ACC_INCLUDE_HYDRA
                        calculate_magnetic_dissipation(&local, &kernel, j, &additional_variables, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // ARTIFICIAL_MAGNETIC_DISSIPATION
#endif // MAGNETIC

#ifdef GT_DUST

#ifdef ACC_INCLUDE_HYDRA
                        calculate_ahydro_dust_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
                        calculate_dust_dissipation_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
#ifdef GT_DUST_DIFFUSION
                        calculate_dust_diffusion_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
#endif // GT_DUST_DIFFUSION

#else // ACC_INCLUDE_HYDRA
                        calculate_ahydro_dust(&local, &kernel, j, &additional_variables, &out);
                        calculate_dust_dissipation(&local, &kernel, j, &additional_variables, &out);
#ifdef GT_DUST_DIFFUSION
                        calculate_dust_diffusion(&local, &kernel, j, &additional_variables, &out);
#endif // GT_DUST_DIFFUSION
#endif // ACC_INCLUDE_HYDRA

#endif // GT_DUST

#if (WAKEUP > 0)

#ifndef GM_MV_MUPPI_NOWU
#ifdef WINDS
                        if (kernel.vsig > WAKEUP * SphP[j].MaxSignalVel && SphP[j].DelayTime <= 0 &&
                            local.DelayTime <= 0)
#else // WINDS
                        if (kernel.vsig > WAKEUP * SphP[j].MaxSignalVel)
#endif // WINDS
#else // GM_MV_MUPPI_NOWU
                        /* GM: NO WAKEUP FROM/TO MULTIPHASE PARTICLES
                           note: I turn WU off for not-conducing particles */
                        if ((kernel.vsig > WAKEUP * SphP[j].MaxSignalVel) && (SphP[j].MultiPhase == 0) &&
                            (local.MultiPhase == 0) && (local.Noartcond == 0) && (SphP[j].NoArtCond == 0))
#endif // GM_MV_MUPPI_NOWU
                        {
                            if (SPHP(j, wakeup) > local.TimeBin + 1 && P[j].TimeBin > local.TimeBin + 1)
                            {
#if defined(_OPENMP) && !defined(ACC_INCLUDE_HYDRA)
#pragma omp critical(__wakeup__)
#elif defined(ACC_NOACC)
#pragma omp critical(__wakeup__)
#endif // if defined(_OPENMP) &&  !defined(ACC_INCLUDE_HYDRA)
                                {
                                    // Nitya adding checks for wakeup variable
#if !defined(AR_SOA_LITE) || defined(AR_SOA_LITE_FROM_RESTART_AOS)
                                    if (SPHP(j, wakeup) > local.TimeBin + 1 && P[j].TimeBin > local.TimeBin + 1)
                                    {
                                        SPHP(j, wakeup) = local.TimeBin;
#ifdef ACC_INCLUDE_HYDRA
                                        int n_ToWakeUp;
#pragma acc atomic capture
                                        n_ToWakeUp = N_ToWakeUp++;
                                        ToWakeupList[n_ToWakeUp] = j;
#else // ACC_INCLUDE_HYDRA
                                        ToWakeupList[N_ToWakeUp++] = j;
#endif // ACC_INCLUDE_HYDRA
                                    }
#endif // AR_SOA_LITE
                                }
                            }
                        }
#endif // WAKEUP > 0

#if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifdef ACC_INCLUDE_HYDRA
                        calculate_energy_and_pressure_gradients_openacc(&local, &kernel, j, P, SphP, cosmo_hydra.a3inv,
                                                                        &out);
#else // ACC_INCLUDE_HYDRA
                        calculate_energy_and_pressure_gradients(&local, &kernel, j, cosmo_hydra.a3inv, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // if defined(CONDUCTION_INCLUDEMAGNETIC) || defined(AA_OUTPUT_GRADS)
                    }
                    else // r == 0
                    {

                    }
                }
            }
        }
    }

    if (mode == 3)
    {
        return 0;
    }

#ifdef ACC_INCLUDE_HYDRA
    target = active_target;
#endif // ACC_INCLUDE_HYDRA
    /* Now collect the result at the right place */
    if (mode == 0 || mode == 2)
    {
#ifdef ACC_INCLUDE_HYDRA
        HydroDataResult[target] = out;
#else // ACC_INCLUDE_HYDRA
        out2particle_hydra(&out, target, 0);
#endif // ACC_INCLUDE_HYDRA
    }
    else
    {
        HydroDataResult[target] = out;
    }
    return 0;
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_HYDRA
#if !defined(ACC_INCLUDE_HYDRA) && defined(ACC)
#define ACC_INCLUDE_HYDRA 1
#include "hydra.cpp"
#undef ACC_INCLUDE_HYDRA
#endif // if !defined(ACC_INCLUDE_HYDRA) && defined(ACC)
#endif // ACC_HYDRA
