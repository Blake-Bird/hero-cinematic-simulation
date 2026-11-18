/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Perform MFM flux calculation.
 *
 * This file contains the core of the MFM flux calculation.
 * Compute Godunov fluxes for all active particles on this step and convert them
 * to changes in hydrodynamical quantities.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/assert.hpp"
#include "../System/communication.hpp"
#include "../Hydro/kernel.hpp"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.hpp"
#include "mfm.hpp"

#ifndef ACC_INCLUDE_HYDRA
#include "../Hydro/hydra_structs.hpp"
#ifdef AR_FIX_TIME_DEP_ART_COND
struct hydro_store_out *HydroStoreOut;
#endif // AR_FIX_TIME_DEP_ART_COND
struct fluxdata_in *FluxDataIn, *FluxDataGet;
struct fluxdata_out *FluxDataResult, *FluxDataOut;
#endif // ACC_INCLUDE_HYDRA
#include "mfm_fluxes.hpp"
#include "../Hydro/hydra_extra.hpp"

#ifdef SHOCKFINDER_SPH
/* Beck+16 SPH-style finder. Never active under HYDRO_MFM, where SHOCKFINDER
 * selects the ray-traced Mfm/mfm_shockfinder.cpp (SHOCKFINDER_MFM) instead. */
#include "../Hydro/shockfinder.hpp"
#endif // SHOCKFINDER_SPH

#include "../CodeBase/greentree.hpp"

#ifndef ACC_INCLUDE_HYDRA

#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
static MyAtLeastDouble hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef ACC_HYDRA
#pragma declare create(hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv)
#endif // ACC_HYDRA

#ifdef ACC_HYDRA
static inline void particle2in_hydra(struct hydrodata_in *in, int i);
#else // ACC_HYDRA
static inline void particle2in_hydra(struct hydrodata_in *in, int i);
#endif // ACC_HYDRA
static inline void out2particle_hydra(struct hydrodata_out *out, int i, int mode, int mfm_mode);

#include "../Hydro/hydra.hpp"

/*! Create reduced particle data structure for communicating to other nodes for gradient computation */
#define particle2in_flux particle2in_hydra

/*! Create data structure with partial sums/results to send back to particle host node */

#if !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA) // we reinclude this same file if ACC is active
#include "../OpenACC/hydra.hpp"
#endif // !(defined ACC_INCLUDE_HYDRA) && defined(ACC_HYDRA)

/**
 * @brief Compute MFM Fluxes.
 *
 * This function is the driver routine for the calculation of hydrodynamical
 * force and rate of change of internal energy / entropy for all active
 * particles if the MFM solver is used.
 */
void mfm_fluxes(void)
{
    TIMER_PUSH(2, "Initializing");

#ifdef AR_FIX_TIME_DEP_ART_COND
    HydroStoreOut = (hydro_store_out *)mymalloc("HydroStoreOut", N_gas * sizeof(hydro_store_out));
#endif // AR_FIX_TIME_DEP_ART_COND

    VERBOSE(6, "ThisTask%d mfm_fluxes.", ThisTask);
#ifdef ACC_FIX_ASSIGN
    if (DoPrimaryOnGPU)
#endif
#ifdef AR_FIX_ASSIGN
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
            if (IS_SPH_ALIVE(P[i]))
            {
                P[i].assigned = 0;
            }
        }
#endif // AR_FIX_ASSIGN

    if (All.ComovingIntegrationOn)
    {
        /* Factors for comoving integration of hydro */
        hubble_a = hubble_function(All.Time);
        hubble_a2 = All.Time * All.Time * hubble_a;

        fac_mu = pow(All.Time, 3 * (GAMMA - 1) / 2) / All.Time;
        fac_vsic_fix = hubble_a * pow(All.Time, 3 * GAMMA_MINUS1);

        a3inv = 1 / (All.Time * All.Time * All.Time);
        atime = All.Time;
    }
    else
    {
        hubble_a = hubble_a2 = atime = fac_mu = fac_vsic_fix = a3inv = 1.0;
    }

    int NTaskTimesNumPart = (static_cast<int>(maxThreads)) * (static_cast<int>(NumPart / DENSITY_LESS_NGB_FACTOR));
    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer;", NTaskTimesNumPart * sizeof(int));

    TIMER_PUSH(1, "MFMFLUX_ZEROING");
    TIMER_POP(1, "MFMFLUX_ZEROING");

    /* allocate buffers to arrange communication */
    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize = static_cast<int>((MyBufferSize * 1024 * 1024) /
                                     (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                      sizeof(struct fluxdata_in) + sizeof(struct fluxdata_out) +
                                      sizemax(sizeof(struct fluxdata_in), sizeof(struct fluxdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(2, "MFM fluxes: using %g MB for buffering, %g MB remaining for other buffers", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    // begin with this index
    NextParticle = FirstActiveParticle;

    int ndone, group_particles = 1;

    TIMER_POP(2, "Initializing");

#ifdef ACC_HYDRA
    struct fluxdata_in *LiteFIn;
    struct fluxdata_out *LiteFOut;

    int *FluxActiveParticleList;
    int NFluxActivePart;
    int should_drift = 0;
    int DoPrimaryOnGPU = 0;
    TIMER_PUSH(1, "ACC_INIT");
    NFluxActivePart = openacc_hydra_init(should_drift, DoPrimaryOnGPU, FluxActiveParticleList, LiteFOut, LiteFIn);
    // note: LiteFIn is in/out property because we need the wakeup properties
    TIMER_POPPUSH(1, "ACC_INIT", "ACC_UPLOAD_AND_I");
#pragma acc enter data create(LiteFIn [0:N_gas], FluxActiveParticleList [0:NFluxActivePart],                           \
                              LiteFOut [0:NFluxActivePart], ToWakeupList [0:N_gas], ) if (DoPrimaryOnGPU)
    /*#pragma acc data copyin(FluxActiveParticleList[0 : NFluxActivePart]) \
      copy(LiteFIn[0 : NumPart], LiteFOut[0 : NFluxActivePart]) COPY_TOWAKEUP_LIST if (DoPrimaryOnGPU)*/
    {
#pragma acc update device(LiteFIn [0:N_gas], FluxActiveParticleList [0:NFluxActivePart], WalkLiteP [0:N_gas],          \
                          N_ToWakeUp, hubble_a, atime, hubble_a2, fac_mu, fac_vsic_fix, a3inv)                         \
    ACC_ASYNC_COPY1 if (DoPrimaryOnGPU)
        openacc_hydra_primary_phase(DoPrimaryOnGPU, NFluxActivePart, FluxActiveParticleList, LiteFOut, LiteFIn);

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

            TIMER_PUSH(1, "MFMFLUX_COMPUTE");

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            ar_green_evaluate_primary(flux_isactive, flux_evaluate, Module_MfmFluxes, TreeWalkType_GreenTreeWalk,
                                      NextParticleLoop_ActiveParticleList, NeighbourSearchCriteria_Density, &ThisThread,
                                      DoPrimaryOnGPU, should_drift, group_particles, ProcessedFlag_mfm_fluxes);

            TIMER_POP(1, "MFMFLUX_COMPUTE");
            TIMER_POPPUSH(2, "Primary", "Exchange");
            TIMER_PUSH(1, "MFMFLUX_PREPARE");

            /* if we are not grouping particles and SendRecv fails, then we panic as there is no memory to process even
             * a single particles */
            int panic_on_error = !group_particles;
            int is_send_receive_ok =
                prepare_sendrec_offset_send(save_NextParticle, panic_on_error); // 1 means panic on bufer too small

            TIMER_POP(1, "MFMFLUX_PREPARE");

            if (!is_send_receive_ok)
            {
                TIMER_POPPUSH(2, "Exchange", "Primary");
                group_particles = 0;
                continue;
            }

            TIMER_PUSH(1, "MFMFLUX_COPY_IN");

            FluxDataGet = (struct fluxdata_in *)mymalloc("FluxDataGet", Nimport * sizeof(struct fluxdata_in));
            DataNodeListGet =
                (struct data_nodelist *)mymalloc("DataNodeListGet", Nimport * sizeof(struct data_nodelist));
            FluxDataIn = (struct fluxdata_in *)mymalloc("FluxDataIn", Nexport * sizeof(struct fluxdata_in));
            DataNodeListIn = (struct data_nodelist *)mymalloc("DataNodeListIn", Nexport * sizeof(struct data_nodelist));
            fluxdata_in FluxDataIn_zero{};
            for (int j = 0; j < Nexport; ++j)
            {
                memcpy(&FluxDataIn[j], &FluxDataIn_zero, sizeof(fluxdata_in));
            }

            /* prepare particle data for export */
            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                particle2in_flux(&FluxDataIn[j], place);
            }

            copy_data_nodelist();

            // exchange particle data
            TIMER_POPPUSH(1, "MFMFLUX_COPY_IN", "MFMFLUX_COMM_EXC");
            templated_exchange_data_in_get(FluxDataIn, FluxDataGet, 1, TAG_FLUX_A);
            TIMER_POP(1, "MFMFLUX_COMM_EXC");

            TIMER_POPPUSH(2, "Exchange", "Secondary");

            TIMER_PUSH(1, "MFMFLUX_ZERO_AND_COPY");
            myfree(DataNodeListIn);
            myfree(FluxDataIn);
            FluxDataResult = (struct fluxdata_out *)mymalloc("FluxDataResult", Nimport * sizeof(struct fluxdata_out));
            FluxDataOut = (struct fluxdata_out *)mymalloc("FluxDataOut", Nexport * sizeof(struct fluxdata_out));
            fluxdata_out FluxDataOut_zero{};
            for (int j = 0; j < Nimport; ++j)
            {
                memcpy(&FluxDataResult[j], &FluxDataOut_zero, sizeof(fluxdata_out));
            }

            report_memory_usage(&HighMark_mfmflux, "MFM_FLUXES");
            TIMER_POPPUSH(1, "MFMFLUX_ZERO_AND_COPY", "MFMFLUX_COMPUTE");

            // now do the particles that were sent to us
            NextJ = 0;

#ifdef ACC_HYDRA
            openacc_hydra_secondary_phase(FluxDataGet, FluxDataResult, LiteFIn, DoPrimaryOnGPU, should_drift);
#else // ACC_HYDRA
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        ar_green_evaluate_secondary(FluxDataGet, flux_evaluate, Module_MfmFluxes, TreeWalkType_GreenTreeWalk,
                                    NeighbourSearchCriteria_Density, &ThisThread, should_drift);
#endif // ACC_HYDRA

            TIMER_POPPUSH(1, "MFMFLUX_COMPUTE", "MFMFLUX_WAIT");

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
            TIMER_POP(1, "MFMFLUX_WAIT");
            TIMER_POPPUSH(2, "Secondary", "Send_Results");
            TIMER_PUSH(1, "MFMFLUX_COMM_EXC");
            templated_exchange_data_Sendrecv(FluxDataResult, FluxDataOut, TAG_FLUX_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POPPUSH(1, "MFMFLUX_COMM_EXC", "MFMFLUX_COPY");

            // add the result to the local particles
            for (int j = 0; j < Nexport; j++)
            {
                int place = DataIndexTable[j].Index;
                out2particle_hydra(&FluxDataOut[j], place, 1, 1);
            }
            TIMER_POP(1, "MFMFLUX_COPY");
            TIMER_POP(2, "Send_Results");

            myfree(FluxDataOut);
            myfree(FluxDataResult);
            myfree(DataNodeListGet);
            myfree(FluxDataGet);
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
#pragma acc wait(1)
            }
            TIMER_POPPUSH(1, "ACC_WAIT", "ACC_DOWNLOAD");
#pragma acc update host(LiteFOut [0:NFluxActivePart], N_ToWakeUp) if (DoPrimaryOnGPU)
            if (N_ToWakeUp > 0)
#pragma acc update host(ToWakeupList [0:N_ToWakeUp]) if (DoPrimaryOnGPU)
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
#pragma acc update host(LiteFIn [0:N_gas]) if (DoPrimaryOnGPU)
            }
            TIMER_POPPUSH(1, "ACC_WAKEUP_DOWNLOAD", "ACC_WAKEUP_MINUS1");

            if (N_ToWakeUp > 0 && n_ToWakeUp_cpu > 0)
            {
#pragma omp parallel for
                for (int i = 0; i < n_ToWakeUp_cpu; i++)
                {
                    int j = to_wakeup_list_cpu[i];
                    // if wakeup is != TIMEBINS then the particle has already been woken up by GPU
                    if (LiteFIn[j].wakeup != TIMEBINS)
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
            for (int counter = 0; counter < NFluxActivePart; counter++)
            {
                int i = FluxActiveParticleList[counter];
                out2particle_hydra(&LiteFOut[counter], i, 1);
            }

            TIMER_POPPUSH(1, "ACC_OUT", "ACC_WAKEUP_MERGE");
            if (WAKEUP > 0 && N_ToWakeUp > 0 && n_ToWakeUp_cpu > 0)
            {
                // LiteFIn is actually an in-out property
                // just becaouse of the wake up stuff
#pragma omp parallel for
                for (int i = 0; i < N_gas; i++)
                {
                    if (P[i].Type == type_gas)
                    {
                        // particle got wakeup in GPU?

                        // Nitya adding checks around variable
#if !defined(AR_SOA_LITE) || defined(AR_SOA_LITE_FROM_RESTART_AOS)
#warning "Nitya in hydra"

                        if (LiteFIn[i].wakeup != TIMEBINS)
                        {
                            // is the GPU wakeup smaller timebin than CPU?
                            if (SPHP(i, wakeup) == TIMEBINS || SPHP(i, wakeup) > LiteFIn[i].wakeup)
                            {

                                SPHP(i, wakeup) = LiteFIn[i].wakeup;
                            }
                        }
#endif // !defined(AR_SOA_LITE) || defined(AR_SOA_LITE_FROM_RESTART_AOS)
                    }
                }
            }
            TIMER_POP(1, "ACC_WAKEUP_MERGE");
        }
    } // and data
#pragma acc exit data delete (ToWakeupList [0:N_gas], FluxActiveParticleList [0:NFluxActivePart], LiteFIn [0:N_gas],   \
                              LiteFOut [0:NFluxActivePart]) if (DoPrimaryOnGPU)
    myfree(LiteFOut);
    myfree(FluxActiveParticleList);
    myfree(LiteFIn);

#endif // ACC_HYDRA

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(RecyclingBuffer);
    myfree(Ngblist);

    TIMER_PUSH(2, "Final_Operations");
    TIMER_PUSH(1, "MFMFLUX_FINAL");

    // do final operations on results
#ifdef _OPENMP
#pragma omp parallel for
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        if (flux_isactive(i))
        {
            // same functions as in hydra.c

#ifdef AR_FIX_TIME_DEP_ART_COND
            SphP[i].MaxSignalVel = HydroStoreOut[i].MaxSignalVelOut;
#endif // AR_FIX_TIME_DEP_ART_COND

            scale_dt_entropy(i, hubble_a2);

#if defined(AA_ATMOSPHERE_FORCE) && !defined(NOACCEL)
            add_aa_atmosphere_force(i);
#endif // if defined(AA_ATMOSPHERE_FORCE) && !defined(NOACCEL)

#ifdef NAVIERSTOKES
            navierstokes_viscosity_final(i, hubble_a2, a3inv);
#endif // NAVIERSTOKES

#if defined(JD_DPP) && !defined(JD_DPPONSNAPSHOTONLY)
            compute_Dpp(i);
#endif // if defined(JD_DPP) && !defined(JD_DPPONSNAPSHOTONLY)

#ifdef TIME_DEP_ART_COND
#if !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0)
            eval_art_cond(i);
#endif // !defined(NOGRAVITY) && (TIME_DEP_ART_COND > 0)
#endif // TIME_DEP_ART_COND

#ifdef TIME_DEP_ART_VISC
#if TIME_DEP_ART_VISC == 0
            eval_art_visc(i, fac_mu, hubble_a);
#endif // TIME_DEP_ART_VISC == 0
#endif // TIME_DEP_ART_VISC


#if (defined(SHOCKFINDER_SPH) && !defined(READ_MACH))
            // Beck+16 finalisation; never active under HYDRO_MFM. The MFM
            // ray-traced finder runs from run.cpp between compute_limiters()
            // and compute_fluxes() instead.
            shockfinder_calculate_shock_properties(i);
#endif // SHOCKFINDER_SPH

#ifdef RT_RAD_PRESSURE
            calculate_rad_pressure(i);
#endif // RT_RAD_PRESSURE

            // now comes the MFM specific part
            double dt_entr_phys;
            integertime ti_step, tstart_step, tend_step;
            ti_step = P[i].TimeBin ? (((integertime)1) << P[i].TimeBin) : 0;
            tstart_step = P[i].Ti_begstep; /* start of step */
            tend_step = P[i].Ti_begstep + ti_step; /* end of step */

            dt_entr_phys = units->timestep_to_physical * (tend_step - tstart_step) * All.Timebase_interval;

            convert_total_to_internal_energy_flux(i, dt_entr_phys, units);
#if defined(LMB_MFM_ENERGY_CLOSURE_LIMITER) && !defined(MFM_CONSERVATIVE_TOTAL_ENERGY)
            // dQdt.GetEnergy() is the internal-energy flux from here on, so scaling it carries the
            // face-closure limit consistently into DtInternalEnergy, DtEntropy and the CR
            // injection cap below, all of which are derived from it. Under
            // MFM_CONSERVATIVE_TOTAL_ENERGY the flux keeps its kinetic part, so the limit is
            // applied to the internal-energy increment in the drift and kick instead.
            SphP[i].dQdt.GetEnergy() *= get_energy_facearea_limiter(i, dt_entr_phys);
#endif // LMB_MFM_ENERGY_CLOSURE_LIMITER && !MFM_CONSERVATIVE_TOTAL_ENERGY
            mfm_fluxes_to_code(i, tstart_step, tend_step);


#ifdef SPH_BND_PARTICLES
            set_bnd_particles(i);
#endif // SPH_BND_PARTICLES

#ifdef WINDS
            do_wind_final(i, fac_mu, a3inv);
#endif // WINDS
        } // if (flux_isactive(i))
    } // for
    TIMER_POP(1, "MFMFLUX_FINAL");
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
 * @brief Evaluates MFM fluxes locally for a target particle.
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
int flux_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                        struct hydrodata_in *HydroDataGet, struct hydrodata_out *HydroDataResult,
                        struct gravity_walkdata *P, struct hydrodata_in *SphP, int i_node, int node)
#else // ACC_INCLUDE_HYDRA
/**
 * @brief Evaluates MFM fluxes for particles in the simulation.
 *
 * This function evaluates MFM fluxes for particles in the simulation, including computation of MFM quantities and
 * interactions with surrounding particles.
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
int flux_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode, int node, int i_node,
                  void *data)
#endif // ACC_INCLUDE_HYDRA
{
#ifdef ACC_INCLUDE_HYDRA
    int target;
    int extended_numngb_inbox = numngb;
#else // ACC_INCLUDE_HYDRA
    int numngb;
#endif // ACC_INCLUDE_HYDRA

    kernel_flux kernel;
    fluxdata_in local{};
    fluxdata_out out{};
#ifdef ACC_INCLUDE_HYDRA
    memset_gpu((unsigned char *)&out, 0, sizeof(struct fluxdata_out));
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
        particle2in_flux(&local, target);
#endif // ACC_INCLUDE_HYDRA
    }
    else
    {
#ifdef ACC_INCLUDE_HYDRA
        memcpy_gpu(&local, &FluxDataGet[target], sizeof(struct fluxdata_in));
#else // ACC_INCLUDE_HYDRA
        local = FluxDataGet[target];
#endif // ACC_INCLUDE_HYDRA
    }

    double dt_entr_phys;
    integertime ti_step, tstart_step, tend_step;
    ti_step = local.TimeBin ? (((integertime)1) << local.TimeBin) : 0;
    tstart_step = local.Ti_begstep; /* start of step */
    tend_step = local.Ti_begstep + ti_step; /* end of step */

    if (All.ComovingIntegrationOn)
    {
        dt_entr_phys = units->timestep_to_physical * (tend_step - tstart_step) *
                       All.Timebase_interval; // used for fluxes, that are all in physical units.
    }
    else
    {
        dt_entr_phys = (tend_step - tstart_step) * All.Timebase_interval;
    }

    struct hydrodata_additional additional_variables;
#ifdef ACC_INCLUDE_HYDRA
    set_additional_variables_and_kernel_i_openacc(&local, &out, P, SphP, atime, &additional_variables, &kernel);
#else // ACC_INCLUDE_HYDRA
    set_additional_variables_and_kernel_i(&local, &out, atime, &additional_variables, &kernel);
#endif // ACC_INCLUDE_HYDRA

    /* Now start the actual MFM computation for this particle */
    int startnode = node;
    if (i_node > 0)
    {
#ifdef ACC_INCLUDE_HYDRA
        memcpy_gpu(&out, &FluxDataResult[active_target], sizeof(struct fluxdata_out));
#else // ACC_INCLUDE_HYDRA
        out = FluxDataResult[target];
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
                numngb, (int)(NumPart / DENSITY_LESS_NGB_FACTOR), DENSITY_LESS_NGB_FACTOR);
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
#ifdef FG_REMOVE_BND
                if (P[j].ID == 0)
                {
                    continue;
                }
#endif // FG_REMOVE_BND

#ifdef ACC_INCLUDE_HYDRA
                set_kernel_distances_ij_openacc(&local, j, P, &kernel);
#else // ACC_INCLUDE_HYDRA
                set_kernel_distances_ij(&local, j, &kernel);
#endif // ACC_INCLUDE_HYDRA

                // Try to avoid self-interaction terms
                if (kernel.r2 < 1.0e-8 * kernel.h_i * kernel.h_i || kernel.r2 < 1.0e-8 * kernel.h_j * kernel.h_j)
                {
                    continue;
                }

                if (kernel.r2 < kernel.h_i * kernel.h_i || kernel.r2 < kernel.h_j * kernel.h_j)
                {
                    kernel.r = sqrt(kernel.r2);
                    if (kernel.r <= 0)
                    {
                        continue;
                    }

#ifdef ACC_INCLUDE_HYDRA
                    set_additional_variables_and_kernel_ij_openacc(&local, j, P, SphP, hubble_a2, &additional_variables,
                                                                   &kernel);
#else // ACC_INCLUDE_HYDRA
                    set_additional_variables_and_kernel_ij(&local, j, hubble_a2, &additional_variables, &kernel);
#endif // ACC_INCLUDE_HYDRA

                    // same functions as in hydra.c (Beck+16 only; never active under HYDRO_MFM)
#if defined(SHOCKFINDER_SPH) && !defined(READ_MACH)
#ifdef ACC_INCLUDE_HYDRA
                    shockfinder_calculate_up_and_downstream_quantities_openacc(&local, &out, &kernel, j, P, SphP);
#else // ACC_INCLUDE_HYDRA
                    shockfinder_calculate_up_and_downstream_quantities(&local, &out, &kernel, j);
#endif // ACC_INCLUDE_HYDRA
#endif // if defined(SHOCKFINDER_SPH) && !defined(READ_MACH)



#ifdef ACC_INCLUDE_HYDRA
                    calculate_vsig_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                    calculate_vsig(&local, &kernel, j, &additional_variables, &out);
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_INCLUDE_HYDRA
                    calculate_art_visc_and_ahydro_openacc(&local, &kernel, j, &additional_variables, fac_mu,
                                                          fac_vsic_fix, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                    calculate_art_visc_and_ahydro(&local, &kernel, j, &additional_variables, fac_mu, fac_vsic_fix,
                                                  &out);
#endif // ACC_INCLUDE_HYDRA

#ifdef NAVIERSTOKES
#ifdef ACC_INCLUDE_HYDRA
                    navierstokes_viscosity_evaluate_openacc(&local, &kernel, j, a3inv, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                    navierstokes_viscosity_evaluate(&local, &kernel, j, a3inv, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // NAVIERSTOKES

#ifdef ARTIFICIAL_CONDUCTIVITY
#ifdef ACC_INCLUDE_HYDRA
                    artificial_conductivity_evaluate_openacc(&local, &kernel, j, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                    artificial_conductivity_evaluate(&local, &kernel, j, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // ARTIFICIAL_CONDUCTIVITY



#ifdef GT_DUST_DIFFUSION
#ifdef ACC_INCLUDE_HYDRA
                    calculate_dust_diffusion_openacc(&local, &kernel, j, &additional_variables, P, SphP, &out);
#else // ACC_INCLUDE_HYDRA
                    calculate_dust_diffusion(&local, &kernel, j, &additional_variables, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // GT_DUST_DIFFUSION

                    if (WAKEUP > 0)
                    {
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
#if defined(_OPENMP) && !defined(ACC_INCLUDE_HYDRA)
#pragma omp critical(__wakeup__)
#elif defined(ACC_NOACC)
#pragma omp critical(__wakeup__)
#endif // defined(_OPENMP) && !defined(ACC_INCLUDE_HYDRA)
                            {
                                if (SPHP(j, wakeup) > local.TimeBin + 1 && P[j].TimeBin > local.TimeBin + 1)
                                {
                                    SPHP(j, wakeup) = local.TimeBin;
#ifdef ACC_INCLUDE_HYDRA
                                    int n_ToWakeUp;
#pragma acc atomic capture
                                    n_ToWakeUp = N_ToWakeUp++;
                                    ToWakeupList[n_ToWakeUp++] = j;
#else // ACC_INCLUDE_HYDRA
                                    ToWakeupList[N_ToWakeUp++] = j;
#endif // ACC_INCLUDE_HYDRA
                                }
                            }
                        }
                    } // if (WAKEUP > 0)

#if defined(CONDUCTION_INCLUDE_MAGNETIC) || defined(AA_OUTPUT_GRADS)
#ifdef ACC_INCLUDE_HYDRA
                    calculate_energy_and_pressure_gradients_openacc(&local, &kernel, j, P, SphP, a3inv, &out);
#else // ACC_INCLUDE_HYDRA
                    calculate_energy_and_pressure_gradients(&local, &kernel, j, a3inv, &out);
#endif // ACC_INCLUDE_HYDRA
#endif // if defined(CONDUCTION_INCLUDE_MAGNETIC) || defined(AA_OUTPUT_GRADS)

                    // now comes the MFM specific part of the flux calculation
#ifndef DONT_BIN_MFM_FLUXES // this should not be active on gpu
                    // Skip neighbours on shorter steps
                    if (P[j].TimeBin < local.TimeBin)
                    {
                        continue;
                    }
                    if ((P[j].TimeBin == local.TimeBin) && (P[j].ID <= local.ID))
                    {
                        continue; // break degeneracy using the ID
                    }
#endif // DONT_BIN_MFM_FLUXES

                    // avoid self-interaction
                    // if (local.ID == P[j].ID) continue;

#ifdef WINDS
                    if (SphP[j].DelayTime > 0) /* decouple wind particles */
                    {
                        continue;
                    }
#endif // WINDS

                    // Skip interaction if both kernel terms are zero
                    // if (kernel.wk_i < 1.0e-10 && kernel.wk_j < 1.0e-10) continue;
#ifdef ACC_INCLUDE_HYDRA
                    calculate_face_openacc(local, kernel, j, additional_variables, P, SphP);
#else // ACC_INCLUDE_HYDRA
                    calculate_face(local, kernel, j, additional_variables);
#endif // ACC_INCLUDE_HYDRA

                    // Skip interaction if both kernel terms are zero
                    if (additional_variables.Amag == 0)
                    {
                        continue;
                    }

#ifdef ACC_INCLUDE_HYDRA
                    set_additional_variables_ij_mfm_openacc(local, j, additional_variables, SphP);
#else // ACC_INCLUDE_HYDRA
                    set_additional_variables_ij_mfm(local, j, additional_variables);
#endif // ACC_INCLUDE_HYDRA
       // Compute primitive fluid vectors for the left-hand state of the Riemann problem
                    slopeLimiter->limit_slopes(local.gradW, local.AlphaSlope, additional_variables.drFacei,
                                               additional_variables.Wi, additional_variables.gradWi);
                    // Compute primitive fluid vectors for the right-hand state of the Riemann problem
                    slopeLimiter->limit_slopes(SphP[j].gradW, SphP[j].AlphaSlope, additional_variables.drFacej,
                                               additional_variables.Wj, additional_variables.gradWj);
                    pairwiseSlopeLimiter->limit_slopes_pairwise(local.Wprim, SphP[j].Wprim, additional_variables.Wi,
                                                                additional_variables.Wj, additional_variables.vface);

#ifdef GT_DUST

                    for (int idust = 0; idust < GT_NUM_DUST; idust++)
                    {

                        slopeLimiter->limit_slopes(local.gradW_dust[idust].data(), local.AlphaSlope_dust[idust],
                                                   additional_variables.drFacei, additional_variables.Wi_dust[idust],
                                                   additional_variables.gradWi_dust[idust].data());

                        slopeLimiter->limit_slopes(SphP[j].gradW_dust[idust].data(), SphP[j].AlphaSlope_dust[idust],
                                                   additional_variables.drFacej, additional_variables.Wj_dust[idust],
                                                   additional_variables.gradWj_dust[idust].data());

                        pairwiseSlopeLimiter->limit_slopes_pairwise(
                            local.Wprim_dust[idust], SphP[j].Wprim_dust[idust], additional_variables.Wi_dust[idust],
                            additional_variables.Wj_dust[idust], additional_variables.vface);
                    }

#endif // GT_DUST
#ifndef NOMUSCL
#ifdef ACC_INCLUDE_HYDRA
                    do_muscl_halfstep_interpolation_openacc(local, kernel, j, additional_variables, P, SphP);
#else // ACC_INCLUDE_HYDRA
                    do_muscl_halfstep_interpolation(local, kernel, j, additional_variables);
#endif // ACC_INCLUDE_HYDRA
#endif // NOMUSCL
       //  check that density and pressure are positive
                    additional_variables.Wi.enforce_positivity();
                    additional_variables.Wj.enforce_positivity();

#ifdef ACC_INCLUDE_HYDRA
                    units_to_physical_openacc(additional_variables, units);
#else // ACC_INCLUDE_HYDRA
                    units_to_physical(additional_variables, units);
#endif // ACC_INCLUDE_HYDRA

                    // Calculate Godunov flux using the selected Riemann solver
#ifdef ACC_INCLUDE_HYDRA
                    evaluate_godunov_fluxes_openacc(local, j, additional_variables, dt_entr_phys, SphP, out);
#else // ACC_INCLUDE_HYDRA
                    evaluate_godunov_fluxes(local, j, additional_variables, dt_entr_phys, out);
#endif // ACC_INCLUDE_HYDRA
                } // if(kernel.r2 < kernel.h_i * kernel.h_i || kernel.r2 < kernel.h_j * kernel.h_j)

            } // for(int n = 0; n < numngb_inbox; n++)
        } // while(startnode >= 0)
    } // while(startnode >= 0)

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
#if defined(_OPENMP) && !defined(DONT_BIN_MFM_FLUXES)
        omp_set_lock(&MfmNgbLockList[target]);
#endif // defined(_OPENMP) && !defined(DONT_BIN_MFM_FLUXES)
#ifdef ACC_INCLUDE_HYDRA
        FluxDataResult[target] = out;
#else // ACC_INCLUDE_HYDRA
        out2particle_hydra(&out, target, 0, 1);
#endif // ACC_INCLUDE_HYDRA
#if defined(_OPENMP) && !defined(DONT_BIN_MFM_FLUXES)
        omp_unset_lock(&MfmNgbLockList[target]);
#endif // defined(_OPENMP) && !defined(DONT_BIN_MFM_FLUXES)
    }
    else
    {
        FluxDataResult[target] = out;
    }
    return 0;
}
#ifdef ACC_INCLUDE_HYDRA
ACC_DECLARE_ROUTINE_AFTER
#endif // ACC_INCLUDE_HYDRA

#ifdef ACC_HYDRA
#if !defined(ACC_INCLUDE_HYDRA) && defined(ACC)
#define ACC_INCLUDE_HYDRA 1
#include "mfm_fluxes.cpp"
#undef ACC_INCLUDE_HYDRA
#endif // !defined(ACC_INCLUDE_HYDRA) && defined(ACC)
#endif // ACC_HYDRA
