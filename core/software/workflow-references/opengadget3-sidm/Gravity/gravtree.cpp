/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Gravity tree.
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
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../System/communication.hpp"

double Ewaldcount, Costtotal;
long long N_nodesinlist;
int Ewald_iter; /* global in file scope, for simplicity */

#if !(defined ACC_INCLUDE_GRAVITY) && defined(ACC) // we reinclude this same file if ACC is active
// note, gravtree also buils the tree and drift so we need some Acc data even if ACC_GRAVITY is off
#include "../OpenACC/gpuallvars_branch.hpp"
#endif // !(defined ACC_INCLUDE_GRAVITY) && defined(ACC_GRAVITY)

#if !(defined ACC_INCLUDE_GRAVITY) && defined(ACC_GRAVITY) // we reinclude this same file if ACC is active
#include "../OpenACC/gravtree.hpp"
#endif // !(defined ACC_INCLUDE_GRAVITY) && defined(ACC_GRAVITY)

#ifdef STATICBRANDT
/**
 * @brief Compute the value of Omega for a given particle position and mode.
 *
 * This function computes the value of Omega for a given particle position and mode.
 * Omega represents a quantity that depends on the distance from a central point
 * (LONG_X/2, LONG_Y/2) and varies based on the specified mode.
 *
 * @param i    Index of the particle for which to compute Omega.
 * @param mode Mode indicating the behavior of Omega calculation:
 *             -1: Constant value of 1.
 *              0: Varies sinusoidally within a range based on distance from the central point.
 *              1: Computes a value based on BRANDT_OmegaBr and BRANDT_R0 constants.
 *              2: Computes a value based on BRANDT_OmegaBr and BRANDT_R0 constants,
 *                 adjusted for distances greater than BRANDT_R0.
 * @return     The computed value of Omega.
 */
inline double OmegaR(int i, int mode)
{
    double m;
    double r = sqrt((P[i].Pos[0] - LONG_X / 2.) * (P[i].Pos[0] - LONG_X / 2.) +
                    (P[i].Pos[1] - LONG_Y / 2.) * (P[i].Pos[1] - LONG_Y / 2.));
    if (r > 0)
    {
        switch (mode)
        {
        case -1:
            m = 1;
            break;
        case 0:
            if (r < 7)
            {
                m = 1.0;
            }
            else
            {
                m = abs(sin((r - 7) * M_PI / (2 * All.Alfa2H)) + 1);
            }
            if (r > 8)
                m = 1.0;
            break;
        case 1:
            m = (Brandt_OmegaBr) / sqrt(1 + (r / Brandt_R0) * (r / Brandt_R0));
            break;
        case 2:
            if (r < Brandt_R0)
            {
                m = Brandt_OmegaBr;
            }
            else
            {
                m = Brandt_OmegaBr * Brandt_R0 / r;
            }
            break;
        default:
            VERBOSE_ALL(0, "Wrong Immplementation of Omega");
            exit(80085);
            break;
        };
    }
    else
    {
        m = 0;
    }
    return m;
};
#endif // STATICBRANDT

void sum_top_level_node_costfactors(void);

/**
 * @brief Compute the gravitational forces for all active particles.
 *
 * This function computes the gravitational forces for all active particles.
 * If needed, a new tree is constructed, otherwise the dynamically updated
 * tree is used. Particles are only exported to other processors when really
 * needed, thereby allowing a good use of the communication buffer.
 */
void gravity_tree(void)
{
    TIMER_PUSH(1, "GRAVITY_TREE");
    double tstart, tend;
    double ErrTolTheta_merk = All.ErrTolTheta;
    long long n_exported = 0;
    int maxnumnodes, iter = 0;
    double sum_costtotal, ewaldtot;
    double plb, plb_max;
    double timetree = 0, timetree1 = 0, timetree2 = 0;

#ifdef FIXEDTIMEINFIRSTPHASE
    int counter;
    double min_time_first_phase, min_time_first_phase_glob;
#endif // FIXEDTIMEINFIRSTPHASE
#ifndef NOGRAVITY
    int ewald_max, save_NextParticle;
    int ndone, ndone_flag; // ngrp;
#endif // NOGRAVITY

    if (header.flag_ic_info == FLAG_SECOND_ORDER_ICS)
    {
        if (All.Ti_Current == 0 && RestartFlag == 0 && All.Previous_Ti_Current == All.Ti_Current)
            if (All.ErrTolTheta == 0)
                All.ErrTolTheta =
                    0.45; /* This will switch to the BH opening criterion for the first force calculation */
    }
    else
    {
        if (All.ErrTolTheta == 0 && All.Previous_Ti_Current == All.Ti_Current)
            All.ErrTolTheta = 0.45; /* This will switch to the BH opening criterion for the first force computation */
    }

    /* set new softening lengths */
    if (All.ComovingIntegrationOn)
    {
        set_softenings();
    }

    /* construct tree if needed */
    if (TreeReconstructFlag)
    {
        VERBOSE(1, "  GRAVITY: tree connstruction in gravtree (ErrTolTheta=%g).  (presently allocated=%g MB)",
                All.ErrTolTheta, AllocatedBytes / (1024.0 * 1024.0));
        on_end_of_timestep(); /* used by OpenACC, empty otherwise */
        VERBOSE(3, "before Force tree build done");
        force_treebuild(NumPart, NULL);
        VERBOSE(3, "Force tree build done");

        TreeReconstructFlag = 0;
        on_beginning_of_timestep();

        VERBOSE(1, "  GRAVITY: tree construction done.");

        /* we have a fresh tree and would like to measure gravity cost */

        /* find the closest level */
        int diff = abs(All.LevelToTimeBin[0] - All.HighestActiveTimeBin);
        TakeLevel = 0;
        for (int i = 1; i < Gravcostlevels; i++)
        {
            if (diff > abs(All.LevelToTimeBin[i] - All.HighestActiveTimeBin))
            {
                TakeLevel = i;
                diff = abs(All.LevelToTimeBin[i] - All.HighestActiveTimeBin);
            }
        }

        if (diff != 0) /* we have not found a matching slot */
        {
            if (All.HighestOccupiedTimeBin - All.HighestActiveTimeBin < Gravcostlevels) /* we should have space */
            {
                /* clear levels that are out of range */
                for (int i = 0; i < Gravcostlevels; i++)
                {
                    if (All.LevelToTimeBin[i] > All.HighestOccupiedTimeBin)
                        All.LevelToTimeBin[i] = 0;
                    if (All.LevelToTimeBin[i] < All.HighestOccupiedTimeBin - (Gravcostlevels - 1))
                        All.LevelToTimeBin[i] = 0;
                }
            }

            TakeLevel = -1;
            for (int i = 0; i < Gravcostlevels; i++)
            {
                if (All.LevelToTimeBin[i] == 0)
                {
                    All.LevelToTimeBin[i] = All.HighestActiveTimeBin;
                    TakeLevel = i;
                    break;
                }
            }

            ENDRUN_IF(TakeLevel < 0 && All.HighestOccupiedTimeBin - All.HighestActiveTimeBin < Gravcostlevels,
                      "TakeLevel < 0, even though we should have a slot");
        }
        VERBOSE(1, "  GRAVITY: Measuring gravity cost ... LevelToTimeBin[TakeLevel=%d]=%d ", TakeLevel,
                All.LevelToTimeBin[TakeLevel]);
    }
    else
    {
        /* in this case we do not measure gravity cost. Check whether this time-level
       has previously mean measured. If yes, then delete it so to make sure that it is not out of time */

        for (int i = 0; i < Gravcostlevels; i++)
        {
            if (All.LevelToTimeBin[i] == All.HighestActiveTimeBin)
            {
                All.LevelToTimeBin[i] = 0;
            }
        }

        TakeLevel = -1;
        VERBOSE(1, "  GRAVITY: No measure of gravity cost in this timebin (ErTolTheta=%g) ...", All.ErrTolTheta);
    }

#ifdef ACC
    // NOTE: this has to be done if ACC is defined in general (not only for ACC_GRAVITY)
    // because we always need the drift and the go_gpu decision
    int is_primary_loop_on_accelerator = 0;
    int should_drift = 1;
    acc_all.go_gpu = NActivePart > ACC_ACTIVEPART_CPU;
    openacc_gravity_drift(is_primary_loop_on_accelerator, should_drift);
#endif // ACC_GRAVITY

#ifndef NOGRAVITY

#ifdef ADAPTGRAVSOFT
    ags_density();
    ags_force_update_hmax();
#endif // ADAPTGRAVSOFT

    /* allocate buffers to arrange communication */
    VERBOSE(1, "  GRAVITY: begin tree force.  (presently allocated=%g MB)", AllocatedBytes / (1024.0 * 1024.0));

    size_t MyBufferSize = 0.5 * KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize =
        (int)((MyBufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct gravdata_in) +
               sizeof(struct gravdata_out) + sizemax(sizeof(struct gravdata_in), sizeof(struct gravdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));

    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeListIn", All.BunchSize * sizeof(struct data_nodelist));
#ifdef AR_NO_NODELIST_IN_DATASTRUCT
    struct data_nodelist *DataNodeListOut =
        (struct data_nodelist *)mymalloc("DataNodeListOut", All.BunchSize * sizeof(struct data_nodelist));
#endif // AR_NO_NODELIST_IN_DATASTRUCT

    VERBOSE(2, "Gravity: using %g MB for buffering, %g MB remaining for other buffers", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    VERBOSE(5, "All.BunchSize=%d", All.BunchSize);

    Ewaldcount = 0;
    Costtotal = 0;
    N_nodesinlist = 0;

#if defined(PERIODIC) && !defined(PMGRID)
    ewald_max = 1;
#else // defined(PERIODIC) && !defined(PMGRID)
    ewald_max = 0;
#endif // defined(PERIODIC) && !defined(PMGRID)

#ifdef ACC_GRAVITY
    struct gravdata_out *LitePOut;
    double *PGravCosts;
    double *NodeGravCosts;
    // Nitya - declare vars here if OMP offload
#ifdef ACC_OPENMP
    double *PGravCostsII;
    double *NodeGravCostsII;
#endif // ACC_OPENMP

    int NodesSize = MaxNodes; // Numnodestree;
#ifdef ACC_OPENMP
    openacc_gravity_init(LitePOut, PGravCosts, NodeGravCosts, PGravCostsII, NodeGravCostsII, NodesSize);
#else // ACC_OPENMP
    openacc_gravity_init(LitePOut, PGravCosts, NodeGravCosts, NodesSize);
#endif // ACC_OPENMP
#endif // ACC_GRAVITY

    TIMER_PUSH(1, "TREE_ZERO");

    if (TakeLevel >= 0)
#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            P[i].GravCost[TakeLevel] = 0;
        }

    TIMER_POP(1, "TREE_ZERO");

#ifdef ACC_GRAVITY
#ifdef ACC_OPENMP
    // Nitya - adding this condition as the alloc of the data on host also happens with condition true
    if (acc_all.go_gpu)
    {
#pragma omp target enter data map(alloc                                                                                \
                                  : NodeGravCosts [0:NodesSize], PGravCosts [0:NumPart], LitePOut [0:NActivePart],     \
                                    NodeGravCostsII [0:NodesSize], PGravCostsII [0:NumPart])
        ACC_PRAGMA(ACC_UPDATE_DEVICE(All));
    }
#endif // ACC_OPENMP
#endif // ACC_GRAVITY
    for (Ewald_iter = 0; Ewald_iter <= ewald_max; Ewald_iter++)
    {
        /* this pragma is ignored if there is no openacc support. But we need that extra block {} here in gravtree
         */
        NextParticle = FirstActiveParticle; /* beginn with this index */
#ifdef ACC_GRAVITY
        int DoPrimaryOnGPU = acc_all.go_gpu;
#ifdef ACC_OPENACC
#pragma acc enter data create(NodeGravCosts [0:NodesSize], PGravCosts [0:NumPart],                                     \
                              LitePOut [0:NActivePart]) if (DoPrimaryOnGPU)
#pragma acc update device(All, NodeGravCosts [0:NodesSize], PGravCosts [0:NumPart], LitePOut [0:NActivePart])          \
    ACC_ASYNC1 if (DoPrimaryOnGPU)
#endif // ACC_OPENACC
#endif // ACC_GRAVITY
        {

#ifdef ACC_GRAVITY
            // note: costTotal_GPU_I  is defined in OpenACC/*.h and zeroed in the gravity init function
            openacc_gravity_primary_loop(LitePOut, PGravCosts, NodeGravCosts, NodesSize);
#endif // ACC_GRAVITY

            do
            {
                iter++;
                BufferFullFlag = 0;
                Nexport = 0;
                save_NextParticle = NextParticle;

                TIMER_PUSH(1, "TREE_WALK");
/* do local particles and prepare export list */
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
                {
                    int mainthreadid = ThisThread;
#ifdef ACC_GRAVITY
                    // gravity_primary_loop(&mainthreadid);
                    // gravity_primary_loop_vect(&mainthreadid, is_primary_loop_on_accelerator, should_drift);
                    //                      gravity_primary_loop_vect(&mainthreadid, 0, should_drift);
                    // NextParticle = -1;
                    gravity_primary_loop(&mainthreadid, is_primary_loop_on_accelerator, should_drift);

#else // ACC_GRAVITY*/
                    gravity_primary_loop(&mainthreadid);
#endif // ACC_GRAVITY
                }

                TIMER_POPPUSH(1, "TREE_WALK", "TREE_PREP_EXPORT");

                if (BufferFullFlag)
                {
                    int last_nextparticle = NextParticle;

                    NextParticle = save_NextParticle;

                    while (NextParticle >= 0)
                    {
                        if (NextParticle == last_nextparticle)
                        {
                            break;
                        }

                        if (ProcessedFlag[NextParticle] != 1)
                        {
                            break;
                        }

                        ProcessedFlag[NextParticle] = 2;

                        NextParticle = NextActiveParticle[NextParticle];
                    }

                    ENDRUN_IF(NextParticle == save_NextParticle,
                              "The buffer is too small to process even a single particle.");

                    int new_export = 0;

                    int k = 0;
                    for (int j = 0; j < Nexport; j++)
                        if (ProcessedFlag[DataIndexTable[j].Index] != 2)
                        {
                            if (k < j + 1)
                            {
                                k = j + 1;
                            }

                            for (; k < Nexport; k++)
                                if (ProcessedFlag[DataIndexTable[k].Index] == 2)
                                {
                                    int old_index = DataIndexTable[j].Index;

                                    DataIndexTable[j] = DataIndexTable[k];
                                    DataNodeList[j] = DataNodeList[k];
                                    DataIndexTable[j].IndexGet = j;
                                    new_export++;

                                    DataIndexTable[k].Index = old_index;
                                    k++;
                                    break;
                                }
                        }
                        else
                        {
                            new_export++;
                        }
                    Nexport = new_export;
                }
                n_exported += Nexport;

                for (int j = 0; j < NTask; j++)
                {
                    Send_count[j] = 0;
                }
                for (int j = 0; j < Nexport; j++)
                {
                    Send_count[DataIndexTable[j].Task]++;
                }

                MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
                tstart = second();
                TIMER_POPPUSH(1, "TREE_PREP_EXPORT", "TREE_WAIT");

#ifndef IMPORT_ALLtoALLv_FROM_G4
                MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
#else // IMPORT_ALLtoALLv_FROM_G4
#ifdef IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
                int non_active_timebin = 0;
                for (int j = TIMEBINS - 1; j >= 0; j--)
                {
                    if (TimeBinActive[j])
                    {
                        break;
                    }
                    else
                    {
                        non_active_timebin++;
                    }
                }
                if (non_active_timebin > IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER)
                {
#endif // IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
                    for (int j = 0; j < NTask; j++)
                    {
                        Recv_count[j] = 0;
                    }
                    MPI_Win_create(Recv_count, NTask * sizeof(int), sizeof(int), MPI_INFO_NULL, MYMPI_COMM_WORLD, &win);
                    MPI_Win_fence(0, win);
                    for (int j = 0; j < NTask; ++j)
                    {
                        if (Send_count[j] > 0)
                            MPI_Put(&Send_count[j], 1, MPI_INT, j, ThisTask, 1, MPI_INT, win);
                    }
                    MPI_Win_fence(0, win);
                    MPI_Win_free(&win);
#ifdef IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
                }
                else
                {
                    MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
                }
#endif // IMPORT_ALLtoALLv_FROM_G4_WHEN_TIMEBIN_ACTIVE_AFTER
#endif // IMPORT_ALLtoALLv_FROM_G4

                TIMER_POPPUSH(1, "TREE_WAIT", "TREE_COPY_OUT");

                Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
                for (int j = 0; j < NTask; j++)
                {
                    Nimport += Recv_count[j];

                    if (j > 0)
                    {
                        Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                        Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                    }
                }

                GravDataGet = (struct gravdata_in *)mymalloc("GravDataGet", Nimport * sizeof(struct gravdata_in));
                GravDataIn = (struct gravdata_in *)mymalloc("GravDataIn", Nexport * sizeof(struct gravdata_in));

                /* prepare particle data for export */

                for (int j = 0; j < Nexport; j++)
                {
                    int place = DataIndexTable[j].Index;

                    for (int k = 0; k < 3; k++)
                    {
                        GravDataIn[j].Pos[k] = P[place].Pos[k];
                    }

                    GravDataIn[j].Type = P[place].Type;
#ifdef FOLLOW_ID
                    GravDataIn[j].ID = P[place].ID;
#endif // FOLLOW_ID
#ifdef ADAPTIVE_GRAVSOFT_FORGAS
                    if (P[place].Type == 0)
                    {
                        GravDataIn[j].Soft = P[place].Hsml;
                    }
#endif // ADAPTIVE_GRAVSOFT_FORGAS
                    GravDataIn[j].OldAcc = P[place].OldAcc;
#ifdef ADAPTGRAVSOFT
                    GravDataIn[j].AGS_zeta = P[place].AGS_zeta;
                    GravDataIn[j].AGS_omega = P[place].AGS_omega;
                    GravDataIn[j].AGS_Hsml = P[place].AGS_Hsml;
                    GravDataIn[j].Mass = P[place].Mass;
#endif // ADAPTGRAVSOFT
                    memcpy(GravDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                           NODELISTLENGTH * sizeof(int));
                }

                /* exchange particle data */
                TIMER_POPPUSH(1, "TREE_COPY_OUT", "TREE_SEND");
                templated_exchange_data_Sendrecv(GravDataIn, GravDataGet, TAG_GRAV_A, Send_offset, Send_count,
                                                 Recv_offset, Recv_count);
                TIMER_POP(1, "TREE_SEND");

                myfree(GravDataIn);
                GravDataResult =
                    (struct gravdata_out *)mymalloc("GravDataResult", Nimport * sizeof(struct gravdata_out));
                GravDataOut = (struct gravdata_out *)mymalloc("GravDataOut", Nexport * sizeof(struct gravdata_out));

                report_memory_usage(&HighMark_gravtree, "GRAVTREE");

                /* now do the particles that were sent to us */
                TIMER_PUSH(1, "TREE_WALK");
                tstart = second();
                NextJ = 0;

#ifdef ACC_GRAVITY

#ifdef ACC_OPENMP
                openacc_gravity_secondary_loop(should_drift, NodeGravCostsII, PGravCostsII);
#elif defined(ACC_OPENACC) // ACC_OPENMP
                openacc_gravity_secondary_loop(should_drift, NodesSize);
#endif // ACC_OPENMP

#else // ACC_GRAVITY
#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
                {
                    int mainthreadid = ThisThread;
                    gravity_secondary_loop(&mainthreadid);
                }
#endif // ACC_GRAVITY
                tend = second();
                timetree2 += timediff(tstart, tend);

                TIMER_POP(1, "TREE_WALK");

                if (NextParticle < 0)
                {
                    ndone_flag = 1;
                }
                else
                {
                    ndone_flag = 0;
                }
                TIMER_PUSH(1, "TREE_WAIT");
                MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
                TIMER_POPPUSH(1, "TREE_WAIT", "TREE_RECV");
                templated_exchange_data_Sendrecv(GravDataResult, GravDataOut, TAG_GRAV_B, Recv_offset, Recv_count,
                                                 Send_offset, Send_count);
                TIMER_POP(1, "TREE_RECV");
                TIMER_PUSH(1, "TREE_WALK");
                tstart = second();

                /* add the results to the local particles */
                for (int j = 0; j < Nexport; j++)
                {
                    int place = DataIndexTable[j].Index;
                    assert(place >= 0);
                    assert(place < NumPart);
                    for (int k = 0; k < 3; k++)
                    {
                        P[place].GravAccel[k] += GravDataOut[j].Acc[k];
                    }
#ifdef FOLLOW_ID
                    if (P[place].ID == FOLLOW_ID)
                    {
                        VERBOSE(0, "Task %d: acc=(%g,%g,%g)  added (%g,%g,%g)\n", ThisTask, P[place].GravAccel[0],
                                P[place].GravAccel[1], P[place].GravAccel[2], GravDataOut[j].Acc[0],
                                GravDataOut[j].Acc[1], GravDataOut[j].Acc[2]);
                    }
#endif // FOLLOW_ID

#ifdef EVALPOTENTIAL
                    P[place].Potential += GravDataOut[j].Potential;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
                    P[place].AGS_corr += GravDataOut[j].AGS_corr;
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
                }

                tend = second();
                timetree1 += timediff(tstart, tend);
                TIMER_POP(1, "TREE_WALK");

                myfree(GravDataOut);
                myfree(GravDataResult);
                myfree(GravDataGet);

            } while (ndone < NTask);
#ifdef ACC_GRAVITY
#ifdef ACC_OPENACC
            if (DoPrimaryOnGPU)
            {
#pragma acc wait(1)
            }
#pragma acc update host(NodeGravCosts [0:NodesSize], PGravCosts [0:NumPart],                                           \
                        LitePOut [0:NActivePart]) if (DoPrimaryOnGPU)
#endif // ACC_OPENACC
#ifdef ACC_OPENMP
            if (acc_all.go_gpu)
            {
#pragma omp taskwait
#ifdef INTEL_UPDATE
#pragma omp target update from(PGravCosts [0:NumPart]) if ((Var_updt & 1) == 1)
#pragma omp target update from(NodeGravCosts [0:NodesSize]) if ((Var_updt & 3) == 3)
#pragma omp target update from(LitePOut [0:NActivePart]) if ((Var_updt & 7) == 7)
#else // INTEL_UPDATE
#pragma omp target update from(LitePOut [0:NActivePart]) depend(inout : LitePOut [0:NActivePart])
#pragma omp target update from(NodeGravCosts [0:NodesSize]) depend(inout : NodeGravCosts [0:NodesSize])
#pragma omp target update from(PGravCosts [0:NumPart]) depend(inout : PGravCosts [0:NumPart])
#endif // INTEL_UPDATE
                VERBOSE(3, "cstTotal = %f", Costtotal);
                VERBOSE(3, "end GPU region");
            }
#endif // ACC_OPENMP

#endif // ACC_GRAVITY
        } // exit GPU region
#ifdef ACC_GRAVITY
#ifdef ACC_OPENACC
#pragma acc exit data delete (NodeGravCosts [0:NodesSize], PGravCosts [0:NumPart],                                     \
                              LitePOut [0:NActivePart]) if (DoPrimaryOnGPU)
#endif // ACC_OPENACC
        if (acc_all.go_gpu)
        {
            openacc_gravity_join_results(LitePOut, PGravCosts, NodeGravCosts, NodesSize);
        }
#endif // ACC_GRAVITY
    } /* Ewald_iter */
// Nitya Check if condition is satisfied
#ifdef ACC_GRAVITY
#ifdef ACC_OPENMP
    if (acc_all.go_gpu)
    {
#pragma omp target exit data map(delete                                                                                \
                                 : NodeGravCosts [0:NodesSize], PGravCosts [0:NumPart], LitePOut [0:NActivePart],      \
                                   NodeGravCostsII [0:NodesSize], PGravCostsII [0:NumPart])
    }
#endif // ACC_OPENMP
#endif // ACC_GRAVITY

    TIMER_PUSH(1, "TREE_FINAL");

#ifdef ACC_GRAVITY
    if (acc_all.go_gpu)
    {
        // Nitya - delete it here if using OpenMP offload, the size doesn't change during the Ewald loop
#ifdef ACC_OPENMP
        myfree(PGravCostsII);
        myfree(NodeGravCostsII);
#endif // ACC_OPENMP
        myfree(LitePOut);
        myfree(PGravCosts);
        myfree(NodeGravCosts);
    }
#endif // ACC_GRAVITY

    myfree(DataNodeList);
    myfree(DataIndexTable);

    /* assign node cost to particles */
    if (TakeLevel >= 0)
    {
        sum_top_level_node_costfactors();

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
#ifdef NEUTRINOS
            if ((P[i].Type != type_neutrino) || (All.Time > All.Time_tree_on_nu))
#endif // NEUTRINOS
            {
                int no = Father[i];

                while (no >= 0)
                {
                    if (Nodes[no].u.d.mass > 0)
                    {
                        P[i].GravCost[TakeLevel] += Nodes[no].GravCost * P[i].Mass / Nodes[no].u.d.mass;
                    }

                    no = Nodes[no].u.d.father;
                }
            }
        }
    }

    /* now add things for comoving integration */

#ifndef PERIODIC
#ifndef PMGRID
    if (All.ComovingIntegrationOn)
    {
        double fac = 0.5 * All.Hubble * All.Hubble * All.Omega0 / All.GravityConstantInternal;

        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            for (int j = 0; j < 3; j++)
            {
                P[i].GravAccel[j] += fac * P[i].Pos[j];
            }
        }
    }
#endif // PMGRID
#endif // PERIODIC

    /* to prevent that we overwrite OldAcc in the first evaluation for 2lpt ICs */
    if (!(header.flag_ic_info == FLAG_SECOND_ORDER_ICS && All.Ti_Current == 0 && RestartFlag == 0))
    {
#ifdef _OPENMP
#pragma omp parallel for
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
#else // _OPENMP
        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
#endif // _OPENMP
            double ax, ay, az;
#ifdef PMGRID
            ax = P[i].GravAccel[0] + P[i].GravPM[0] / All.GravityConstantInternal;
            ay = P[i].GravAccel[1] + P[i].GravPM[1] / All.GravityConstantInternal;
            az = P[i].GravAccel[2] + P[i].GravPM[2] / All.GravityConstantInternal;
#else // PMGRID
            ax = P[i].GravAccel[0];
            ay = P[i].GravAccel[1];
            az = P[i].GravAccel[2];
#endif // PMGRID
            P[i].OldAcc = sqrt(ax * ax + ay * ay + az * az);
        } // for
    } // if

    if (ErrTolTheta_merk == 0)
    {
        All.ErrTolTheta = ErrTolTheta_merk;
    }

    /*  muliply by G */
#ifdef _OPENMP
#pragma omp parallel for // if(NActivePart>40)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        for (int j = 0; j < 3; j++)
        {
            P[i].GravAccel[j] *= All.GravityConstantInternal;
        }

#ifdef EVALPOTENTIAL
        /* remove self-potential */
        P[i].Potential += P[i].Mass / All.SofteningTable[P[i].Type];

        if (All.ComovingIntegrationOn)
        {
            if (All.PeriodicBoundariesOn)
            {
                P[i].Potential -=
                    2.8372975 * pow(P[i].Mass, 2.0 / 3) *
                    pow(All.Omega0 * 3 * All.Hubble * All.Hubble / (8 * M_PI * All.GravityConstantInternal), 1.0 / 3);
            }
        }

        P[i].Potential *= All.GravityConstantInternal;

#ifdef PMGRID
        P[i].Potential += P[i].PM_Potential; /* add in long-range potential */
#endif // PMGRID

        if (All.ComovingIntegrationOn)
        {
#ifndef PERIODIC
            double fac = -0.5 * All.Omega0 * All.Hubble * All.Hubble;
            double r2 = 0;
            for (int k = 0; k < 3; k++)
            {
                r2 += P[i].Pos[k] * P[i].Pos[k];
            }

            P[i].Potential += fac * r2;
#endif // PERIODIC
        }
        else
        {
            double r2 = 0;
            double fac = -0.5 * All.OmegaLambda * All.Hubble * All.Hubble;
            if (fac != 0)
            {
                for (int k = 0; k < 3; k++)
                {
                    r2 += P[i].Pos[k] * P[i].Pos[k];
                }

                P[i].Potential += fac * r2;
            }
        }
#endif // EVALPOTENTIAL
    } // for

    /* Finally, the following factor allows a computation of a cosmological simulation
       with vacuum energy in physical coordinates */
#ifndef PERIODIC
#ifndef PMGRID
    if (All.ComovingIntegrationOn == 0)
    {
        double fac = All.OmegaLambda * All.Hubble * All.Hubble;

        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            for (int j = 0; j < 3; j++)
            {
                P[i].GravAccel[j] += fac * P[i].Pos[j];
            }
        }
    }
#endif // PMGRID
#endif // PERIODIC

    VERBOSE(5, "tree is done.");
    TIMER_POP(1, "TREE_FINAL");

#else // NOGRAVITY
    TIMER_PUSH(1, "TREE_FORCE_ZERO");
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        for (int j = 0; j < 3; j++)
        {
            P[i].GravAccel[j] = 0;
        }
    }
    TIMER_POP(1, "TREE_FORCE_ZERO");
#endif // NOGRAVITY

#ifdef CONSTANT_GRAVITY_Y
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#ifdef SPH_BND_PARTICLES
        if (P[i].ID != 0)
#endif // SPH_BND_PARTICLES
            P[i].GravAccel[1] += Constant_Gravity_Y;
    }
#endif // CONSTANT_GRAVITY_Y

#ifdef STATICISO
    {
        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            double m;
            double dx = P[i].Pos[0];
            double dy = P[i].Pos[1];
            double dz = P[i].Pos[2];

            double r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r > Iso_R200)
            {
                m = Iso_M200;
            }
            else
            {
                m = Iso_M200 * r / Iso_R200;
            }

#ifdef ISO_FRACTION
            m *= ISO_FRACTION;
#endif // ISO_FRACTION
            if (r > 0)
            {
                P[i].GravAccel[0] += -All.GravityConstantInternal * m * dx / r / (r * r + Iso_Eps * Iso_Eps);
                P[i].GravAccel[1] += -All.GravityConstantInternal * m * dy / r / (r * r + Iso_Eps * Iso_Eps);
                P[i].GravAccel[2] += -All.GravityConstantInternal * m * dz / r / (r * r + Iso_Eps * Iso_Eps);
            }
        }
    }
#endif // STATICISO

#ifdef DISKPOT
    gravity_tree_subfunc_diskpot();
#endif // DISKPOT

#ifdef GROWING_DISK_POTENTIAL
    {
        double mdisk, aR, az;
        growing_disk_init();
        mdisk = get_disk_mass(All.Time);

        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            double dx = P[i].Pos[0];
            double dy = P[i].Pos[1];
            double dz = P[i].Pos[2];

            double r = sqrt(dx * dx + dy * dy);
            double z = abs(dz);

            get_disk_forces(r, z, &aR, &az);

            aR *= mdisk;
            az *= mdisk;

            if (r > 0)
            {
                P[i].GravAccel[0] += -dx / r * aR;
                P[i].GravAccel[1] += -dy / r * aR;
                P[i].GravAccel[2] += -dz / z * az;
            }
        }
    }
#endif // GROWING_DISK_POTENTIAL

#ifdef STATICNFW
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        double dx, dy, dz;

#ifdef NFW_BOXCENTERED
        dx = P[i].Pos[0] - boxHalf_X;
        dy = P[i].Pos[1] - boxHalf_Y;
        dz = P[i].Pos[2] - boxHalf_Z;
#else // NFW_BOXCENTERED
        dx = P[i].Pos[0];
        dy = P[i].Pos[1];
        dz = P[i].Pos[2];
#endif // NFW_BOXCENTERED

        MyAtLeastDouble r = sqrt(dx * dx + dy * dy + dz * dz);
        MyAtLeastDouble m = enclosed_mass(r);

#ifdef NFW_DARKFRACTION
        m *= NFW_DARKFRACTION;
#endif // NFW_DARKFRACTION
        if (r > 0)
        {
            P[i].GravAccel[0] += -All.GravityConstantInternal * m * dx / (r * r * r);
            P[i].GravAccel[1] += -All.GravityConstantInternal * m * dy / (r * r * r);
            P[i].GravAccel[2] += -All.GravityConstantInternal * m * dz / (r * r * r);
        }
    }
#endif // STATICNFW

#ifdef STATICPLUMMER
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        double r = sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]);

        for (int l = 0; l < 3; l++)
        {
            P[i].GravAccel[l] += -P[i].Pos[l] / pow(r * r + 1, 1.5);
        }
    }
#endif // STATICPLUMMER

#ifdef STATICHQ
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        double r = sqrt(P[i].Pos[0] * P[i].Pos[0] + P[i].Pos[1] * P[i].Pos[1] + P[i].Pos[2] * P[i].Pos[2]);
        double a = pow(All.GravityConstantInternal * Hq_M200 / (100 * All.Hubble * All.Hubble), 1.0 / 3) / Hq_C *
                   sqrt(2 * (log(1 + Hq_C) - Hq_C / (1 + Hq_C)));

        double m = Hq_M200 * pow(r / (r + a), 2);
#ifdef HQ_DARKFRACTION
        m *= HQ_DARKFRACTION;
#endif // HQ_DARKFRACTION
        if (r > 0)
        {
            for (int l = 0; l < 3; l++)
            {
                P[i].GravAccel[l] += -All.GravityConstantInternal * m * P[i].Pos[l] / (r * r * r);
            }
        }
    }
#endif // STATICHQ

#ifdef STATICBRANDT
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        /* note there is no acceleration in z */
        /* to be additive, this is not explicitely set to zero. */
        P[i].GravAccel[0] -= OmegaR(i, BRANDT_MODE) * OmegaR(i, BRANDT_MODE) * (P[i].Pos[0] - LONG_X / 2.);
        P[i].GravAccel[1] -= OmegaR(i, BRANDT_MODE) * OmegaR(i, BRANDT_MODE) * (P[i].Pos[1] - LONG_Y / 2.);
    }
#endif // STATICBRANDT

#ifdef KEPLER_DISK
    MyLongDouble x00 = 4.0, y00 = 4.0; /* 2D center of orbit: the is hard-coded for the relevant test problem */
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        // add the constant field
        MyFloat r = pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 0.5);
        if (r <= 0.35)
        {
            P[i].GravAccel[0] += -(P[i].Pos[0] - x00) * pow(r / 0.35, 2) /
                                 pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[0] += +(P[i].Pos[0] - x00) * (0.35 - r) / 0.35 /
                                 pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[1] += -(P[i].Pos[1] - y00) * pow(r / 0.35, 2) /
                                 pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[1] += +(P[i].Pos[1] - y00) * (0.35 - r) / 0.35 /
                                 pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[2] += 0;
        }
        else if ((r > 0.35) && (r < 2.1))
        {
            P[i].GravAccel[0] += -(P[i].Pos[0] - x00) / pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[1] += -(P[i].Pos[1] - y00) / pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[2] += 0;
        }
        else if (r >= 2.1)
        {
            P[i].GravAccel[0] += -(P[i].Pos[0] - x00) * (1 + (r - 2.1) / 0.1) /
                                 pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[1] += -(P[i].Pos[1] - y00) * (1 + (r - 2.1) / 0.1) /
                                 pow(pow(P[i].Pos[1] - y00, 2) + pow(P[i].Pos[0] - x00, 2), 1.5);
            P[i].GravAccel[2] += 0;
        }
    }
#endif // KEPLER_DISK

#ifdef DISK_SLICE
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        // add the centrifugal force
        P[i].GravAccel[0] += 2 * All.qDisk * All.OmegaDisk * All.OmegaDisk * (P[i].Pos[0] - boxHalf_X);
        // add the coriolis foce
        MyFloat vr, vphi;
        if (P[i].Type == 0)
        {
            vr = SphP[i].VelPred[0];
            vphi = SphP[i].VelPred[2];
        }
        else
        {
            vr = P[i].Vel[0];
            vphi = P[i].Vel[2];
        }
        P[i].GravAccel[0] += 2.0 * All.OmegaDisk * vphi;
        P[i].GravAccel[2] += -2.0 * All.OmegaDisk * vr;
    }
#endif // DISK_SLICE

#ifdef ANALYTIC_POTENTIALS
    analytic_potentials.apply_accelerations();
#endif // ANALYTIC_POTENTIALS

    TIMER_PUSH(1, "TREE_FINAL");
    double maxt, sumt, maxt1, sumt1, maxt2, sumt2;
    timetree = timetree1 + timetree2;

    /* max-reduced values only; sum-reduced values (incl. Costtotal/Ewaldcount) share one array reduce below */
    double timetree_local[3] = {timetree, timetree1, timetree2};
    double timetree_max[3];
    MPI_Reduce(timetree_local, timetree_max, 3, MPI_DOUBLE, MPI_MAX, 0, MYMPI_COMM_WORLD);
    maxt = timetree_max[0];
    maxt1 = timetree_max[1];
    maxt2 = timetree_max[2];

    double sum_local[5] = {timetree, timetree1, timetree2, Costtotal, Ewaldcount};
    double sum_global[5];
    MPI_Reduce(sum_local, sum_global, 5, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
    sumt = sum_global[0];
    sumt1 = sum_global[1];
    sumt2 = sum_global[2];
    sum_costtotal = sum_global[3];
    ewaldtot = sum_global[4];

    /* n_exported/N_nodesinlist are only read on rank 0 below, so a plain sum-reduce (rather than
       sumup_longs()'s MPI_Allgather, which distributes the result to every rank) is sufficient. */
    long long exported_local[2] = {n_exported, N_nodesinlist};
    long long exported_global[2];
    MPI_Reduce(exported_local, exported_global, 2, MPI_LONG_LONG, MPI_SUM, 0, MYMPI_COMM_WORLD);
    n_exported = exported_global[0];
    N_nodesinlist = exported_global[1];

    All.TotNumOfForces += GlobNumForceUpdate;

    plb = (NumPart / ((double)All.TotNumPart)) * NTask;
    /* Numnodestree fits exactly in a double, so it can share this MAX reduce with plb instead of a separate call */
    double max_local[2] = {plb, (double)Numnodestree};
    double max_global[2];
    MPI_Reduce(max_local, max_global, 2, MPI_DOUBLE, MPI_MAX, 0, MYMPI_COMM_WORLD);
    plb_max = max_global[0];
    maxnumnodes = (int)max_global[1];

#ifdef FIXEDTIMEINFIRSTPHASE
    MPI_Reduce(&min_time_first_phase, &min_time_first_phase_glob, 1, MPI_DOUBLE, MPI_MIN, 0, MYMPI_COMM_WORLD);
    VERBOSE(5, "FIXEDTIMEINFIRSTPHASE=%g  min_time_first_phase_glob=%g", FIXEDTIMEINFIRSTPHASE,
            min_time_first_phase_glob);
#endif // FIXEDTIMEINFIRSTPHASE

    if (ThisTask == 0)
    {
        fprintf(FdTimings, "Step= %d  t= %g  dt= %g \n", All.NumCurrentTiStep, All.Time, All.TimeStep);
        fprintf(FdTimings, "Nf= %d%09d  total-Nf= %d%09d  ex-frac= %g (%g) iter= %d\n",
                (int)(GlobNumForceUpdate / 1000000000), (int)(GlobNumForceUpdate % 1000000000),
                (int)(All.TotNumOfForces / 1000000000), (int)(All.TotNumOfForces % 1000000000),
                n_exported / ((double)GlobNumForceUpdate), N_nodesinlist / ((double)n_exported + 1.0e-10), iter);
        /* note: on Linux, the 8-byte integer could be printed with the format identifier "%qd", but doesn't work on AIX
         */

        fprintf(FdTimings, "work-load balance: %g (%g %g) rel1to2=%g   max=%g avg=%g\n", maxt / (1.0e-6 + sumt / NTask),
                maxt1 / (1.0e-6 + sumt1 / NTask), maxt2 / (1.0e-6 + sumt2 / NTask), sumt1 / (1.0e-6 + sumt1 + sumt2),
                maxt, sumt / NTask);
        fprintf(FdTimings, "particle-load balance: %g\n", plb_max);
        fprintf(FdTimings, "max. nodes: %d, filled: %g\n", maxnumnodes,
                maxnumnodes / (All.TreeAllocFactor * All.MaxPart + NTopnodes));
        fprintf(FdTimings, "part/sec=%g | %g  ia/part=%g (%g)\n", GlobNumForceUpdate / (sumt + 1.0e-20),
                GlobNumForceUpdate / (1.0e-6 + maxt * NTask),
                ((double)(sum_costtotal)) / (1.0e-20 + GlobNumForceUpdate),
                ((double)ewaldtot) / (1.0e-20 + GlobNumForceUpdate));
        fprintf(FdTimings, "\n");

        fflush(FdTimings);
    }

    double costtotal_new = 0, sum_costtotal_new;
    if (TakeLevel >= 0)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : costtotal_new)
#endif // _OPENMP
        for (int i = 0; i < NumPart; i++)
        {
            costtotal_new += P[i].GravCost[TakeLevel];
        }
        MPI_Reduce(&costtotal_new, &sum_costtotal_new, 1, MPI_DOUBLE, MPI_SUM, 0, MYMPI_COMM_WORLD);
        if (ThisTask == 0)
        {
            if (sum_costtotal)
            {
                VERBOSE(2, "  GRAVITY: relative error in the total number of tree-gravity interactions = %g",
                        (sum_costtotal - sum_costtotal_new) / sum_costtotal);
            }
        }
        /* can be non-zero if THREAD_SAFE_COSTS is not used (and due to round-off errors). */
    }
    TIMER_POP(1, "TREE_FINAL");
    TIMER_POP(1, "GRAVITY_TREE");
    VERBOSE(1, "  GRAVITY: tree force done.");
}

/**
 * @brief Primary gravity computation loop for a single thread.
 *
 * This function performs the primary loop for computing gravitational forces between particles.
 * It iterates over particles, evaluating gravitational forces between them using the tree structure.
 * For each particle, it evaluates either the short-range or Ewald correction forces, depending on
 * compilation flags and system configuration.
 *
 * @param p A pointer to the thread ID for identifying the current thread.
 * @return Void pointer indicating the completion of the loop.
 */
#ifdef ACC_GRAVITY
void *gravity_primary_loop(void *p, int is_primary_on_gpu, int should_drift)
#else // ACC_GRAVITY
void *gravity_primary_loop(void *p)
#endif // ACC_GRAVITY
{
    int ret;
    int thread_id = *(int *)p;

    int *exportflag, *exportnodecount, *exportindex;

    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    /* Note: exportflag is local to each thread */
    for (int j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }

#ifdef FIXEDTIMEINFIRSTPHASE
    int counter = 0;
    double tstart;

    if (thread_id == 0)
    {
        tstart = second();
    }
#endif // FIXEDTIMEINFIRSTPHASE

#ifdef ACC_GRAVITY
    int mode = 0;
    if (is_primary_on_gpu)
    {

        mode = 3;
    }
#endif // ACC_GRAVITY

    while (1)
    {
        int i;
        int exitFlag = 0;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        if (BufferFullFlag != 0 || NextParticle < 0)
        {
            exitFlag = 1;
        }
        else
        {
            i = NextParticle;
            ProcessedFlag[i] = 0;
            NextParticle = NextActiveParticle[NextParticle];
        }

        if (exitFlag)
        {
            break;
        }

#ifndef PMGRID
#ifdef PERIODIC
        if (Ewald_iter)
        {
#ifdef ACC_GRAVITY
            ret = force_treeevaluate_ewald_correction(i, mode, exportflag, exportnodecount, exportindex, should_drift);
#else // ACC_GRAVITY
            ret = force_treeevaluate_ewald_correction(i, 0, exportflag, exportnodecount, exportindex);
#endif // ACC_GRAVITY
            if (ret >= 0)
            {

#ifdef _OPENMP
#pragma omp atomic
#endif // _OPENMP
                Ewaldcount +=
                    ret; /* note: ewaldcount may be slightly incorrect for multiple threads if buffer gets filled up */
            }
            else
            {
                break; /* export buffer has filled up */
            }
        }
        else
#endif // PERIODIC
        {
#ifdef ACC_GRAVITY
            ret = force_treeevaluate(i, mode, exportflag, exportnodecount, exportindex, should_drift);
#else // ACC_GRAVITY
            ret = force_treeevaluate(i, 0, exportflag, exportnodecount, exportindex);
#endif // ACC_GRAVITY
            if (ret < 0)
            {
                break; /* export buffer has filled up */
            }

#ifdef _OPENMP
#pragma omp atomic
#endif // _OPENMP
            Costtotal += ret;
        }
#else // PMGRID

#ifdef NEUTRINOS
        if ((P[i].Type != type_neutrino) || (All.Time > All.Time_tree_on_nu))
#endif // NEUTRINOS
        {
#ifdef ACC_GRAVITY
            ret = force_treeevaluate_shortrange(i, mode, exportflag, exportnodecount, exportindex,
                                                should_drift); // 1= drift!
#else // ACC
            ret = force_treeevaluate_shortrange(i, 0, exportflag, exportnodecount, exportindex);
#endif // ACC
            if (ret < 0)
            {
                break; /* export buffer has filled up */
            }

#ifdef _OPENMP
#pragma omp atomic
#endif // _OPENMP
            Costtotal += ret;
        }
#endif // PMGRID

        ProcessedFlag[i] = 1; /* particle successfully finished */

#ifdef FIXEDTIMEINFIRSTPHASE
        if (thread_id == 0)
        {
            counter++;
            if ((counter & 255) == 0)
            {
                if (timediff(tstart, second()) > FIXEDTIMEINFIRSTPHASE)
                {
                    TimerFlag = 1;
                    break;
                }
            }
        }
        else
        {
            if (TimerFlag)
            {
                break;
            }
        }
#endif // FIXEDTIMEINFIRSTPHASE
    }
    return NULL;
}

/**
 * @brief Secondary gravity computation loop for handling imported particles.
 *
 * This function performs the secondary loop for computing gravitational forces between imported particles.
 * It iterates over imported particles and evaluates gravitational forces between them and local particles.
 * Depending on compilation flags and system configuration, it computes either short-range or Ewald correction forces.
 *
 * @param p A pointer to the thread ID for identifying the current thread.
 * @return Void pointer indicating the completion of the loop.
 */
void *gravity_secondary_loop(void *p)
{
    int nodesinlist, dummy, ret;

    while (1)
    {
        int j;
#ifdef _OPENMP
#pragma omp atomic capture
#endif // _OPENMP
        j = NextJ++;

        if (j >= Nimport)
        {
            break;
        }

#ifndef PMGRID
#ifdef PERIODIC
        if (Ewald_iter)
        {
#ifdef ACC_GRAVITY
            int cost = force_treeevaluate_ewald_correction(j, 1, &dummy, &dummy, &dummy, 1);
#else // ACC_GRAVITY
            int cost = force_treeevaluate_ewald_correction(j, 1, &dummy, &dummy, &dummy);
#endif // ACC_GRAVITY
#ifdef _OPENMP
#pragma omp atomic
#endif // _OPENMP
            Ewaldcount += cost;
        }
        else
#endif // PERIODIC
        {
#ifdef ACC_GRAVITY
            ret = force_treeevaluate(j, 1, &nodesinlist, &dummy, &dummy, 1);
#else // ACC_GRAVITY
            ret = force_treeevaluate(j, 1, &nodesinlist, &dummy, &dummy);
#endif // ACC_GRAVITY
#ifdef _OPENMP
#pragma omp critical(__cost__)
#endif // _OPENMP
            {
                N_nodesinlist += nodesinlist;
                Costtotal += ret;
            }
        }
#else // PMGRID
#ifdef ACC_GRAVITY
        ret = force_treeevaluate_shortrange(j, 1, &nodesinlist, &dummy, &dummy, 1);
#else // ACC
        ret = force_treeevaluate_shortrange(j, 1, &nodesinlist, &dummy, &dummy);
#endif // ACC
#ifdef _OPENMP
#pragma omp critical(__cost__)
#endif // _OPENMP
        {
            N_nodesinlist += nodesinlist;
            Costtotal += ret;
        }
#endif // PMGRID
    }
    return NULL;
}

/**
 * @brief Sums the cost factors of the top-level nodes.
 *
 * This function calculates the total gravitational cost by summing the cost factors of the top-level nodes.
 * It allocates memory for temporary arrays to store the cost factors, performs an MPI reduction operation
 * to compute the sum across all MPI processes, and then updates the gravitational cost of each top-level node
 * with the summed values.
 */
void sum_top_level_node_costfactors(void)
{
    double *costlist = (double *)mymalloc("costlist", NTopnodes * sizeof(double));
    double *costlist_all = (double *)mymalloc("costlist_all", NTopnodes * sizeof(double));

    for (int i = 0; i < NTopnodes; i++)
    {
        costlist[i] = Nodes[All.MaxPart + i].GravCost;
    }

    MPI_Allreduce(costlist, costlist_all, NTopnodes, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    for (int i = 0; i < NTopnodes; i++)
    {
        Nodes[All.MaxPart + i].GravCost = costlist_all[i];
    }

    myfree(costlist_all);
    myfree(costlist);
}

/**
 * @brief Set softening lenghts.
 *
 * This function sets the (comoving) softening length of all particle
 * types in the table All.SofteningTable[...]. We check that the physical
 * softening length is bounded by the Softening-MaxPhys values.
 */
void set_softenings(void)
{
    if (All.ComovingIntegrationOn)
    {
        if (All.SofteningGas * All.Time > All.SofteningGasMaxPhys)
        {
            All.SofteningTable[0] = All.SofteningGasMaxPhys / All.Time;
        }
        else
        {
            All.SofteningTable[0] = All.SofteningGas;
        }

        if (All.SofteningHalo * All.Time > All.SofteningHaloMaxPhys)
        {
            All.SofteningTable[1] = All.SofteningHaloMaxPhys / All.Time;
        }
        else
        {
            All.SofteningTable[1] = All.SofteningHalo;
        }
        if (All.SofteningDisk * All.Time > All.SofteningDiskMaxPhys)
        {
            All.SofteningTable[2] = All.SofteningDiskMaxPhys / All.Time;
        }
        else
        {
            All.SofteningTable[2] = All.SofteningDisk;
        }

        if (All.SofteningBulge * All.Time > All.SofteningBulgeMaxPhys)
        {
            All.SofteningTable[3] = All.SofteningBulgeMaxPhys / All.Time;
        }
        else
        {
            All.SofteningTable[3] = All.SofteningBulge;
        }

        if (All.SofteningStars * All.Time > All.SofteningStarsMaxPhys)
        {
            All.SofteningTable[4] = All.SofteningStarsMaxPhys / All.Time;
        }
        else
        {
            All.SofteningTable[4] = All.SofteningStars;
        }

        if (All.SofteningBndry * All.Time > All.SofteningBndryMaxPhys)
        {
            All.SofteningTable[5] = All.SofteningBndryMaxPhys / All.Time;
        }
        else
        {
            All.SofteningTable[5] = All.SofteningBndry;
        }
    }
    else
    {
        All.SofteningTable[0] = All.SofteningGas;
        All.SofteningTable[1] = All.SofteningHalo;
        All.SofteningTable[2] = All.SofteningDisk;
        All.SofteningTable[3] = All.SofteningBulge;
        All.SofteningTable[4] = All.SofteningStars;
        All.SofteningTable[5] = All.SofteningBndry;
    }

    for (int i = 0; i < 6; i++)
    {
        All.ForceSoftening[i] = 2.8 * All.SofteningTable[i];
    }

    All.MinGasHsml = All.MinGasHsmlFractional * All.ForceSoftening[0];
}

/**
 * @brief
 *

 */

/**
 * @brief Compare function for sorting data_index structures based on Task, Index, and IndexGet.
 *
 * This function compares two data_index structures based on their Task, Index, and IndexGet fields.
 * It is used to group particles in the communication buffer that are going to be sent to the same CPU.
 * It returns -1 if the Task of the first structure is less than the Task of the second structure,
 * +1 if the Task of the first structure is greater than the Task of the second structure,
 * -1 if the Index of the first structure is less than the Index of the second structure,
 * +1 if the Index of the first structure is greater than the Index of the second structure,
 * -1 if the IndexGet of the first structure is less than the IndexGet of the second structure,
 * +1 if the IndexGet of the first structure is greater than the IndexGet of the second structure,
 * and 0 if all fields are equal.
 *
 * @param a Pointer to the first data_index structure.
 * @param b Pointer to the second data_index structure.
 * @return -1 if a < b, +1 if a > b, 0 if a == b.
 */
int data_index_compare(const void *a, const void *b)
{
    if (((struct data_index *)a)->Task < (((struct data_index *)b)->Task))
    {
        return -1;
    }

    if (((struct data_index *)a)->Task > (((struct data_index *)b)->Task))
    {
        return +1;
    }

    if (((struct data_index *)a)->Index < (((struct data_index *)b)->Index))
    {
        return -1;
    }

    if (((struct data_index *)a)->Index > (((struct data_index *)b)->Index))
    {
        return +1;
    }

    if (((struct data_index *)a)->IndexGet < (((struct data_index *)b)->IndexGet))
    {
        return -1;
    }

    if (((struct data_index *)a)->IndexGet > (((struct data_index *)b)->IndexGet))
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Merge sort function for sorting data_index structures using a temporary array.
 *
 * This function implements the merge sort algorithm to sort an array of data_index structures.
 * It recursively divides the array into smaller subarrays, sorts each subarray, and then merges
 * the sorted subarrays using a temporary array.
 *
 * @param b Pointer to the array of data_index structures to be sorted.
 * @param n Size of the array.
 * @param t Pointer to the temporary array used for merging.
 */
static void msort_dataindex_with_tmp(struct data_index *b, size_t n, struct data_index *t)
{
    if (n <= 1)
    {
        return;
    }

    size_t n1 = n / 2;
    size_t n2 = n - n1;
    struct data_index *b1 = b;
    struct data_index *b2 = b + n1;

    msort_dataindex_with_tmp(b1, n1, t);
    msort_dataindex_with_tmp(b2, n2, t);

    struct data_index *tmp = t;

    while (n1 > 0 && n2 > 0)
    {
        if (b1->Task < b2->Task || (b1->Task == b2->Task && b1->Index <= b2->Index))
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
        memcpy(tmp, b1, n1 * sizeof(struct data_index));
    }

    memcpy(b, t, (n - n2) * sizeof(struct data_index));
}

/**
 * @brief Sorts an array of data_index structures using merge sort.
 *
 * This function sorts an array of data_index structures using merge sort algorithm.
 *
 * @param b Pointer to the array of data_index structures to be sorted.
 * @param n Number of elements in the array.
 * @param s Size of each element in bytes.
 * @param cmp Pointer to the comparison function.
 */
void mysort_dataindex(void *b, size_t n, size_t s, int (*cmp)(const void *, const void *))
{
    const size_t size = n * s;

    struct data_index *tmp = (struct data_index *)mymalloc("struct data_index *tmp", size);

    msort_dataindex_with_tmp((struct data_index *)b, n, tmp);

    myfree(tmp);
}

#ifdef ACC_GRAVITY
/**
 * @brief Secondary gravity computation loop for handling imported particles with vectorization support.
 *
 * This function performs the secondary loop for computing gravitational forces between imported particles.
 * It iterates over imported particles and evaluates gravitational forces between them and local particles.
 * Depending on compilation flags and system configuration, it computes either short-range or Ewald correction forces.
 * This version of the function supports vectorization for improved performance.
 *
 * @param p A pointer to the thread ID for identifying the current thread.
 * @param should_drift Flag indicating whether drift should be performed.
 * @return Void pointer indicating the completion of the loop.
 */

static double costTotal_CPU_II = 0.; // Nitya - this will be shared across threads

void *gravity_secondary_loop_vect(void *p, int should_drift)
{
    // gcc wants reduction variables to be public if omp region is opened outside function
    int nodesinlist, dummy, ret;

#pragma omp for reduction(+ : costTotal_CPU_II) reduction(+ : Ewaldcount) reduction(+ : N_nodesinlist)
    for (int j = 0; j < Nimport; j++)
    {
#ifndef PMGRID
#if defined(PERIODIC)
        if (Ewald_iter)
        {
            int cost = force_treeevaluate_ewald_correction(j, 1, &dummy, &dummy, &dummy, should_drift);
            Ewaldcount += cost;
        }
        else
#endif // defined(PERIODIC)
        {
            ret = force_treeevaluate(j, 1, &nodesinlist, &dummy, &dummy, should_drift);
            N_nodesinlist += nodesinlist;
            costTotal_CPU_II += ret;
        }
#else // PMGRID
        ret = force_treeevaluate_shortrange(j, 1, &nodesinlist, &dummy, &dummy, should_drift);
        N_nodesinlist += nodesinlist;
        costTotal_CPU_II += ret;
#endif // PMGRID
    } // end for

    //  VERBOSE(-1,"costTotal_CPU_II=%g;  Costtotal=%g (before update)",costTotal_CPU_II, Costtotal);
#pragma omp atomic
    Costtotal += costTotal_CPU_II;
    //  VERBOSE(-1,"costTotal_CPU_II=%g;  Costtotal=%g (after update)",costTotal_CPU_II, Costtotal);

    return NULL;
}

/**
 * @brief Primary gravity computation loop for handling local particles with vectorization support.
 *
 * This function performs the primary loop for computing gravitational forces between local particles.
 * It iterates over active particles and evaluates gravitational forces between them and target particles.
 * Depending on compilation flags and system configuration, it computes either short-range or Ewald correction forces.
 * This version of the function supports vectorization for improved performance.
 *
 * @param p A pointer to the thread ID for identifying the current thread.
 * @param is_primary_on_gpu Flag indicating whether primary computation is performed on GPU.
 * @param should_drift Flag indicating whether drift should be performed.
 * @return Void pointer indicating the completion of the loop.
 */
static double costTotal_CPU_I = 0.; // Nitya - this will be shared across threads
void *gravity_primary_loop_vect(void *p, int is_primary_on_gpu, int should_drift)
{
    int thread_id = *(int *)p;
    int *exportflag, *exportnodecount, *exportindex;

    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    for (int j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }
    int mode = 0;
#ifdef ACC_GRAVITY
    if (is_primary_on_gpu)
    {
        mode = 3;
    }
#endif // ACC_GRAVITY

    for (int chunk_counter = 0; chunk_counter < NActivePart; chunk_counter++)
    {
        if (BufferFullFlag != 0)
        {
            // wait. GCC doesn't want break in parallel regions
#ifdef ACC_PGI
            break;
#endif // ACC_PGI
        }
        else
        {
            int target_id = chunk_counter;
            int target = ActiveParticleList[target_id];
            int ret = 0;
            if (ProcessedFlag[target] == 1)
            {
                // still to be tested, but, ewald=1 needs to re-evaluate the particles so I cannot skip =1 particles
                // on the other hand, if I do not continue, I will re-do all particles at different buffer-iterations
                // probably this whole function cannot work for ewald iterations
                //
                // continue;
            }
            if (ProcessedFlag[target] == 2)
            {
                continue;
            }
            ProcessedFlag[target] = 0;
#ifndef PMGRID
#if defined(PERIODIC)
            if (Ewald_iter)
            {
                ret = force_treeevaluate_ewald_correction(target, mode, exportflag, exportnodecount, exportindex,
                                                          should_drift);
            }
            else
#endif // defined(PERIODIC)
            {
                ret = force_treeevaluate(target, mode, exportflag, exportnodecount, exportindex, should_drift);
            }
#else // PMGRID
#ifdef NEUTRINOS
            if ((P[target].Type != type_neutrino) || (All.Time > All.Time_tree_on_nu))
#endif // NEUTRINOS

                ret =
                    force_treeevaluate_shortrange(target, mode, exportflag, exportnodecount, exportindex, should_drift);
#endif // PMGRID

            if (ret < 0)
            {
                BufferFullFlag = 1;
#ifdef ACC_PGI
                break;
#endif // ACC_PGI
            }
            else
            {
                ProcessedFlag[target] = 1;
                if (mode == 0)
                {
                    costTotal_CPU_I += ret;
                }
            }
        }
    }
    NextParticle = -1;
    if (mode == 0)
    {
        // only if complete I phase is in GPU then we add costtoal, otherwise it is accounted by GPU
        // #pragma omp atomic
        Costtotal += costTotal_CPU_I;
    }
    if (Nexport > All.BunchSize)
    {
        Nexport = All.BunchSize;
    }
    return NULL;
}

/**
 * @brief Joins the results of gravitational computation for OpenACC implementation.
 *
 * This function joins the results of gravitational computation from LitePOut, PGravCosts, and NodeGravCosts
 * into the respective particle and node structures.
 *
 * @param LitePOut Pointer to the array of gravitational data for active particles.
 * @param PGravCosts Array of gravitational costs for particles.
 * @param NodeGravCosts Array of gravitational costs for nodes.
 * @param NodesSize Size of the node array.
 */
inline void openacc_gravity_join_results(struct gravdata_out *LitePOut, double *PGravCosts, double *NodeGravCosts,
                                         int NodesSize)
{
#pragma omp parallel for
    for (int counter = 0; counter < NActivePart; counter++)
    {
        int i = ActiveParticleList[counter];
#ifdef NEUTRINOS
        if (!((P[i].Type != type_neutrino) || (All.Time > All.Time_tree_on_nu)))
        {
            continue;
        }
#endif // NEUTRINOS
        P[i].GravAccel[0] += LitePOut[counter].Acc[0];
        P[i].GravAccel[1] += LitePOut[counter].Acc[1];
        P[i].GravAccel[2] += LitePOut[counter].Acc[2];
#ifdef EVALPOTENTIAL
        P[i].Potential += LitePOut[counter].Potential;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
        P[i].AGS_corr += LitePOut[counter].AGS_corr;
#endif // defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
    }
    if (TakeLevel >= 0)
    {
#pragma omp parallel for
        for (int counter = 0; counter < NumPart; counter++)
        {
            P[counter].GravCost[TakeLevel] += PGravCosts[counter];
        }
    }
    Costtotal += costTotal_GPU_I; // finally add I phase (probably async) gpu grav cost
#pragma omp parallel for
    for (int counter = 0; counter < NodesSize; counter++)
    {
        Nodes_base[counter].GravCost += NodeGravCosts[counter];
    }
}
#endif // ACC_GRAVITY
