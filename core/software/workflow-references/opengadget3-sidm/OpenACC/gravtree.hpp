/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Antonio Ragagnin.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifdef ACC_OPENACC
#include <openacc.h>
#elif defined(ACC_OPENMP)
#include <omp.h>
#endif // ACC_OPENACC

ACC_DECLARE_ROUTINE_BEFORE
int force_treeevaluate_shortrange_local(int actove_target, int p_target, int mode, struct gravity_walkdata *P,
                                        struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult,
                                        double *NodeGravCosts, double *PGravCosts);
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int force_treeevaluate_local(int actove_target, int p_target, int mode, struct gravity_walkdata *P,
                             struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult,
                             double *NodeGravCosts, double *PGravCosts);
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int force_treeevaluate_ewald_correction_local(int actove_target, int p_target, int mode, struct gravity_walkdata *P,
                                              struct gravdata_in *GravDataGet, struct gravdata_out *GravDataResult,
                                              double *NodeGravCosts, double *PGravCosts);
ACC_DECLARE_ROUTINE_AFTER

void openacc_gravity_primary_loop(struct gravdata_out *LitePOut, double *PGravCosts, double *NodeGravCosts,
                                  int NodesSize);
inline void openacc_gravity_join_results(struct gravdata_out *LitePOut, double *PGravCosts, double *NodeGravCosts,
                                         int NodesSize);
inline void *gravity_primary_loop_vect(void *p, int is_primary_on_gpu, int should_drift);
inline void *gravity_secondary_loop_vect(void *p, int should_drift);

// cost total of I phase is global bc it is upadte in a async GPU region and merged to GPU at the end of gravity
double costTotal_GPU_I;
#ifdef ACC_OPENMP
inline void openacc_gravity_init(struct gravdata_out *&LitePOut, double *&PGravCosts, double *&NodeGravCosts,
                                 double *&PGravCostsII, double *&NodeGravCostsII, int &NodesSize)
#else // ACC_OPENACC
inline void openacc_gravity_init(struct gravdata_out *&LitePOut, double *&PGravCosts, double *&NodeGravCosts,
                                 int &NodesSize)
#endif // ACC_OPENMP
{
    costTotal_GPU_I = 0.;

    VERBOSE(3, "go_gpu I = %d; NActivePart = %d;  ACC_ACTIVEPART_CPU = %d", acc_all.go_gpu, NActivePart,
            ACC_ACTIVEPART_CPU);
    if (acc_all.go_gpu)
    {
        // this cost stuff is only in gravity, it is needed only for the domain decomposition
        NodeGravCosts = (double *)mymalloc("NGC", sizeof(double) * NodesSize);
        PGravCosts = (double *)mymalloc("PGravCosts", NumPart * sizeof(double));
        LitePOut = (struct gravdata_out *)mymalloc("LitePDO", NActivePart * sizeof(struct gravdata_out));

#ifdef ACC_OPENMP
        NodeGravCostsII = (double *)mymalloc("NGCII", sizeof(double) * NodesSize);
        PGravCostsII = (double *)mymalloc("PGravCostsII", NumPart * sizeof(double));
#endif // ACC_OPENMP
#pragma omp parallel for
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
            P[i].GravAccel[0] = 0;
            P[i].GravAccel[1] = 0;
            P[i].GravAccel[2] = 0;
#ifdef EVALPOTENTIAL
            P[i].Potential = 0;
#endif // EVALPOTENTIAL

#if defined(ADAPTGRAVSOFT) && defined(AGS_OUTPUTCORR)
            P[i].AGS_corr = correction;
#endif // ADAPTGRAVSOFT && AGS_OUTPUTCORR

        } // end for nactivepart
    } // do primary on GPU
}

void openacc_gravity_primary_loop(struct gravdata_out *LitePOut, double *PGravCosts, double *NodeGravCosts,
                                  int NodesSize)
{

    if (acc_all.go_gpu)
    {
        // for some reason nvhpc>21 hangs at the first loop if I dont make this data region
#ifdef ACC_OPENACC
#pragma acc data present(NodeGravCosts [0:NodesSize], PGravCosts [0:NumPart], LitePOut [0:NActivePart],                \
                         ActiveParticleList [0:NActivePart], NodeGravCosts [0:NodesSize],                              \
                         WalkLiteP [0:NumPart]) if (acc_all.go_gpu)
#endif // ACC_OPENACC
        {
#ifdef ACC_OPENACC
#pragma acc parallel loop if (acc_all.go_gpu) ACC_ASYNC1
#endif // ACC_OPENACC
#ifdef ACC_OPENMP
#pragma omp target teams distribute parallel for ACC_ASYNC1 // if(acc_all.go_gpu)
#endif // ACC_OPENMP
            for (int i = 0; i < NodesSize; i++)
            {
                NodeGravCosts[i] = 0.0;
            }
#ifdef ACC_OPENACC
#pragma acc parallel loop if (acc_all.go_gpu) ACC_ASYNC1
#endif // ACC_OPENACC
#ifdef ACC_OPENMP
#pragma omp target teams distribute parallel for ACC_ASYNC1 // if(acc_all.go_gpu)
#endif // ACC_OPENMP
            for (int i = 0; i < NumPart; i++)
            {
                PGravCosts[i] = 0.0;
            }

            // Nitya - activating async phase here
            VERBOSE(3, "Starting GPU region");
#ifdef ACC_OPENACC
#pragma acc update device(WalkLiteP [0:NumPart]) ACC_ASYNC1 if (acc_all.go_gpu)
#endif // ACC_OPENACC

#ifndef ACC_NOACC
#ifdef ACC_OPENACC
#pragma acc parallel loop reduction(+ : costTotal_GPU_I) if (acc_all.go_gpu) ACC_ASYNC1
#endif // ACC_OPENACC
#ifdef ACC_OPENMP
#ifdef INTEL_UPDATE
#pragma omp target teams distribute parallel for reduction(+ : costTotal_GPU_I) map(always, from : Var_updt) ACC_ASYNC1
#else // INTEL_UPDATE
#pragma omp target teams distribute parallel for nowait reduction(+ : costTotal_GPU_I)                                        \
    depend(inout : LitePOut[0 : NActivePart], NodeGravCosts[0 : NodesSize], PGravCosts[0 : NumPart])
#endif // INTEL_UPDATE
#endif // ACC_OPENMP
#else // ACC_NOACC
#pragma omp parallel for reduction(+ : costTotal_GPU_I) if (acc_all.go_gpu)
#endif // ACC_NOACC
            for (int i = 0; i < NActivePart; i++)
            {
                double *NodeGravCosts_shift = NodeGravCosts - All.MaxPart;
                int active_target = i;
                int p_target = ActiveParticleList[i];
                // struct NODE *Nodi = Nodes_base;  // Nitya why is this here? Its causing extra data transfers
                int ret = 0;
                struct gravity_walkdata *W = WalkLiteP;
#ifndef PMGRID
#ifdef PERIODIC
                if (Ewald_iter)
                {
                    ret = force_treeevaluate_ewald_correction_local(active_target, p_target, 2, W, NULL, LitePOut,
                                                                    NodeGravCosts_shift, PGravCosts);
                }
                else
#endif // PERIODIC
                {

                    ret = force_treeevaluate_local(active_target, p_target, 2, W, NULL, LitePOut, NodeGravCosts_shift,
                                                   PGravCosts);
                }
#else // PMGRID
#ifdef NEUTRINOS
                if ((W[p_target].Type != type_neutrino) || (All.Time > All.Time_tree_on_nu))
#endif // NEUTRINOS
       // Nitya - call with var update to copy back arrays only if updated on GPU
#if defined(ACC_OPENMP) && defined(INTEL_UPDATE)
                    ret = force_treeevaluate_shortrange_local(active_target, p_target, 2, W, NULL, LitePOut,
                                                              NodeGravCosts_shift, PGravCosts, Var_updt);
#else // ACC_OPENMP && INTEL_UPDATE
                ret = force_treeevaluate_shortrange_local(active_target, p_target, 2, W, NULL, LitePOut,
                                                          NodeGravCosts_shift, PGravCosts);
#endif // ACC_OPENMP && INTEL_UPDATE

#endif // PMGRID
                costTotal_GPU_I += ret;
            } // end parallel for
        }

    } // acc_all.go_gpu

    //  Costtotal += costTotal_GPU_I; we can't add costotal here if the GPU loop is async
}

// Nitya - adding 2 more variables that we allocated at the top of the loop
#ifdef ACC_OPENMP
inline void openacc_gravity_secondary_loop(int should_drift, double *NodeGravCostsII, double *PGravCostsII)
#else // ACC_OPENMP
inline void openacc_gravity_secondary_loop(int should_drift, int NodesSize)
#endif // ACC_OPENMP
{

#if defined(ACC_OPENMP) && defined(INTEL_UPDATE)
    int Var_updt = 0;
#endif // ACC_OPENMP && INTEL_UPDATE

    int DoSecondaryOnGPU = 0;
#ifdef ACC_GRAVITY_II
    DoSecondaryOnGPU = acc_all.go_gpu && (Nimport > ACC_ACTIVEPART_II_CPU);
#ifdef ACC_ASYNC_TEST_II
    int queue_test = acc_async_test(1);
    DoSecondaryOnGPU &=
        queue_test; // if first phase still in GPU, it may be better to start overlapping CPU/GPU computation now
#endif // ACC_ASYNC
#endif // ACC_GRAVITY_II

    if (!DoSecondaryOnGPU)
    {
#pragma omp parallel
        {
            int mainthreadid = omp_get_thread_num(); // can't use __thread ThisThread
            gravity_secondary_loop_vect(&mainthreadid, should_drift);
        }
    }
    else
    {

#ifdef ACC_OPENACC
#pragma acc wait(1)
#endif // ACC_OPENACC
       // Nitya - Allocate if not using OpenMP
#ifndef ACC_OPENMP
        double *NodeGravCostsII = (double *)mymalloc("NGC", sizeof(double) * NodesSize);
        double *PGravCostsII = (double *)mymalloc("PGravCosts", NumPart * sizeof(double));
#endif // ACC_OPENMP
        VERBOSE(3, "Nimport: %d, NodesSize=%d Numnodestree=%d", Nimport, NodesSize, Numnodestree);
        // Nitya - adding openmp pragmas
        // OpenMP doesnt have an API to find out the free available memory in a device, so having to transfer Nimport
        // particles for now. We can use a function call similar to acc_get_property later on to reduce the H2D transfer
        // For OpenMP offload, defined(ACC_GRAVITY_MAX_NIMPORT) && defined(OPENACC_IMPORT) is not true, so chunk_size =
        // Nimport
#ifdef ACC_OPENACC
#pragma acc data copyout(PGravCostsII [0:NumPart], NodeGravCostsII [0:NodesSize])
#elif defined(ACC_OPENMP)
#pragma omp target data map(from : PGravCostsII [0:NumPart], NodeGravCostsII [0:NodesSize])
#endif // ACC_OPENACC
        {
#ifdef ACC_OPENACC
#pragma acc parallel loop default(present)
#endif // ACC_OPENACC
#ifdef ACC_OPENMP
#pragma omp target teams distribute parallel for
#endif // ACC_OPENMP
            for (int i = 0; i < NodesSize; i++)
            {
                NodeGravCostsII[i] = 0.0;
            }
#ifdef ACC_OPENACC
#pragma acc parallel loop default(present)
#endif // ACC_OPENACC
#ifdef ACC_OPENMP
#pragma omp target teams distribute parallel for
#endif // ACC_OPENMP
            for (int i = 0; i < NumPart; i++)
            {
                PGravCostsII[i] = 0.0;
            }

            TIMER_PUSH(8, "ACC");

            double costTotal_GPU_II = 0.;
#ifdef ACC_OPENACC
#pragma acc parallel loop reduction(+ : costTotal_GPU_II) copyin(GravDataGet[0 : Nimport])                         \
    copyout(GravDataResult[0 : Nimport])
#endif // ACC_OPENACC
#ifdef ACC_OPENMP
#pragma omp target teams distribute parallel for map(to : GravDataGet[0 : Nimport])                                \
    map(from : GravDataResult[0 : Nimport]) reduction(+ : costTotal_GPU_II)
#endif // ACC_OPENMP
            for (int i = 0; i < Nimport; i++)
            {
                memset_gpu((unsigned char *)&GravDataResult[i], 0, sizeof(struct gravdata_out));
                double *NodeGravCost_shiftII = NodeGravCostsII - All.MaxPart;
                int ret = 0;
#ifndef PMGRID
#ifdef PERIODIC
                if (Ewald_iter)
                {
                    ret = force_treeevaluate_ewald_correction_local(i, i, 1, WalkLiteP, GravDataGet, GravDataResult,
                                                                    NodeGravCost_shiftII, PGravCostsII);
                }
                else
#endif // PERIODIC
                {

                    ret = force_treeevaluate_local(i, i, 1, WalkLiteP, GravDataGet, GravDataResult,
                                                   NodeGravCost_shiftII, PGravCostsII);
                }
#else // PMGRID

#if defined(ACC_OPENMP) && defined(INTEL_UPDATE)
                ret = force_treeevaluate_shortrange_local(i, i, 1, WalkLiteP, GravDataGet, GravDataResult,
                                                          NodeGravCost_shiftII, PGravCostsII, Var_updt);
#else // ACC_OPENMP && INTEL_UPDATE
                ret = force_treeevaluate_shortrange_local(i, i, 1, WalkLiteP, GravDataGet, GravDataResult,
                                                          NodeGravCost_shiftII, PGravCostsII);
#endif // ACC_OPENMP && INTEL_UPDATE
#endif // PMGRID

                costTotal_GPU_II += ret;

            } // end parallel for
            TIMER_POP(8, "ACC");
            // for every chunk and for every II phase iteration we update costtotal with the GPU counting
            Costtotal += costTotal_GPU_II;
        }

        // Niitya - Update values on host, WalkLiteP is updated for mode 1
        if (acc_all.go_gpu)
        {
#ifdef ACC_OPENMP
#ifdef INTEL_UPDATE
#pragma omp target update from(PGravCostsII [0:NumPart]) if ((Var_updt & 1) == 1)
#pragma omp target update from(NodeGravCostsII [0:NodesSize]) if ((Var_updt & 3) == 3)
#else // INTEL_UPDATE
#pragma omp target update from(PGravCostsII [0:NumPart])
#pragma omp target update from(NodeGravCostsII [0:NodesSize])
#endif // INTEL_UPDATE
#endif // ACC_OPENMP
        }

        // note that if takelevel is -1 we could avoid sending `PGravCostsII` array
        // one should check
        if (TakeLevel >= 0)
        {
#pragma omp parallel for
            for (int counter = 0; counter < NumPart; counter++)
            {
                P[counter].GravCost[TakeLevel] += PGravCostsII[counter]; // only GPU updates it
            }
        }

#pragma omp parallel for
        for (int counter = 0; counter < NodesSize; counter++)
        {
            Nodes_base[counter].GravCost += NodeGravCostsII[counter];
        }

        // Nitya delete if not using OpenMP. Else we delete it outside the loop
#ifndef ACC_OPENMP
        myfree(PGravCostsII);
        myfree(NodeGravCostsII);
#endif // ACC_OPENMP
    }
}
