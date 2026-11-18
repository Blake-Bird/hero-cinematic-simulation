/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Offloading of the conduction.
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
int ngb_treefind_pairs_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *aP, int *ngblist, int maxnumngb, int i_node);

ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int conduction_evaluate_local(int target, int p_target, int mode, double *in, MyFloat *Kappa, MyFloat *KappaB,

                              int *ngblist, int extended_numngb_inbox, struct conductiondata_in *ConductionDataGet,
                              struct conductiondata_out *ConductionDataResult, struct gravity_walkdata *P,
                              struct conductiondata_in *SphP, int node, int i_node);
ACC_DECLARE_ROUTINE_AFTER

inline int openacc_conduction_init(int &should_drift, int &DoPrimaryOnGPU, int *&ConductionActiveParticleList,
                                   struct conductiondata_out *&LiteCOut, struct conductiondata_in *&LiteCIn,
                                   double *out, double *sum)
{

    ENDRUN_IF((acc_all.go_gpu != 0 && acc_all.go_gpu != 1), "acc_all.go_gpu should be 0 or 1 ", acc_all.go_gpu);

    DoPrimaryOnGPU = acc_all.go_gpu;
    should_drift = !acc_all.go_gpu;

    LiteCIn = (struct conductiondata_in *)mymalloc("LiteDI", NumPart * sizeof(struct conductiondata_in));

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int i = 0; i < N_gas; i++)
        {
            // #ifdef BLACK_HOLES
            //             if (P[i].Type == 0 && P[i].Mass > 0)
            // #else // BLACK_HOLES
            //             if (P[i].Type == 0)
            // #endif // BLACK_HOLES
            particle2in_conduction(&LiteCIn[i], i);
        }
    }

    ConductionActiveParticleList = (int *)mymalloc("DAPL", N_gas * sizeof(int));

    int NConductionActivePart = 0;

    if (DoPrimaryOnGPU)
    {
        // conduction uses all particles from 0 to N_gas
        // BUT: maybe one day I will unify all OpenACC portings
        // and thus I still create a ActiveParticleList array as in hydro and density
#pragma omp parallel for
        for (int i = 0; i < N_gas; i++)
        {
            // #ifdef BLACK_HOLES
            //             if (P[i].Type == 0 && P[i].Mass > 0)
            // #else // BLACK_HOLES
            //             if (P[i].Type == 0)
            // #endif // BLACK_HOLES
            {
                int iConductionActivePart;
#pragma omp atomic capture
                iConductionActivePart = NConductionActivePart++;
                ConductionActiveParticleList[iConductionActivePart] = i;
            }
        }
    }

    if (DoPrimaryOnGPU)
    {
        DoPrimaryOnGPU = NConductionActivePart > ACC_ACTIVEPART_CPU;
    }

    LiteCOut =
        (struct conductiondata_out *)mymalloc("LiteCO", NConductionActivePart * sizeof(struct conductiondata_out));

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel
        {
            struct conductiondata_out myout;
            memset(&myout, 0, sizeof(struct conductiondata_out));
#pragma omp for
            for (int il = 0; il < NConductionActivePart; il++)
            {
                int i = ConductionActiveParticleList[il];
                memset(&LiteCOut[il], 0, sizeof(struct conductiondata_out));
                out2particle_conduction(&myout, i, 0);
            }
        }
    }
    return NConductionActivePart;
}

inline void openacc_conduction_primary_phase(int DoPrimaryOnGPU, int NConductionActivePart,
                                             int *ConductionActiveParticleList, struct conductiondata_out *LiteCOut,
                                             conductiondata_in *LiteCIn, double *in, MyFloat *Kappa, MyFloat *KappaB)
{

    if (DoPrimaryOnGPU)
    {
        int MiniNGBList[ACC_GPU_NGB_CONDUCTION];
        for (int i = 0; i < ACC_GPU_NGB_CONDUCTION; i++)
        {
            MiniNGBList[i] = 0;
        }
        VERBOSE(3, "Conduction on GPU NumPart=%d N_gas=%d NActivePAst=%d", NumPart, N_gas, NConductionActivePart);

#if defined(ACC_OPENACC)
#pragma acc parallel loop present(WalkLiteP [0:N_gas], COPY_KAPPAB Kappa [0:N_gas], in [0:N_gas]) private(MiniNGBList) \
    async(1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for private(MiniNGBList) nowait
#elif defined(ACC_NOACC) // ACC_NOACC
#pragma omp parallel for private(MiniNGBList)
#endif // ACC_NOACC
        for (int il = 0; il < NConductionActivePart; il++)
        {
            int p_target = ConductionActiveParticleList[il];
            int startnode = All.MaxPart;
            int endnode = startnode;
            int i_node = 0;

            while (endnode >= 0)
            {
                startnode = endnode;
                int numngb =
                    ngb_treefind_pairs_threads_local(WalkLiteP[p_target].Pos, WalkLiteP[p_target].Hsml, 2, &endnode,
                                                     WalkLiteP, MiniNGBList, ACC_GPU_NGB_CONDUCTION, i_node);
                int ret = conduction_evaluate_local(il, p_target, 2, in, Kappa, KappaB, MiniNGBList, numngb, NULL,
                                                    LiteCOut, WalkLiteP, LiteCIn, i_node, startnode);

                i_node++;
            }
        } // end parallel for
    } // end do primary on gpu

    VERBOSE(3, "Conduction on GPU NumPart=%d N_gas=%d NActivePAst=%d done", NumPart, N_gas, NConductionActivePart);
}

inline void openacc_conduction_secondary_phase(struct conductiondata_in *ConductionDataGet,
                                               struct conductiondata_out *ConductionDataResult,
                                               struct conductiondata_in *LiteCIn, int DoPrimaryOnGPU, int should_drift,
                                               double *in, MyFloat *Kappa, MyFloat *KappaB, double *out, double *sum)
{

    int DoSecondaryOnGPU = DoPrimaryOnGPU && (Nimport > ACC_ACTIVEPART_II_CPU);
#ifdef ACC_II_TEST
    DoSecondaryOnGPU = DoSecondaryOnGPU && acc_async_test(1);
#endif
#if defined(ACC_OPENMP)
    DoSecondaryOnGPU = 0;
#endif
    if (!DoSecondaryOnGPU)
    {
        VERBOSE(1, " ar_conduction_evaluate_secondary(%d)m Nimport=%d loop on CPU", !acc_all.go_gpu, Nimport);
#pragma omp parallel
        {
            struct thread_params threadparams;
            threadparams.tid = omp_get_thread_num(); // can't use __thread ThisThread
            threadparams.in = in;
            threadparams.out = out;
            threadparams.sum = sum;

            ar_green_evaluate_secondary(ConductionDataGet, conduction_evaluate, Module_Conduction,
                                        TreeWalkType_GreenTreeWalk, NeighbourSearchCriteria_Hydro, &threadparams,
                                        should_drift);
        }
    }
    else if (Nimport > 0)
    { // do secondary on GPU
        {

            int MiniNGBListII[ACC_GPU_NGB_CONDUCTION];
            TIMER_PUSH(1, "ACC_II");
#if defined(ACC_OPENACC)
#pragma acc wait(1)
#pragma acc parallel loop copyin(ConductionDataGet [0:Nimport], DataNodeListGet [0:Nimport])                           \
    present(LiteCIn [0:N_gas], in [0:N_gas], COPY_KAPPAB Kappa [0:N_gas])                                              \
        copyout(ConductionDataResult [0:Nimport]) private(MiniNGBListII)
#elif defined(ACC_OPENMP) // ACC_OPENACC
#pragma omp taskwait
#pragma omp target teams distribute parallel for private(MiniNGBListII)                                                \
    map(to                                                                                                             \
        : ConductionDataGet [0:Nimport], DataNodeListGet [0:Nimport])                                                  \
        map(from                                                                                                       \
            : ConductionDataResult [0:Nimport]) // nowait Nitya nowait no implemented yet
#endif // ACC_OPENMP

            for (int i = 0; i < Nimport; i++)
            {
                memset_gpu((unsigned char *)&ConductionDataResult[i], 0, sizeof(struct conductiondata_out));

                int i_node = 0;

                for (int node_list_i = 0; node_list_i < NODELISTLENGTH; node_list_i++)
                {
                    if (DataNodeListGet[i].NodeList[node_list_i] == -1)
                        break;
                    int startnode = DataNodeListGet[i].NodeList[node_list_i];
                    int endnode = startnode;
                    int first_iter = 1;

                    while (endnode >= 0)
                    {
                        startnode = endnode;

                        int numngb = ngb_treefind_pairs_threads_local(
                            ConductionDataGet[i].Pos, ConductionDataGet[i].Hsml, 1, &endnode, WalkLiteP, MiniNGBListII,
                            ACC_GPU_NGB_CONDUCTION, first_iter);
                        int ret = conduction_evaluate_local(i, i, 1, in, Kappa, KappaB, MiniNGBListII, numngb,
                                                            ConductionDataGet, ConductionDataResult, WalkLiteP, LiteCIn,
                                                            i_node, startnode);

                        i_node++;
                        first_iter = 0;
                    } // end while
                } // end for nodes
            } // end parallel for
            TIMER_POP(1, "ACC_II");

        } // end data//else secondary phae on GPU
    } // end do secondary on GPU
}
