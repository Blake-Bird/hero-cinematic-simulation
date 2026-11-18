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

#ifdef LT_METAL_COOLING_WAL
#define set_metallicities_gpu(i, Metallicities, grad_to_phys_fact)                                                     \
    {                                                                                                                  \
        double nonHydmass;                                                                                             \
        double mass;                                                                                                   \
        int pos, j, k;                                                                                                 \
        grad_to_phys_fact *= All.UnitDensity_in_cgs * All.HubbleParam * All.HubbleParam;                               \
        mass = P[i].Mass;                                                                                              \
        for (pos = 0, j = 0; j < WalCool_n_Ab; j++, pos++)                                                             \
            if (SpeciesIdx[j] >= 0)                                                                                    \
            {                                                                                                          \
                if (j == myHyd)                                                                                        \
                {                                                                                                      \
                    for (nonHydmass = 0, k = 0; k < LT_NMetP; k++)                                                     \
                    {                                                                                                  \
                        nonHydmass += SphP[i].Metals[k];                                                               \
                    }                                                                                                  \
                    Metallicities[SpeciesPos[j]] =                                                                     \
                        (double)(mass - nonHydmass) / mass * SphP[i].Density * grad_to_phys_fact;                      \
                }                                                                                                      \
                else                                                                                                   \
                {                                                                                                      \
                    Metallicities[SpeciesPos[j]] = (double)SphP[i].Metals[SpeciesIdx[j]] / mass;                       \
                }                                                                                                      \
                if (j == myHel && UseHeNumberRatio)                                                                    \
                {                                                                                                      \
                    double d = grad_to_phys_fact * SphP[i].Density;                                                    \
                    Metallicities[SpeciesPos[j]] = d;                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        Metallicities[HydPos] /= PROTONMASS;                                                                           \
        if (UseHeNumberRatio)                                                                                          \
        {                                                                                                              \
            Metallicities[HelPos] /= (4 * PROTONMASS * Metallicities[HydPos]);                                         \
        }                                                                                                              \
        Metallicities;                                                                                                 \
    }
#endif // LT_METAL_COOLING_WAL

inline int openacc_gradients_init(int &should_drift, int &DoPrimaryOnGPU, int *&GradientsActiveParticleList,
                                  struct graddata_out *&LiteGOut, struct graddata_in *&LiteGIn)
{

    ENDRUN_IF((acc_all.go_gpu != 0 && acc_all.go_gpu != 1), "acc_all.go_gpu should be 0 or 1 ", acc_all.go_gpu);

    DoPrimaryOnGPU = acc_all.go_gpu;
    should_drift = !acc_all.go_gpu;

    LiteGIn = (struct graddata_in *)mymalloc("LiteGI", NumPart * sizeof(struct graddata_in));

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int i = 0; i < NumPart; i++)
        {
            particle2in_gradients(&LiteGIn[i], i);
        }
    }

    GradientsActiveParticleList = (int *)mymalloc("DAPL", NActivePart * sizeof(int));

    int NGradientsActivePart = 0;

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
            if (gradients_isactive(i))
            {
                int iGradientsActivePart;
#pragma omp atomic capture
                iGradientsActivePart = NGradientsActivePart++;
                GradientsActiveParticleList[iGradientsActivePart] = i;
            }
        }
        if (DoPrimaryOnGPU)
        {
            DoPrimaryOnGPU = NGradientsActivePart > ACC_ACTIVEPART_CPU;
        }
    }
    if (DoPrimaryOnGPU)
    {
#pragma omp parallel
        {
            struct graddata_out out;
            memset(&out, 0, sizeof(struct graddata_out));
#pragma omp for
            for (int il = 0; il < NGradientsActivePart; il++)
            {
                int i = GradientsActiveParticleList[il];
                out2particle_gradients(&out, i, 0);
            }
        }
    }

    LiteGOut = (struct graddata_out *)mymalloc("LiteGO", NGradientsActivePart * sizeof(struct graddata_out));

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel
        {
            struct graddata_out out;
            memset((unsigned char *)&out, 0, sizeof(struct graddata_out));
#pragma omp for
            for (int il = 0; il < NGradientsActivePart; il++)
            {
                int i = GradientsActiveParticleList[il];
                memset(&LiteGOut[il], 0, sizeof(struct graddata_out));
                out2particle_gradients(&out, i, 0);
            }
        }
    }

    return NGradientsActivePart;
}

ACC_DECLARE_ROUTINE_BEFORE
inline int gradients_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                                    struct graddata_in *GradDataGet, struct graddata_out *GradDataResult,
                                    struct gravity_walkdata *GP, struct graddata_in *P, int i_node, int node);
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int ngb_treefind_variable_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                        struct gravity_walkdata *P, int *ngblist, int maxnumngb, int i_node);
ACC_DECLARE_ROUTINE_AFTER

void openacc_gradients_primary_phase(int DoPrimaryOnGPU, int NGradientsActivePart, int *GradientsActiveParticleList,
                                     struct graddata_out *LiteGOut, struct graddata_in *LiteGIn)
{

    if (DoPrimaryOnGPU && NGradientsActivePart > 0)
    {
        int MiniNGBList[ACC_GPU_NGB];

#ifdef ACC_OPENACC
#pragma acc parallel loop present(WalkLiteP [0:NumPart], LiteGOut [0:NGradientsActivePart]) private(MiniNGBList)       \
    async(1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for map(                                                                  \
    LiteGIn [0:NumPart], WalkLiteP [0:NumPart], LiteGOut [0:NGradientsActivePart],                                     \
    GradientsActiveParticleList [0:NGradientsActivePart]) private(MiniNGBList) // nowait
#elif defined(ACC_NOACC) // ACC_NOACC
#pragma omp parallel for private(MiniNGBList)
#endif // ACC_NOACC
        for (int i1 = 0; i1 < NGradientsActivePart; i1++)
        {
            int p_target = GradientsActiveParticleList[i1];
            int startnode = All.MaxPart;
            int endnode = startnode;
            int i_node = 0;
            while (endnode >= 0)
            {
                startnode = endnode;
                int numngb = ngb_treefind_variable_threads_local(WalkLiteP[p_target].Pos, WalkLiteP[p_target].Hsml, 2,
                                                                 &endnode, WalkLiteP, MiniNGBList, ACC_GPU_NGB, i_node);

                int ret = gradients_evaluate_local(i1, p_target, 2, MiniNGBList, numngb, NULL, LiteGOut, WalkLiteP,
                                                   LiteGIn, i_node, startnode);
                i_node++;
            }
        } // end parallel for
    } // end do primary on gpu
}

inline void openacc_gradients_secondary_phase(struct graddata_in *GradDataGet, struct graddata_out *GradDataResult,
                                              struct graddata_in *LiteGIn, int DoPrimaryOnGPU, int should_drift)
{

    int DoSecondaryOnGPU = DoPrimaryOnGPU && DoPrimaryOnGPU && (Nimport > ACC_ACTIVEPART_II_CPU);
#ifdef ACC_II_TEST
    DoSecondaryOnGPU = DoSecondaryOnGPU && acc_async_test(1);
#endif // ACC_II_TEST
#if defined(ACC_OPENMP)
    DoSecondaryOnGPU = 0;
#endif
    if (!DoSecondaryOnGPU)
    {

#pragma omp parallel
        {
            int mainthreadid = omp_get_thread_num(); // can't use __thread ThisThread
            ar_green_evaluate_secondary(GradDataGet, gradients_evaluate, Module_Gradients, TreeWalkType_GreenTreeWalk,
                                        NeighbourSearchCriteria_Density, &mainthreadid, should_drift);
        }
    }
    else if (Nimport > 0)
    { // do secondary on GPU
        {

            int MiniNGBListII[ACC_GPU_NGB];
            TIMER_PUSH(1, "ACC_II");
#ifdef ACC_OPENACC
#pragma acc wait(1)
#pragma acc parallel loop present(LiteGIn [0:N_gas]) copyin(GradDataGet [0:Nimport], DataNodeListGet [0:Nimport])      \
    copyout(GradDataResult [0:Nimport]) private(MiniNGBListII)
#elif defined(ACC_OPENMP) // ACC_OPENACC
#pragma omp taskwait
#pragma omp target parallel for map(to                                                                                 \
                                    : LiteGIn [0:N_gas]) map(to                                                        \
                                                             : GradDataGet [0:Nimport], DataNodeListGet [0:Nimport])   \
    map(from                                                                                                           \
        : GradDataResult [0:Nimport]) private(MiniNGBListII)
#endif // ACC_OPENMP

            for (int i = 0; i < Nimport; i++)
            {
                memset_gpu((unsigned char *)&GradDataResult[i], 0, sizeof(struct graddata_out));
                int i_node = 0;

                for (int node_list_i = 0; node_list_i < NODELISTLENGTH; node_list_i++)
                {
                    if (DataNodeListGet[i].NodeList[node_list_i] == -1)
                    {
                        break;
                    }
                    int startnode = DataNodeListGet[i].NodeList[node_list_i];
                    int endnode = startnode;
                    int first_iter = 1;

                    while (endnode >= 0)
                    {
                        startnode = endnode;

                        int numngb =
                            ngb_treefind_variable_threads_local(GradDataGet[i].Pos, GradDataGet[i].Hsml, 1, &endnode,
                                                                WalkLiteP, MiniNGBListII, ACC_GPU_NGB, first_iter);

                        int ret = gradients_evaluate_local(i, i, 1, MiniNGBListII, numngb, GradDataGet, GradDataResult,
                                                           WalkLiteP, LiteGIn, i_node, startnode);

                        i_node++;
                        first_iter = 0;
                    } // end while
                } // end for nodes
            } // end parallel for
            TIMER_POP(1, "ACC_II");
        } // end data//else secondary phae on GPU
    } // end do secondary on GPU
}
