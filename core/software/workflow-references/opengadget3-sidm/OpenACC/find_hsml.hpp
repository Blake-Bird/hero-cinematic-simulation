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
inline int hsml_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                               struct hsmldata_in *HsmlDataGet, struct hsmldata_out *HsmlDataResult,
                               struct gravity_walkdata *GP, struct hsmldata_in *P, int i_node, int node);
ACC_DECLARE_ROUTINE_AFTER

inline struct hsmldata_in *openacc_hsml_init(int &should_drift, int &DoPrimaryFirstIterationOnGPU)
{

    TIMER_PUSH(8, "ACC-hsml_init");

    if (acc_all.go_gpu == -1)
    {
        // first iteration starts with a density smoothing length finder:
        update_walk_lite_p();

        acc_all.go_gpu = 1;
        // next time we will run or not on GPU depending on what the gravity loop decided.
    }

    if (!acc_all.inside_data_region)
    {
        update_openacc_structures();
    }

    DoPrimaryFirstIterationOnGPU = acc_all.go_gpu;
    should_drift = !acc_all.go_gpu;

    struct hsmldata_in *LiteDIn = (hsmldata_in *)mymalloc("LiteDIn", NumPart * sizeof(struct hsmldata_in));

    if (DoPrimaryFirstIterationOnGPU)
    {
#pragma omp parallel for
        for (int i = 0; i < NumPart; i++)
        {
            WalkLiteP[i].Hsml = P[i].Hsml;
            particle2in_hsml(&LiteDIn[i], i);
        }
    }

    TIMER_POP(8, "ACC-hsml_init");

    return LiteDIn;
}

inline int openacc_hsml_iteration_init(int &should_drift, int &DoPrimaryFirstIterationOnGPU, int &DoPrimaryOnGPU,
                                       int *&DensityActiveParticleList, struct hsmldata_out *&LiteDOut, int iter)
{

    DoPrimaryFirstIterationOnGPU = acc_all.go_gpu;
    should_drift = !acc_all.go_gpu;

    DensityActiveParticleList = (int *)mymalloc("DAPL", NActivePart * sizeof(int));
    int NDensityActivePart = 0;

    if (iter > 0)
    {
        DoPrimaryOnGPU = DoPrimaryOnGPU && DoPrimaryFirstIterationOnGPU;
    }
    else
    {
        DoPrimaryOnGPU = DoPrimaryFirstIterationOnGPU;
    }

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
            if (hsml_isactive(i))
            {
                int iDensityActivePart;
#pragma omp atomic capture
                iDensityActivePart = NDensityActivePart++;
                DensityActiveParticleList[iDensityActivePart] = i;
            }
        }
    }
    if (DoPrimaryOnGPU)
    {
        DoPrimaryOnGPU = NDensityActivePart > ACC_ACTIVEPART_CPU;
    }

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel
        {
            struct hsmldata_out out;
            memset(&out, 0, sizeof(struct hsmldata_out));

#pragma omp for
            for (int il = 0; il < NDensityActivePart; il++)
            {
                int i = DensityActiveParticleList[il];
                out2particle_hsml(&out, i, 0);
            }
        }

        LiteDOut = (struct hsmldata_out *)mymalloc("LiteDO", NDensityActivePart * sizeof(struct hsmldata_out));
        // let's zero all output-data.
    }

    return NDensityActivePart;
}

ACC_DECLARE_ROUTINE_BEFORE
int density_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                           struct hsmldata_in *HsmlDataGet, struct hsmldata_out *HsmlDataResult,
                           struct gravity_walkdata *GP, struct hsmldata_in *P, int i_node, int node);
ACC_DECLARE_ROUTINE_AFTER

ACC_DECLARE_ROUTINE_BEFORE
int ngb_treefind_variable_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                        struct gravity_walkdata *P, int *ngblist, int maxnumngb, int i_node);
ACC_DECLARE_ROUTINE_AFTER

void openacc_hsml_first_phase(int DoPrimaryOnGPU, int NDensityActivePart, int *HsmlActiveParticleList,
                              struct hsmldata_out *LiteDOut, struct hsmldata_in *LiteDIn)
{

    if (DoPrimaryOnGPU && NDensityActivePart > 0)
    {

#ifndef ACC_NOACC
        int MiniNGBList[ACC_GPU_NGB];

#ifdef ACC_OPENACC
#pragma acc parallel loop gang vector present(WalkLiteP [0:N_gas],                                                     \
                                              LiteDOut [0:NDensityActivePart]) private(MiniNGBList) async(1)
#elif defined(ACC_OPENMP)

#pragma omp target teams distribute parallel for map(to                                                                \
                                                     : WalkLiteP [0:N_gas],                                            \
                                                       HsmlActiveParticleList [0:NDensityActivePart])                  \
    map(tofrom                                                                                                         \
        : LiteDOut [0:NDensityActivePart]) private(MiniNGBList) nowait
#endif // ACC_OPENMP
#else // ACC_NOACC
#pragma omp parallel for
#endif // ACC_NOACC

        for (int i = 0; i < NDensityActivePart; i++)
        {

#ifdef ACC_NOACC
            int MiniNGBList[ACC_GPU_NGB];
#endif // ACC_NOACC
            int p_target = HsmlActiveParticleList[i];
            int startnode = All.MaxPart;
            int endnode = startnode;
            int i_node = 0;

            while (endnode >= 0)
            {
#ifdef FOLLOW_ID
                if (LiteDIn[p_target].ID == FOLLOW_ID)
                {
                    VERBOSE(
                        0, "FOLLOW_ID %d %s %d, start ngb_treefind_variable_threads_local node=%d inode=%d, rho = %g\n",
                        ThisTask, __FILE__, __LINE__, endnode, i_node, LiteDOut[i].Rho);
                }
#endif // FOLLOW_ID
                startnode = endnode;
                int numngb = ngb_treefind_variable_threads_local(WalkLiteP[p_target].Pos, WalkLiteP[p_target].Hsml, 2,
                                                                 &endnode, WalkLiteP, MiniNGBList, ACC_GPU_NGB, i_node);

                int ret = hsml_evaluate_local(i, p_target, 2, MiniNGBList, numngb, NULL, LiteDOut, WalkLiteP, LiteDIn,
                                              i_node, startnode);

                // int ret = 0;
                i_node++;
            }
            // mode is 2 because the computation is very ismilar to the second phae (mode==1) where you have
            // datain/dataout structures and the local P array
            //   break;
        } // end parallel for
    }
}

inline void openacc_hsml_secondary_phase(struct hsmldata_in *HsmlDataGet, struct hsmldata_out *HsmlDataResult,
                                         struct hsmldata_in *LiteDIn, int DoPrimaryOnGPU, int should_drift)
{

    int DoSecondaryOnGPU =
        DoPrimaryOnGPU && DoPrimaryOnGPU && (Nimport > ACC_ACTIVEPART_II_CPU); // && acc_async_test(1);
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
            ar_green_evaluate_secondary(HsmlDataGet, hsml_evaluate, Module_HSML, TreeWalkType_GreenTreeWalk,
                                        NeighbourSearchCriteria_Density, &mainthreadid, should_drift);
        }
    }
    else if (Nimport > 0)
    {
        int MiniNGBListII[ACC_GPU_NGB];

        TIMER_PUSH(1, "ACC_II");
#ifdef ACC_OPENACC
#pragma acc wait(1)
#pragma acc parallel loop gang vector copyin(HsmlDataGet [0:Nimport], DataNodeListGet [0:Nimport])                     \
    present(WalkLiteP [0:NumPart]) present_or_copy(LiteDIn [0:NumPart])                                                \
        copyout(HsmlDataResult [0:Nimport]) private(MiniNGBListII)
#elif defined(ACC_OPENMP)
#pragma omp taskwait
#pragma omp target teams distribute parallel for map(WalkLiteP [0:NumPart], LiteDIn [0:NumPart])                       \
    map(to                                                                                                             \
        : HsmlDataGet [0:Nimport], DataNodeListGet [0:Nimport])                                                        \
        map(from                                                                                                       \
            : HsmlDataResult [0:Nimport]) private(MiniNGBListII)
#endif // ACC_OPENMP

        for (int i = 0; i < Nimport; i++)
        {
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

                    int numngb = ngb_treefind_variable_threads_local(
                        HsmlDataGet[i].Pos, HsmlDataGet[i].Hsml, 1, &endnode, WalkLiteP,
                        /* WalkLiteSPH,*/ MiniNGBListII, ACC_GPU_NGB, first_iter);

                    int ret = hsml_evaluate_local(i, i, 1, MiniNGBListII, numngb, HsmlDataGet, HsmlDataResult,
                                                  WalkLiteP, LiteDIn, i_node, startnode);

                    i_node++;
                    first_iter = 0;
                } // end while
            } // end for nodes
        } // end parallel for
        TIMER_POP(1, "ACC_II");
    }
}
