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
#endif // ACC_OPENMP

ACC_DECLARE_ROUTINE_BEFORE
int ngb_treefind_pairs_threads_local(MyLongDouble *searchcenter, MyFloat hsml, int mode, int *startnode,
                                     struct gravity_walkdata *aP, int *ngblist, int maxnumngb, int i_node);
ACC_DECLARE_ROUTINE_AFTER

#if GADGET_HYDRO == HYDRO_MFM
ACC_DECLARE_ROUTINE_BEFORE
int flux_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                        struct hydrodata_in *HydroDataGet, struct hydrodata_out *HydroDataResult,
                        struct gravity_walkdata *P, struct hydrodata_in *SphP, int i_node, int node);
ACC_DECLARE_ROUTINE_AFTER
#else // GADGET_HYDRO
ACC_DECLARE_ROUTINE_BEFORE
int hydro_evaluate_local(int active_target, int p_target, int mode, int *ngblist, int numngb,
                         struct hydrodata_in *HydroDataGet, struct hydrodata_out *HydroDataResult,
                         struct gravity_walkdata *P, struct hydrodata_in *, int i_node, int node);
ACC_DECLARE_ROUTINE_AFTER
#endif // GADGET_HYDRO

ACC_DECLARE_ROUTINE_BEFORE
void hydra_evaluate_shockfinder_openacc(struct hydrodata_in *local, struct hydrodata_out *out,
                                        struct kernel_hydra *kernel, int j, struct gravity_walkdata *P,
                                        struct hydrodata_in *SphP);
ACC_DECLARE_ROUTINE_AFTER

inline int openacc_hydra_init(int &should_drift, int &DoPrimaryOnGPU, int *&HydraActiveParticleList,
                              struct hydrodata_out *&LiteHOut, struct hydrodata_in *&LiteHIn)
{

    ENDRUN_IF((acc_all.go_gpu != 0 && acc_all.go_gpu != 1), "acc_all.go_gpu should be 0 or 1 ", acc_all.go_gpu);

    DoPrimaryOnGPU = acc_all.go_gpu;
    should_drift = !acc_all.go_gpu;

    LiteHIn = (struct hydrodata_in *)mymalloc("LiteDI", N_gas * sizeof(struct hydrodata_in));

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int i = 0; i < N_gas; i++)
        {
#ifdef BLACK_HOLES
            if (P[i].Type == 0 && P[i].Mass > 0)
#else // BLACK_HOLES
            if (P[i].Type == 0)
#endif // BLACK_HOLES
                particle2in_hydra(&LiteHIn[i], i);
        }
    }

    HydraActiveParticleList = (int *)mymalloc("DAPL", NActivePart * sizeof(int));

    int NHydraActivePart = 0;

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel for
        for (int il = 0; il < NActivePart; il++)
        {
            int i = ActiveParticleList[il];
#ifdef BLACK_HOLES
            if (P[i].Type == 0 && P[i].Mass > 0)
#else // BLACK_HOLES
            if (P[i].Type == 0)
#endif // BLACK_HOLES
            {
                int iHydraActivePart;
#pragma omp atomic capture
                iHydraActivePart = NHydraActivePart++;
                HydraActiveParticleList[iHydraActivePart] = i;
            }
        }

        if (DoPrimaryOnGPU)
        {
            DoPrimaryOnGPU = NHydraActivePart > ACC_ACTIVEPART_CPU;
        }
    }
    if (DoPrimaryOnGPU) // we need this re-update because between density and hydro we update hmax between nodes
    {
        // update_walk_lite_p();
        // update_openacc_structures();
        ACC_PRAGMA(ACC_UPDATE_DEVICE(Nodes_base [0:NodesSize]));
    }

    LiteHOut = (struct hydrodata_out *)mymalloc("LiteHO", NHydraActivePart * sizeof(struct hydrodata_out));

    if (DoPrimaryOnGPU)
    {
#pragma omp parallel
        {
            struct hydrodata_out out;
            memset((unsigned char *)&out, 0, sizeof(struct hydrodata_out));
#pragma omp for
            for (int il = 0; il < NHydraActivePart; il++)
            {
                int i = HydraActiveParticleList[il];
#ifdef AR_FIX_ASSIGN
                P[i].assigned = 0;
#endif // AR_FIX_ASSIGN
                memset(&LiteHOut[il], 0, sizeof(struct hydrodata_out));
                out2particle_hydra(&out, i, 0);
            }
        }
    }

    return NHydraActivePart;
}

inline void openacc_hydra_primary_phase(int DoPrimaryOnGPU, int NHydraActivePart, int *HydraActiveParticleList,
                                        struct hydrodata_out *LiteHOut, hydrodata_in *LiteHIn)
{
    if (DoPrimaryOnGPU && NHydraActivePart > 0)
    {
        int MiniNGBList[ACC_GPU_NGB];
        for (int i = 0; i < ACC_GPU_NGB; i++)
        {
            MiniNGBList[i] = 0;
        }

#if defined(ACC_OPENACC)
#pragma acc parallel loop present(WalkLiteP [0:N_gas], LiteHOut [0:NHydraActivePart]) private(MiniNGBList) async(1)
#elif defined(ACC_OPENMP)
#pragma omp target teams distribute parallel for map(WalkLiteP [0:N_gas], LiteHOut [0:NHydraActivePart],               \
                                                     HydraActiveParticleList                                           \
                                                     [0:NHydraActivePart]) private(MiniNGBList) nowait
#elif defined(ACC_NOACC) // ACC_NOACC
#pragma omp parallel for private(MiniNGBList) if (DoPrimaryOnGPU)
#endif // ACC_NOACC
        for (int i1 = 0; i1 < NHydraActivePart; i1++)
        {
            int p_target = HydraActiveParticleList[i1];
            int startnode = All.MaxPart;
            int endnode = startnode;
            int i_node = 0;
            while (endnode >= 0)
            {
                startnode = endnode;
                int numngb = ngb_treefind_pairs_threads_local(WalkLiteP[p_target].Pos, WalkLiteP[p_target].Hsml, 2,
                                                              &endnode, WalkLiteP, MiniNGBList, ACC_GPU_NGB, i_node);
#if GADGET_HYDRO == HYDRO_MFM
                int ret = flux_evaluate_local(i1, p_target, 2, MiniNGBList, numngb, NULL, LiteHOut, WalkLiteP, LiteHIn,
                                              i_node, startnode);
#else // GADGET_HYDRO
                int ret = hydro_evaluate_local(i1, p_target, 2, MiniNGBList, numngb, NULL, LiteHOut, WalkLiteP, LiteHIn,
                                               i_node, startnode);
#endif // GADGET_HYDRO
                i_node++;
            }
        } // end parallel for
    } // end do primary on gpu
}

inline void openacc_hydra_secondary_phase(struct hydrodata_in *HydroDataGet, struct hydrodata_out *HydroDataResult,
                                          struct hydrodata_in *LiteHIn, int DoPrimaryOnGPU, int should_drift)
{

    int DoSecondaryOnGPU = DoPrimaryOnGPU && (Nimport > ACC_ACTIVEPART_II_CPU);
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
#if GADGET_HYDRO == HYDRO_MFM
            ar_green_evaluate_secondary(HydroDataGet, flux_evaluate, Module_Hydro, TreeWalkType_GreenTreeWalk,
                                        NeighbourSearchCriteria_Hydro, &mainthreadid, should_drift);
#else // GADGET_HYDRO
            ar_green_evaluate_secondary(HydroDataGet, hydro_evaluate, Module_Hydro, TreeWalkType_GreenTreeWalk,
                                        NeighbourSearchCriteria_Hydro, &mainthreadid, should_drift);
#endif // GADGET_HYDRO
        }
    }
    else if (Nimport > 0)
    { // do secondary on GPU
        {

            int MiniNGBListII[ACC_GPU_NGB];
            TIMER_PUSH(1, "ACC_II");
#if defined(ACC_OPENACC)
#pragma acc wait(1)
#pragma acc parallel loop present(LiteHIn [0:N_gas]) copyin(HydroDataGet [0:Nimport], DataNodeListGet [0:Nimport])     \
    copyout(HydroDataResult [0:Nimport]) private(MiniNGBListII)
#elif defined(ACC_OPENMP) // ACC_OPENACC
#pragma omp taskwait
#pragma omp target teams distribute parallel for map(LiteHIn [0:N_gas])                                                \
    map(to                                                                                                             \
        : HydroDataGet [0:Nimport], DataNodeListGet [0:Nimport])                                                       \
        map(from                                                                                                       \
            : HydroDataResult [0:Nimport]) private(MiniNGBListII)
            // nowait Nitya nowait no implemented yet
#endif // ACC_OPENMP

            for (int i = 0; i < Nimport; i++)
            {
                memset_gpu((unsigned char *)&HydroDataResult[i], 0, sizeof(struct hydrodata_out));

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
                            ngb_treefind_pairs_threads_local(HydroDataGet[i].Pos, HydroDataGet[i].Hsml, 1, &endnode,
                                                             WalkLiteP, MiniNGBListII, ACC_GPU_NGB, first_iter);
#if GADGET_HYDRO == HYDRO_MFM
                        int ret = flux_evaluate_local(i, i, 1, MiniNGBListII, numngb, HydroDataGet, HydroDataResult,
                                                      WalkLiteP, LiteHIn, i_node, startnode);
#else // GADGET_HYDRO
                        int ret = hydro_evaluate_local(i, i, 1, MiniNGBListII, numngb, HydroDataGet, HydroDataResult,
                                                       WalkLiteP, LiteHIn, i_node, startnode);
#endif // GADGET_HYDRO

                        i_node++;
                        first_iter = 0;
                    } // end while
                } // end for nodes
            } // end parallel for
            TIMER_POP(1, "ACC_II");

        } // end data//else secondary phae on GPU
    } // end do secondary on GPU
}
