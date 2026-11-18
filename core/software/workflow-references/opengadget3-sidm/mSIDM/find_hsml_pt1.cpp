/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Definitions for the dark matter kernel size determination.
 *
 * @author OpenGadget3-development team.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "find_hsml_pt1.hpp"
#include "../System/communication.hpp"
#include "../CodeBase/greentree.hpp"
#include "tools.hpp"

#define HSML_DEBUG 43728420
#define HSML_TOLERANCE 0.0001

using namespace scatter;
using namespace scatter::tools;
using namespace scatter::pxp_execute;

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
treewalk_filter_dm_only_class treewalk_filter_dm_only;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/** Data structures for MPI communication */
struct hsmldata_in *HsmlDataIn, *HsmlDataGet;
struct hsmldata_out *HsmlDataResult, *HsmlDataOut;
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of reference class for hsmldata_in.
 */
hsmldata_in_ref::hsmldata_in_ref(struct hsmldata_in &p_in)
    : Pos(p_in.Pos), Hsml(p_in.Hsml)
#ifdef TYPE_FOR_DATASTRUCT
      ,
      Type(p_in.Type)
#endif // TYPE_FOR_DATASTRUCT
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of reference class for hsmldata_in.
 */
hsmldata_in_ref::hsmldata_in_ref(struct particle_data &p_in)
    : Pos(p_in.Pos), Hsml(p_in.Hsml)
#ifdef TYPE_FOR_DATASTRUCT
      ,
      Type(p_in.Type)
#endif // TYPE_FOR_DATASTRUCT
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to copy particle data to class.
 */
void particle2in_pt1_hsml(struct hsmldata_in *in, int i)
{
    for (int k = 0; k < 3; ++k)
        in->Pos[k] = P[i].Pos[k];

    in->Hsml = P[i].Hsml;
#ifdef TYPE_FOR_DATASTRUCT
    in->Type = P[i].Type;
#endif // TYPE_FOR_DATASTRUCT
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to copy data from class to global particle data.
 */
void out2particle_pt1_hsml(struct hsmldata_out *out, int i, int mode)
{
    ASSIGN_ADD(P[i].TrueNGB, out->TrueNGB, mode);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This struct contains the properties for the hsml evaluation.
 */
struct kernel_pt1_hsml
{
    MyAtLeastDouble dx, dy, dz;
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Evaluate the kernel size (HSML) for a dark matter particle.
 *
 * This function calculates the kernel size (HSML) for a particle specified by the target index. The HSML is
 * evaluated based on the particle's neighbors within the specified search radius. The target particle may either be
 * local, or reside in the communication buffer. The computation can be performed in different modes:
 * - Mode 0: Regular evaluation of HSML.
 * - Mode 1: Special mode for handling active particles.
 * - Mode 2: Special mode for handling target particles.
 * - Mode 3: Count the number of neighbors within the search radius without evaluating HSML.
 *
 * @param target The index of the target particle for which to evaluate HSML.
 * @param mode The evaluation mode (0, 1, 2, or 3).
 * @param ngblist An array containing the indices of neighbors of the target particle.
 * @param extended_numngb_inbox The number of extended neighbors in the inbox.
 * @param i_startnode The start node index for the neighbor search.
 * @param node The node index.
 * @param i_node The node index for the target particle.
 * @param data Additional data pointer.
 * @return An integer indicating the success of the evaluation process.
 */
#ifndef ADAPTGRAVSOFT
inline int scatter::pt1_hsml_evaluate(int target, int mode, int *ngblist, int extended_numngb_inbox, int *i_startnode,
                                      int node, int i_node, void *data)
{
    int startnode = node;
    int numngb;

    struct kernel_pt1_hsml kernel;
    struct hsmldata_out out;
    memset(&out, 0, sizeof(struct hsmldata_out));

    if (i_node > 0)
    {
        out = HsmlDataResult[target];
    }

    auto inner_eval = [&startnode, &numngb, &i_startnode, &extended_numngb_inbox, &ngblist, &kernel,
                       &out](hsmldata_in_ref &local) -> int {
        const MyAtLeastDouble h2 = local.Hsml * local.Hsml;

        while (startnode >= 0)
        {
            if (extended_numngb_inbox == -1)
            {
                return -1;
            }

            numngb = extended_numngb_inbox;
            startnode = *i_startnode;

#if defined(DENSITY_LESS_NGB_FACTOR) && !defined(ACC_INCLUDE_DENSITY)
            if (numngb > NumPart / DENSITY_LESS_NGB_FACTOR)
            {
                cout(error) << "pt1_hsml: Number of neighbours found (" << numngb << ") exceeds allowed number ("
                            << (NumPart / DENSITY_LESS_NGB_FACTOR)
                            << ") derived from NumPart/DENSITY_LESS_NGB_FACTOR, to continue, decreas "
                               "DENSITY_LESS_NGB_FACTOR (currently "
                            << DENSITY_LESS_NGB_FACTOR << ", min possible val would be 1) !!" << endl;
                ENDRUN_MAIN("12198");
            }
#endif // DENSITY_LESS_NGB_FACTOR && !ACC_INCLUDE_DENSITY

            for (int n = 0; n < numngb; ++n)
            {
                const int &j = ngblist[n];

                kernel.dx = local.Pos[0] - P[j].Pos[0];
                kernel.dy = local.Pos[1] - P[j].Pos[1];
                kernel.dz = local.Pos[2] - P[j].Pos[2];

#ifdef PERIODIC /*  now find the closest image in the given box size  */
                kernel.dx = NEAREST_X(kernel.dx);
                kernel.dy = NEAREST_Y(kernel.dy);
                kernel.dz = NEAREST_Z(kernel.dz);
#endif // PERIODIC
                const MyAtLeastDouble r2 = kernel.dx * kernel.dx + kernel.dy * kernel.dy + kernel.dz * kernel.dz;

                if (r2 < h2)
                {
                    ++out.TrueNGB;
                }
            }
        }
        return 0;
    };

    if (mode == 0 || mode == 2 || mode == 3)
    {
        class hsmldata_in_ref local(P[target]);
        if (inner_eval(local) == -1)
            return -1;
    }
    else
    {
        class hsmldata_in_ref local(HsmlDataGet[target]);
        if (inner_eval(local) == -1)
            return -1;
    }

    if (mode == 3)
        return 0;

    if (mode == 0)
    {
        out2particle_pt1_hsml(&out, target, 0);
    }
    else
    {
        HsmlDataResult[target] = out;
    }
    return 0;
}
#endif // !ADAPTGRAVSOFT

///////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the kernel size for each active dark matter particle.
 */
#ifndef ADAPTGRAVSOFT
void scatter::pt1_find_hsml()
{
    TIMER_PUSH(1, "HSML_PT1");
    MyFloat *Left, *Right;
    MyFloat *LeftN, *RightN;
    int *LeftNtrue, *RightNtrue;
    int ndone, ndone_flag, iter = 0;
    int ngrp, recvTask, place;
    long long ntot = 0;
    int save_NextParticle;

    /* Here we set the limits for the NGB search:
       Defaults are always set to allow any possible number
       Therefore all combinations of NGB and weigthed NGB
       can be implemented */
    MyFloat DesN[6], MinHsml[6], MaxHsml[6];
    int MinNtrue[6], MaxNtrue[6];

    for (int i = 0; i < 6; ++i)
    {
        MinNtrue[i] = MinHsml[i] = DesN[i] = 0;
        MaxNtrue[i] = NumPart;
        MaxHsml[i] = All.BoxSize;
    }

    DesN[1] = All.DesNumNgb_pt1;

    MinNtrue[1] = All.DesNumNgb_pt1 - All.MaxNumNgbDeviation_pt1;
    MaxNtrue[1] = All.DesNumNgb_pt1 + All.MaxNumNgbDeviation_pt1;

    MinHsml[1] = 0.0;
    MaxHsml[1] = 1.0 / 0.0;

    int NTaskTimesNumPart;
    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer;", NTaskTimesNumPart * sizeof(int));

    Left = (MyFloat *)mymalloc("Left", NumPart * sizeof(MyFloat));
    Right = (MyFloat *)mymalloc("Right", NumPart * sizeof(MyFloat));
    LeftN = (MyFloat *)mymalloc("LeftN", NumPart * sizeof(MyFloat));
    RightN = (MyFloat *)mymalloc("RightN", NumPart * sizeof(MyFloat));
    LeftNtrue = (int *)mymalloc("LeftNtrue", NumPart * sizeof(int));
    RightNtrue = (int *)mymalloc("RightNtrue", NumPart * sizeof(int));

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if (pt1_hsml_isactive(P[i].TimeBin, P[i].Type))
        {
            Left[i] = Right[i] = LeftN[i] = RightN[i] = 0;
            LeftNtrue[i] = RightNtrue[i] = 0;
        }
    }

    /* allocate buffers to arrange communication */

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize =
        (int)((MyBufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + 3 * sizeof(struct data_nodelist) + sizeof(struct hsmldata_in) +
               sizeof(struct hsmldata_out) + sizemax(sizeof(struct hsmldata_in), sizeof(struct hsmldata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    coutr(extra) << "PT1 Hsml: using " << MyBufferSize / 1.0 << " MB for buffering, " << FreeBytes / (1024.0 * 1024.0)
                 << " MB remaining for other buffers" << endl;

    int DoPrimaryOnGPU = 0;
    int should_drift = 1;

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */

    do
    {
        NextParticle = FirstActiveParticle; /* beginn with this index */
        int group_particles = 1;
        do
        {
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextParticle = NextParticle;

            TIMER_PUSH(1, "HSMLCOMPUTE");
#ifdef PXP_OPENMP
#pragma omp parallel
#endif // PXP_OPENMP
            {
                auto isactive_pt1 = [](int n) -> int { return pt1_hsml_isactive(P[n].TimeBin, P[n].Type); };

                auto treewalk_filter_pt1 = [](particle_data *P, int p, void *DataGet, int *targets, int n_targets,
                                              enum IterationPhase mode) -> int {
                    return treewalk_filter_dm_only(P[p].Type);
                };

                ar_green_evaluate_primary(isactive_pt1, pt1_hsml_evaluate, Module_Hsml_pt1, TreeWalkType_GreenTreeWalk,
                                          NextParticleLoop_ActiveParticleList, NeighbourSearchCriteria_Hsml_pt1,
                                          group_criteria_distance, treewalk_filter_pt1, &ThisThread, DoPrimaryOnGPU,
                                          should_drift, group_particles);
            }

            TIMER_POP(1, "HSMLCOMPUTE");
            /* if we are not grouping particles and SendRecv fails, then we panic as there is no memory to process even
             * a single particles */
            /* and if green tree fails, we retry without it */

            int panic_on_error = !group_particles;
            int is_send_receive_ok = prepare_sendrec_offset_send(save_NextParticle, panic_on_error);
            if (!is_send_receive_ok)
            {
                group_particles = 0;
                continue;
            }

            HsmlDataGet = (struct hsmldata_in *)mymalloc("HsmlDataGet", Nimport * sizeof(struct hsmldata_in));
            DataNodeListGet =
                (struct data_nodelist *)mymalloc("DataNodeListGet", Nimport * sizeof(struct data_nodelist));
            HsmlDataIn = (struct hsmldata_in *)mymalloc("HsmlDataIn", Nexport * sizeof(struct hsmldata_in));
            DataNodeListIn = (struct data_nodelist *)mymalloc("DataNodeListIn", Nexport * sizeof(struct data_nodelist));

            /* prepare particle data for export */
            for (int j = 0; j < Nexport; ++j)
            {
                place = DataIndexTable[j].Index;
                particle2in_pt1_hsml(&HsmlDataIn[j], place);
            }

            copy_data_nodelist();

            /* exchange particle data */
            TIMER_PUSH(1, "HSML_COMM");
            templated_exchange_data_in_get(HsmlDataIn, HsmlDataGet, 1, TAG_N);
            TIMER_POP(1, "HSML_COMM");

            myfree(DataNodeListIn);
            myfree(HsmlDataIn);
            HsmlDataResult = (struct hsmldata_out *)mymalloc("HsmlDataResult", Nimport * sizeof(struct hsmldata_out));
            HsmlDataOut = (struct hsmldata_out *)mymalloc("HsmlDataOut", Nexport * sizeof(struct hsmldata_out));

            /* now do the particles that were sent to us */
            TIMER_PUSH(1, "HSMLCOMPUTE");
            NextJ = 0;

#ifdef PXP_OPENMP
#pragma omp parallel
#endif // PXP_OPENMP
            {
                auto treewalk_filter_pt1 = [](particle_data *P, int p, void *DataGet, int *targets, int n_targets,
                                              enum IterationPhase mode) -> int {
                    return treewalk_filter_dm_only(P[p].Type);
                };

                ar_green_evaluate_secondary(HsmlDataGet, pt1_hsml_evaluate, Module_Hsml_pt1, TreeWalkType_GreenTreeWalk,
                                            NeighbourSearchCriteria_Hsml_pt1, group_criteria_distance,
                                            treewalk_filter_pt1, &ThisThread, should_drift);
            }

            TIMER_POP(1, "HSMLCOMPUTE");
            if (NextParticle < 0)
            {
                ndone_flag = 1;
            }

            TIMER_PUSH(1, "HSMLWAIT");
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            TIMER_POPPUSH(1, "HSMLWAIT", "HSML_COMM");

            /* get the result */
            for (ngrp = 1; ngrp < (1 << PTask); ++ngrp)
            {
                recvTask = ThisTask ^ ngrp;
                if (recvTask < NTask)
                {
                    if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                    {
                        /* send the results */
                        MPI_Sendrecv(&HsmlDataResult[Recv_offset[recvTask]],
                                     Recv_count[recvTask] * sizeof(struct hsmldata_out), MPI_BYTE, recvTask, TAG_DENS_B,
                                     &HsmlDataOut[Send_offset[recvTask]],
                                     Send_count[recvTask] * sizeof(struct hsmldata_out), MPI_BYTE, recvTask, TAG_DENS_B,
                                     MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
            TIMER_POPPUSH(1, "HSML_COMM", "HSMLCOMPUTE");

            /* add the result to the local particles */
            for (int j = 0; j < Nexport; ++j)
            {
                place = DataIndexTable[j].Index;
                out2particle_pt1_hsml(&HsmlDataOut[j], place, 1);
            }
            TIMER_POP(1, "HSMLCOMPUTE");
            myfree(HsmlDataOut);
            myfree(HsmlDataResult);
            myfree(DataNodeListGet);
            myfree(HsmlDataGet);
        } while (ndone < NTask);

        /* do final operations on results */
        TIMER_PUSH(1, "HSMLCOMPUTE");

        for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            if (pt1_hsml_isactive(P[i].TimeBin, P[i].Type))
            {
                /* now check whether we had enough neighbours */
                int redo_particle = 0;
                MyFloat hsml_old = P[i].Hsml;

                if (iter == (MAXITER - 50))
                {
                    for (int iii = 0; iii < 6; iii++)
                    {
                        cout(info) << "Bounds [" << iii << "]: hsml: " << MinHsml[iii] << "/" << MaxHsml[iii]
                                   << " TNgb: " << MinNtrue[iii] << "/" << DesN[iii] << "/" << MaxNtrue[iii]
                                   << "  Ngb: " << DesN[iii] << endl;
                    }
                }

                if ((iter >= MAXITER - 50) || P[i].ID == HSML_DEBUG)
                {
                    cout(info) << "Before: i=" << i << " task=" << ThisTask << " ID=" << P[i].ID
                               << " Hsml=" << P[i].Hsml << "/" << P[i].NumNgb << "/" << P[i].TrueNGB
                               << " Left=" << Left[i] << "/" << LeftN[i] << "/" << LeftNtrue[i] << " Right=" << Right[i]
                               << "/" << RightN[i] << "/" << RightNtrue[i] << " Right-Left=" << (Right[i] - Left[i])
                               << "/" << abs(hsml_old - P[i].Hsml) << endl;
                }

                /* First Check if the change in bounds are already very small */
                if ((Right[i] - Left[i]) < P[i].Hsml * (HSML_TOLERANCE * iter) && Left[i] > 0 && Right[i] > 0)
                {
                    redo_particle = 0; /* therefore we stop here */
                }
                else

                {
                    /* Case all requironments are meat, outer loop is True neighbours, inner loop is Weighted neighbours
                     */
                    if (P[i].TrueNGB >= MinNtrue[P[i].Type] &&
                        P[i].TrueNGB <= MaxNtrue[P[i].Type]) /* Total Neighbours are fine */
                    {
                        redo_particle = 0; /* Nothing to do */
                    }
                    else if (P[i].TrueNGB < MinNtrue[P[i].Type]) /* Would like to grow hsml */
                    {
                        redo_particle = 1;
                        if (false) // P[i].NumNgb > MaxN[P[i].Type])                                           /*
                                   // Weighted Ngb shrink hsml (!!) */
                        {
                            redo_particle = 0; /* therefore we stop here */
                        }
                        else
                        {
                            Left[i] = P[i].Hsml;
                            LeftN[i] = P[i].NumNgb;
                            LeftNtrue[i] = P[i].TrueNGB;
                            if (Right[i] > 0)
                            {
                                P[i].Hsml = 0.5 * (Right[i] + Left[i]);
                            }
                            else
                            {
                                P[i].Hsml *= 1.26;
                            }
                        }
                    }
                    else if (P[i].TrueNGB > MaxNtrue[P[i].Type]) /* Would like to shrink hsml */
                    {
                        redo_particle = 1;
                        if (false) // P[i].NumNgb < MinN[P[i].Type])                                           /*
                                   // Weighted Ngb grow hsml (!!) */
                        {
                            redo_particle = 0; /* therefore we stop here */
                        }
                        else
                        {
                            Right[i] = P[i].Hsml;
                            RightN[i] = P[i].NumNgb;
                            RightNtrue[i] = P[i].TrueNGB;
                            if (Left[i] > 0)
                            {
                                P[i].Hsml = 0.5 * (Right[i] + Left[i]);
                            }
                            else
                            {
                                P[i].Hsml /= 1.26;
                            }
                        }
                    }
                    else
                    {
                        ENDRUN("This can not happen !!!!");
                    }
                }

                if (redo_particle == 1) /* Check special Cases */
                {
                    if (abs(hsml_old - P[i].Hsml) < P[i].Hsml * (HSML_TOLERANCE * iter)) /* Too small change in hsml */
                    {
                        redo_particle = 0;
                        P[i].Hsml = hsml_old;
                    }
                    if (P[i].Hsml < MinHsml[P[i].Type] && P[i].Hsml < hsml_old) /* Running against lower bound */
                    {
                        redo_particle = 0;
                        P[i].Hsml = hsml_old;
                    }
                    if (P[i].Hsml > MaxHsml[P[i].Type] && P[i].Hsml > hsml_old) /* Running against lower bound */
                    {
                        redo_particle = 0;
                        P[i].Hsml = hsml_old;
                    }
                }

                if (redo_particle == 1) /* need to redo this particle */
                {
                    if (iter >= (MAXITER - 50) || P[i].ID == HSML_DEBUG)
                    {
                        cout(info) << "After: i=" << i << " task=" << ThisTask << " ID=" << P[i].ID
                                   << " Hsml=" << P[i].Hsml << "/" << P[i].NumNgb << "/" << P[i].TrueNGB
                                   << " Left=" << Left[i] << "/" << LeftN[i] << "/" << LeftNtrue[i]
                                   << " Right=" << Right[i] << "/" << RightN[i] << "/" << RightNtrue[i]
                                   << " Right-Left=" << (Right[i] - Left[i]) << "/" << abs(hsml_old - P[i].Hsml)
                                   << endl;
                    }
                }
                else
                {
                    P[i].TimeBin = -P[i].TimeBin - 1; /* Mark as inactive */
                }
            } // if
        } // for

        TIMER_POP(1, "HSMLCOMPUTE");

        if (ntot > 0)
        {
            ++iter;

            if (iter > 0 && ThisTask == 0)
            {
                VERBOSE_ALL(info, "ngb iteration %d: need to repeat for %d%09d particles.\n", iter,
                            (int)(ntot / 1000000000), (int)(ntot % 1000000000));
            }

            if (iter > MAXITER)
            {
                cout(error) << "failed to converge in neighbour iteration in pt1_find_hsml()" << endl;
                ENDRUN_MAIN("1155");
            }
        }
    } while (ntot > 0);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(RightNtrue);
    myfree(LeftNtrue);
    myfree(RightN);
    myfree(LeftN);
    myfree(Right);
    myfree(Left);
    myfree(RecyclingBuffer);

    myfree(Ngblist);

    /* mark as active again */
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if (P[i].TimeBin < 0)
        {
            P[i].TimeBin = -P[i].TimeBin - 1;
        }
    }

    std::cout.clear();

    TIMER_POP(1, "HSML_PT1");
}
#endif // !ADAPTGRAVSOFT

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Update hmax-values in tree nodes that hold dark matter particles.
 *
 * This function updates the hmax-values in tree nodes that hold dark matter
 * particles. These values are needed to find all neighbors in the hsml
 * computation.
 *
 */
#ifndef ADAPTGRAVSOFT
void scatter::pt1_force_update_hmax(void)
{
    int totDomainNumChanged;
    int *domainList_all;
    int *counts, *offset_list, *offset_hmax;
    MyFloat *domainHmax_loc, *domainHmax_all;

    GlobFlag++;

    DomainNumChanged = 0;
    DomainList = (int *)mymalloc("DomainList", NTopleaves * sizeof(int));

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        int no = Father[i];

        while (no >= 0)
        {
            force_drift_node(no, All.Ti_Current);

            if (P[i].Hsml > Nodes[no].maxsoft)
            {

                Nodes[no].maxsoft = P[i].Hsml;

                if (Nodes[no].u.d.bitflags & (1 << BITFLAG_TOPLEVEL)) /* we reached a top-level node */
                {
                    if (Nodes[no].Flag != GlobFlag)
                    {
                        Nodes[no].Flag = GlobFlag;
                        DomainList[DomainNumChanged++] = no;
                    }
                    break;
                }
            }
            else
            {
                break;
            }

            no = Nodes[no].u.d.father;
        }
    }
    /* share the hmax-data of the pseudo-particles accross CPUs */

    counts = (int *)mymalloc("counts", sizeof(int) * NTask);
    offset_list = (int *)mymalloc("offset_list", sizeof(int) * NTask);
    offset_hmax = (int *)mymalloc("offset_hmax", sizeof(int) * NTask);

    domainHmax_loc = (MyFloat *)mymalloc("domainHmax_loc", DomainNumChanged * 2 * sizeof(MyFloat));

    for (int i = 0; i < DomainNumChanged; ++i)
    {
        domainHmax_loc[2 * i] = Nodes[DomainList[i]].maxsoft;
    }

    MPI_Allgather(&DomainNumChanged, 1, MPI_INT, counts, 1, MPI_INT, MYMPI_COMM_WORLD);

    {
        int ta;
        for (ta = 0, totDomainNumChanged = 0, offset_list[0] = 0, offset_hmax[0] = 0; ta < NTask; ++ta)
        {
            totDomainNumChanged += counts[ta];
            if (ta > 0)
            {
                offset_list[ta] = offset_list[ta - 1] + counts[ta - 1];
                offset_hmax[ta] = offset_hmax[ta - 1] + counts[ta - 1] * 2 * sizeof(MyFloat);
            }
        }
    }

    coutr(info) << "*pt1_find_hsml* Hmax exchange: " << totDomainNumChanged << " topleaves out of " << NTopleaves
                << endl;

    domainHmax_all = (MyFloat *)mymalloc("domainHmax_all", totDomainNumChanged * 2 * sizeof(MyFloat));
    domainList_all = (int *)mymalloc("domainList_all", totDomainNumChanged * sizeof(int));

    MPI_Allgatherv(DomainList, DomainNumChanged, MPI_INT, domainList_all, counts, offset_list, MPI_INT,
                   MYMPI_COMM_WORLD);

    for (int ta = 0; ta < NTask; ++ta)
    {
        counts[ta] *= 2 * sizeof(MyFloat);
    }

    MPI_Allgatherv(domainHmax_loc, 2 * DomainNumChanged * sizeof(MyFloat), MPI_BYTE, domainHmax_all, counts,
                   offset_hmax, MPI_BYTE, MYMPI_COMM_WORLD);

    for (int i = 0; i < totDomainNumChanged; ++i)
    {
        int no = domainList_all[i];

        if (Nodes[no].u.d.bitflags & (1 << BITFLAG_DEPENDS_ON_LOCAL_MASS)) /* to avoid that the hmax is updated twice */
        {
            no = Nodes[no].u.d.father;
        }

        while (no >= 0)
        {
            force_drift_node(no, All.Ti_Current);

            if (domainHmax_all[2 * i] > Nodes[no].maxsoft)
            {
                Nodes[no].maxsoft = domainHmax_all[2 * i];
            }
            else
            {
                break;
            }
            no = Nodes[no].u.d.father;
        }
    }

    myfree(domainList_all);
    myfree(domainHmax_all);
    myfree(domainHmax_loc);
    myfree(offset_hmax);
    myfree(offset_list);
    myfree(counts);
    myfree(DomainList);
}
#endif // !ADAPTGRAVSOFT

//////////////////////////////////////////////////////////////////////////////////////////////////
