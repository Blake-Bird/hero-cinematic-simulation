/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file for the class template for pairwise interactions.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_EXECUTE_PXP_CPP
#define MSIDM_EXECUTE_PXP_CPP

#include "execute_pxp.hpp"
#include "comm.hpp"

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#ifdef NODELISTLENGTH
#undef NODELISTLENGTH
#endif // NODELISTLENGTH

#define NODELISTLENGTH NODELISTLENGTH_PXP
#define DataNodeList scatter::pxp_nodelist::DataNodeList_pxp
#define DataNodeListIn scatter::pxp_nodelist::DataNodeListIn_pxp
#define DataNodeListGet scatter::pxp_nodelist::DataNodeListGet_pxp
#include "../CodeBase/greentree.hpp"
#include "../System/communication.hpp"
#include "tools.hpp"

// select method for OpenMP parallelisation
// #define PXP_OPENMP1
#define PXP_OPENMP2

using namespace scatter::pxp_execute;

//////////////////////////////////////////////////////////////////////////////////////////////////

treewalk_filter_dm_only_class treewalk_filter_dm_only;
treewalk_filter_bary_only_class treewalk_filter_bary_only;

//////////////////////////////////////////////////////////////////////////////////////////////////

data_nodelist_pxp *pxp_nodelist::DataNodeList_pxp(nullptr);
data_nodelist_pxp *pxp_nodelist::DataNodeListIn_pxp(nullptr);
data_nodelist_pxp *pxp_nodelist::DataNodeListGet_pxp(nullptr);

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of the constructor
 */
template <typename BASE> execute_pxp<BASE>::execute_pxp()
{
    B::CPU_PXPCOMPUTE = B::module_name + "_compute";
    B::CPU_PXPPREP = B::module_name + "_prep";
    B::CPU_PXPCOMM = B::module_name + "_comm";
    B::CPU_PXPWAIT = B::module_name + "_wait";
    B::CPU_PXPNETWORK = B::module_name + "_network";
    B::CPU_PXPLOCAL = B::module_name + "_local";
    B::CPU_PXPPARALLEL = B::module_name + "_nonlocal";
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief The 'all' method, which is called by the physics module to execute the interactions.
 *
 * This method allocates Ngblist and uses greentree to execute evaluate(). It
 * also does the MPI communication.
 *
 */
template <typename BASE> void execute_pxp<BASE>::all()
{
    using namespace tools;

    int ndone_flag, ndone;

    int save_NextParticle;

    /* allocate buffers to arrange communication */
    int NTaskTimesNumPart;

    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    RecyclingBuffer = (int *)mymalloc("RecyclingBuffer;", NTaskTimesNumPart * sizeof(int));

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize = (int)((MyBufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + 3 * sizeof(struct data_nodelist_pxp) + sizeof(DATA_IN) +
                           sizeof(DATA_OUT) + sizemax(sizeof(DATA_IN), sizeof(DATA_OUT))));

    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    B::DataNodeList_pxp =
        (struct data_nodelist_pxp *)mymalloc("DataNodeListIn_PXP", All.BunchSize * sizeof(struct data_nodelist_pxp));

    coutr(extra) << "PXP: using " << MyBufferSize / 1.0 << " MB for buffering, " << FreeBytes / (1024.0 * 1024.0)
                 << " MB remaining for other buffers" << endl;

    NextParticle = FirstActiveParticle; /* beginn with this index */

    int DoPrimaryOnGPU = 0;
    int should_drift = 1;
    int group_particles = 1;

    do
    {
        BufferFullFlag = 0;
        Nexport = 0;
        save_NextParticle = NextParticle;

        for (int j = 0; j < NTask; ++j)
        {
            Send_count[j] = 0;
        }

        for (int j = 0; j < NTask * maxThreads; ++j)
        {
            Exportflag[j] = -1;
        }

        /** do local particles and prepare export list */

        TIMER_PUSH(1, B::CPU_PXPLOCAL);
        TIMER_PUSH(1, B::CPU_PXPCOMPUTE);

#ifdef PXP_OPENMP
#pragma omp parallel
#endif // PXP_OPENMP
        {
            int ThisThread = omp_get_thread_num();

            auto isactive_pXp = [this](int n) { return B::isactive(P[n].Type); };

            auto evaluate_pXp = [this](int target, int mode, const int *ngblist, int extended_numngb_inbox,
                                       int *i_startnode, int node, int i_node, void *data) -> int {
                return evaluate(target, mode, ngblist, extended_numngb_inbox, i_startnode, node, i_node, data);
            };

            auto treewalk_filter_pXp = [this](particle_data *P, int p, void *DataGet, int *targets, int n_targets,
                                              enum IterationPhase mode) -> int {
                return B::treewalk_filter(P[p].Type);
            };

            ar_green_evaluate_primary(isactive_pXp, evaluate_pXp, Module(B::module), TreeWalkType_GreenTreeWalk,
                                      NextParticleLoop_ActiveParticleList,
                                      NeighbourSearchCriteria(B::neighboursearchcriteria), group_criteria_distance,
                                      treewalk_filter_pXp, &ThisThread, DoPrimaryOnGPU, should_drift, group_particles);
        }

        TIMER_POP(1, B::CPU_PXPCOMPUTE);
        TIMER_POP(1, B::CPU_PXPLOCAL);

        /** if we are not grouping particles and SendRecv fails, then we panic as there is no memory
           to process even a single particles and if green tree fails, we retry without it */

        TIMER_PUSH(1, B::CPU_PXPPREP);

        int panic_on_error = !group_particles;
        int is_send_receive_ok = prepare_sendrec_offset_send(save_NextParticle, panic_on_error);
        // here Nimport is determined
        if (!is_send_receive_ok)
        {
            group_particles = 0;
            TIMER_POP(1, B::CPU_PXPPREP);
            continue;
        }

        const int Recv_count_max = get_max_of_array(Recv_count, NTask);
        const int Send_count_max = get_max_of_array(Send_count, NTask);

        B::DataGet = (DATA_IN *)mymalloc("PXPDataGet", Recv_count_max * sizeof(DATA_IN));
        B::DataNodeListGet_pxp = (struct data_nodelist_pxp *)mymalloc(
            "DataNodeListGet_PXP", Recv_count_max * sizeof(struct data_nodelist_pxp));
        B::DataIn = (DATA_IN *)mymalloc("PXPDataIn", Send_count_max * sizeof(DATA_IN));
        B::DataNodeListIn_pxp = (struct data_nodelist_pxp *)mymalloc("DataNodeListIn_PXP",
                                                                     Send_count_max * sizeof(struct data_nodelist_pxp));
        B::DataResult = (DATA_OUT *)mymalloc("PXPDataResult", Recv_count_max * sizeof(DATA_OUT));
        B::DataOut = (DATA_OUT *)mymalloc("PXPDataOut", Send_count_max * sizeof(DATA_OUT));

        report_memory_usage(&HighMark_pxp, "PXP");

        TIMER_POP(1, B::CPU_PXPPREP);

        // do the MPI communication and the scattering
#ifdef PXP_OLDPARALLEL
        PB::secondary_loop_one_way(toLargerTaskNum, DoPrimaryOnGPU, should_drift, group_particles, ndone);

        PB::secondary_loop_one_way(toSmallerTaskNum, DoPrimaryOnGPU, should_drift, group_particles, ndone);
#else // PXP_OLDPARALLEL
        PB::secondary_loop_both_ways(should_drift);
#endif // PXP_OLDPARALLEL

        if (NextParticle < 0)
            ndone_flag = 1;
        else
            ndone_flag = 0;

        TIMER_PUSH(1, B::CPU_PXPWAIT);

        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

        TIMER_POP(1, B::CPU_PXPWAIT);

        myfree(B::DataOut);
        myfree(B::DataResult);
        myfree(B::DataNodeListIn_pxp);
        myfree(B::DataIn);
        myfree(B::DataNodeListGet_pxp);
        myfree(B::DataGet);
    } while (ndone < NTask);

    myfree(B::DataNodeList_pxp);
    myfree(DataIndexTable);
    myfree(RecyclingBuffer);
    myfree(Ngblist);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief General 'evaluate' method for computing local and non-local interactions.
 *
 * This method is processed in parallel and calls pxp_physics(), via primary_phase and
 * secondary_phase, which takes two particles and computes the scattering.
 *
 */
template <typename BASE>
int execute_pxp<BASE>::evaluate(int target, int mode, const int *ngblist, int extended_numngb_inbox, int *i_startnode,
                                int node, int i_node, void *data)
{
    if (mode == 0 || mode == 2 || mode == 3)
    {
        return primary_phase(target, ngblist, extended_numngb_inbox, i_startnode, node, i_node, data);
    }
    else
    {
        return secondary_phase(target, ngblist, extended_numngb_inbox, i_startnode, node, i_node, data);
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This method is called by evaluate and compute the local interactions.
 */
template <typename BASE>
int execute_pxp<BASE>::primary_phase(int target, const int *ngblist, int extended_numngb_inbox, int *i_startnode,
                                     int node, int i_node, void *data)
{
    int startnode = node;
    int numngb;

    DATA_IN_REF local(target);

    while (startnode >= 0)
    {
        if (extended_numngb_inbox < 0)
        {
            return -1;
        }

        numngb = extended_numngb_inbox;
        startnode = *i_startnode;

        ENDRUN_IF(numngb > NumPart / DENSITY_LESS_NGB_FACTOR, "PXP: Number of neighbours found (%d)\
             exceeds allowed number (%d) derived from NumPart/DENSITY_LESS_NGB_FACTOR, to continue,\
             decreas DENSITY_LESS_NGB_FACTOR (currently %f, min possible val would be 1) !!\n",
                  numngb, (int)(NumPart / DENSITY_LESS_NGB_FACTOR), DENSITY_LESS_NGB_FACTOR);

        // target particle scatters with its neighbours
#ifdef PXP_OPENMP
#ifdef PXP_OPENMP1
        for (int n = 0; n < numngb; ++n)
        {
            // the scattering is done here - velocity update of local and P
            if (P[target].ID > P[ngblist[n]].ID)
            {
                omp_set_lock(&pxpPartLockList[target]);
                omp_set_lock(&pxpPartLockList[ngblist[n]]);
                pxp_physics(local, ngblist[n]);
                omp_unset_lock(&pxpPartLockList[ngblist[n]]);
                omp_unset_lock(&pxpPartLockList[target]);
            }
#ifdef PXP_DOACTIVEONCE
            else if ((P[target].ID < P[ngblist[n]].ID) && (!TimeBinActive[P[ngblist[n]].TimeBin]))
#else // PXP_DOACTIVEONCE
            else if (P[target].ID < P[ngblist[n]].ID)
#endif // PXP_DOACTIVEONCE
            {
                omp_set_lock(&pxpPartLockList[ngblist[n]]);
                omp_set_lock(&pxpPartLockList[target]);
                pxp_physics(local, ngblist[n]);
                omp_unset_lock(&pxpPartLockList[target]);
                omp_unset_lock(&pxpPartLockList[ngblist[n]]);
            }
        }
#elif defined(PXP_OPENMP2)
        if (numngb == 0)
        {
            continue;
        }
        int ngbl[numngb];
        for (int n = 0; n < numngb; ++n)
        {
            ngbl[n] = ngblist[n];
        }

        for (int n = 0, remaining = numngb; remaining > 0; n = (n + 1) % numngb)
        {
            // the scattering is done here - velocity update of local and P
            if (P[target].ID > P[ngbl[n]].ID)
            {
                omp_set_lock(&pxpPartLockList[target]);
                if (omp_test_lock(&pxpPartLockList[ngbl[n]]))
                {
                    pxp_physics(local, ngbl[n]);
                    omp_unset_lock(&pxpPartLockList[ngbl[n]]);
                    omp_unset_lock(&pxpPartLockList[target]);
                    --remaining;
                }
                else
                {
                    omp_unset_lock(&pxpPartLockList[target]);
                    ngbl[(n + remaining) % numngb] = ngbl[n];
                }
            }
#ifdef PXP_DOACTIVEONCE
            else if ((P[target].ID < P[ngbl[n]].ID) && (!TimeBinActive[P[ngbl[n]].TimeBin]))
#else // PXP_DOACTIVEONCE
            else if (P[target].ID < P[ngbl[n]].ID)
#endif // PXP_DOACTIVEONCE
            {
                if (omp_test_lock(&pxpPartLockList[ngbl[n]]))
                {
                    omp_set_lock(&pxpPartLockList[target]);
                    pxp_physics(local, ngbl[n]);
                    omp_unset_lock(&pxpPartLockList[target]);
                    omp_unset_lock(&pxpPartLockList[ngbl[n]]);
                    --remaining;
                }
                else
                {
                    ngbl[(n + remaining) % numngb] = ngbl[n];
                }
            }
            else
            {
                --remaining;
            }
        }
#endif // PXP_OPENMP2
#else // PXP_OPENMP
        for (int n = 0; n < numngb; ++n)
        {
            // the scattering is done here - velocity update of local and P
#ifdef PXP_DOACTIVEONCE
            if (!((local.ID <= P[ngblist[n]].ID) && (TimeBinActive[P[ngblist[n]].TimeBin])))
            {
                pxp_physics(local, ngblist[n]);
            }
#else // PXP_DOACTIVEONCE
            pxp_physics(local, ngblist[n]);
#endif // PXP_DOACTIVEONCE
        }
#endif // PXP_OPENMP
    }

    return 0;
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This method is called by evaluate and compute the non-local interactions.
 */
template <typename BASE>
int execute_pxp<BASE>::secondary_phase(int target, const int *ngblist, int extended_numngb_inbox, int *i_startnode,
                                       int node, int i_node, void *data)
{
    int startnode = node;
    int numngb;

    DATA_IN_REF local(B::DataGet[target]);

    while (startnode >= 0)
    {
        if (extended_numngb_inbox == -1)
        {
            return -1;
        }

        numngb = extended_numngb_inbox;
        startnode = *i_startnode;

        ENDRUN_IF(numngb > NumPart / DENSITY_LESS_NGB_FACTOR, "PXP: Number of neighbours found (%d)\
             exceeds allowed number (%d) derived from NumPart/DENSITY_LESS_NGB_FACTOR, to continue,\
             decreas DENSITY_LESS_NGB_FACTOR (currently %f, min possible val would be 1) !!\n",
                  numngb, (int)(NumPart / DENSITY_LESS_NGB_FACTOR), DENSITY_LESS_NGB_FACTOR);

        if (numngb < 0)
        {
            return -1;
        }

        // target particle scatters with its neighbours
#ifdef PXP_OPENMP
#ifdef PXP_OPENMP1
        for (int n = 0; n < numngb; ++n)
        {
            // the scattering is done here - velocity update of local and P
#ifdef PXP_DOACTIVEONCE
            if (!((local.ID <= P[ngblist[n]].ID) && (TimeBinActive[P[ngblist[n]].TimeBin])))
#endif // PXP_DOACTIVEONCE
            {
                omp_set_lock(&pxpPartLockList[ngblist[n]]);
                pxp_physics(local, ngblist[n]);
                omp_unset_lock(&pxpPartLockList[ngblist[n]]);
            }
        }
#elif defined(PXP_OPENMP2)
        if (numngb == 0)
        {
            continue;
        }

        int ngbl[numngb]; // this requires numngb > 0
        for (int n = 0; n < numngb; ++n)
        {
            ngbl[n] = ngblist[n];
        }

        for (int n = 0, remaining = numngb; remaining > 0; n = (n + 1) % numngb)
        {
            // the scattering is done here - velocity update of local and P
#ifdef PXP_DOACTIVEONCE
            if (!((local.ID <= P[ngbl[n]].ID) && (TimeBinActive[P[ngbl[n]].TimeBin])))
#endif // PXP_DOACTIVEONCE
            {
                if (omp_test_lock(&pxpPartLockList[ngbl[n]]))
                {
                    pxp_physics(local, ngbl[n]);
                    omp_unset_lock(&pxpPartLockList[ngbl[n]]);
                    --remaining;
                }
                else
                {
                    ngbl[(n + remaining) % numngb] = ngbl[n];
                }
            }
#ifdef PXP_DOACTIVEONCE
            else
            {
                --remaining;
            }
#endif // PXP_DOACTIVEONCE
        }
#endif // PXP_OPENMP2
#else // PXP_OPENMP
        for (int n = 0; n < numngb; ++n)
        {
            // the scattering is done here - velocity update of local and P
#ifdef PXP_DOACTIVEONCE
            if (!((local.ID <= P[ngblist[n]].ID) && (TimeBinActive[P[ngblist[n]].TimeBin])))
#endif // PXP_DOACTIVEONCE
            {
                pxp_physics(local, ngblist[n]);
            }
        }
#endif // PXP_OPENMP
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
#ifdef mSIDM
/**
 * @brief Template specialization for mSIDM.
 */
template class pxp_execute::execute_pxp<msidm_base>;
#endif

} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif // MSIDM_EXECUTE_PXP_CPP
