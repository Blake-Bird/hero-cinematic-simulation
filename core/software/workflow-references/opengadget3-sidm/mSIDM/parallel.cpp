/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file for the class to parallelise the computations of pair-wise interactions.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "parallel.hpp"
#include "tools.hpp"
#include <cstring>

#ifndef PXP_OLDPARALLEL
#include "comm.hpp"
#endif // !PXP_OLDPARALLEL

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "execute_pxp.hpp"
#undef NODELISTLENGTH
#define NODELISTLENGTH NODELISTLENGTH_PXP
#define DataNodeList scatter::pxp_nodelist::DataNodeList_pxp
#define DataNodeListIn scatter::pxp_nodelist::DataNodeListIn_pxp
#define DataNodeListGet scatter::pxp_nodelist::DataNodeListGet_pxp
#include "../CodeBase/greentree.hpp"

using namespace scatter::tools;

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Method to determine which processes communicate with each other.
 */
#ifdef PXP_OLDPARALLEL
template <typename BASE> int parallel<BASE>::process_to_comm_with(const int b)
{
    return (b + 1) ^ ThisTask;
}
#endif // PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Method to do secondary loop.
 */
#ifdef PXP_OLDPARALLEL
template <typename BASE>
void parallel<BASE>::secondary_loop_one_way(SendDirection senddir, int DoPrimaryOnGPU, int should_drift,
                                            int group_particles, int &ndone)
{
    for (int b = 0; b < (1 << PTask) - 1; ++b) // begin loop over processor communication
    {
        int commTask = process_to_comm_with(b);

        CommMode commode;

        if ((senddir == toLargerTaskNum && ThisTask > commTask) || (senddir == toSmallerTaskNum && ThisTask < commTask))
        {
            commode = RecvFirst;
        }
        else
        {
            commode = SendFirst;
        }

        if (commTask < NTask)
        {
            // exchange particle data
            if (commode == SendFirst && Send_count[commTask] > 0)
            {
                TIMER_PUSH(1, B::CPU_PXPNETWORK);

                // prepare particle data for export
                const int j_pmax = Send_offset[commTask] + Send_count[commTask];
                for (int j = Send_offset[commTask], i = 0; j < j_pmax; ++j, ++i)
                {
                    int &place = DataIndexTable[j].Index;
                    B::DataIn[i].particle2in(place);
                    memcpy(&B::DataNodeListIn_pxp[i], &B::DataNodeList_pxp[DataIndexTable[j].IndexGet],
                           sizeof(struct data_nodelist_pxp));
                }

                TIMER_POPPUSH(1, B::CPU_PXPNETWORK, B::CPU_PXPCOMM);

                /* send the particles */
                MPI_Send(&B::DataIn[0], Send_count[commTask] * sizeof(DATA_IN), MPI_BYTE, commTask, TAG_PXP_A,
                         MYMPI_COMM_WORLD);

                /* send nodelist */
                MPI_Send(&B::DataNodeListIn_pxp[0], Send_count[commTask] * sizeof(struct data_nodelist_pxp), MPI_BYTE,
                         commTask, TAG_GRAV_A, MYMPI_COMM_WORLD);

                TIMER_POP(1, B::CPU_PXPCOMM);
            }
            else if (commode == RecvFirst && Recv_count[commTask] > 0)
            {
                TIMER_PUSH(1, B::CPU_PXPWAIT);

                /* recv the particles */
                MPI_Recv(&B::DataGet[0], Recv_count[commTask] * sizeof(DATA_IN), MPI_BYTE, commTask, TAG_PXP_A,
                         MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                /* recv nodelist */
                MPI_Recv(&B::DataNodeListGet_pxp[0], Recv_count[commTask] * sizeof(struct data_nodelist_pxp), MPI_BYTE,
                         commTask, TAG_GRAV_A, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                TIMER_POP(1, B::CPU_PXPWAIT);
            }

            // now do the particles that were sent to us
            if (commode == RecvFirst) // only one process should compute the scattering
            {
                NextJ = 0;
                int NimportTrue = Nimport;
                Nimport = Recv_count[commTask];

#ifdef PXP_DEBUG
                // check for pseudo particles
                for (int loop = 0; loop < Recv_count[commTask]; ++loop)
                {
                    int loop2 = 0;
                    while (loop2 < loop)
                    {
                        if (B::DataGet[loop].ID == B::DataGet[loop2].ID)
                        {
                            cout(warning)
                                << "PXP Warning: pseudo particle found, NODELISTLENGTH_PXP is too small." << endl;
                            break;
                        }
                        ++loop2;
                    }
                }
#endif // PXP_DEBUG

                TIMER_PUSH(1, B::CPU_PXPPARALLEL);
                TIMER_PUSH(1, B::CPU_PXPCOMPUTE);

#ifdef PXP_OPENMP
#pragma omp parallel
#endif // PXP_OPENMP
                {
                    int ThisThread = omp_get_thread_num();

                    auto evaluate_pxp = [this](int target, int mode, const int *ngblist, int extended_numngb_inbox,
                                               int *i_startnode, int node, int i_node, void *data) -> int {
                        return evaluate(target, mode, ngblist, extended_numngb_inbox, i_startnode, node, i_node, data);
                    };

                    auto treewalk_filter_pXp = [this](particle_data *P, int p, void *DataGet, int *targets,
                                                      int n_targets, enum IterationPhase mode) -> int {
                        return B::treewalk_filter(P[p].Type);
                    };

                    ar_green_evaluate_secondary(B::DataGet, evaluate_pxp, Module(B::module), TreeWalkType_GreenTreeWalk,
                                                NeighbourSearchCriteria(B::neighboursearchcriteria),
                                                group_criteria_distance, treewalk_filter_pXp, &ThisThread,
                                                should_drift);
                }

                TIMER_POP(1, B::CPU_PXPCOMPUTE);
                TIMER_POP(1, B::CPU_PXPPARALLEL);

                Nimport = NimportTrue;
            }
        }

        if (commTask < NTask)
        {
            if (commode == SendFirst && Send_count[commTask] > 0)
            {
                TIMER_PUSH(1, B::CPU_PXPWAIT);

                /* recv the results */
                MPI_Recv(&B::DataOut[0], Send_count[commTask] * sizeof(DATA_OUT), MPI_BYTE, commTask, TAG_PXP_B,
                         MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

                TIMER_POP(1, B::CPU_PXPWAIT);
            }
            else if (commode == RecvFirst && Recv_count[commTask] > 0)
            {
                TIMER_PUSH(1, B::CPU_PXPNETWORK);

                for (int loop = 0; loop < Recv_count[commTask]; ++loop)
                {
                    B::DataResult[loop].in2out(B::DataGet[loop]);
                }

                TIMER_POPPUSH(1, B::CPU_PXPNETWORK, B::CPU_PXPCOMM);

                /* send the results */
                MPI_Send(&B::DataResult[0], Recv_count[commTask] * sizeof(DATA_OUT), MPI_BYTE, commTask, TAG_PXP_B,
                         MYMPI_COMM_WORLD);

                TIMER_POP(1, B::CPU_PXPCOMM);
            }

            /* add the result to the local particles */
            TIMER_PUSH(1, B::CPU_PXPNETWORK);

            if (commode == SendFirst)
            {
                const int j_pmax = Send_offset[commTask] + Send_count[commTask];
                for (int j = Send_offset[commTask], i = 0; j < j_pmax; ++j, ++i)
                {
                    int &place = DataIndexTable[j].Index;
                    B::DataOut[i].out2particle(place);
                }
            }

            TIMER_POP(1, B::CPU_PXPNETWORK);
        }
    } // end loop over processor communication
}
#endif // PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Method to do secondary loop.
 */
#ifndef PXP_OLDPARALLEL
template <typename BASE> void parallel<BASE>::secondary_loop_both_ways(int should_drift)
{
    comm_list my_comm_list;
    my_comm_list.create();
    int num_b = my_comm_list.get_num_comm_rows();

    for (int b = 0; b < num_b; ++b) // begin loop over processor communication
    {
        int commTask = my_comm_list.process_to_comm_with(b);

        if (commTask < NTask)
        {
            // exchange particle data
            inner_comm_one_way(commTask, toLargerTaskNum, should_drift);
            inner_comm_one_way(commTask, toSmallerTaskNum, should_drift);
        }
    } // end loop over processor communication
}
#endif // !PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Communication and computations of the secondary loop for a given pair of MPI processes.
 */
#ifndef PXP_OLDPARALLEL
template <typename BASE>
void parallel<BASE>::inner_comm_one_way(const int &commTask, SendDirection senddir, int should_drift)
{
    // from here we assume that commTask<NumTasks

    CommMode commode;
    if ((senddir == toLargerTaskNum && ThisTask > commTask) || (senddir == toSmallerTaskNum && ThisTask < commTask))
    {
        commode = RecvFirst;
    }
    else
    {
        commode = SendFirst;
    }

    if (commode == SendFirst && Send_count[commTask] > 0)
    {
        pre_calc_send(commTask);
    }
    else if (commode == RecvFirst && Recv_count[commTask] > 0)
    {
        pre_calc_recv(commTask);
    }

    // now do the particles that were sent to us
    if (commode == RecvFirst) // only one process should compute the scattering
    {
        NextJ = 0;
        int NimportTrue = Nimport;
        Nimport = Recv_count[commTask];

#ifdef PXP_DEBUG
        // check for pseudo particles
        for (int loop = 0; loop < Recv_count[commTask]; ++loop)
        {
            int loop2 = 0;
            while (loop2 < loop)
            {
                if (B::DataGet[loop].ID == B::DataGet[loop2].ID)
                {
                    cout(warning) << "PXP Warning: pseudo particle found, NODELISTLENGTH_PXP is too small." << endl;
                    break;
                }
                ++loop2;
            }
        }
#endif // PXP_DEBUG

        evaluate_secondary(should_drift);

        Nimport = NimportTrue;
    }

    if (commode == SendFirst && Send_count[commTask] > 0)
    {
        post_calc_recv(commTask);
    }
    else if (commode == RecvFirst && Recv_count[commTask] > 0)
    {
        post_calc_send(commTask);
    }

    /* add the result to the local particles */
    TIMER_PUSH(1, B::CPU_PXPNETWORK);

    if (commode == SendFirst)
    {
        const int j_pmax = Send_offset[commTask] + Send_count[commTask];
        for (int j = Send_offset[commTask], i = 0; j < j_pmax; ++j, ++i)
        {
            int &place = DataIndexTable[j].Index;
            B::DataOut[i].out2particle(place);
        }
    }

    TIMER_POP(1, B::CPU_PXPNETWORK);
}
#endif // !PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Method to do the actual interactions for non-local particles.
 */
#ifndef PXP_OLDPARALLEL
template <typename BASE> inline void parallel<BASE>::evaluate_secondary(const int &should_drift)
{
    TIMER_PUSH(1, B::CPU_PXPPARALLEL);
    TIMER_PUSH(1, B::CPU_PXPCOMPUTE);

#ifdef PXP_OPENMP
#pragma omp parallel
#endif // PXP_OPENMP
    {
        int ThisThread = omp_get_thread_num();
        auto evaluate_pxp = [this](int target, int mode, const int *ngblist, int extended_numngb_inbox,
                                   int *i_startnode, int node, int i_node, void *data) -> int {
            return evaluate(target, mode, ngblist, extended_numngb_inbox, i_startnode, node, i_node, data);
        };

        auto treewalk_filter_pXp = [this](particle_data *P, int p, void *DataGet, int *targets, int n_targets,
                                          enum IterationPhase mode) -> int { return B::treewalk_filter(P[p].Type); };

        ar_green_evaluate_secondary(B::DataGet, evaluate_pxp, Module(B::module), TreeWalkType_GreenTreeWalk,
                                    NeighbourSearchCriteria(B::neighboursearchcriteria), group_criteria_distance,
                                    treewalk_filter_pXp, &ThisThread, should_drift);
    }

    TIMER_POP(1, B::CPU_PXPCOMPUTE);
    TIMER_POP(1, B::CPU_PXPPARALLEL);
}
#endif // !PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief The method to send particles before the interaction computations.
 */
#ifndef PXP_OLDPARALLEL
template <typename BASE> void parallel<BASE>::pre_calc_send(const int &commTask) const
{
    TIMER_PUSH(1, B::CPU_PXPNETWORK);

    // prepare particle data for export
    const int j_pmax = Send_offset[commTask] + Send_count[commTask];
    for (int j = Send_offset[commTask], i = 0; j < j_pmax; ++j, ++i)
    {
        int &place = DataIndexTable[j].Index;
        B::DataIn[i].particle2in(place);
        memcpy(&B::DataNodeListIn_pxp[i], &B::DataNodeList_pxp[DataIndexTable[j].IndexGet],
               sizeof(struct data_nodelist_pxp));
    }

    TIMER_POPPUSH(1, B::CPU_PXPNETWORK, B::CPU_PXPCOMM);

    /* send the particles */
    MPI_Send(&B::DataIn[0], Send_count[commTask] * sizeof(DATA_IN), MPI_BYTE, commTask, TAG_PXP_A, MYMPI_COMM_WORLD);

    /* send nodelist */
    MPI_Send(&B::DataNodeListIn_pxp[0], Send_count[commTask] * sizeof(struct data_nodelist_pxp), MPI_BYTE, commTask,
             TAG_GRAV_A, MYMPI_COMM_WORLD);

    TIMER_POP(1, B::CPU_PXPCOMM);
}
#endif // !PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief The method to receive particles before the interaction computations.
 */
#ifndef PXP_OLDPARALLEL
template <typename BASE> void parallel<BASE>::pre_calc_recv(const int &commTask) const
{
    TIMER_PUSH(1, B::CPU_PXPWAIT);

    /* recv the particles */
    MPI_Recv(&B::DataGet[0], Recv_count[commTask] * sizeof(DATA_IN), MPI_BYTE, commTask, TAG_PXP_A, MYMPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    /* recv nodelist */
    MPI_Recv(&B::DataNodeListGet_pxp[0], Recv_count[commTask] * sizeof(struct data_nodelist_pxp), MPI_BYTE, commTask,
             TAG_GRAV_A, MYMPI_COMM_WORLD, MPI_STATUS_IGNORE);

    TIMER_POP(1, B::CPU_PXPWAIT);
}
#endif // !PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief The method to send particles after the interaction computations.
 */
#ifndef PXP_OLDPARALLEL
template <typename BASE> void parallel<BASE>::post_calc_send(const int &commTask) const
{
    TIMER_PUSH(1, B::CPU_PXPNETWORK);

    for (int loop = 0; loop < Recv_count[commTask]; ++loop)
    {
        B::DataResult[loop].in2out(B::DataGet[loop]);
    }

    TIMER_POPPUSH(1, B::CPU_PXPNETWORK, B::CPU_PXPCOMM);

    /* send the results */
    MPI_Send(&B::DataResult[0], Recv_count[commTask] * sizeof(DATA_OUT), MPI_BYTE, commTask, TAG_PXP_B,
             MYMPI_COMM_WORLD);

    TIMER_POP(1, B::CPU_PXPCOMM);
}
#endif // !PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief The method to receive particles after the interaction computations.
 */
#ifndef PXP_OLDPARALLEL
template <typename BASE> void parallel<BASE>::post_calc_recv(const int &commTask) const
{
    TIMER_PUSH(1, B::CPU_PXPWAIT);

    /* recv the results */
    MPI_Recv(&B::DataOut[0], Send_count[commTask] * sizeof(DATA_OUT), MPI_BYTE, commTask, TAG_PXP_B, MYMPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    TIMER_POP(1, B::CPU_PXPWAIT);
}
#endif // !PXP_OLDPARALLEL
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Template specialization for mSIDM.
 */
#ifdef mSIDM
template class pxp_execute::parallel<msidm_base>;
#endif // mSIDM

} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////
