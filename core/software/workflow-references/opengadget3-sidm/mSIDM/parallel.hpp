/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file for the class to parallelise the computations of pair-wise interactions.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_PARALLEL_HPP
#define MSIDM_PARALLEL_HPP

#include "../CodeBase/precision.hpp"

#ifdef mSIDM
#include "msidm_base.hpp"
#endif


#include "../CodeBase/switches.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Enumeration to specify the direction for sending particles to other MPI ranks.
 */
enum SendDirection
{
    toLargerTaskNum,
    toSmallerTaskNum
};
/**
 * @brief Enumeration to specify whether an MPI ranks first sends or receives particles.
 */
enum CommMode
{
    SendFirst,
    RecvFirst
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Declaration of the class for the MPI parallelisation.
 */
template <typename BASE> class parallel : virtual public BASE
{
  public:
    parallel(){};

  private:
    typedef BASE B;
    typedef typename B::tDATA_IN DATA_IN;
    typedef typename B::tDATA_OUT DATA_OUT;
    typedef typename B::tDATA_IN_REF DATA_IN_REF;

  public:
    virtual int evaluate(int target, int mode, const int *ngblist, int extended_numngb_inbox, int *i_startnode,
                         int node, int i_node, void *data) = 0;

#ifndef PXP_OLDPARALLEL
    void secondary_loop_both_ways(int should_drift);
    void inner_comm_one_way(const int &commTask, SendDirection senddir, int should_drift);
    inline void evaluate_secondary(const int &should_drift);
    inline void pre_calc_send(const int &commTask) const;
    inline void pre_calc_recv(const int &commTask) const;
    inline void post_calc_send(const int &commTask) const;
    inline void post_calc_recv(const int &commTask) const;
#else // PXP_OLDPARALLEL
    int process_to_comm_with(const int b);
    void secondary_loop_one_way(SendDirection senddir, int DoPrimaryOnGPU, int should_drift, int group_particles,
                                int &ndone);
#endif // PXP_OLDPARALLEL
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_PARALLEL_HPP */
