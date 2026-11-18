/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file with declarations to build the communication schedule.
 *
 * This header file is part of the MPI parallelization of the SIDM and IDM modules.
 * It contains the declarations of classes used to create the communication schedule,
 * i.e., the information about the order in which the MPI ranks communicate with each
 * other to exchange particle data and perform dependent calculations.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer and Antonio Ragagnin.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_COMM_HPP
#define MSIDM_COMM_HPP

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Declaration of the base class for a communication task.
 */
class comm_list_element_base
{
  public:
    comm_list_element_base();

    int priority;
    int task_j;

    bool operator<(const comm_list_element_base &cle) const;
    bool operator>(const comm_list_element_base &cle) const;
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Declaration of the class for a communication task.
 */
class comm_list_element : public comm_list_element_base
{
  public:
    comm_list_element();
    comm_list_element(comm_list_element_base &cleb);

    int task_i;
    comm_list_element *prev_element;
    comm_list_element *next_element;

    void append_at_end(
        comm_list_element *new_element); // append at end without changing next_element of appended element
    void append_as_last(comm_list_element *new_element); // append at end with next_element as nullptr
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Declaration of the class for a row of the communication schedule.
 */
class comm_schedule_row
{
  public:
    comm_schedule_row(int num_tasks);
    ~comm_schedule_row();

    int &operator[](int task);

    comm_schedule_row *next_row;

  private:
    int *ctasks;
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Declaration of the class for the communication schedule.
 */
class comm_list
{
  public:
    comm_list();
    ~comm_list();

    int process_to_comm_with(int b) const;
    int get_num_comm_rows() const;
    void create();

  private:
    int rank, numtasks;
    int num_comm_rows;
    int *comm_schedule;
    bool created;
    const float exchange_sort_stop_criterium;

    inline void create_step1(comm_list_element_base *my_comm_list);
    inline void create_step2(comm_list_element_base *my_comm_list, int &max_num_comm);
    inline void create_step3(comm_list_element_base *my_comm_list, int &max_num_comm,
                             comm_list_element_base *recv_comm_list);
    inline void create_step4(int &max_num_comm, comm_list_element_base *recv_comm_list,
                             comm_list_element &head_element);
    inline void create_step5(comm_list_element &head_element);
    inline void create_step6(comm_list_element &head_element, comm_schedule_row *first_schedule_row);
    inline void create_step7(comm_schedule_row *first_schedule_row);
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_COMM_HPP */
