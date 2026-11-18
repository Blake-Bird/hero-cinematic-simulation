/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file with definitions to build the communication schedule.
 *
 * This source file is part of the MPI parallelization of the pairwise interaction
 * class template. It contains the definitions of classes used to create the
 * communication schedule, i.e., the information about the order in which the MPI
 * ranks communicate with each other to exchange particle data and perform dependent
 * calculations.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer and Antonio Ragagnin.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "comm.hpp"
#include "tools.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////

using namespace scatter;
using namespace scatter::pxp_execute;

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor for the base class of a communication task.
 */
comm_list_element_base::comm_list_element_base() : priority(0)
{
    unsigned int i = 0;
    task_j = i - 1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Operator for comparing priority of communication tasks.
 */
bool comm_list_element_base::operator<(const comm_list_element_base &cle) const
{
    return priority < cle.priority;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Operator for comparing priority of communication tasks.
 */
bool comm_list_element_base::operator>(const comm_list_element_base &cle) const
{
    return priority > cle.priority;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor for the class of a communication task.
 */
comm_list_element::comm_list_element() : prev_element(nullptr), next_element(nullptr)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor for the class of a communication task.
 */
comm_list_element::comm_list_element(comm_list_element_base &cleb)
    : comm_list_element_base(cleb), prev_element(nullptr), next_element(nullptr)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to append a communication task at the end.
 */
void comm_list_element::append_at_end(comm_list_element *new_element)
{
    if (new_element != nullptr)
    {
        if (next_element == nullptr)
        {
            next_element = new_element;
            new_element->prev_element = this;
        }
        else
        {
            next_element->append_at_end(new_element);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to append a communication task as last element (no subsequent elements).
 */
void comm_list_element::append_as_last(comm_list_element *new_element)
{
    if (new_element != nullptr)
    {
        if (next_element == nullptr)
        {
            next_element = new_element;
            new_element->prev_element = this;
            new_element->next_element = nullptr;
        }
        else
        {
            next_element->append_as_last(new_element);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor for a row of the communication schedule.
 */
comm_schedule_row::comm_schedule_row(int num_tasks) : next_row(nullptr)
{
    ctasks = new int[num_tasks];
    for (int loop = 0; loop < num_tasks; ++loop)
        ctasks[loop] = num_tasks;
}

/**
 * @brief Destructor for a row of the communication schedule.
 */
comm_schedule_row::~comm_schedule_row()
{
    delete[] ctasks;
}

/**
 * @brief Operator for accessing a specific communication task.
 */
int &comm_schedule_row::operator[](int task)
{
    return ctasks[task];
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor for the communication schedule.
 */
comm_list::comm_list() : num_comm_rows(0), created(false), exchange_sort_stop_criterium(0.7)
{
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Destructor for deleting communication schedule.
 */
comm_list::~comm_list()
{
    if (created)
        delete[] comm_schedule;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method that reads from the schedule with which process to communicate in step b.
 */
int comm_list::process_to_comm_with(int b) const
{
    return comm_schedule[b];
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to read the number of rows in the communication schedule.
 */
int comm_list::get_num_comm_rows() const
{
    return num_comm_rows;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method for creating communication schedule.
 */
void comm_list::create()
{
    // 1. every task creates list with communications
    comm_list_element_base my_comm_list[numtasks];
    create_step1(my_comm_list);

    // 2. every task sorts the list (my_comm_list)
    int max_num_comm = 0;
    create_step2(my_comm_list, max_num_comm);

    // 3. make mpi exchane using max_num_comm
    comm_list_element_base *recv_comm_list = new comm_list_element_base[max_num_comm * numtasks];
    create_step3(my_comm_list, max_num_comm, recv_comm_list);

    // 4. build global list
    comm_list_element head_element;
    create_step4(max_num_comm, recv_comm_list, head_element);
    delete[] recv_comm_list;

    // 5. sort list
    create_step5(head_element);

    // 6. build comm plan
    comm_schedule_row *first_schedule_row = new comm_schedule_row(numtasks);
    create_step6(head_element, first_schedule_row);

    // 7. build individual comm list
    create_step7(first_schedule_row);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of step 1 for creating the communication schedule.
 *
 * Description of step 1: Given that every process knows how many particles it will
 * send to and receive from other processes, it creates a list with all its communications
 * to other processes. The list contains also the number of particles that are exchanged
 * with each process (the sum of send and receive). It is used later on to prioritise the
 * communication pairs of processes.
 *
 */
inline void comm_list::create_step1(comm_list_element_base *my_comm_list)
{
    for (int loop = 0; loop < numtasks; ++loop)
    {
        my_comm_list[loop].task_j = loop;
    }
    my_comm_list[rank].priority = 0;
    for (int loop = 0; loop < rank; ++loop)
    {
        my_comm_list[loop].priority = Send_count[loop];
        my_comm_list[loop].priority += Recv_count[loop];
    }
    for (int loop = rank; loop < numtasks; ++loop)
    {
        my_comm_list[loop].priority = Send_count[loop];
        my_comm_list[loop].priority += Recv_count[loop];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of step 2 for creating the communication schedule.
 *
 * Description of step 2: Each process sorts the previously created list according
 * to the number of particles to be exchanged, i.e. according to priority.
 *
 */
inline void comm_list::create_step2(comm_list_element_base *my_comm_list, int &max_num_comm)
{
    scatter::tools::sort_large_to_small(my_comm_list, numtasks);
    int num_comm = 0;
    for (int loop = 0; loop < numtasks; ++loop)
    {
        if (my_comm_list[loop].priority > 0)
        {
            ++num_comm;
        }
        else
        {
            break;
        }
    }

    unsigned long send_ncomm = num_comm, recv_ncomm;
    MPI_Allreduce(&send_ncomm, &recv_ncomm, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    max_num_comm = recv_ncomm;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of step 3 for creating the communication schedule.
 *
 * Description of step 3: Next, all processes exchange their communication lists.
 * As a result, every process knows about all communications that should be executed.
 * We note that the length of these lists must not scale with $N^2$ (N being the
 * number of processes). In case communication is not required between all possible
 * pairs of processes (this is typically the case), it can be shorter.
 *
 */
inline void comm_list::create_step3(comm_list_element_base *my_comm_list, int &max_num_comm,
                                    comm_list_element_base *recv_comm_list)
{
    comm_list_element_base *send_comm_list = new comm_list_element_base[max_num_comm];
    for (int loop = 0; loop < max_num_comm; ++loop)
    {
        send_comm_list[loop] = my_comm_list[loop];
    }
    int cnt = sizeof(comm_list_element_base) * max_num_comm;
    MPI_Allgather(send_comm_list, cnt, MPI_BYTE, recv_comm_list, cnt, MPI_BYTE, MPI_COMM_WORLD);
    delete[] send_comm_list;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of step 4 for creating the communication schedule.
 *
 * Description of step 4: From the received lists every process builds a single list
 * with all communications. Here, the processes start with the highest priority elements
 * of all individual lists and keep for each communication pair of processes only one
 * entry. Moreover, the final list contains only communication pairs for which a positive
 * number of particles is exchanged.
 *
 */
inline void comm_list::create_step4(int &max_num_comm, comm_list_element_base *recv_comm_list,
                                    comm_list_element &head_element)
{
    comm_list_element *last_element = &head_element;
    for (int loopC = 0; loopC < max_num_comm; ++loopC)
    {
        for (int loopT = 0; loopT < numtasks; ++loopT)
        {
            comm_list_element_base &rcle = recv_comm_list[loopT * max_num_comm + loopC];
            if (rcle.priority > 0 && loopT > rcle.task_j) // append element to list
            {
                comm_list_element *new_cle = new comm_list_element(rcle);
                new_cle->task_i = loopT;
                new_cle->prev_element = last_element;
                last_element->next_element = new_cle;
                last_element = new_cle;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of step 5 for creating the communication schedule.
 *
 * Description of step 5: The list containing all communications from the previous
 * step is sorted according to priority. We note that we don't require that the list
 * be perfectly sorted. Instead, we interrupt the sorting when the list is sufficiently
 * ordered.
 *
 */
inline void comm_list::create_step5(comm_list_element &head_element)
{
    using namespace scatter::tools;

    // sort list
    comm_list_element *container = &head_element;
    int iter = 0;
    comm_list_element *node = container->next_element; // the next element of the head is the first one
    while (node != nullptr) // iterate over the top unsorted node
    {
        int priority_max = node->priority;
        comm_list_element *node_with_max_priority = node;
        comm_list_element *test = node->next_element;

        double mean = priority_max;
        double squares = mean * mean;
        int size = 1;
        while (test != nullptr) // iterate over the unsorted part of the list
        {
            const int &priority = test->priority;
            if (priority > priority_max)
            {
                priority_max = priority;
                node_with_max_priority = test;
            }
            {
                const double my_priority = static_cast<double>(priority);
                mean += my_priority;
                squares += my_priority * my_priority;
            }
            ++size;
            test = test->next_element;
        }

        // now we swap the first unsorted node with the one with max priority
        if (node_with_max_priority != node)
        {
            // update neighbouting nodes
            // see https://www.javatpoint.com/program-to-swap-nodes-in-a-singly-linked-list-without-swapping-data
            comm_list_element *prev_node = node->prev_element;
            comm_list_element *&prev_max = node_with_max_priority->prev_element;
            comm_list_element *next_node = node->next_element;
            comm_list_element *&next_max = node_with_max_priority->next_element;

            prev_node->next_element = node_with_max_priority;
            prev_max->next_element = node;
            next_node->prev_element = node_with_max_priority;
            if (next_max != nullptr) // only next_max can be NULL, as it comes after node
            {
                next_max->prev_element = node;
            }
            comm_list_element *tmp_next = node->next_element;

            node->next_element = next_max;
            next_max = tmp_next;
            node->prev_element = prev_max;
            node_with_max_priority->prev_element = prev_node;
        }
        container = container->next_element;
        node = container->next_element;
        ++iter;

        // compute statistics
        mean /= size;
        squares /= size;
        const double stddev = sqrt(squares - mean * mean);
        const bool to_break = (stddev / mean) < exchange_sort_stop_criterium;

        if (node == head_element.next_element->next_element || node == nullptr || to_break)
        {
            coutr(extra) << " ThisTask" << rank << "; iter:" << iter << "; len:" << size << "; mean:" << mean
                         << "; fstddev:" << stddev / mean << endl;
        }
        if (to_break)
        {
            break;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of step 6 for creating the communication schedule.
 *
 * Description of step 6: Based on the partially sorted list, every process builds
 * the overall communication plan on its own, no further communication is involved.
 * The plan consists of several steps that are meant to be executed in a consecutive
 * order. At each step, several pairs do their communication and also the scattering
 * computations of the exchanged particles. These pairs do not conflict with each
 * other, i.e. no process can be assigned more than once per step. The plan is built
 * by trying to fulfil the highest priorities first. The steps are filled after each
 * other with pair-wise communications. If a communication does not fit in a step
 * because a corresponding process is already busy, it is queued and retried for the
 * next step. We turn to the next step when the queue for the communication that
 * should be assigned for the current step is empty or when all processes are already
 * assigned a communication for the current step.
 *
 */
inline void comm_list::create_step6(comm_list_element &head_element, comm_schedule_row *first_schedule_row)
{
    comm_list_element *queue[2] = {head_element.next_element, nullptr};
    comm_list_element *queue_last[2] = {head_element.next_element, nullptr};
    comm_schedule_row *csr = first_schedule_row;
    int i = 0;
    bool do_rows = true;
    while (do_rows)
    {
        int num_assigned_comm = 0;
        while (queue[i % 2] != nullptr) // if false we need to create next row
        {
            if ((*csr)[queue[i % 2]->task_i] >= numtasks && (*csr)[queue[i % 2]->task_j] >= numtasks)
            {
                (*csr)[queue[i % 2]->task_i] = queue[i % 2]->task_j;
                (*csr)[queue[i % 2]->task_j] = queue[i % 2]->task_i;
                num_assigned_comm += 2;
                // delete comm_list element
                comm_list_element *cled = queue[i % 2];
                queue[i % 2] = cled->next_element;
                if (num_assigned_comm >= numtasks - 1)
                {
                    if (queue[(i + 1) % 2] != nullptr)
                    {
                        if (queue_last[(i + 1) % 2] != nullptr)
                        {
                            queue_last[(i + 1) % 2]->append_at_end(queue[i % 2]);
                        }
                        else
                        {
                            queue[(i + 1) % 2]->append_at_end(queue[i % 2]);
                        }
                    }
                    else
                    {
                        queue[(i + 1) % 2] = queue[i % 2];
                    }
                    queue_last[(i + 1) % 2] = queue_last[i % 2];
                    queue[i % 2] = nullptr;
                }
                delete cled;
            }
            else
            {
                // move comm_list element to other queue
                comm_list_element *next_element = queue[i % 2]->next_element;

                if (queue[(i + 1) % 2] == nullptr)
                {
                    queue[(i + 1) % 2] = queue[i % 2];
                    queue[(i + 1) % 2]->next_element = nullptr;
                }
                else
                {
                    if (queue_last[(i + 1) % 2] != nullptr)
                    {
                        queue_last[(i + 1) % 2]->append_as_last(queue[i % 2]);
                    }
                    else
                    {
                        queue[(i + 1) % 2]->append_as_last(queue[i % 2]);
                    }
                }
                queue_last[(i + 1) % 2] = queue[i % 2];
                queue[i % 2] = next_element;
            }
        }
        queue_last[i % 2] = nullptr;
        ++i;
        if (queue[i % 2] != nullptr)
        {
            // create new comm-row
            csr->next_row = new comm_schedule_row(numtasks);
            csr = csr->next_row;
        }
        else
        {
            do_rows = false;
        }
    }
    num_comm_rows = i;
    // by now all comm_list_elements should be deleted
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Definition of step 7 for creating the communication schedule.
 *
 * Description of step 7: Every process extracts its communication schedule from
 * the overall communication plan.
 *
 */
inline void comm_list::create_step7(comm_schedule_row *first_schedule_row)
{
    comm_schedule = new int[num_comm_rows];
    comm_schedule_row *csr = first_schedule_row;
    for (int loop = 0; loop < num_comm_rows; ++loop)
    {
        comm_schedule[loop] = (*csr)[rank];
        comm_schedule_row *csrd = csr;
        csr = csr->next_row;
        delete csrd;
    }
    created = true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
