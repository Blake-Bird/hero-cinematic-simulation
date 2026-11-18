/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Base class and other basic things for the class template for pairwise interactions.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_EXECUTE_PXP_BASE_HPP
#define MSIDM_EXECUTE_PXP_BASE_HPP

#include <string>

//////////////////////////////////////////////////////////////////////////////////////////////////
namespace scatter
{
/**
 * @brief enumeration for particle types.
 */
enum particle_type
{
    type_gas = 0,
    type_dm = 1
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Class for defining a tree walk filter for exclusively picking DM particles.
 */
extern class treewalk_filter_dm_only_class
{
  public:
    int operator()(short int &Type) const
    {
        return (Type != type_dm);
    };
} treewalk_filter_dm_only;
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Class for defining a tree walk filter for exclusively picking gas particles.
 */
extern class treewalk_filter_bary_only_class
{
  public:
    int operator()(short int &Type) const
    {
        return (Type != type_gas);
    };
} treewalk_filter_bary_only;
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Data type for node lists used in MPI communication.
 */
struct data_nodelist_pxp
{
    int NodeList[NODELISTLENGTH_PXP];
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Node lists for pairwise particle interactions.
 */
class pxp_nodelist
{
  public:
    pxp_nodelist()
    {
        DataNodeList_pxp = nullptr;
        DataNodeListIn_pxp = nullptr;
        DataNodeListGet_pxp = nullptr;
    };

    static data_nodelist_pxp *DataNodeList_pxp, *DataNodeListIn_pxp, *DataNodeListGet_pxp;
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter::pxp_execute
{
/**
 * @brief Template of the base class for pairwise interactions.
 *
 * The template parameters specify the data types used in MPI communication and criteria
 * for searching for neighbouring particles with which the interactions are to be performed.
 * The required instances are specified for each physics module. For SIDM, this is done in
 * msidm_base.hpp.
 *
 */
template <typename DATA_IN, typename DATA_OUT, typename DATA_IN_REF, typename ISACTIVE, typename TREEWALK_FILTER>
class execute_pxp_base : public pxp_nodelist
{
  public:
    execute_pxp_base(const int module_in, const int neighboursearchcriteria_in, ISACTIVE isactive_in,
                     TREEWALK_FILTER treewalk_filter_in)
        : pxp_nodelist(), module(module_in), neighboursearchcriteria(neighboursearchcriteria_in), DataIn(nullptr),
          DataGet(nullptr), DataResult(nullptr), DataOut(nullptr), isactive(isactive_in),
          treewalk_filter(treewalk_filter_in)
    {
        CPU_PXPCOMPUTE = "";
        CPU_PXPPREP = "";
        CPU_PXPCOMM = "";
        CPU_PXPWAIT = "";
        CPU_PXPNETWORK = "";
        CPU_PXPLOCAL = "";
        CPU_PXPPARALLEL = "";
    };

    typedef DATA_IN tDATA_IN;
    typedef DATA_OUT tDATA_OUT;
    typedef DATA_IN_REF tDATA_IN_REF;

    const int module;
    const int neighboursearchcriteria;
    DATA_IN *DataIn, *DataGet;
    DATA_OUT *DataResult, *DataOut;

  protected:
    std::string CPU_PXPCOMPUTE, CPU_PXPPREP, CPU_PXPCOMM, CPU_PXPWAIT, CPU_PXPNETWORK, CPU_PXPLOCAL, CPU_PXPPARALLEL;
    std::string module_name;

    ISACTIVE isactive;
    TREEWALK_FILTER treewalk_filter;
};
} // namespace scatter::pxp_execute

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_EXECUTE_PXP_BASE_HPP */
