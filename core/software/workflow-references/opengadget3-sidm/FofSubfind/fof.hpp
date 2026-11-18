/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header for the parallel FoF group finder.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef FOF_HPP
#define FOF_HPP

#include "../CodeBase/allvars.hpp"

void fof_exchange_group_data(void);
int fof_compare_FOF_PList_MinID(const void *a, const void *b);
int fof_compare_FOF_GList_MinID(const void *a, const void *b);
int fof_compare_FOF_GList_MinIDTask(const void *a, const void *b);
int fof_compare_FOF_GList_LocCountTaskDiffMinID(const void *a, const void *b);
int fof_compare_FOF_GList_ExtCountMinID(const void *a, const void *b);
int fof_compare_Group_GrNr(const void *a, const void *b);
int fof_compare_Group_MinIDTask(const void *a, const void *b);
int fof_compare_Group_MinID(const void *a, const void *b);
int fof_compare_ID_list_GrNrID(const void *a, const void *b);

void fof_finish_group_properties(void);

extern int Ngroups, TotNgroups;
extern long long TotNids;
extern int NAllocatedGroups; // we log the mymalloc size of Group for debugging purposes;

typedef struct
{
    MyIDType MinID;
    MyIDType MinIDTask;
    int LocCount;
    int ExtCount;
#ifdef DENSITY_SPLIT_BY_TYPE
    int LocDMCount;
    int ExtDMCount;
#endif // DENSITY_SPLIT_BY_TYPE
    int GrNr;
} fof_group_list;

typedef struct
{
    int Len;
    unsigned int Offset;
    MyIDType MinID;
    MyIDType MinIDTask;
    int GrNr;
#ifndef FOF_EXTENDED_PROPERTIES
    int LenType[6];
    MyOutputFloat MassType[6];
#endif // FOF_EXTENDED_PROPERTIES
    MyOutputFloat Mass;
    MyOutputFloat CM[3];
    MyOutputFloat Vel[3];
    MyLongDouble FirstPos[3];
#ifdef SFR
    double Sfr;
#endif // SFR
#ifdef BLACK_HOLES
    MyOutputFloat BH_Mass;
    MyOutputFloat BH_Mdot;
    MyOutputFloat MaxDens;
    int index_maxdens, task_maxdens;
#endif // BLACK_HOLES
#ifdef LMB_CR_OUTPUT_SYNCHROTRON
    MyAtLeastDouble Synchrotron[8];
#endif // LMB_CR_OUTPUT_SYNCHROTRON

#ifdef SUBFIND
    int Nsubs;
    int FirstSub;
    MyLongDouble Pos[3];
    MyOutputFloat M_TopHat200, R_TopHat200;
    MyOutputFloat M_Mean200, R_Mean200;
    MyOutputFloat M_Crit200, R_Crit200;
#ifdef SO_VEL_DISPERSIONS
    MyOutputFloat VelDisp_TopHat200, VelDisp_Mean200, VelDisp_Crit200;
#endif // SO_VEL_DISPERSIONS
#ifdef SO_BAR_INFO
    MyOutputFloat M_Mean500, R_Mean500;
    MyOutputFloat M_Crit500, R_Crit500;
    MyOutputFloat M_Crit2500, R_Crit2500;
#ifdef SO_VEL_DISPERSIONS
    MyOutputFloat VelDisp_Mean500, VelDisp_Crit500, VelDisp_Crit2500;
#endif // SO_VEL_DISPERSIONS
#endif // SO_BAR_INFO
    int ContaminationLen;
    MyOutputFloat ContaminationMass;
#ifdef SO_BAR_INFO
    MyOutputFloat gas_mass[6], star_mass[6], temp[6], xlum[6], ygas[6];
#endif // SO_BAR_INFO
#endif // SUBFIND

#ifdef FOF_EXTENDED_PROPERTIES
    MyOutputFloat VelDisp, Rmax, Vmax;
    MyOutputFloat ToI[9];
    MyOutputFloat AngMom[9];
    MyOutputFloat Pos[3];
    unsigned short Origintask;
#endif // FOF_EXTENDED_PROPERTIES
} group_properties;

typedef struct
{
    MyIDType ID;
    unsigned int GrNr;
} fof_id_list;

extern group_properties *Group;

#ifdef ALTERNATIVE_PSORT
void fof_sort_FOF_GList_LocCountTaskDiffMinID(fof_group_list *data, int ndata);

void fof_sort_FOF_GList_ExtCountMinID(fof_group_list *data, int ndata);

void fof_sort_Group_GrNr(group_properties *data, int ndata);

void fof_sort_ID_list_GrNrID(fof_id_list *data, int ndata);
#endif // ALTERNATIVE_PSORT

#endif // FOF_HPP
