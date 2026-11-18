/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Variables for the substructure finding algorithm (Subfind).
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../CodeBase/allvars.hpp"

#ifdef SUBFIND
#include "subfind.hpp"
#include "fof.hpp"

int Ncollective;
int MaxNsubgroups;
int Nsubgroups;
int TotNsubgroups;

#ifdef SUBFIND_COUNT_BIG_HALOS
int Nbiggroups;
int TotNbiggroups;
#endif // SUBFIND_COUNT_BIG_HALOS

int *grouptasks, ntaskspergroup;
int SubThisTask, SubNTask, color, ncolors;
MPI_Comm SUB_COMM;

#ifdef LT_ADD_GAL_TO_SUB
int bc_use_index[LT_ADD_GAL_TO_SUB];
char bc_name[LT_ADD_GAL_TO_SUB][2];
double CB07_used_lumdist, H0Filters, OMFilters, OLFiletrs;
#endif // LT_ADD_GAL_TO_SUB

struct subgroup_properties *SubGroup;

struct nearest_r2_data *R2Loc;

struct nearest_ngb_data *NgbLoc;

struct r2data *R2list;

// struct nearest_ngb_data *NgbLoc;
// struct nearest_r2_data *R2Loc;

MyFloat *Dist2list;
#endif // SUBFIND
