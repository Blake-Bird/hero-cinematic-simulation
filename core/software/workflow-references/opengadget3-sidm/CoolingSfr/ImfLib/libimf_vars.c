/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particluar Luca Tornatore.
 *
 * @copyright Copyright (c) 2025, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "libimf_vars.h"

const char *IMF_Spec_Labels[] = {"Power Law", "Whatever"};

IMF_SPEC IMF_Spec;

IMF_Type *IMFs, IMFu;

int IMFs_dim;

int Nof_TimeDep_IMF;

EXTERNALIMF **externalIMFs_byMass;
EXTERNALIMF **externalIMFs_byNum;
EXTERNALIMF **externalIMFs_byEgy;
char **externalIMFs_names;
int NexternalIMFs = 0;
