/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This header file contains the declaration of the domain decomposition functions.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include <stdio.h>

void domain_print(void);
void domain_findSplit_work_balanced(int ncpu, int ndomain);
void domain_findSplit_load_balanced(int ncpu, int ndomain);
int domain_sort_loadorigin(const void *a, const void *b);
int domain_sort_segments(const void *a, const void *b);
void domain_assign_load_or_work_balanced(int mode, int multipledomains);
void domain_allocate(void);
int domain_check_memory_bound(int multipledomains);
int domain_compare_key(const void *a, const void *b);
int domain_compare_key(const void *a, const void *b);
int domain_compare_toplist(const void *a, const void *b);
int domain_countToGo(size_t nlimit);
void domain_Decomposition(int UseAllTimeBins, int SaveKeys);

#ifdef WRITE_KEY_FILES
int domain_decompose(int SaveKeys);
int domain_determineTopTree(int SaveKeys, int multipledomains);
void domain_findExtent(int SaveKeys);
#else // WRITE_KEY_FILES
int domain_decompose(void);
int domain_determineTopTree(void);
void domain_findExtent(void);
#endif // WRITE_KEY_FILES

void domain_exchange(void);
void domain_findExchangeNumbers(int task, int partner, int sphflag, int *send, int *recv);
void domain_findSplit(int cpustart, int ncpu, int first, int last);
void domain_findSplit_balanced(int cpustart, int ncpu, int first, int last);
void domain_free(void);
void domain_shiftSplit(void);
void domain_sumCost(void);
int domain_topsplit(int node, peanokey startkey);
int domain_topsplit_local(int node, peanokey startkey, int mode);
int domain_topsplit_special(void);
int domain_compare_key(const void *a, const void *b);

#ifdef AR_PARALLEL_LOCAL_REFINE
int domain_check_for_local_refine(int i, double countlimit, double costlimit, int layer);
#else // AR_PARALLEL_LOCAL_REFINE
int domain_check_for_local_refine(int i, double countlimit, double costlimit);
#endif // AR_PARALLEL_LOCAL_REFINE

void domain_free_trick(void);
void domain_allocate_trick(void);
int domain_recursively_combine_topTree(int start, int ncpu);
void domain_walktoptree(int no);
void mysort_domain(void *b, size_t n, size_t s);

void domain_recomposition();

#endif // DOMAIN_HPP
