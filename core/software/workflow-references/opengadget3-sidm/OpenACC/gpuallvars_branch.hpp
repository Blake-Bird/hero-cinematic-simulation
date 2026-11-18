/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Antonio Ragagnin.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef GPUALLVARS_H
#define GPUALLVARS_H

#define STRINGIFY(a) #a
#define ACC_APPLY_PRAGMA(...)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        VERBOSE(2, STRINGIFY(__VA_ARGS__));                                                                            \
        _Pragma(STRINGIFY(__VA_ARGS__))                                                                                \
    } while (0)

#if defined(ACC_OPENACC)

#define ACC_UPDATE_DEVICE(...) update device(__VA_ARGS__)
#define ACC_IF(condition, ...) __VA_ARGS__ if (condition)
#define ACC_ENTER(...) enter data __VA_ARGS__
#define ACC_EXIT(...) exit data __VA_ARGS__
#define ACC_CREATE(...) create(__VA_ARGS__)
#define ACC_DELETE(...) delete (__VA_ARGS__)

#elif defined(ACC_OPENMP)

#define ACC_UPDATE_DEVICE(...) target update to(__VA_ARGS__)
#define ACC_IF(condition, ...) __VA_ARGS__ if (condition)
#define ACC_ENTER(...) target enter data __VA_ARGS__
#define ACC_EXIT(...) target exit data __VA_ARGS__
#define ACC_CREATE(...) map(alloc : __VA_ARGS__)
// release decrease the counter similarly to openacc exit, delete zeros it
#define ACC_DELETE(...) map(release : __VA_ARGS__)

#endif // ACC_OPENMP

// Nitya
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern double costTotal;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef ACC_OPENACC
#pragma acc declare create(costTotal)
#endif // ACC_OPENACC

#ifdef C_SIMPLE
extern
#endif // C_SIMPLE
    struct gravity_walkdata
{
    MyLongDouble Pos[3];
    short int Type;
    MyFloat Mass;
    MyFloat OldAcc;
    short int TimeBin;
    MyFloat Hsml;
#ifdef ACC_ID
    int particle_index;
    MyIDType ID;
#endif // ACC_ID
};

#ifdef C_SIMPLE
extern
#endif // C_SIMPLE
    struct AccAll
{
    int no_tree_flag;
    int wrong_next_node_flag;
    int inside_data_region;
    int len_nextnode;
    int len_nodes_base;
    struct NODE *nodes_base_pointer;
    struct extNODE *extnodes_base_pointer;
    struct gravity_walkdata *walklitep_pointer;
    int *nextnode_pointer;
    int *active_particle_list_pointer;
    int last_numpart;
    int len_p;
    int len_ap;
    int *walk_lite_p_pointer;
    int go_gpu;
    int entered_gpu;
    int *SpeciesPos_pointer;
    int *SpeciesIdx_pointer;
#ifdef LT_STELLAREVOLUTION
    double *Acc_SFs_PDT_pointer;
    SF_Type *SFs_pointer;
    double *CoolZvalue_pointer;
#endif // LT_STELLAREVOLUTION
    int should_drift;
    int is_primary_loop_on_accelerator;
    // Nitya - keep track of old value
};

#ifdef LT_STELLAREVOLUTION
// Nitya
#ifdef ACC_OPENMP
#pragma omp declare target
#endif // ACC_OPENMP
extern double *Acc_SFs_PhysDensThresh_table;
#ifdef ACC_OPENMP
#pragma omp end declare target
#endif // ACC_OPENMP

#ifdef ACC_OPENACC
#pragma acc declare create(Acc_SFs_PhysDensThresh_table)
#endif // ACC_OPENACC

ACC_DECLARE_ROUTINE_BEFORE
double INLINE_FUNC get_PhysDensThresh_gpu(int i);
ACC_DECLARE_ROUTINE_AFTER

#endif // LT_STELLAREVOLUTION

void update_openacc_structures(void);
void update_walk_lite_p(void);
void openacc_setup_gravity(void);
void update_openacc_structures_exit(void);
void update_openacc_structures_enter(void);
void update_openacc_walklite_exit(void);

extern struct gravity_walkdata *WalkLiteP;
#ifdef ACC_OPENACC
#pragma acc declare create(WalkLiteP)
#endif // ACC_OPENACC
extern struct AccAll acc_all;

#define UCC unsigned char *
#define UC unsigned char
#define memset_gpu(dest, val, len)                                                                                     \
    {                                                                                                                  \
        UCC _dest = (UCC)(dest);                                                                                       \
        UC _val = (UC)(val);                                                                                           \
        size_t _len = (len);                                                                                           \
        while (_len-- > 0)                                                                                             \
        {                                                                                                              \
            *_dest++ = _val;                                                                                           \
        }                                                                                                              \
    }
#define memcpy_gpu(dest, src, len)                                                                                     \
    {                                                                                                                  \
        UCC _src = (UCC)(src);                                                                                         \
        UCC _dest = (UCC)(dest);                                                                                       \
        size_t _len = (len);                                                                                           \
        while (_len-- > 0)                                                                                             \
        {                                                                                                              \
            *_dest++ = *_src++;                                                                                        \
        }                                                                                                              \
    }

inline void openacc_gravity_drift(int &is_primary_loop_on_accelerator, int &should_drift)
{
    VERBOSE(3, "going to drift!");
    // we only drift if it is a large timestep, otherwise drifting goes as usual
    if (acc_all.go_gpu)
    {
#pragma omp parallel for
        for (int il = 0; il < NActivePart; il++)
        {
            ProcessedFlag[ActiveParticleList[il]] = 0;
        }

        VERBOSE(3, "going to drift!");
        is_primary_loop_on_accelerator = 1;
        should_drift = 0;
#pragma omp parallel for
        for (int no = All.MaxPart; no < All.MaxPart + Numnodestree; no++)
        {
            force_drift_node(no, All.Ti_Current);
        }
#pragma omp parallel for
        for (int counter = 0; counter < NumPart; counter++)
        {
            if (P[counter].Ti_current != All.Ti_Current)
            {
                drift_particle(counter, All.Ti_Current);
            }
        }

        update_walk_lite_p();

        update_openacc_structures();
    }
    acc_all.should_drift = should_drift;
    acc_all.is_primary_loop_on_accelerator = is_primary_loop_on_accelerator;
}

#endif // GPUALLVARS_H
