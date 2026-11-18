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

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "gpuallvars_branch.hpp"

struct gravity_walkdata *WalkLiteP;

int NodesSize;
// #pragma acc declare create(NodesSize);

struct AccAll acc_all = {.no_tree_flag = 0,
                         .wrong_next_node_flag = 0,
                         .inside_data_region = 0,
                         .len_nextnode = -1,
                         .len_nodes_base = -1,
                         .last_numpart = -1,
                         .len_p = -1,
                         .len_ap = -1,
                         .go_gpu = -1,
                         .entered_gpu = 0,

#ifdef LT_STELLAREVOLUTION
                         .SFs_pointer = NULL
#endif // LT_STELLAREVOLUTION
};

#if defined(LT_STELLAREVOLUTION) && defined(ACC_GRADIENTS)

#define OTHER_ENTER                                                                                                    \
    , WalCool_n_Ab, SpeciesIdx [0:LT_NMet], SpeciesPos [0:WalCool_n_Ab], CoolZvalue [0:ZBins],                         \
    Acc_SFs_PhysDensThresh_table [0:SFs_dim * ZBins], SFs [0:SFs_dim]
#define OTHER_EXIT                                                                                                     \
    , WalCool_n_Ab, SpeciesIdx [0:LT_NMet], SpeciesPos [0:WalCool_n_Ab], CoolZvalue [0:ZBins],                         \
    Acc_SFs_PhysDensThresh_table [0:SFs_dim * ZBins], SFs [0:SFs_dim]

#else // LT_STELLAREVOLUTION
#define OTHER_ENTER , NodesSize
#define OTHER_EXIT , NodesSize
#endif // LT_STELLAREVOLUTION

#ifdef ACC
int AccErrorCode = 0;
#endif // ACC

#if defined(LT_STELLAREVOLUTION) && defined(ACC_GRADIENTS)
double *Acc_SFs_PhysDensThresh_table;
#define SFs_PhysDensThresh(i, j) Acc_SFs_PhysDensThresh_table[i * ZBins + j]

/**
 * @brief Computes the physical density threshold for star formation on the GPU.
 *
 * This function retrieves the star formation index and IMF (Initial Mass Function)
 * index for the given particle `i`, and calculates the physical density threshold
 * (`myPhysDensThresh`) based on metallicity-dependent star formation models, if applicable.
 * The function performs interpolation when the threshold depends on metallicity values.
 *
 * @param i The particle index for which to compute the physical density threshold.
 *
 * @return The computed physical density threshold for star formation.
 */
double INLINE_FUNC get_PhysDensThresh_gpu(int i)
{
    double myPhysDensThresh;
    int mySFi, myIMFi, flag;
    double Zcool, p;

    get_SF_index(i, &mySFi, &myIMFi);

    if (SFs[mySFi].SFTh_Zdep)
    {
        /*
                                                                                                    if the effective
           model is allowed to depend on metallicity, thresholds and evaporation factors differ for different Z.
        */
        getindex(&CoolZvalue[0], 0, ZBins - 1, &Zcool, &flag);

        if (flag == 0 || flag == ZBins - 1)
        {
            myPhysDensThresh = SFs_PhysDensThresh(mySFi, flag);
        }
        else
        {
            p = (Zcool - CoolZvalue[flag + 1]) / (CoolZvalue[flag + 1] - CoolZvalue[flag]);
            /* interpolate */
            myPhysDensThresh = SFs_PhysDensThresh(mySFi, flag) * (1 - p) + SFs_PhysDensThresh(mySFi, flag + 1) * p;
        }
    }
    else
    {
        myPhysDensThresh = SFs_PhysDensThresh(mySFi, 0);
    }

    return (myPhysDensThresh);
}
#endif // LT_STELLAREVOLUTION

/**
 * @brief Updates OpenACC structures and manages transitions between CPU and GPU.
 *
 * This function checks the current status of the OpenACC data structures and determines
 * whether to enter or exit the GPU data region based on the flags and conditions. It ensures
 * that the necessary tree structures are set, and handles potential errors related to
 * restarts or incorrect configurations.
 */
void update_openacc_structures(void)
{
    NodesSize = Numnodestree;
    VERBOSE(3, "inside data region? %d - notree flag=%d", acc_all.inside_data_region, acc_all.no_tree_flag);
    if (acc_all.no_tree_flag < 0 || acc_all.len_nodes_base < 0)
    {
        ENDRUN_MAIN(
            "Lenght of the Tree is not set. I Never tested the code with restart. \n Probably viriables Acc* must "
            "be set in restart.c if the tree is not rebuilt on runtime. notreeflag = %d, len_nodes_base=%d",
            acc_all.no_tree_flag, acc_all.len_nodes_base);
    }

    // dump_memory_table();
    if (acc_all.go_gpu)
    {
        if (acc_all.inside_data_region)
        {
            update_openacc_structures_exit();
        }
        update_openacc_structures_enter();
    }
}

/**
 * @brief Prepares and enters the GPU data region, updating OpenACC structures.
 *
 * This function initializes and transfers key data structures (such as particle arrays,
 * tree nodes, and force tables) to the GPU for computation. It checks and sets necessary
 * flags, ensuring consistency between the CPU and GPU environments. If tree rebuilding
 * is required, it handles the associated pointers and updates them accordingly.
 */
void update_openacc_structures_enter(void)
{
    NodesSize = Numnodestree;

    VERBOSE(3, "=> acc enter data NumPart=%d", NumPart);

    acc_all.len_nextnode = (All.MaxPart + NTopnodes);
    acc_all.len_nodes_base = (MaxNodes + 1);
    acc_all.len_p = NumPart;
    acc_all.len_ap = NumPart;

    VERBOSE(3, "Entering device");
    ACC_PRAGMA(ACC_ENTER(ACC_CREATE(WalkLiteP [0:NumPart], ActiveParticleList [0:NumPart])));
    ACC_PRAGMA(ACC_ENTER(ACC_CREATE(Extnodes_base [0:acc_all.len_nodes_base], Nextnode [0:acc_all.len_nextnode],
                                    Nodes_base [0:acc_all.len_nodes_base], shortrange_table [0:NTAB],
                                    shortrange_table_potential [0:NTAB] OTHER_ENTER)));

    acc_all.entered_gpu = 1;

    VERBOSE(3, "Updating device");

    ACC_PRAGMA(ACC_UPDATE_DEVICE(NodesSize));

    ACC_PRAGMA(ACC_UPDATE_DEVICE(ActiveParticleList [0:NActivePart]));
    ACC_PRAGMA(ACC_UPDATE_DEVICE(WalkLiteP [0:NumPart]));
    ACC_PRAGMA(ACC_UPDATE_DEVICE(Extnodes_base[:acc_all.len_nodes_base]));
    ACC_PRAGMA(ACC_IF(acc_all.no_tree_flag, ACC_UPDATE_DEVICE(Nextnode [0:NumPart])));
#if defined(LT_STELLAREVOLUTION) && defined(ACC_GRADIENTS)
    ACC_PRAGMA(ACC_UPDATE_DEVICE(SpeciesIdx [0:LT_NMet]));
    ACC_PRAGMA(ACC_UPDATE_DEVICE(SpeciesPos [0:WalCool_n_Ab]));
    ACC_PRAGMA(ACC_UPDATE_DEVICE(CoolZvalue [0:ZBins]));
    ACC_PRAGMA(ACC_UPDATE_DEVICE(SFs [0:SFs_dim], Acc_SFs_PhysDensThresh_table [0:SFs_dim * ZBins]));
    // myHyd/myHel/HydPos/HelPos and WalCool_n are "acc declare create" globals (allvars.hpp) that are
    // otherwise never synced to the device: without this, the device copies of HydPos/HelPos stay
    // uninitialized and are used as array indices into Metallicities[] in set_metallicities_gpu(),
    // which causes an illegal-address fault in the gradients kernel.
    ACC_PRAGMA(ACC_UPDATE_DEVICE(myHyd, myHel, HydPos, HelPos, UseHeNumberRatio, WalCool_n [0:6]));
#endif // LT_STELLAREVOLUTION

#ifdef PERIODIC
    ACC_PRAGMA(ACC_UPDATE_DEVICE(boxHalf, boxSize, fac_intp, fcorrx, fcorry, fcorrz, potcorr));
#endif // PERIODIC

    ACC_PRAGMA(ACC_UPDATE_DEVICE(Nextnode [All.MaxPart:acc_all.len_nextnode - All.MaxPart]) if (acc_all.no_tree_flag));
    ACC_PRAGMA(ACC_UPDATE_DEVICE(Nodes_base [0:NodesSize]));

    ACC_PRAGMA(ACC_UPDATE_DEVICE(shortrange_table [0:NTAB], shortrange_table_potential [0:NTAB]));
    ACC_PRAGMA(ACC_UPDATE_DEVICE(All, TakeLevel, MaxNodes, Numnodestree, NActivePart, AccErrorCode, NumPart, NodesSize,
                                 N_gas));
#ifdef PERIODIC
#ifdef LONG_X
    ACC_PRAGMA(ACC_UPDATE_DEVICE(boxHalf_X, boxSize_X));
#endif // LONG_X
#ifdef LONG_Y
    ACC_PRAGMA(ACC_UPDATE_DEVICE(boxHalf_Y, boxSize_Y));
#endif // LONG_Y
#ifdef LONG_Z
    ACC_PRAGMA(ACC_UPDATE_DEVICE(boxHalf_Z, boxSize_Z));
#endif // LONG_Z
#endif // PERIODIC

    VERBOSE(-1, "Copied data to GPU region");

    /*save pointers*/
    // WalkLiteP (and, under LT_STELLAREVOLUTION+ACC_GRADIENTS, Acc_SFs_PhysDensThresh_table) are
    // myfree()'d/mymalloc()'d again every timestep in on_end_of_timestep()/on_beginning_of_timestep(),
    // independently of whether the tree was rebuilt. The "enter data create" above always runs against
    // whatever the CURRENT host pointers are, so the matching delete in update_openacc_structures_exit()
    // must always delete against those same current pointers too - gating this save behind no_tree_flag
    // (which only tracks tree rebuilds) left exit() deleting a stale pointer whenever WalkLiteP moved
    // without the tree being rebuilt, permanently orphaning the freshly created device buffer.
    acc_all.active_particle_list_pointer = ActiveParticleList;
    acc_all.walklitep_pointer = WalkLiteP;
    acc_all.nodes_base_pointer = Nodes_base;
    acc_all.nextnode_pointer = Nextnode;
    acc_all.extnodes_base_pointer = Extnodes_base;

#if defined(LT_STELLAREVOLUTION) && defined(ACC_GRADIENTS)
    acc_all.SpeciesIdx_pointer = SpeciesIdx;
    acc_all.SpeciesPos_pointer = SpeciesPos;
    acc_all.SFs_pointer = (SF_Type *)SFs; // boh ma viene fatto in vari luoghi
    acc_all.CoolZvalue_pointer = CoolZvalue;
    acc_all.Acc_SFs_PDT_pointer = Acc_SFs_PhysDensThresh_table;
#endif // LT_STELLAREVOLUTION
    acc_all.no_tree_flag = 0;
    acc_all.inside_data_region = 1;
    acc_all.wrong_next_node_flag = 0;

    VERBOSE(3, "Set pointer");
}

/**
 * @brief Safely exits the GPU region and updates relevant OpenACC structures.
 *
 * This function handles the proper exit from a GPU computation region, checking
 * and updating OpenACC structures, ensuring consistency between the CPU and GPU.
 * It verifies if key pointers have changed or if the tree needs to be rebuilt,
 * and sets flags accordingly. Additionally, it performs cleanup by deleting
 * OpenACC-managed structures and resetting tracking variables.
 *
 * @note This function also checks if the implementation is compatible with
 * restart scenarios or cases involving changes to runtime structures like `Nextnode`.
 */
void update_openacc_structures_exit(void)
{
    VERBOSE(2, "=>Exiting from GPU region NumPart=%d", NumPart);

    NodesSize = Numnodestree;
    VERBOSE(2, "inside data region? %d", acc_all.inside_data_region);
    if (acc_all.no_tree_flag < 0 || acc_all.len_nodes_base < 0)
    {
        ENDRUN_MAIN(
            "Lenght of the Tree is not set. I Never tested the code with restart. \n Probably viriables Acc* must "
            "be set in restart.c if the tree is not rebuilt on runtime. notreeflag = %d, len_nodes_base=%d",
            acc_all.no_tree_flag, acc_all.len_nodes_base);
    }

    if (Nodes_base != acc_all.nodes_base_pointer || Nextnode != acc_all.nextnode_pointer ||
        Extnodes_base != acc_all.extnodes_base_pointer || ActiveParticleList != acc_all.active_particle_list_pointer)
    {
        acc_all.no_tree_flag = 1;
    }

    if (acc_all.wrong_next_node_flag)
    {
        ENDRUN_MAIN(
            "OpenACC implementation still not tested with star formation (so with runtime Nextnode changes). In "
            "doubt add if (AccHasWrongNextnode) AccHasNoTree=1; But you will lose performance.");
        acc_all.no_tree_flag = 1;
        // you will lose performance. If AccHasWrongNextnode then you need to send only Nextnode to the GPU. Pay
        // attention that its size changed (there is a new particle.
    }
    if (acc_all.last_numpart != NumPart)
    {
        acc_all.no_tree_flag = 1; // if numpart changed, we reload stuff    acc_all.last_numpart=NumPart;
    }

    VERBOSE(2, "no flag? %d", acc_all.no_tree_flag);

    ACC_PRAGMA(ACC_EXIT(ACC_DELETE(acc_all.extnodes_base_pointer [0:acc_all.len_nodes_base],
                                   acc_all.nodes_base_pointer [0:acc_all.len_nodes_base],
                                   acc_all.nextnode_pointer [0:acc_all.len_nextnode], shortrange_table [0:NTAB],
                                   shortrange_table_potential [0:NTAB] OTHER_EXIT)));
    // WARNING: AR: maybe adding the sizes in the delete (required by openmp) may break the openacc, this must be
    // checked
    ACC_PRAGMA(ACC_EXIT(ACC_DELETE(acc_all.walklitep_pointer [0:acc_all.len_p],
                                   acc_all.active_particle_list_pointer [0:acc_all.len_ap])));

    acc_all.inside_data_region = 0;
    acc_all.entered_gpu = 0;
    VERBOSE(2, "Exited from GPU region");
}

/**
 * @brief Update the properties of WalkLiteP particles based on P particles.
 *
 * This function updates the position, type, smoothing length, mass, and old acceleration
 * of each WalkLiteP particle from the corresponding properties of P particles.
 *
 * @note Assumes P and WalkLiteP arrays are already allocated and have compatible structures.
 */
void update_walk_lite_p(void)
{
#pragma omp parallel for
    for (int i = 0; i < NumPart; i++)
    {
        WalkLiteP[i].Pos[0] = P[i].Pos[0];
        WalkLiteP[i].Pos[1] = P[i].Pos[1];
        WalkLiteP[i].Pos[2] = P[i].Pos[2];
        WalkLiteP[i].Type = P[i].Type;
        WalkLiteP[i].Hsml = P[i].Hsml;
        WalkLiteP[i].Mass = P[i].Mass;
        WalkLiteP[i].OldAcc = P[i].OldAcc;
        WalkLiteP[i].TimeBin = P[i].TimeBin;
#ifdef ACC_ID
        WalkLiteP[i].ID = P[i].ID;
#endif // ACC_ID
    }
}
