/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the functions for handling the swallowing of gas particles and black holes by black holes.
 *
 * @author OpenGadget3-development team, in particular Alice Damiano, Klaus Dolag, Luca Sala and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../Hydro/kernel.hpp"
#include "blackhole.hpp"
#include "../System/communication.hpp"

#ifdef BLACK_HOLES

void *blackhole_evaluate_swallow_secondary(void *p);
void *blackhole_evaluate_swallow_primary(void *p);

/***********************************************************************************/
/************************** Data Structures for loop (accreation) ******************/
/***********************************************************************************/

struct blackholeswallowdata_in
{
    MyLongDouble Pos[3];
    MyFloat Hsml;
    MyFloat Mass;
    MyAtLeastDouble BH_Mass;
    MyIDType ID;
    MyAtLeastDouble BH_Spin[3];
    MyFloat Vel[3];
    int BH_AccrCounter;
    MyAtLeastDouble BH_Mass_Sum;
    MyAtLeastDouble BH_accr_mass_Sum;
    MyAtLeastDouble BH_dt_Sum;
    MyAtLeastDouble Jgas_in_Kernel_Sum[3];
    int NodeList[NODELISTLENGTH];
#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
    MyFloat BH_Mdot;
    MyFloat Dt;
    MyFloat GasMass_Cone_Up;
    MyFloat GasMass_Cone_Dn;
    int Np_Cone_Up;
    int Np_Cone_Dn;
    MyFloat UnitVector_AGN_Wind[3];
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
};

struct blackholeswallowdata_out
{
    MyAtLeastDouble BH_accreted_Mass;
    MyAtLeastDouble BH_accreted_BH_Mass;
    MyAtLeastDouble AccretedMomentum[3];
    int BH_CountProgs;
    MyFloat Accreted_Age;
    MyAtLeastDouble BH_final_spin[3];
    MyAtLeastDouble BH_EtaDisk;
    int BH_AccrCounter;
    int BH_flag_merged;
    MyAtLeastDouble BH_Mass_Sum;
    MyAtLeastDouble BH_accr_mass_Sum;
    MyAtLeastDouble BH_dt_Sum;
    MyAtLeastDouble Jgas_in_Kernel_Sum[3];
};

/**
 * @brief Transfers particle data into a structure for black hole swallowing.
 *
 * This function copies relevant data from a particle represented by index `i` into the provided
 * `blackholeswallowdata_in` structure. This is used for managing the swallowing of particles by
 * black holes, and includes both standard particle properties and additional black hole-specific data
 * if certain conditions are met.
 *
 * @param in Pointer to the `blackholeswallowdata_in` structure where the particle data will be copied.
 * @param i  Index of the particle in the global particle array `P` whose data is to be transferred.
 */
void particle2in_blackholeswallow(struct blackholeswallowdata_in *in, int i)
{
    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
        in->UnitVector_AGN_Wind[k] = BPP(i).UnitVector_AGN_Wind[k];
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
    }
    in->Hsml = P[i].Hsml;
    in->Mass = P[i].Mass;
    in->BH_Mass = BPP(i).BH_Mass;
    in->ID = P[i].ID;
#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
    in->BH_Mdot = BPP(i).BH_Mdot;
    in->Dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / hubble_a;
    in->GasMass_Cone_Up = BPP(i).GasMass_Cone_Up;
    in->GasMass_Cone_Dn = BPP(i).GasMass_Cone_Dn;
    in->Np_Cone_Up = BPP(i).Np_Cone_Up;
    in->Np_Cone_Dn = BPP(i).Np_Cone_Dn;
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
    for (int k = 0; k < 3; k++)
    {
        in->Vel[k] = P[i].Vel[k];
    }

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    for (int k = 0; k < 3; k++)
    {
        in->BH_Spin[k] = BPP(i).BH_Spin[k];
    }
    in->BH_AccrCounter = BPP(i).BH_AccrCounter;
    in->BH_Mass_Sum = BPP(i).BH_Mass_Sum;
    in->BH_accr_mass_Sum = BPP(i).BH_accr_mass_Sum;
    in->BH_dt_Sum = BPP(i).BH_dt_Sum;
    for (int k = 0; k < 3; k++)
        in->Jgas_in_Kernel_Sum[k] = BPP(i).Jgas_in_Kernel_Sum[k];
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
}

/**
 * @brief Updates the properties of a black hole particle based on the output data from swallowing.
 *
 * This function updates the properties of a black hole particle, represented by index `i`, using
 * the information contained in the `blackholeswallowdata_out` structure. The updates include changes
 * to mass, momentum, and other black hole-specific parameters, depending on the provided `mode` and
 * defined conditions.
 *
 * @param out Pointer to the `blackholeswallowdata_out` structure containing the updated data
 *            for the black hole particle.
 * @param i      Index of the particle in the global black hole particle array `BPP` to be updated.
 * @param mode   Mode of the update (used for conditional processing, if applicable).
 */
void out2particle_blackholeswallow(struct blackholeswallowdata_out *out, int i, int mode)
{
    ASSIGN_ADD(BPP(i).BH_accreted_Mass, out->BH_accreted_Mass, 1);
    ASSIGN_ADD(BPP(i).BH_accreted_BHMass, out->BH_accreted_BH_Mass, 1);

    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(BPP(i).BH_accreted_momentum[k], out->AccretedMomentum[k], 1);
    }

    ASSIGN_ADD(BPP(i).BH_CountProgs, out->BH_CountProgs, 1);

    if (BPP(i).StellarAge > out->Accreted_Age)
    {
        BPP(i).StellarAge = out->Accreted_Age;
    }
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    for (int k = 0; k < 3; k++)
    {
        ASSIGN_ADD(BPP(i).BH_final_spin[k], out->BH_final_spin[k], 1);
    }

    BPP(i).BH_AccrCounter = out->BH_AccrCounter;
    BPP(i).BH_Mass_Sum = out->BH_Mass_Sum;
    BPP(i).BH_accr_mass_Sum = out->BH_accr_mass_Sum;
    BPP(i).BH_dt_Sum = out->BH_dt_Sum;
    for (int k = 0; k < 3; k++)
    {
        BPP(i).Jgas_in_Kernel_Sum[k] = out->Jgas_in_Kernel_Sum[k];
    }
    BPP(i).BH_flag_merged = out->BH_flag_merged;

#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
}

static struct blackholeswallowdata_in *BlackholeSwallowDataIn, *BlackholeSwallowDataGet;
static struct blackholeswallowdata_out *BlackholeSwallowDataResult, *BlackholeSwallowDataOut;

static int N_gas_swallowed, N_BH_swallowed;
static int N_gas_slices_swallowed;

#if (defined(PB_BH_KINETIC_FEEDBACK) || defined(PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT))
static int N_BHwind_Kicked, tot_BHwind_Kicked;
#endif // PB_BH_KINETIC_FEEDBACK || PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT

/**
 * @brief Executes the loop for handling the swallowing of gas particles and black holes.
 *
 * This function performs the main loop for managing the swallowing of gas particles and black holes
 * by black holes in the simulation.
 */
void blackhole_swallow_loop(void)
{
    TIMER_PUSH(1, "PERFORM_BH_SWALLOW");
    int ndone_flag, ndone;
    int Ntot_gas_swallowed, Ntot_BH_swallowed;
    int save_NextParticle;
    int Ntot_gas_slices_swallowed;

    N_gas_swallowed = N_BH_swallowed = 0;
    N_gas_slices_swallowed = 0;
#if (defined(PB_BH_KINETIC_FEEDBACK) || defined(PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT))
    N_BHwind_Kicked = 0;
#endif // PB_BH_KINETIC_FEEDBACK || PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT

    int NTaskTimesNumPart;

    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize = (int)((MyBufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                           sizeof(struct blackholeswallowdata_in) + sizeof(struct blackholeswallowdata_out) +
                           sizemax(sizeof(struct blackholeswallowdata_in), sizeof(struct blackholeswallowdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(1, "BH: Start swallowing of gas particles and black holes");

    NextParticle = FirstActiveParticle; /* first particle for this task */

    do
    {

        /* do local particles and prepare export list */
        BufferFullFlag = 0;
        Nexport = 0;
        save_NextParticle = NextParticle;

        TIMER_PUSH(2, "Primary");
        TIMER_PUSH(1, "BHSW_COMPUTE");

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
#ifdef _OPENMP
            int thread_id = omp_get_thread_num();
#else // _OPENMP
            int thread_id = 0;
#endif // _OPENMP
            blackhole_evaluate_swallow_primary(&thread_id); /* do local particles and prepare export list */
        }

        TIMER_POP(1, "BHSW_COMPUTE");
        TIMER_POPPUSH(2, "Primary", "Exchange");
        TIMER_PUSH(1, "BHSW_PREPARE");

        prepare_sendrec_offset_send(save_NextParticle, 1); // 1 means panic on bufer too small

        BlackholeSwallowDataGet = (struct blackholeswallowdata_in *)mymalloc(
            "BlackholeDataGet", Nimport * sizeof(struct blackholeswallowdata_in));
        BlackholeSwallowDataIn = (struct blackholeswallowdata_in *)mymalloc(
            "BlackholeDataIn", Nexport * sizeof(struct blackholeswallowdata_in));

        TIMER_POPPUSH(1, "BHSW_PREPARE", "BHSW_COPY_IN");

        for (int j = 0; j < Nexport; j++)
        {
            int place = DataIndexTable[j].Index;
            particle2in_blackholeswallow(&BlackholeSwallowDataIn[j], place);
            memcpy(BlackholeSwallowDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                   NODELISTLENGTH * sizeof(int));
        }

        /* exchange particle data */
        TIMER_POPPUSH(1, "BHSW_COPY_IN", "BHSW_COMM");
        templated_exchange_data_Sendrecv(BlackholeSwallowDataIn, BlackholeSwallowDataGet, TAG_DENS_A, Send_offset,
                                         Send_count, Recv_offset, Recv_count);
        TIMER_POP(1, "BHSW_COMM");

        TIMER_POPPUSH(2, "Exchange", "Secondary");

        TIMER_PUSH(1, "BHSW_COMPUTE");

        myfree(BlackholeSwallowDataIn);
        BlackholeSwallowDataResult = (struct blackholeswallowdata_out *)mymalloc(
            "BlackholeDataResult", Nimport * sizeof(struct blackholeswallowdata_out));
        BlackholeSwallowDataOut = (struct blackholeswallowdata_out *)mymalloc(
            "BlackholeDataOut", Nexport * sizeof(struct blackholeswallowdata_out));

        /* now do the particles that were sent to us */
        NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
#ifdef _OPENMP
            int thread_id = omp_get_thread_num();
#else // _OPENMP
            int thread_id = 0;
#endif // _OPENMP
            blackhole_evaluate_swallow_secondary(&thread_id); /* do local particles and prepare export list */
        }

        TIMER_POPPUSH(1, "BHSW_COMPUTE", "BHSW_WAIT");

        if (NextParticle < 0)
        {
            ndone_flag = 1;
        }
        else
        {
            ndone_flag = 0;
        }

        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
        TIMER_POP(1, "BHSW_WAIT");

        TIMER_POPPUSH(2, "Secondary", "Send_Results");

        TIMER_PUSH(1, "BHSW_COMM");
        templated_exchange_data_Sendrecv(BlackholeSwallowDataResult, BlackholeSwallowDataOut, TAG_DENS_B, Recv_offset,
                                         Recv_count, Send_offset, Send_count);
        TIMER_POPPUSH(1, "BHSW_COMM", "BHSW_COPY");

        /* add the result to the particles */
        for (int j = 0; j < Nexport; j++)
        {
            int place = DataIndexTable[j].Index;
            out2particle_blackholeswallow(&BlackholeSwallowDataOut[j], place, 1);
        }

        TIMER_POP(1, "BHSW_COPY");
        TIMER_POP(2, "Send_Results");

        myfree(BlackholeSwallowDataOut);
        myfree(BlackholeSwallowDataResult);
        myfree(BlackholeSwallowDataGet);
    } while (ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);

    MPI_Reduce(&N_gas_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&N_BH_swallowed, &Ntot_BH_swallowed, 1, MPI_INT, MPI_SUM, 0, MYMPI_COMM_WORLD);
    MPI_Reduce(&N_gas_slices_swallowed, &Ntot_gas_slices_swallowed, 1, MPI_INT, MPI_SUM, 0, MYMPI_COMM_WORLD);
    VERBOSE(1,
            "BH: Accretion done, %d gas slices swallowed, %d last-slices swallowed (gas particles disappeared), %d BH "
            "particles swallowed",
            Ntot_gas_slices_swallowed, Ntot_gas_swallowed, Ntot_BH_swallowed);
    TIMER_POP(1, "PERFORM_BH_SWALLOW");

#if (defined(PB_BH_KINETIC_FEEDBACK) || defined(PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT))
    MPI_Reduce(&N_BHwind_Kicked, &tot_BHwind_Kicked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (tot_BHwind_Kicked > 0)
    {
        if (ThisTask == 0)
        {
            fprintf(FdBlackHolesDetails, "PB_BH_KINETIC_FEEDBACK: Kicked total %d gas particles into AGN wind ... \n",
                    tot_BHwind_Kicked);
        }
    }
#endif // PB_BH_KINETIC_FEEDBACK || PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT
}

/**
 * @brief Evaluates the swallowing process of a black hole.
 *
 * This function handles the swallowing of particles by black holes, including both gas particles
 * and other black holes. It updates the properties of the black hole, accumulates the mass of the
 * swallowed particles, and calculates the final spin of the black hole if a merger occurs.
 * The function operates in different modes, where mode 0 is for initial evaluations and mode 1 is for
 * subsequent evaluations based on previously collected data.
 *
 * @param target The ID of the black hole that is swallowing particles.
 * @param mode Determines the mode of operation:
 *        - 0: Initial evaluation mode.
 *        - 1: Subsequent evaluation mode (using previously collected data).
 * @param exportflag Pointer to an integer flag used for exporting data.
 * @param exportnodecount Pointer to an integer that keeps track of the number of nodes to export.
 * @param exportindex Pointer to an integer array for indexing the export nodes.
 * @param ngblist Pointer to an integer array to store the list of neighboring particles.
 *
 * @return Returns 0 on successful completion. If an error occurs during neighbor finding or if the number
 *         of neighbors exceeds the allowed limit, it returns -1.
 */
int blackhole_evaluate_swallow(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                               int *ngblist)
{
    struct blackholeswallowdata_in local;
    struct blackholeswallowdata_out out;
    memset(&out, 0, sizeof(struct blackholeswallowdata_out));

    if (mode == 0)
    {
        particle2in_blackholeswallow(&local, target);
    }
    else
    {
        local = BlackholeSwallowDataGet[target];
    }

    out.Accreted_Age = 1;
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
    out.BH_AccrCounter = local.BH_AccrCounter;
    out.BH_Mass_Sum = local.BH_Mass_Sum;
    out.BH_accr_mass_Sum = local.BH_accr_mass_Sum;
    out.BH_dt_Sum = local.BH_dt_Sum;
    for (int k = 0; k < 3; k++)
    {
        out.Jgas_in_Kernel_Sum[k] = local.Jgas_in_Kernel_Sum[k];
    }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION

    int startnode, listindex = 0;
    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = BlackholeSwallowDataGet[target].NodeList[listindex];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            int numngb = ngb_treefind_blackhole_threads(local.Pos, local.Hsml, target, &startnode, mode, exportflag,
                                                        exportnodecount, exportindex, ngblist);

            ENDRUN_IF(numngb > NumPart / DENSITY_LESS_NGB_FACTOR,
                      "black hole: Number of neighbours found (%d) exceeds allowed number (%d) derived from "
                      "NumPart/DENSITY_LESS_NGB_FACTOR, to continue, increase DENSITY_LESS_NGB_FACTOR (currently %f, "
                      "max possible val would be 1) !!",
                      numngb, (int)(NumPart / DENSITY_LESS_NGB_FACTOR), DENSITY_LESS_NGB_FACTOR);

            if (numngb < 0)
            {
                return -1;
            }

            for (int n = 0; n < numngb; n++)
            {
                int j = ngblist[n];

                if (P[j].Type == 5 && P[j].Mass > 0) /* we have a black hole merger */
                {
                    if (BPP(j).SwallowID == local.ID)
                    {
#ifdef _OPENMP
#pragma omp critical(_bh5)
#endif // _OPENMP
                        {
                            if (All.BlackHoleDetails >= 2)
                            {
                                fprintf(FdBlackHolesDetails, "ThisTask=%d, time=%g: id=%llu swallows %llu (%g %g)\n",
                                        ThisTask, All.Time, (unsigned long long)local.ID, (unsigned long long)P[j].ID,
                                        local.BH_Mass, BPP(j).BH_Mass);
                            }

#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                            if (local.BH_AccrCounter > 0)
                            {
                                out.BH_AccrCounter = 0;
                                out.BH_Mass_Sum = 0;
                                out.BH_accr_mass_Sum = 0;
                                out.BH_dt_Sum = 0;
                                for (int k = 0; k < 3; k++)
                                {
                                    out.Jgas_in_Kernel_Sum[k] = 0;
                                }
                            }
                            MyAtLeastDouble a_f[3] = {0, 0, 0}; /* Store the post-merger spin */
                            MyAtLeastDouble a1[3] = {0, 0, 0};
                            MyAtLeastDouble a2[3] = {0, 0, 0};
                            MyAtLeastDouble a1_mod = 0;
                            MyAtLeastDouble a2_mod = 0;
                            MyAtLeastDouble l[3] = {0, 0, 0}; /* Analytical fit bt Rezzolla+08 */
                            MyAtLeastDouble l_mod = 0;
                            MyAtLeastDouble L_dir[3] = {0, 0, 0};
                            MyAtLeastDouble L1[3] = {0, 0, 0};
                            MyAtLeastDouble L2[3] = {0, 0, 0};
                            MyAtLeastDouble q = 0;
                            MyAtLeastDouble mu = 0;
                            MyAtLeastDouble M1 = 0;
                            MyAtLeastDouble M2 = 0;
                            MyAtLeastDouble r1[3] = {0, 0, 0};
                            MyAtLeastDouble r2[3] = {0, 0, 0};
                            MyAtLeastDouble v1[3] = {0, 0, 0};
                            MyAtLeastDouble v2[3] = {0, 0, 0};
                            MyAtLeastDouble rcom[3] = {0, 0, 0};
                            MyAtLeastDouble vcom[3] = {0, 0, 0};
                            /* Parameters of the fit by Rezzolla+08 */
                            MyAtLeastDouble s4 = -0.129;
                            MyAtLeastDouble s5 = -0.384;
                            MyAtLeastDouble t0 = -2.686;
                            MyAtLeastDouble t2 = -3.454;
                            MyAtLeastDouble t3 = 2.353;
                            if (local.BH_Mass >= BPP(j).BH_Mass) /* the mass ratio q is defined as a quantity <= 1 */
                            {
                                M1 = local.BH_Mass;
                                M2 = BPP(j).BH_Mass;
                                q = BPP(j).BH_Mass / local.BH_Mass;
                                for (int k = 0; k < 3; k++)
                                {
                                    a1[k] = local.BH_Spin[k] / All.GravityConstantInternal /
                                            (local.BH_Mass * local.BH_Mass) * (C_LIGHT / All.UnitVelocity_in_cm_per_s);
                                    r1[k] = local.Pos[k];
                                    v1[k] = local.Vel[k];
                                    a2[k] = BPP(j).BH_Spin[k] / All.GravityConstantInternal /
                                            (BPP(j).BH_Mass * BPP(j).BH_Mass) *
                                            (C_LIGHT / All.UnitVelocity_in_cm_per_s);
                                    r2[k] = P[j].Pos[k];
                                    v2[k] = P[j].Vel[k];
                                }
                            }
                            else
                            {
                                M1 = BPP(j).BH_Mass;
                                M2 = local.BH_Mass;
                                q = local.BH_Mass / BPP(j).BH_Mass;
                                for (int k = 0; k < 3; k++)
                                {
                                    a1[k] = BPP(j).BH_Spin[k] / All.GravityConstantInternal /
                                            (BPP(j).BH_Mass * BPP(j).BH_Mass) *
                                            (C_LIGHT / All.UnitVelocity_in_cm_per_s);
                                    r1[k] = P[j].Pos[k];
                                    v1[k] = P[j].Vel[k];
                                    a2[k] = local.BH_Spin[k] / All.GravityConstantInternal /
                                            (local.BH_Mass * local.BH_Mass) * (C_LIGHT / All.UnitVelocity_in_cm_per_s);
                                    r2[k] = local.Pos[k];
                                    v2[k] = local.Vel[k];
                                }
                            }

                            /* printf("local.ID = %d, a1 = [%e,%e,%e], a2 = [%e,%e,%e] ",
                             * local.ID,a1[0],a1[1],a1[2],a2[0],a2[1],a2[2]);
                             */
                            mu = q / ((1 + q) * (1 + q));
                            /* printf("local.ID = %d, mu= %e ", local.ID, mu);
                             */
                            /* Here compute L and L_dir.... */
                            for (int k = 0; k < 3; k++)
                            {
                                rcom[k] = (r1[k] * M1 + r2[k] * M2) / (M1 + M2);
                                vcom[k] = (v1[k] * M1 + v2[k] * M2) / (M1 + M2);
                            }

                            /* Compute relative positions and velocities */
                            MyAtLeastDouble dr1[3], dv1[3], dr2[3], dv2[3];
                            for (int k = 0; k < 3; k++)
                            {
                                dr1[k] = r1[k] - rcom[k];
                                dv1[k] = v1[k] - vcom[k];
                                dr2[k] = r2[k] - rcom[k];
                                dv2[k] = v2[k] - vcom[k];
                            }

                            /* Compute Cross Product */
                            L1[0] = M1 * (dr1[1] * dv1[2] - dr1[2] * dv1[1]);
                            L1[1] = M1 * (dr1[2] * dv1[0] - dr1[0] * dv1[2]);
                            L1[2] = M1 * (dr1[0] * dv1[1] - dr1[1] * dv1[0]);

                            L2[0] = M2 * (dr2[1] * dv2[2] - dr2[2] * dv2[1]);
                            L2[1] = M2 * (dr2[2] * dv2[0] - dr2[0] * dv2[2]);
                            L2[2] = M2 * (dr2[0] * dv2[1] - dr2[1] * dv2[0]);

                            /* Compute magnitude */
                            MyAtLeastDouble L_mod =
                                sqrt(pow(L1[0] + L2[0], 2) + pow(L1[1] + L2[1], 2) + pow(L1[2] + L2[2], 2));

                            /* printf("local.ID = %d, L_mod = %e ", local.ID, L_mod);
                             */
                            for (int k = 0; k < 3; k++)
                            {
                                L_dir[k] = (L1[k] + L2[k]) / (L_mod);
                            }

                            a1_mod = sqrt(a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2]);
                            a2_mod = sqrt(a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2]);

                            MyAtLeastDouble cosphi1 = (a1[0] * L_dir[0] + a1[1] * L_dir[1] + a1[2] * L_dir[2]) /
                                                      (a1_mod); /* In the denominator should be a1_mod*L_dir_mod, but
                                                                   L_dir_mod=1 by construction */
                            MyAtLeastDouble cosphi2 =
                                (a2[0] * L_dir[0] + a2[1] * L_dir[1] + a2[2] * L_dir[2]) / (a2_mod);

                            /* printf("local.ID = %d, cosphi1 = %e, cosphi2 = %e ", local.ID, cosphi1, cosphi2);
                             */
                            l_mod = s4 / ((1 + q * q) * (1 + q * q)) *
                                        (a1_mod * a1_mod + a2_mod * a2_mod * q * q * q * q +
                                         2 * (a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2]) * q * q) +
                                    (s5 * mu + t0 + 2) / (1 + q * q) * (a1_mod * cosphi1 + a2_mod * q * q * cosphi2) +
                                    2 * sqrt(3) + t2 * mu + t3 * mu * mu;

                            /* printf("local.ID = %d, l_mod = %e ", local.ID, l_mod);
                             */
                            for (int k = 0; k < 3; k++)
                            {
                                l[k] = l_mod * L_dir[k];
                            }

                            for (int k = 0; k < 3; k++)
                            {
                                a_f[k] = 1 / ((1 + q) * (1 + q)) * (a1[k] + a2[k] * q * q + l[k] * q);
                            }

                            MyAtLeastDouble a_f_mod = 0.;
                            MyAtLeastDouble a_f_dir[3] = {0, 0, 0};
                            for (int k = 0; k < 3; k++)
                            {
                                a_f_mod += a_f[k] * a_f[k];
                            }
                            a_f_mod = sqrt(a_f_mod);
                            for (int k = 0; k < 3; k++)
                            {
                                a_f_dir[k] = a_f[k] / a_f_mod;
                            }
                            if (a_f_mod < MinimumAllowedDimensionlessSpinValue)
                            {
                                a_f_mod = MinimumAllowedDimensionlessSpinValue;
                            }

                            /* printf("local.ID = %d, a_f = [%e,%e,%e] ",local.ID, a_f[0],a_f[1],a_f[2]);
                             */
                            MyAtLeastDouble final_BH_mass = M1 + M2;
                            for (int k = 0; k < 3; k++)
                            {
                                out.BH_final_spin[k] = a_f_mod * All.GravityConstantInternal *
                                                       (final_BH_mass * final_BH_mass) /
                                                       (C_LIGHT / All.UnitVelocity_in_cm_per_s) * a_f_dir[k];
                            }
                            out.BH_flag_merged = 1;
                            /* printf("local.ID = %d, BH_final_spin = [%e,%e,%e] ",local.ID,
                             * out.BH_final_spin[0],out.BH_final_spin[1],out.BH_final_spin[2]);
                             */
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
                            out.BH_accreted_Mass += BPP(j).BH_Mass;
                            out.BH_accreted_BH_Mass += BPP(j).BH_Mass;
                            if (All.BlackHoleIgnoreMomentum == 0)
                            {
                                for (int k = 0; k < 3; k++)
                                {
                                    out.AccretedMomentum[k] += BPP(j).BH_Mass * P[j].Vel[k];
                                }
                            }

                            out.BH_CountProgs += BPP(j).BH_CountProgs;

                            int bin = P[j].TimeBin;
                            TimeBin_BH_mass[bin] -= BPP(j).BH_Mass;
                            TimeBin_BH_dynamicalmass[bin] -= P[j].Mass;
                            TimeBin_BH_Mdot[bin] -= BPP(j).BH_Mdot;
                            double temp_meddington = Mdot_Eddington(BPP(j).BH_Mass);
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                            if (All.BlackHoleSubgridADOn == 1)
                            {
                                temp_meddington = temp_meddington / (BPP(j).BH_EtaDisk / 0.1);
                            }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION
                            if (BPP(j).BH_Mass > 0)
                            {
                                TimeBin_BH_Medd[bin] -= BPP(j).BH_Mdot / temp_meddington;
                            }

                            P[j].Mass = 0;
                            BPP(j).BH_Mass = 0;
                            BPP(j).BH_Mdot = 0;

                            out.Accreted_Age = BPP(j).StellarAge;

                            N_BH_swallowed++;
                        } // critical
                    } // end of BH merger
                }

                if (P[j].Type == 0)
                {
                    if (SphP[j].SwallowID == local.ID)
                    {
#ifdef _OPENMP
#pragma omp critical(_bh0)
#endif // _OPENMP
                        {
                            double myaccreted_mass = P[j].Mass;
                            int number_of_slices_generated = 0;
                            if (All.bits > 0 && All.BlackHoleAccretionSlicesOn == 1)
                            {
                                number_of_slices_generated = (P[j].ID >> (sizeof(MyIDType) * 8 - All.bits));
                                myaccreted_mass /= (All.Generations - number_of_slices_generated);
                            }
                            if (number_of_slices_generated == (All.Generations - 1))
                            {
                                myaccreted_mass = P[j].Mass;
                                P[j].Mass = 0;
#ifdef LT_STELLAREVOLUTION
                                for (int k = 0; k < LT_NMetP; k++)
                                {
                                    SphP[j].Metals[k] = 0;
                                }
#endif // LT_STELLAREVOLUTION
                                N_gas_swallowed++;
                            }
                            else
                            {
#ifdef LT_STELLAREVOLUTION
                                MyAtLeastDouble dec_factor = myaccreted_mass / P[j].Mass;
                                for (int k = 0; k < LT_NMetP; k++)
                                {
                                    SphP[j].Metals[k] *= dec_factor;
                                }
#endif // LT_STELLAREVOLUTION
#if GADGET_HYDRO == HYDRO_MFM
                                mfm_add_mass(j, -myaccreted_mass);
#endif // GADGET_HYDRO == HYDRO_MFM
                                P[j].Mass -= myaccreted_mass;
                                P[j].ID += ((MyIDType)1 << (sizeof(MyIDType) * 8 - All.bits));
                            }

                            out.BH_accreted_Mass += myaccreted_mass;
                            if (All.BlackHoleIgnoreMomentum == 0)
                            {
                                for (int k = 0; k < 3; k++)
                                {
                                    out.AccretedMomentum[k] += myaccreted_mass * P[j].Vel[k];
                                }
                            }

                            N_gas_slices_swallowed++;
                            SphP[j].SwallowID = 0;
                        } // critical
                    } // if (SphP[j].SwallowID == local.ID
                    else
                    {
                        /* If NOT a just-swallowed gas particle, then consider to give PB_* Kinetic Feedback. */
#if (defined(PB_BH_KINETIC_FEEDBACK) || defined(PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT) || defined(PB_BH_WIND_VELOCITY))
                        MyAtLeastDouble Edd_Ratio = local.BH_Mdot / Mdot_Eddington(local.BH_Mass);

                        if ((All.BlackHoleKinFeedback == 1 && All.BlackHoleSwitchThermalKinetic == 0) ||
                            (All.BlackHoleKinFeedback == 1 && All.BlackHoleSwitchThermalKinetic == 1 &&
                             Edd_Ratio < All.BHEddRatioSwitchThermalKinetic))
                        {
                            double Mdot_Wind, prob_KinFeed, norm_KinFeed, dir[3];
                            int ii;
#ifdef PB_BH_KIN_ISOTROPIC_WIND
                            double theta_KinFeed, phi_KinFeed;
#endif // PB_BH_KIN_ISOTROPIC_WIND
                            MyFloat mdot = local.BH_Mdot;

                            MyAtLeastDouble dx = NEAREST_X(local.Pos[0] - P[j].Pos[0]);
                            MyAtLeastDouble dy = NEAREST_Y(local.Pos[1] - P[j].Pos[1]);
                            MyAtLeastDouble dz = NEAREST_Z(local.Pos[2] - P[j].Pos[2]);

                            MyAtLeastDouble r2 = (dx * dx) + (dy * dy) + (dz * dz);

                            MyFloat dt = local.Dt;
                            MyFloat h_i = local.Hsml;
                            MyIDType id = local.ID;

#ifdef PB_BH_KIN_ENERGY_DRIVEN
                            Mdot_Wind = mdot * 2.0 * All.BlackHoleFeedbackFactor * All.BlackHoleRadiativeEfficiency *
                                        pow(C_LIGHT / (PB_BH_WIND_VELOCITY * All.UnitVelocity_in_cm_per_s), 2);
#endif // PB_BH_KIN_ENERGY_DRIVEN
#ifdef PB_BH_KIN_MOMENTUM_DRIVEN
                            Mdot_Wind = mdot * All.BlackHoleFeedbackFactor * All.BlackHoleRadiativeEfficiency *
                                        C_LIGHT / (PB_BH_WIND_VELOCITY * All.UnitVelocity_in_cm_per_s);
#endif // PB_BH_KIN_MOMENTUM_DRIVEN

#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
                            MyFloat Mgas_Inside_Cone_Up, Mgas_Inside_Cone_Dn, dr_dot_KickVec, Angle_deg;
                            MyFloat UnitVector_WindKick[3];
                            int Np_Inside_Cone_Up, Np_Inside_Cone_Dn;

                            Mgas_Inside_Cone_Up = local.GasMass_Cone_Up;
                            Mgas_Inside_Cone_Dn = local.GasMass_Cone_Dn;
                            Np_Inside_Cone_Up = local.Np_Cone_Up;
                            Np_Inside_Cone_Dn = local.Np_Cone_Dn;
                            UnitVector_WindKick = local.UnitVector_AGN_Wind;

                            dr_dot_KickVec = -((dx * UnitVector_WindKick[0]) + (dy * UnitVector_WindKick[1]) +
                                               (dz * UnitVector_WindKick[2]));

                            Angle_deg = (acos(dr_dot_KickVec / sqrt(r2))) * 180.0 / M_PI;

                            if (Angle_deg < PB_BH_KIN_CONE_ANGLE_DEG)
                            {
                                prob_KinFeed = Mdot_Wind * dt * Mgas_Inside_Cone_Dn /
                                               pow((Mgas_Inside_Cone_Up + Mgas_Inside_Cone_Dn), 2);
                            }
                            else if (180.0 - Angle_deg < PB_BH_KIN_CONE_ANGLE_DEG)
                            {
                                prob_KinFeed = Mdot_Wind * dt * Mgas_Inside_Cone_Up /
                                               pow((Mgas_Inside_Cone_Up + Mgas_Inside_Cone_Dn), 2);
                            }
                            else
                            {
                                prob_KinFeed = 0.0;
                            }
#else // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                            if (Mdot_Wind > 0 && rho > 0)
                            {
                                prob_KinFeed = Mdot_Wind * dt * wk / rho; /* Probability for Kinetic Feedback. */
                            }
                            else
                            {
                                prob_KinFeed = 0.0;
                            }
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                            if (prob_KinFeed > 1.0)
                            {
#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                VERBOSE(0,
                                        "PB_BH_KINETIC_FEEDBACK ERROR: prob > 1, prob_KinFeed=%e, ID=%u, Mdot_Wind=%e, "
                                        "dt=%e, Mgas_Inside_Cone_Up=%e, h_i=%e \n",
                                        prob_KinFeed, P[j].ID, Mdot_Wind, dt, Mgas_Inside_Cone_Up, h_i);
#else // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                VERBOSE(0,
                                        "PB_BH_KINETIC_FEEDBACK ERROR: prob > 1, prob_KinFeed=%e, ID=%u, Mdot_Wind=%e, "
                                        "dt=%e, h_i=%e \n",
                                        prob_KinFeed, P[j].ID, Mdot_Wind, dt, h_i);
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                            }
                            /* If NOT a just-swallowed particle,
                             * AND random number - uniform in [0, 1] - is less than probability;
                             * then give Kinetic Feedback Velocity Kick to particle, along relevant direction.
                             */
                            if ((SphP[j].SwallowID != id) && (get_random_number(P[j].ID + 1) < prob_KinFeed))
                            {
#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
#ifdef PB_BH_KIN_KICK_RADIALLY_OUTWARD_FROM_BH
                                dir[0] = P[j].Pos[0] - local.Pos[0];
                                dir[1] = P[j].Pos[1] - local.Pos[1];
                                dir[2] = P[j].Pos[2] - local.Pos[2];
#else // PB_BH_KIN_KICK_RADIALLY_OUTWARD_FROM_BH
                                dir[0] = UnitVector_WindKick[0];
                                dir[1] = UnitVector_WindKick[1];
                                dir[2] = UnitVector_WindKick[2];
#endif // PB_BH_KIN_KICK_RADIALLY_OUTWARD_FROM_BH
#else // PB_BH_KIN_EJECT_PART_INSIDE_CONE
#ifdef PB_BH_KIN_ISOTROPIC_WIND
                                theta_KinFeed = acos(2 * get_random_number(P[j].ID + 3) - 1);
                                phi_KinFeed = 2 * M_PI * get_random_number(P[j].ID + 4);

                                dir[0] = sin(theta_KinFeed) * cos(phi_KinFeed);
                                dir[1] = sin(theta_KinFeed) * sin(phi_KinFeed);
                                dir[2] = cos(theta_KinFeed);
#else // PB_BH_KIN_ISOTROPIC_WIND
                                dir[0] = P[j].GravAccel[1] * P[j].Vel[2] - P[j].GravAccel[2] * P[j].Vel[1];
                                dir[1] = P[j].GravAccel[2] * P[j].Vel[0] - P[j].GravAccel[0] * P[j].Vel[2];
                                dir[2] = P[j].GravAccel[0] * P[j].Vel[1] - P[j].GravAccel[1] * P[j].Vel[0];
#endif // PB_BH_KIN_ISOTROPIC_WIND
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                for (ii = 0, norm_KinFeed = 0; ii < 3; ii++)
                                {
                                    norm_KinFeed += dir[ii] * dir[ii];
                                }
                                norm_KinFeed = sqrt(norm_KinFeed);

#ifndef PB_BH_KIN_KICK_RADIALLY_OUTWARD_FROM_BH
#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                if (dz > 0.0)
                                {
                                    norm_KinFeed = -norm_KinFeed;
                                }
#else // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                if (get_random_number(P[j].ID + 2) < 0.5)
                                {
                                    norm_KinFeed = -norm_KinFeed;
                                }
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
#endif // PB_BH_KIN_KICK_RADIALLY_OUTWARD_FROM_BH
                                if (norm_KinFeed != 0)
                                {
                                    for (ii = 0; ii < 3; ii++)
                                    {
                                        dir[ii] /= norm_KinFeed;
                                    }
                                    for (ii = 0; ii < 3; ii++)
                                    {
                                        P[j].Vel[ii] += PB_BH_WIND_VELOCITY * ascale * dir[ii];
                                        SphP[j].VelPred[ii] += PB_BH_WIND_VELOCITY * ascale * dir[ii];
                                    }
                                    N_BHwind_Kicked++;
#ifdef PB_BH_KIN_FEED_DECOUPLE_WIND
                                    SphP[j].DecoupleTime_AGN = PB_BH_WIND_FREE_TRAVEL_LENGTH / PB_BH_WIND_VELOCITY;
#endif // PB_BH_KIN_FEED_DECOUPLE_WIND
                                    int dummyKick = 44556677;

#ifdef PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                    fprintf(FdBlackHolesDetails,
                                            "PB_BH_KINETIC_FEEDBACK Kick Gas Part: %e   %e   %d   %d   %e   %e   %u \n",
                                            All.Time, prob_KinFeed, Np_Inside_Cone_Up, Np_Inside_Cone_Dn, Mdot_Wind,
                                            local.BH_Mass, id);
#else // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                    fprintf(FdBlackHolesDetails,
                                            "PB_BH_KINETIC_FEEDBACK Kick Gas Part: %e   %e   %e   %e   %e   %e   %u \n",
                                            All.Time, prob_KinFeed, norm_KinFeed, Mdot_Wind, local.BH_Mass, h_i, id);
#endif // PB_BH_KIN_EJECT_PART_INSIDE_CONE
                                }
                            }
                        } // if(BlackHoleKinFeedback==1 && BlackHoleSwitchThermalKinetic==1 && Edd<BHEddRatioSwitch)
#endif // PB_BH_KINETIC_FEEDBACK || PB_BH_NOACCR_FIXED_KIN_ENERGY_OUTPUT || PB_BH_WIND_VELOCITY

                    } /* else at beginning of PB_* Kinetic Feedback part. */

                } // if (P[j].Type == 0)
            } // for
        } // while
        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = BlackholeSwallowDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    } // while

    /* Now collect the result at the right place */
    if (mode == 0)
    {
        out2particle_blackholeswallow(&out, target, 0);
    }
    else
    {
        BlackholeSwallowDataResult[target] = out;
    }

    return 0;
}

/**
 * @brief Thread function to evaluate black hole swallowing for a given set of particles.
 *
 * This function handles the evaluation of black hole swallowing of particles assigned.
 * It processes particles of type 5 (black holes) to determine
 * if they are involved in swallowing processes and calls `blackhole_evaluate_swallow` for those particles.
 *
 * @param p Pointer to an integer representing the thread ID. This ID is used to access thread-specific
 *          data and arrays.
 *
 * @return Returns NULL upon completion.
 */
void *blackhole_evaluate_swallow_primary(void *p)
{
    int thread_id = *((int *)p);
    int i;
    int *exportflag, *exportnodecount, *exportindex, *ngblist;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;

    /* Note: exportflag is local to each thread */
    for (int j = 0; j < NTask; j++)
    {
        exportflag[j] = -1;
    }

    while (1)
    {
        int exitFlag = 0;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            if (BufferFullFlag != 0 || NextParticle < 0)
            {
                exitFlag = 1;
            }
            else
            {
#ifdef _OPENMP
                do
                {
                    i = NextParticle;
                    NextParticle = NextActiveParticle[i];
                    if (NextParticle < 0)
                    {
                        break;
                    }
                    ProcessedFlag[i] = 1;
                } while (P[i].Type != 5);
                ProcessedFlag[i] = 0;
#else // _OPENMP
                i = NextParticle;
                ProcessedFlag[NextParticle] = 0;
                NextParticle = NextActiveParticle[i];
#endif // _OPENMP
            }
        }
        if (exitFlag)
            break;

        if (P[i].Type == 5)
        {
            if (BPP(i).SwallowID == 0)
                if (blackhole_evaluate_swallow(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0)
                {
                    break;
                }
        }
        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Secondary thread function to evaluate black hole swallowing for imported particles.
 *
 * This function is executed by secondary threads to handle the evaluation of black hole swallowing
 * for particles that have been imported into the local buffer. It processes each particle by calling
 * `blackhole_evaluate_swallow` to check if it is involved in swallowing events.
 *
 * @param p Pointer to an integer representing the thread ID. This ID is used to access thread-specific
 *          data and arrays.
 *
 * @return Returns NULL upon completion.
 */
void *blackhole_evaluate_swallow_secondary(void *p)
{
    int thread_id = *((int *)p);
    int j, dummy, *ngblist;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));

    while (1)
    {
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            j = NextJ;
            NextJ++;
        }

        if (j >= Nimport)
        {
            break;
        }

        blackhole_evaluate_swallow(j, 1, &dummy, &dummy, &dummy, ngblist);
    }
    return NULL;
}
#endif // BLACK_HOLES
