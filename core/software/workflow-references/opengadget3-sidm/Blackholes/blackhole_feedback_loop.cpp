/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains the implementation of the black hole feedback loop.
 *
 * @author OpenGadget3-development team, in particular Alice Damiano, Giuseppe Murante, Klaus Dolag, Luca Sala, Stefano
 * Borgani and Volker Springel.
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

void *blackhole_evaluate_primary(void *p);
void *blackhole_evaluate_secondary(void *p);

/***********************************************************************************/
/************************** Data Structures for loop (accreation) ******************/
/***********************************************************************************/

struct blackholedata_in
{
    MyLongDouble Pos[3];
    MyFloat BH_Density;
    MyFloat BH_Mdot;
    MyFloat Dt;
    MyFloat Hsml;
    MyFloat Mass;
    MyAtLeastDouble BH_Mass;
    MyFloat Vel[3];
    MyFloat Csnd;
    MyIDType ID;
    int NodeList[NODELISTLENGTH];
    MyFloat BH_TotalFeedbackEfficiency;
    MyFloat Potential;

#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE_AD_MASSWEIGHTS
    MyFloat BH_TotalSurroundingMass;
#endif
};

struct blackholedata_out
{
    MyLongDouble BH_MinPotPos[3];
    MyFloat BH_MinPot;
#ifdef AD_DYNFRIC
    MyFloat BHDF_DynFric[3];
    int BHDF_ngb;
#endif // AD_DYNFRIC
#ifdef GM_REPOSITION_ON_STARDENSITY_MAX
    MyLongDouble BH_StarDensPos[3];
    MyLongDouble BH_StarDens;
#ifdef GM_USE_VELS_ON_REPOSITIONING
    MyLongDouble BH_StarDensVel[3];
#endif // GM_USE_VELS_ON_REPOSITIONING
#endif // GM_REPOSITION_ON_STARDENSITY_MAX
};

/**
 * @brief Transfers particle data to a black hole data structure.
 *
 * This function copies relevant particle information into the `blackholedata_in` structure for a specific black hole.
 * It ensures that the particle is of the correct type and associated with the correct black hole ID before transferring
 * the data. This is typically used for simulations or calculations involving black holes where particle data needs to
 * be processed or reported.
 *
 * @param in Pointer to a `blackholedata_in` structure where the particle data will be copied.
 * @param i Index of the particle in the global particle array.
 */
void particle2in_blackhole(struct blackholedata_in *in, int i)
{
    assert(P[i].Type == 5);
    assert(BHP[P[i].pt.BHID].PID == i);
    for (int k = 0; k < 3; k++)
    {
        in->Pos[k] = P[i].Pos[k];
        in->Vel[k] = P[i].Vel[k];
    }
    in->BH_Density = BPP(i).BH_Density;
    in->BH_Mdot = BPP(i).BH_Mdot;
    in->Dt = (P[i].TimeBin ? (1 << P[i].TimeBin) : 0) * All.Timebase_interval / hubble_a;
    in->Hsml = P[i].Hsml;
    in->Mass = P[i].Mass;
    in->BH_Mass = BPP(i).BH_Mass;
    in->Csnd = sqrt(GAMMA * BPP(i).BH_Entropy * pow(BPP(i).BH_Density / (ascale * ascale * ascale), GAMMA_MINUS1));
    in->ID = P[i].ID;
    in->BH_TotalFeedbackEfficiency = BPP(i).BH_TotalFeedbackEfficiency;
    in->Potential = P[i].Potential;
#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE_AD_MASSWEIGHTS
    in->BH_TotalSurroundingMass = BPP(i).BH_TotalSurroundingMass;
#endif
}

/**
 * @brief Updates particle data from black hole output data.
 *
 * This function transfers data from the `blackholedata_out` structure to a particle's black hole properties
 * based on a specified mode. It updates various properties of the black hole, including potential and density
 * information, depending on whether the mode is 0 or 1.
 *
 * @param out Pointer to a `blackholedata_out` structure containing the updated black hole data.
 * @param i Index of the particle in the global particle array.
 * @param mode Mode of operation, which affects how data is updated:
 *             - `0`: Directly assigns values from `out` to `BPP(i)`.
 *             - `1`: Updates values only if the new data is better (e.g., lower potential or higher density).
 */
void out2particle_blackhole(struct blackholedata_out *out, int i, int mode)
{
    assert(P[i].Type == 5);
    assert(BHP[P[i].pt.BHID].PID == i);
    if (mode == 0)
    {
        for (int k = 0; k < 3; k++)
        {
            BPP(i).BH_MinPotPos[k] = out->BH_MinPotPos[k];
        }
        BPP(i).BH_MinPot = out->BH_MinPot;
#ifdef AD_DYNFRIC
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(BPP(i).BHDF_DynFric[k], out->BHDF_DynFric[k], mode);
        }
        ASSIGN_ADD(BPP(i).BHDF_ngb, out->BHDF_ngb, mode);
#endif // AD_DYNFRIC
#ifdef GM_REPOSITION_ON_STARDENSITY_MAX
        for (int k = 0; k < 3; k++)
        {
            BPP(i).BH_StarDensPos[k] = out->BH_StarDensPos[k];
#ifdef GM_USE_VELS_ON_REPOSITIONING
            BPP(i).BH_StarDensVel[k] = out->BH_StarDensVel[k];
#endif // GM_USE_VELS_ON_REPOSITIONING
        }
        BPP(i).BH_StarDens = out->BH_StarDens;
#endif // GM_REPOSITION_ON_STARDENSITY_MAX
    }
    else
    {
        if (BPP(i).BH_MinPot > out->BH_MinPot)
        {
            for (int k = 0; k < 3; k++)
            {
                BPP(i).BH_MinPotPos[k] = out->BH_MinPotPos[k];
            }
            BPP(i).BH_MinPot = out->BH_MinPot;
        }
#ifdef GM_REPOSITION_ON_STARDENSITY_MAX
        if (BPP(i).BH_StarDens < out->BH_StarDens)
        {
            BPP(i).BH_StarDens = out->BH_StarDens;
            for (int k = 0; k < 3; k++)
            {
                BPP(i).BH_StarDensPos[k] = out->BH_StarDensPos[k];
#ifdef GM_USE_VELS_ON_REPOSITIONING
                BPP(i).BH_StarDensVel[k] = out->BH_StarDensVel[k];
#endif // GM_USE_VELS_ON_REPOSITIONING
            }
        }
#endif // GM_REPOSITION_ON_STARDENSITY_MAX
#ifdef AD_DYNFRIC
        for (int k = 0; k < 3; k++)
        {
            ASSIGN_ADD(BPP(i).BHDF_DynFric[k], out->BHDF_DynFric[k], mode);
        }
        ASSIGN_ADD(BPP(i).BHDF_ngb, out->BHDF_ngb, mode);
#endif // AD_DYNFRIC
    }
}

static struct blackholedata_in *BlackholeDataIn, *BlackholeDataGet;
static struct blackholedata_out *BlackholeDataResult, *BlackholeDataOut;

/**
 * @brief Executes the feedback loop for black holes, computing and applying feedback effects.
 *
 * This function performs the main feedback loop for black holes, which includes calculating
 * feedback energy, preparing for the swallowing of gas particles and black holes, and managing
 * inter-process communication.
 */
void blackhole_feedback_loop(void)
{
    TIMER_PUSH(1, "COMPUTE_BH_FEEDBACK");
    int ndone_flag, ndone;
    int save_NextParticle;

    VERBOSE(1, "BH: Compute feedback and prepare swallowing of gas particles and black holes");

    /* allocate buffers to arrange communication */

    int NTaskTimesNumPart;

    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR));
    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize = (int)((MyBufferSize * 1024 * 1024) /
                          (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct blackholedata_in) +
                           sizeof(struct blackholedata_out) +
                           sizemax(sizeof(struct blackholedata_in), sizeof(struct blackholedata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(1, "BH: using %g MB for buffering, %g MB remaining for other buffers ", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

    /** Let's first spread the feedback energy, and determine which particles may be swalled by whom */

    NextParticle = FirstActiveParticle; /* first particle for this task */

    do
    {
        /* do local particles and prepare export list */
        BufferFullFlag = 0;
        Nexport = 0;
        save_NextParticle = NextParticle;

        TIMER_PUSH(2, "Primary");
        TIMER_PUSH(1, "BHFB_COMPUTE");

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
        {
#ifdef _OPENMP
            int thread_id = omp_get_thread_num();
#else // _OPENMP
            int thread_id = 0;
#endif // _OPENMP
            blackhole_evaluate_primary(&thread_id); /* do local particles and prepare export list */
        }

        TIMER_POP(1, "BHFB_COMPUTE");
        TIMER_POPPUSH(2, "Primary", "Exchange");
        TIMER_PUSH(1, "BHFB_PREPARE");

        prepare_sendrec_offset_send(save_NextParticle, 1); // 1 means panic on bufer too small

        BlackholeDataGet =
            (struct blackholedata_in *)mymalloc("BlackholeDataGet", Nimport * sizeof(struct blackholedata_in));
        BlackholeDataIn =
            (struct blackholedata_in *)mymalloc("BlackholeDataIn", Nexport * sizeof(struct blackholedata_in));

        TIMER_POPPUSH(1, "BHFB_PREPARE", "BHFB_COPY_IN");

        for (int j = 0; j < Nexport; j++)
        {
            int place = DataIndexTable[j].Index;
            particle2in_blackhole(&BlackholeDataIn[j], place);
            memcpy(BlackholeDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                   NODELISTLENGTH * sizeof(int));
        }

        /* exchange particle data */
        TIMER_POPPUSH(1, "BHFB_COPY_IN", "BHFB_COMM");
        templated_exchange_data_Sendrecv(BlackholeDataIn, BlackholeDataGet, TAG_DENS_A, Send_offset, Send_count,
                                         Recv_offset, Recv_count);
        TIMER_POP(1, "BHFB_COMM");

        TIMER_POPPUSH(2, "Exchange", "Secondary");

        TIMER_PUSH(1, "BHFB_COMPUTE");

        myfree(BlackholeDataIn);

        BlackholeDataResult =
            (struct blackholedata_out *)mymalloc("BlackholeDataResult", Nimport * sizeof(struct blackholedata_out));
        BlackholeDataOut =
            (struct blackholedata_out *)mymalloc("BlackholeDataOut", Nexport * sizeof(struct blackholedata_out));

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
            blackhole_evaluate_secondary(&thread_id); /* do local particles and prepare export list */
        }

        TIMER_POPPUSH(1, "BHFB_COMPUTE", "BHFB_WAIT");

        if (NextParticle < 0)
        {
            ndone_flag = 1;
        }
        else
        {
            ndone_flag = 0;
        }

        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

        TIMER_POP(1, "BHFB_WAIT");

        TIMER_POPPUSH(2, "Secondary", "Send_Results");

        TIMER_PUSH(1, "BHFB_COMM");
        templated_exchange_data_Sendrecv(BlackholeDataResult, BlackholeDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                         Send_offset, Send_count);
        TIMER_POPPUSH(1, "BHFB_COMM", "BHFB_COPY");

        /* add the result to the particles */
        for (int j = 0; j < Nexport; j++)
        {
            int place = DataIndexTable[j].Index;
            out2particle_blackhole(&BlackholeDataOut[j], place, 1);
        }

        TIMER_POP(1, "BHFB_COPY");
        TIMER_POP(2, "Send_Results");

        myfree(BlackholeDataOut);
        myfree(BlackholeDataResult);
        myfree(BlackholeDataGet);
    } while (ndone < NTask);

    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);

#ifdef AD_DYNFRIC
    TIMER_PUSH(2, "Final");
    TIMER_PUSH(1, "BHFB_DYN_FRIC");
#ifdef _OPENMP
#pragma omp parallel for if (NActivePart > 80)
    for (int il = 0; il < NActivePart; il++)
    {
        int i = ActiveParticleList[il];
#else // _OPENMP
    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
#endif // _OPENMP
        if (P[i].Type == 5 && P[i].Mass > 0 && BPP(i).BHDF_ngb > 0)
        {
            assert(BHP[P[i].pt.BHID].PID == i);
            MyAtLeastDouble ad_prefac = 0.;
            if (All.BlackHoleFrictionForceOn == 1)
            {
                ad_prefac = 3. / 2. * All.GravityConstantInternal * All.GravityConstantInternal /
                            (All.ForceSoftening[5] * All.ForceSoftening[5] * All.ForceSoftening[5]);
            }
            else if (All.BlackHoleFrictionForceOn == 2)
            {
                ad_prefac = (2 * M_PI) * All.GravityConstantInternal * All.GravityConstantInternal;
            }
            for (int k = 0; k < 3; k++)
            {
                BPP(i).BHDF_DynFric[k] = ad_prefac * BPP(i).BHDF_DynFric[k];
#ifdef DEBUG
                if (abs(BPP(i).BHDF_DynFric[k]) > abs(P[i].GravAccel[k]) && abs(BPP(i).BHDF_DynFric[k]) > 1e6)
                {
                    VERBOSE_ALL(0, "WARNING DYNFRIC: P[%d].ID=%llu a_frc=%e > a_grav=%e  (total)", i,
                                (unsigned long long)P[i].ID, BPP(i).BHDF_DynFric[k], P[i].GravAccel[k]);
                }
#endif // DEBUG

                P[i].GravAccel[k] += BPP(i).BHDF_DynFric[k];
            }

            fprintf(FdBlackHolesDetails,
                    "DYNFRIC:At time : %g  on ID: %lu Particles found: %d Smoothing Length : "
                    "%g Dynamical friction : (%g %g %g) GravAccel (%g %g %g)  \n",
                    All.Time, (unsigned long int)P[i].ID, BPP(i).BHDF_ngb, P[i].Hsml, BPP(i).BHDF_DynFric[0],
                    BPP(i).BHDF_DynFric[1], BPP(i).BHDF_DynFric[2], P[i].GravAccel[0], P[i].GravAccel[1],
                    P[i].GravAccel[2]);
            for (int k = 0; k < 3; k++)
            {
                BPP(i).BHDF_DynFric[k] = 0;
            }
        }
    }
    TIMER_POP(1, "BHFB_DYN_FRIC");
    TIMER_POP(2, "Final");
#endif // AD_DYNFRIC
    TIMER_POP(1, "COMPUTE_BH_FEEDBACK");
}

#if (defined(PB_BH_KINETIC_FEEDBACK) || defined(PB_BH_KIN_CONE_ANGLE_DEG))
/* Search among gas neighbors within [factor * h_sml_BH] to find gas mass inside Bi-Cone with BH at apex.
 */
int blackhole_Gas_Mass_Inside_Cone(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                                   int *ngblist)
{
    struct blackholedata_in local;
    struct blackholedata_out out;
    memset(&out, 0, sizeof(struct blackholedata_out));

    int startnode, numngb, j, n, listindex = 0, Nparticle_Cone_Up = 0, Nparticle_Cone_Dn = 0;
    MyLongDouble *pos;
    MyFloat h_i, mass, *UnitVector_WindKick, dr_dot_KickVec, Angle_deg;
    MyFloat dx, dy, dz, h_i2, r2, Added_Mass_Cone_Up = 0.0, Added_Mass_Cone_Dn = 0.0;

    if (mode == 0)
    {
        particle2in_blackhole(&local, target);
    }
    else
    {
        local = BlackholeDataGet[target];
    }

    pos = local.Pos;
    h_i = local.Hsml;
    mass = local.Mass;
    UnitVector_WindKick = local.UnitVector_AGN_Wind;

#ifdef PB_BH_KIN_SET_MIN_BH_SMOOTHING /* Limiting distance to find neighbors. */
    if (h_i < PB_MIN_BH_SMOOTHING_FACTOR * All.SofteningBndry)
    {
        h_i = PB_MIN_BH_SMOOTHING_FACTOR * All.SofteningBndry;
    }
#endif // PB_BH_KIN_SET_MIN_BH_SMOOTHING

    h_i2 = h_i * h_i;

    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = BlackholeDataGet[target].NodeList[listindex];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            numngb = ngb_treefind_blackhole_threads(pos, h_i, target, &startnode, mode, exportflag, exportnodecount,
                                                    exportindex, ngblist);
            if (numngb < 0)
            {
                return -1;
            }

            for (n = 0; n < numngb; n++)
            {
                j = ngblist[n];

                if ((P[j].Type == 0) && (P[j].Mass > 0) && (mass > 0))
                { /* Gas particle, existing, around an existing BH. */
                    dx = NEAREST_X(pos[0] - P[j].Pos[0]);
                    dy = NEAREST_Y(pos[1] - P[j].Pos[1]);
                    dz = NEAREST_Z(pos[2] - P[j].Pos[2]);

                    r2 = dx * dx + dy * dy + dz * dz;

                    dr_dot_KickVec = -((dx * UnitVector_WindKick[0]) + (dy * UnitVector_WindKick[1]) +
                                       (dz * UnitVector_WindKick[2]));

                    Angle_deg = (acos(dr_dot_KickVec / sqrt(r2))) * 180.0 / M_PI;

                    if (r2 < h_i2)
                    {
                        if (Angle_deg < PB_BH_KIN_CONE_ANGLE_DEG)
                        {
                            Added_Mass_Cone_Up += P[j].Mass;
                            Nparticle_Cone_Up++;
                        }
                        if (180.0 - Angle_deg < PB_BH_KIN_CONE_ANGLE_DEG)
                        {
                            Added_Mass_Cone_Dn += P[j].Mass;
                            Nparticle_Cone_Dn++;
                        }
                    }
                }
            }
        } // while

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = BlackholeDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    } // while

    out.GasMass_Cone_Up = Added_Mass_Cone_Up;
    out.GasMass_Cone_Dn = Added_Mass_Cone_Dn;

    out.Np_Cone_Up = Nparticle_Cone_Up;
    out.Np_Cone_Dn = Nparticle_Cone_Dn;

    /* Now collect the result at the right place */
    if (mode == 0)
    {
        out2particle_blackhole(&out, target, 0);
    }
    else
    {
        BlackholeDataResult[target] = out;
    }
    return 0;
}
#endif // PB_BH_KINETIC_FEEDBACK || PB_BH_KIN_CONE_ANGLE_DEG

/**
 * @brief Evaluates the properties and interactions of a black hole particle with its neighbors.
 *
 * This function performs the evaluation of a black hole particle based on its type, properties, and
 * interactions with neighboring particles. It calculates physical quantities such as potential,
 * relative velocities, and energy feedback.
 *
 * @param target The index of the target black hole particle to evaluate.
 * @param mode Indicates the mode of operation:
 *              - 0: Process an individual particle.
 *              - 1: Collect results from previously stored evaluations.
 * @param exportflag An array used to mark particles for export. The size of the array is determined
 *                    by the number of tasks.
 * @param exportnodecount An array to count the number of nodes to be exported for each task.
 * @param exportindex An array of indices for the export nodes.
 * @param ngblist An array that will be filled with indices of neighboring particles.
 *
 * @return Returns 0 on success. If an error occurs (e.g., too many neighbors found), it returns -1.
 */
int blackhole_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist)
{
    struct blackholedata_in local;
    struct blackholedata_out out;
    memset(&out, 0, sizeof(struct blackholedata_out));
#ifdef GM_REPOSITION_ON_STARDENSITY_MAX
    MyLongDouble stardenspos[3] = {0., 0., 0.}, stardens = -1.0;
#ifdef GM_USE_VELS_ON_REPOSITIONING
    MyLongDouble stardensvel[3] = {0., 0., 0.};
#endif // GM_USE_VELS_ON_REPOSITIONING
#endif // GM_REPOSITION_ON_STARDENSITY_MAX

    out.BH_MinPot = BHPOTVALUEINIT;

    if (mode == 0)
    {
        particle2in_blackhole(&local, target);
    }
    else
    {
        local = BlackholeDataGet[target];
    }

    /* initialize variables before SPH loop is started */
    MyAtLeastDouble h_i2 = local.Hsml * local.Hsml;

    /* Now start the actual SPH computation for this particle */
    int startnode, listindex = 0;
    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = BlackholeDataGet[target].NodeList[listindex];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            int numngb = ngb_treefind_blackhole_threads(local.Pos, local.Hsml, target, &startnode, mode, exportflag,
                                                        exportnodecount, exportindex, ngblist);

            ENDRUN_IF(numngb > NumPart / DENSITY_LESS_NGB_FACTOR,
                      "black_hole: Number of neighbours found (%d) exceeds allowed number (%d) derived from "
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
#ifdef WINDS
                int Valid = 1;
                if (P[j].Mass <= 0)
                {
                    Valid = 0;
                }
                if (P[j].Type == 0)
                {
                    if (SphP[j].DelayTime > 0)
                    {
                        Valid = 0;
                    }
                }
                if (Valid == 1)
#else // WINDS
                if (P[j].Mass > 0)
#endif // WINDS
                {
                    if (local.Mass > 0)
                    {
                        MyAtLeastDouble dx = NEAREST_X(local.Pos[0] - P[j].Pos[0]);
                        MyAtLeastDouble dy = NEAREST_Y(local.Pos[1] - P[j].Pos[1]);
                        MyAtLeastDouble dz = NEAREST_Z(local.Pos[2] - P[j].Pos[2]);

                        MyAtLeastDouble r2 = dx * dx + dy * dy + dz * dz;
#ifdef AD_DYNFRIC
                        MyAtLeastDouble h_soft = All.ForceSoftening[5] * All.ForceSoftening[5];
                        MyAtLeastDouble h_inv = 1.0 / All.ForceSoftening[5];

                        if (All.BlackHoleFrictionForceOn > 0)
                        {
                            if (r2 < h_soft)
                            {
                                if (P[j].Type == 4 || P[j].Type == 1)
                                {

                                    out.BHDF_ngb++;
                                    MyAtLeastDouble vrel_2 = 0;
                                    MyAtLeastDouble vrel[3] = {0, 0, 0};
                                    for (int k = 0; k < 3; k++)
                                    {
                                        vrel[k] = P[j].Vel[k] - local.Vel[k];
                                        vrel_2 += vrel[k] * vrel[k];
                                    }
                                    MyAtLeastDouble Lambda = All.ForceSoftening[5] * vrel_2 /
                                                             (All.GravityConstantInternal * (P[j].Mass + local.Mass));
                                    MyAtLeastDouble dyn_fric_fac = 0.;

                                    if (All.BlackHoleFrictionForceOn == 2)
                                    {
                                        MyAtLeastDouble r = sqrt(r2);
                                        MyAtLeastDouble u = r / All.ForceSoftening[5];
                                        MyAtLeastDouble fac = 0.;
                                        MyAtLeastDouble prefac = 8. / (M_PI) * (h_inv * h_inv * h_inv);

                                        if (u > 1.)
                                        {
                                            fac = 3. / (4. * M_PI * (r2 * r));
                                        }
                                        else if (u < 0.5)
                                        {
                                            fac = prefac * (1 - 6 * (u * u) + 6 * (u * u * u));
                                        }
                                        else if (u > 0.5 && u < 1)
                                        {
                                            fac = prefac * (2 * (1 - u) * (1 - u) * (1 - u));
                                        }
                                        dyn_fric_fac = fac * log(1 + Lambda * Lambda) * P[j].Mass *
                                                       (local.Mass + P[j].Mass) / vrel_2 / sqrt(vrel_2);
                                    }
                                    else
                                    {
                                        dyn_fric_fac = log(1 + Lambda * Lambda) * P[j].Mass * (local.Mass + P[j].Mass) /
                                                       vrel_2 / sqrt(vrel_2);
                                    }

                                    for (int k = 0; k < 3; k++)
                                    {
#ifdef DEBUG
                                        MyAtLeastDouble AccDynThis =
                                            dyn_fric_fac * vrel[k] * 3.0 / 2.0 * All.GravityConstantInternal *
                                            All.GravityConstantInternal /
                                            (All.ForceSoftening[5] * All.ForceSoftening[5] * All.ForceSoftening[5]);
                                        if (abs(AccDynThis) > abs(P[j].GravAccel[k]) && abs(AccDynThis) > 1e6)
                                        {
                                            VERBOSE_ALL(
                                                0, "WARNING DYNFRIC: a_frc=%e > a_grav=%e  dv=%e  sqrt(gm/r)=%e",
                                                AccDynThis, P[j].GravAccel[k], sqrt(vrel_2),
                                                sqrt(All.GravityConstantInternal * local.Mass / All.ForceSoftening[5]));
                                        }
#endif // DEBUG
                                        out.BHDF_DynFric[k] += dyn_fric_fac * vrel[k];
                                    }
                                }
                            }
                        }
#endif // AD_DYNFRIC

                        if (r2 < h_i2)
                        {
                            MyAtLeastDouble vrel;
                            if (All.BlackHoleRepositioningOn == 1) /* if this option is switched on, we may also
                                                                      encounter dark matter particles or stars */
                            {
#ifdef GM_BH_MOVE_ON_GRAV_SOFT
                                if (r2 < All.SofteningTable[5] * All.SofteningTable[5] * GM_BH_MOVE_ON_GRAV_SOFT *
                                             GM_BH_MOVE_ON_GRAV_SOFT)
#endif // GM_BH_MOVE_ON_GRAV_SOFT
                                    if (P[j].Potential < out.BH_MinPot)
                                    {
                                        if (P[j].Type == 0 || P[j].Type == 1 || P[j].Type == 4 || P[j].Type == 5)
                                        {
                                            vrel = 0;
                                            for (int k = 0; k < 3; k++)
                                            {
                                                vrel += (P[j].Vel[k] - local.Vel[k]) *
                                                        (P[j].Vel[k] - local.Vel[k]); /* compute relative velocities */
                                            }
                                            vrel = sqrt(vrel) / ascale;
                                            if (vrel <= 0.25 * local.Csnd)
                                            {
                                                out.BH_MinPot = P[j].Potential;
                                                for (int k = 0; k < 3; k++)
                                                {
                                                    out.BH_MinPotPos[k] = P[j].Pos[k];
                                                }
                                            }
                                        }
                                    }
                            }

#ifdef GM_REPOSITION_ON_STARDENSITY_MAX
                            /* if this option is switched on, stars are used here */
                            if (All.BlackHoleRepositioningOn == 1)
                            {
                                if (P[j].Type == 4) // here we only consider stars
                                {
                                    if (P[j].StarDensity > out.BH_StarDens)
                                    {
                                        {
                                            /* compute relative velocities */
                                            for (int k = 0, vrel = 0; k < 3; k++)
                                            {
                                                vrel += (P[j].Vel[k] - local.Vel[k]) * (P[j].Vel[k] - local.Vel[k]);
                                            }

                                            vrel = sqrt(vrel) / ascale;

                                            if (vrel <= GM_VREL_REPOSITION)
                                            {
                                                out.BH_StarDens = P[j].StarDensity;
                                                for (int k = 0; k < 3; k++)
                                                {
                                                    out.BH_StarDensPos[k] = P[j].Pos[k];
#ifdef GM_USE_VELS_ON_REPOSITIONING
                                                    out.BH_StarDensVel[k] = P[j].Vel[k];
#endif // GM_USE_VELS_ON_REPOSITIONING
                                                }
                                                if (All.BlackHoleDetails >= 2)
                                                {
                                                    fprintf(FdBlackHolesDetails,
                                                            "ThisTask=%d, time=%g: reposition on stellar densiy max BH "
                                                            "%llu vrel=%g dens=%g \n",
                                                            ThisTask, All.Time, (unsigned long long)local.ID, vrel,
                                                            out.BH_StarDens);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
#endif /* ends GM_REPOSITION_ON_STARDENSITY_MAX */

                            if (P[j].Type == 5) /* we have a black hole merger */
                            {
                                if (local.ID != P[j].ID)
                                {
                                    assert(BHP[P[j].pt.BHID].PID == j);
                                    vrel = 0;
                                    for (int k = 0; k < 3; k++)
                                    {
                                        vrel += (P[j].Vel[k] - local.Vel[k]) *
                                                (P[j].Vel[k] - local.Vel[k]); /* compute relative velocity of BHs */
                                    }

                                    MyAtLeastDouble rrel = sqrt(r2);
                                    int flag_merg = 0; /* flag for merging BHs */
                                    MyAtLeastDouble brel;

                                    if (All.BlackHoleMergerCondition == 2)
                                    {
                                        brel = abs(P[j].Potential - local.Potential);
                                        if ((0.5 * vrel < (brel)) &&
                                            (rrel < All.BlackHoleMergeDistFrac * All.SofteningTable[5]))
                                        {
                                            flag_merg = 1;
                                        }
                                    }

                                    else if (All.BlackHoleMergerCondition == 1)
                                    {

                                        vrel = sqrt(vrel) / ascale;

                                        brel = abs(P[j].Potential - local.Potential) / ascale + vrel * vrel;
                                        // BlackHoles should meger if fullfill soundspeed, distance and binding criteria
                                        // (if set)
                                        if (
#ifdef GM_USE_ABSVAL_IN_VREL
                                            ((vrel < All.BlackHoleMergeCsndFrac / ascale) ||
                                             (All.BlackHoleMergeCsndFrac <= 0))
#else // GM_USE_ABSVAL_IN_VREL
                                            ((vrel < All.BlackHoleMergeCsndFrac * local.Csnd) ||
                                             (All.BlackHoleMergeCsndFrac <= 0))
#endif // GM_USE_ABSVAL_IN_VREL
                                            && ((rrel < All.BlackHoleMergeDistFrac * All.SofteningTable[5]) ||
                                                (All.BlackHoleMergeDistFrac <= 0)) &&
                                            ((brel < All.BlackHoleMergeBindingFrac * local.Csnd * local.Csnd) ||
                                             All.BlackHoleMergeBindingFrac <= 0))
                                        {
                                            flag_merg = 1;
                                        }
                                    }
                                    if (flag_merg == 1)
                                    {
#ifdef _OPENMP
#pragma omp critical(_bh5)
#endif // _OPENMP
                                        { // Second condition captures the right selection in rare cases of multiple
                                          // (>2) swallow posibilities
                                            MyAtLeastDouble P_j_Pot = P[j].Potential, l_Pot = local.Potential;
#ifndef DO_NOT_COMBINE_POT_AND_MASS_FOR_MERGER
                                            if (P_j_Pot > 0)
                                            {
                                                P_j_Pot /= log(BPP(j).BH_Mass);
                                            }
                                            else
                                            {
                                                P_j_Pot *= log(BPP(j).BH_Mass);
                                            }
                                            if (l_Pot > 0)
                                            {
                                                l_Pot /= log(local.BH_Mass);
                                            }
                                            else
                                            {
                                                l_Pot *= log(local.BH_Mass);
                                            }
#endif // !DO_NOT_COMBINE_POT_AND_MASS_FOR_MERGER
                                            if (P_j_Pot > l_Pot && BPP(j).SwallowPot < l_Pot)
                                            {
                                                BPP(j).SwallowID = local.ID;
                                                BPP(j).SwallowPot = l_Pot;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if (All.BlackHoleDetails >= 2)
                                        {
                                            if (All.BlackHoleMergerCondition == 1)
                                            {
                                                fprintf(FdBlackHolesDetails,
                                                        "SWALLOW: ThisTask=%d, time=%g: id=%llu would like to swallow "
                                                        "%llu, but vrel=%g csnd=%g rrel=%g soft=%g brel=%g\n",
                                                        ThisTask, All.Time, (unsigned long long)local.ID,
                                                        (unsigned long long)P[j].ID, vrel, local.Csnd, rrel,
                                                        All.SofteningTable[5], brel);
                                            }
                                            else if (All.BlackHoleMergerCondition == 2)
                                            {
                                                fprintf(FdBlackHolesDetails,
                                                        "SWALLOW: ThisTask=%d, time=%g: id=%llu would like to swallow "
                                                        "%llu, but vrel2=%g rrel=%g soft=%g brel=%g\n",
                                                        ThisTask, All.Time, (unsigned long long)local.ID,
                                                        (unsigned long long)P[j].ID, vrel, rrel, All.SofteningTable[5],
                                                        brel);
                                            }
                                        }
                                    }

                                } // local.ID != P[j].ID
                            } // P[j].Type == 5

                            if (P[j].Type == 0) /* here we have a gas particle */
                            {
                                MyAtLeastDouble r = sqrt(r2);
                                MyAtLeastDouble hinv, hinv3, hinv4;
                                kernel_hinv(local.Hsml, &hinv, &hinv3, &hinv4);
                                MyAtLeastDouble u = r * hinv;
                                MyAtLeastDouble wk, dwk;
                                kernel_main(u, hinv3, hinv4, &wk, &dwk, 0);
#ifdef _OPENMP
#pragma omp critical(_bh0)
#endif // _OPENMP
                                {
                                    int number_of_slices_generated = 0;
                                    if (All.BlackHoleSwallowGasOn >= 1) /* compute accretion probability */
                                    {
                                        MyAtLeastDouble acc_mass = P[j].Mass;
                                        if (All.bits > 0 && All.BlackHoleAccretionSlicesOn == 1)
                                        {
                                            number_of_slices_generated = (P[j].ID >> (sizeof(MyIDType) * 8 - All.bits));
                                            acc_mass = acc_mass / (All.Generations - number_of_slices_generated);
                                        }

                                        MyAtLeastDouble sm = local.BH_Mdot * local.Dt;
                                        double p = -1;
                                        if (local.BH_Mass > local.Mass)
                                        {
                                            p = P[j].Mass / acc_mass * (1 - exp(-sm / P[j].Mass));
                                        }

                                        double w =
                                            get_random_number(P[j].ID); /* compute random number, uniform in [0,1] */

                                        if (All.BlackHoleSwallowGasOn == 2)
                                        {
#ifdef LT_STELLAREVOLUTION
                                            MyFloat xclouds = SphP[j].XColdCloud;
                                            if (xclouds >
                                                0.01) // Here we want to swallow only the star forming particles
#else // LT_STELLAREVOLUTION
      // FIX IT: need to compute xclouds and Temperature here !!
      // get_starformation_rate(j, &Temperature, &xclouds);
                                            ENDRUN_MAIN(
                                                "BlackHoleSwallowGasOn == 2, need xcloud to decide which particles "
                                                "to swallow, but LT_STELLAREVOLUTION is off. xclouds computation "
                                                "not yet implemented in this case.");
#endif // LT_STELLAREVOLUTION
                                                if (w < p)
                                                {
                                                    if (SphP[j].SwallowID < local.ID)
                                                    {
                                                        SphP[j].SwallowID = local.ID;
                                                    }
                                                }
                                        }
                                        else if (All.BlackHoleSwallowGasOn ==
                                                 1) // In this case we swallow the particles regardless their state
                                        {
                                            if (w < p)
                                            {
                                                if (SphP[j].SwallowID < local.ID)
                                                {
                                                    SphP[j].SwallowID = local.ID;
                                                }
#ifdef PB_CHECK_SWALLOW_PROBABILITY
                                                fprintf(FdBlackHolesDetails,
                                                        "PB_CHECK_SWALLOW_PROBABILITY: ThisTask=%d, id=%u, bh_mass=%e, "
                                                        "mass=%e, rho=%e, P[j].ID=%u, P[j].Mass=%e, wk=%e, p=%e \n",
                                                        ThisTask, id, bh_mass, mass, rho, P[j].ID, P[j].Mass, wk, p);
#endif // PB_CHECK_SWALLOW_PROBABILITY
                                            }
                                        }
                                    }

                                    if (P[j].Mass > 0 && local.BH_Density > 0 &&
                                        (All.BlackHoleThermalFeedbackOn == 1 || All.BlackHoleKineticFeedbackOn == 1))
                                    {
                                        MyAtLeastDouble energy =
                                            local.BH_TotalFeedbackEfficiency * local.BH_Mdot * local.Dt * C_LIGHT *
                                            C_LIGHT / (All.UnitVelocity_in_cm_per_s * All.UnitVelocity_in_cm_per_s);

                                        MyAtLeastDouble meddington = Mdot_Eddington(local.BH_Mass);
#ifdef LS_BLACK_HOLES_SPIN_EVOLUTION
                                        if (All.BlackHoleSubgridADOn == 1)
                                        {
                                            meddington = meddington / (BPP(n).BH_EtaDisk / 0.1);
                                        }
#endif // LS_BLACK_HOLES_SPIN_EVOLUTION

// Weighting the distribution
#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE
                                        if ((local.BH_Mdot / meddington) < All.BlackHoleRadioModeTreshold)
                                        {
#ifdef KD_BH_USE_TOPHAT_FOR_RADIO_MODE_AD_MASSWEIGHTS
                                            energy *= P[j].Mass / local.BH_TotalSurroundingMass;
#else
                                            energy *= 1. / P[j].TrueNGB;
#endif
                                        }
                                        else
                                        {
                                            energy *= P[j].Mass * wk / local.BH_Density;
                                        }
#else // KD_BH_USE_TOPHAT_FOR_RADIO_MODE
                                        energy *= P[j].Mass * wk / local.BH_Density;
#endif // KD_BH_USE_TOPHAT_FOR_RADIO_MODE

                                        if (All.BlackHoleLimitMaxTemp > 0) /* Limit maximum temperature */
                                        {
                                            MyAtLeastDouble u_to_temp_fac = (4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))) *
                                                                            PROTONMASS / BOLTZMANN * GAMMA_MINUS1 *
                                                                            All.UnitEnergy_in_cgs / All.UnitMass_in_g;
                                            MyAtLeastDouble temp = u_to_temp_fac * energy / P[j].Mass;
                                            if (temp > All.BlackHoleLimitMaxTemp)
                                            {
                                                VERBOSE(1,
                                                        "BH: WARNING: task=%d: id=%llu, m=%e large Injected_BH_Energy "
                                                        "in blackhole: %e+%e would be temp=%e ",
                                                        ThisTask, (unsigned long long)P[j].ID, (float)P[j].Mass,
                                                        (float)SphP[j].Injected_BH_Energy, (double)energy,
                                                        (double)temp);
                                                energy = All.BlackHoleLimitMaxTemp / u_to_temp_fac * P[j].Mass;
                                            }
                                        }

                                        int was_heated = 1;
                                        if (All.BlackHoleLimitFeedbackOn == 0 ||
                                            (All.BlackHoleLimitFeedbackOn & 2) == 2 ||
                                            (local.BH_Mdot / meddington) <
                                                All.BlackHoleRadioModeTreshold) /* no restriction in heating */
                                        {
                                            SphP[j].Injected_BH_Energy += energy;
                                        }
                                        else
                                        {
                                            MyFloat Temperature, xclouds;
#ifdef LT_STELLAREVOLUTION
                                            Temperature = SphP[j].Temperature;
                                            xclouds = SphP[j].XColdCloud;
#else // LT_STELLAREVOLUTION
      // FIX IT: need to compute Temperature here !!
      // get_starformation_rate(j, &Temperature, &xclouds);
                                            ENDRUN_MAIN("Trying to limit feedback but LT_STELLAREVOLUTION is off. "
                                                        "Temperature computation not yet implemented in this case.");
#endif // LT_STELLAREVOLUTION
                                            if ((All.BlackHoleLimitFeedbackOn & 1) == 1 &&
                                                Temperature > All.BlackHoleColdGasTemperatureTresh && xclouds < 0.01)
                                            {
                                                SphP[j].Injected_BH_Energy += energy;
                                            }
                                            else
                                            {
                                                was_heated = 0;
                                            }
                                        }

                                        if (SphP[j].SwallowID == local.ID && was_heated == 1)
                                        {
                                            SphP[j].Injected_BH_Energy -=
                                                energy / (All.Generations - number_of_slices_generated);
                                        }

                                        if (All.BlackHoleDetails >= 3)
                                        {
                                            fprintf(FdBlackHolesDetails,
                                                    "ENERGY: idBH=%llu time=%g: mbh=%g mdot=%g dt=%g: idGas=%llu gets "
                                                    "%g (of %g available)\n",
                                                    (unsigned long long)local.ID, All.Time, local.BH_Mass,
                                                    local.BH_Mdot, local.Dt, (unsigned long long)P[j].ID,
                                                    SphP[j].Injected_BH_Energy, energy);
                                        }
                                    } // if(P[j].Mass > 0 && local.BH_Density > 0)
                                } // critical
                            } // if(P[j].Type == 0)
                        } // if(r2 < h_i2)
                    } // if(mass > 0)
                } // (P[j].Mass > 0)
            } // for
        } // while

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = BlackholeDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
            }
        }
    } // while

    /* Now collect the result at the right place */
    if (mode == 0)
    {
        out2particle_blackhole(&out, target, 0);
    }
    else
    {
        BlackholeDataResult[target] = out;
    }

    return 0;
}

/**
 * @brief Evaluates black hole properties and manages particle exports in parallel threads.
 *
 * This function evaluates black hole properties and handles the export of particle data.
 * It processes particles based on their type, performs
 * evaluations, and manages flags and counts for exporting data.
 *
 * @param p Pointer to an integer containing the thread ID. The thread ID determines the segment of
 *          global arrays that the thread will work on.
 */
void *blackhole_evaluate_primary(void *p)
{
    int i;
    int thread_id = *((int *)p);
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
        {
            break;
        }

        if (P[i].Type == 5)
        {
            if (blackhole_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist) < 0)
            {
                break; // buffer was too small to add this BH, we exit without flagging it as processed.
            }
        }

        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Evaluates black hole properties in parallel threads.
 *
 * This function performs evaluations of black hole properties.
 * It processes particles in a parallel fashion, updating their properties based on the black hole evaluations.
 * Each thread processes a subset of the particles and utilizes the `blackhole_evaluate` function to perform
 * the actual computation.
 *
 * @param p Pointer to an integer containing the thread ID. The thread ID is used to determine the segment of
 *          the neighbor list to be processed by this thread.
 */
void *blackhole_evaluate_secondary(void *p)
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
        blackhole_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist);

    }
    return NULL;
}
#endif // BLACK_HOLES
