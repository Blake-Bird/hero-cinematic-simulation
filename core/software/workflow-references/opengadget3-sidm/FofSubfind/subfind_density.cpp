/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Functions to compute densities in Subfind.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
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
#include <sys/stat.h>
#include <sys/types.h>

#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"
#include "../Hydro/kernel.hpp"
#include "../System/communication.hpp"

#ifdef SUBFIND

#include "fof.hpp"
#include "subfind.hpp"

#define LOCK_NEXPORT
#define UNLOCK_NEXPORT

/**
 * @brief Structure for communication during the density computation. Holds data that is sent to other processors.
 */
static struct densdata_in
{
    MyLongDouble Pos[3];
    MyFloat Hsml;
    int NodeList[NODELISTLENGTH];
} *DensDataIn, *DensDataGet;

/**
 * @brief Structure for communication during the density computation. Holds data that is recieved from other processors.
 */
static struct densdata_out
{
    MyFloat Rho;
    MyFloat VelDisp, Vx, Vy, Vz;
    int Ngb;
} *DensDataResult, *DensDataOut;

#ifdef SUBFIND_RESHUFFLE_CATALOGUE
static MyFloat *DM_Vx, *DM_Vy, *DM_Vz;
#endif // SUBFIND_RESHUFFLE_CATALOGUE
static long long Ntotal;

/**
 * @brief Structure representing arguments for density evaluation.
 */
struct dens_eval_arg_t
{
    int threadid;
    char *todo;
#ifdef DENSITY_SPLIT_BY_TYPE
    int j_target;
    int j_in;
#endif // DENSITY_SPLIT_BY_TYPE
};

/**
 * @brief Computes density and other related quantities for particles.
 *
 * This function computes the density and other related quantities for particles, including
 * the number of neighbors (`DM_NumNgb`), density (`u.DM_Density`), and velocity dispersion
 * (`v.DM_VelDisp`). It iterates over particles until convergence is achieved, ensuring that
 * each particle has the desired number of neighbors (`All.DesNumNgb`).
 *
 * If density splitting by particle type is enabled (`DENSITY_SPLIT_BY_TYPE`), this function
 * only operates on particles of the specified target type (`j_target`), as well as input type
 * (`j_in`). Otherwise, it processes all particles.
 *
 * if DENSITY_SPLIT_BY_TYPE:
 * @param j_in Input particle type.
 */
#ifdef DENSITY_SPLIT_BY_TYPE
void subfind_density(int j_in)
#else // DENSITY_SPLIT_BY_TYPE
void subfind_density(void)
#endif // DENSITY_SPLIT_BY_TYPE
{
    long long ntot;
    int k, ndone, ndone_flag, npleft, iter = 0;
    MyFloat *Left, *Right;
    // char *Todo;
    int place;
    double t0, t1;

    int save_NextParticle;
    // long long n_exported = 0;

    struct dens_eval_arg_t dens_eval_arg;

#ifdef DENSITY_SPLIT_BY_TYPE
    int j_target;

    if (j_in < 0)
    {
        j_target = -j_in - 1;
    }
    else
    {
        j_target = j_in;
    }
#endif // DENSITY_SPLIT_BY_TYPE

    if (ThisTask == 0)
    {
#ifdef DENSITY_SPLIT_BY_TYPE
        VERBOSE_ALL(2, "finding densities, mode %d ", j_in);
#else // DENSITY_SPLIT_BY_TYPE
        VERBOSE_ALL(2, "finding densities for all particles.");
#endif // DENSITY_SPLIT_BY_TYPE
    }

    /* allocate buffers to arrange communication */

    int NTaskTimesNumPart;

    NTaskTimesNumPart = ((int)(maxThreads)) * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

    Ngblist = (int *)mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
    Dist2list = (MyFloat *)mymalloc("Dist2list", NTaskTimesNumPart * sizeof(MyFloat));

    Left = (MyFloat *)mymalloc("Left", sizeof(MyFloat) * NumPart);
    Right = (MyFloat *)mymalloc("Right", sizeof(MyFloat) * NumPart);
    dens_eval_arg.todo = (char *)mymalloc("Todo", sizeof(char) * NumPart);

    size_t MyBufferSize = KD_BUFFER_MANAGEMENT * FreeBytes / (1024.0 * 1024.0);

    All.BunchSize =
        (int)((MyBufferSize * 1024 * 1024) /
              (sizeof(struct data_index) + sizeof(struct data_nodelist) + sizeof(struct densdata_in) +
               sizeof(struct densdata_out) + sizemax(sizeof(struct densdata_in), sizeof(struct densdata_out))));
    DataIndexTable = (struct data_index *)mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *)mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

    VERBOSE(0, "Subfind Density: using %g MB for buffering, %g MB remaining for other buffers\n", MyBufferSize / 1.0,
            FreeBytes / (1024.0 * 1024.0));

#ifdef SUBFIND_RESHUFFLE_CATALOGUE
    DM_Vx = (MyFloat *)mymalloc("DM_Vx", sizeof(MyFloat) * NumPart);
    DM_Vy = (MyFloat *)mymalloc("DM_Vy", sizeof(MyFloat) * NumPart);
    DM_Vz = (MyFloat *)mymalloc("DM_Vz", sizeof(MyFloat) * NumPart);
#endif // SUBFIND_RESHUFFLE_CATALOGUE

#ifdef _OPENMP
#pragma omp parallel for
#endif // _OPENMP
    for (int i = 0; i < NumPart; i++)
    {
        Left[i] = Right[i] = 0;
        P[i].DM_NumNgb = 0;
#ifdef KD_SUBFIND_HSML_FOR_GROUP_MEMBER_ONLY
        // KD: Place ifdef here !!
        if (P[i].GrNr < TotNgroups + 1)
        {
            dens_eval_arg.todo[i] = 1;
        }
        else
        {
            dens_eval_arg.todo[i] = 0;
        }
#else // KD_SUBFIND_HSML_FOR_GROUP_MEMBER_ONLY
        dens_eval_arg.todo[i] = 1;
#endif // KD_SUBFIND_HSML_FOR_GROUP_MEMBER_ONLY
    }

    /* we will repeat the whole thing for those particles where we didn't find enough neighbours */
    do
    {
        t0 = second();
        NextParticle = 0; /* begin with this index */
        do
        {
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextParticle = NextParticle;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mythreadid = omp_get_thread_num();
#else // _OPENMP
                int mythreadid = 0;
#endif // _OPENMP
#ifdef DENSITY_SPLIT_BY_TYPE
                subfind_density_evaluate_primary(mythreadid, dens_eval_arg.todo, j_in, j_target);
#else // DENSITY_SPLIT_BY_TYPE
                subfind_density_evaluate_primary(mythreadid, dens_eval_arg.todo);
#endif // DENSITY_SPLIT_BY_TYPE
            }

            if (BufferFullFlag)
            {
                int last_nextparticle = NextParticle;

                NextParticle = save_NextParticle;

                while (NextParticle < NumPart)
                {
                    if (NextParticle == last_nextparticle)
                    {
                        break;
                    }

                    if (ProcessedFlag[NextParticle] != 1)
                    {
                        break;
                    }

                    ProcessedFlag[NextParticle] = 2;

                    NextParticle += 1;
                }

                if (NextParticle == save_NextParticle)
                {
                    /* in this case, the buffer is too small to process even a single particle */
                    VERBOSE_ALL(2, "Task %d: Type=%d pos=(%g,%g,%g) mass=%g", ThisTask, P[NextParticle].Type,
                                P[NextParticle].Pos[0], P[NextParticle].Pos[1], P[NextParticle].Pos[2],
                                P[NextParticle].Mass);
                    if (P[NextParticle].Type == 0)
                    {
                        VERBOSE_ALL(2, "   rho=%g hsml=%g", SphP[NextParticle].Density, P[NextParticle].Hsml);
                    }
                    /*if(P[NextParticle].Type == 4)
                       printf("   Nstars=%d hsml=%g\n", N_stars, P[NextParticle].Hsml); */

                    ENDRUN("buffer for density() too small");
                }

                int new_export = 0;
                k = 0;
                for (int j = 0; j < Nexport; j++)
                {
                    if (ProcessedFlag[DataIndexTable[j].Index] != 2)
                    {
                        if (k < j + 1)
                        {
                            k = j + 1;
                        }

                        for (; k < Nexport; k++)
                        {
                            if (ProcessedFlag[DataIndexTable[k].Index] == 2)
                            {
                                int old_index = DataIndexTable[j].Index;

                                DataIndexTable[j] = DataIndexTable[k];
                                DataNodeList[j] = DataNodeList[k];
                                DataIndexTable[j].IndexGet = j;
                                new_export++;

                                DataIndexTable[k].Index = old_index;
                                k++;
                                break;
                            }
                        }
                    }
                    else
                    {
                        new_export++;
                    }
                }

                Nexport = new_export;
            }

            // n_exported += Nexport;

            for (int j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
            }
            for (int j = 0; j < Nexport; j++)
            {
                Send_count[DataIndexTable[j].Task]++;
            }

            MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);

            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);
            Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0;
            for (int j = 0; j < NTask; j++)
            {
                Nimport += Recv_count[j];

                if (j > 0)
                {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }

            DensDataGet = (struct densdata_in *)mymalloc("	  DensDataGet", Nimport * sizeof(struct densdata_in));
            DensDataIn = (struct densdata_in *)mymalloc("	  DensDataIn", Nexport * sizeof(struct densdata_in));

            /* prepare particle data for export */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;

                DensDataIn[j].Pos[0] = P[place].Pos[0];
                DensDataIn[j].Pos[1] = P[place].Pos[1];
                DensDataIn[j].Pos[2] = P[place].Pos[2];
                DensDataIn[j].Hsml = P[place].DM_Hsml;

                memcpy(DensDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList,
                       NODELISTLENGTH * sizeof(int));
            }

            /* exchange particle data */
            TIMER_PUSH(4, "subfind_comm");
            templated_exchange_data_Sendrecv(DensDataIn, DensDataGet, TAG_DENS_A, Send_offset, Send_count, Recv_offset,
                                             Recv_count);
            TIMER_POP(4, "subfind_comm");

            myfree(DensDataIn);
            DensDataResult =
                (struct densdata_out *)mymalloc("	  DensDataResult", Nimport * sizeof(struct densdata_out));
            DensDataOut = (struct densdata_out *)mymalloc("	  DensDataOut", Nexport * sizeof(struct densdata_out));

            /* now do the particles that were sent to us */

            NextJ = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif // _OPENMP
            {
#ifdef _OPENMP
                int mythreadid = omp_get_thread_num();
#else // _OPENMP
                int mythreadid = 0;
#endif // _OPENMP
#ifdef DENSITY_SPLIT_BY_TYPE
                subfind_density_evaluate_secondary(mythreadid, j_in);
#else // DENSITY_SPLIT_BY_TYPE
                subfind_density_evaluate_secondary(mythreadid);
#endif // DENSITY_SPLIT_BY_TYPE
            }

            if (NextParticle >= NumPart)
            {
                ndone_flag = 1;
            }
            else
            {
                ndone_flag = 0;
            }

            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
            TIMER_PUSH(4, "subfind_comm");
            templated_exchange_data_Sendrecv(DensDataResult, DensDataOut, TAG_DENS_B, Recv_offset, Recv_count,
                                             Send_offset, Send_count);
            TIMER_POP(4, "subfind_comm");
            /* add the result to the local particles */
            for (int j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;
                P[place].DM_NumNgb += DensDataOut[j].Ngb;
                P[place].u.DM_Density += DensDataOut[j].Rho;
#ifdef DENSITY_SPLIT_BY_TYPE
                if (j_in >= 0) /* sum up VelDisp only for own species */
#endif // DENSITY_SPLIT_BY_TYPE
                    P[place].v.DM_VelDisp += DensDataOut[j].VelDisp;
#ifdef SUBFIND_RESHUFFLE_CATALOGUE
                DM_Vx[place] += DensDataOut[j].Vx;
                DM_Vy[place] += DensDataOut[j].Vy;
                DM_Vz[place] += DensDataOut[j].Vz;
#endif // SUBFIND_RESHUFFLE_CATALOGUE
            }

            myfree(DensDataOut);
            myfree(DensDataResult);
            myfree(DensDataGet);

        } while (ndone < NTask);

#ifdef DENSITY_SPLIT_BY_TYPE
        VERBOSE(2, "checking smoothing lenghs ... ");
#endif // DENSITY_SPLIT_BY_TYPE

        /* do final operations on results */
        npleft = 0;
        for (int i = 0; i < NumPart; i++)
        {
            /* now check whether we had enough neighbours */
#ifdef DENSITY_SPLIT_BY_TYPE
            if (j_in < 0)
            {
                dens_eval_arg.todo[i] = 0;
            }
            if ((P[i].Type != j_target) || (j_in < 0))
            {
                continue;
            }
#else // DENSITY_SPLIT_BY_TYPE
            if (!((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
            {
                continue;
            }
#endif // DENSITY_SPLIT_BY_TYPE

            if (dens_eval_arg.todo[i])
            {
                if (P[i].DM_NumNgb != All.DesNumNgb &&
                    ((Right[i] - Left[i]) > 1.0e-4 * Left[i] || Left[i] == 0 || Right[i] == 0))
                {
                    /* need to redo this particle */
                    npleft++;

                    if (P[i].DM_NumNgb < All.DesNumNgb)
                    {
                        Left[i] = max(P[i].DM_Hsml, Left[i]);
                    }
                    else
                    {
                        if (Right[i] != 0)
                        {
                            if (P[i].DM_Hsml < Right[i])
                            {
                                Right[i] = P[i].DM_Hsml;
                            }
                        }
                        else
                        {
                            Right[i] = P[i].DM_Hsml;
                        }
                    }

                    if (iter >= MAXITER - 10)
                    {
                        VERBOSE_ALL(
                            2, "i=%d task=%d ID=%d Hsml=%g Left=%g Right=%g Ngbs=%g Right-Left=%g\n   pos=(%g|%g|%g)",
                            i, ThisTask, (int)P[i].ID, P[i].DM_Hsml, Left[i], Right[i], (double)P[i].DM_NumNgb,
                            Right[i] - Left[i], P[i].Pos[0], P[i].Pos[1], P[i].Pos[2]);
                    }

                    if (Right[i] > 0 && Left[i] > 0)
                    {
                        P[i].DM_Hsml = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
                    }
                    else
                    {
                        if (Right[i] == 0 && Left[i] == 0)
                        {
                            ENDRUN("Error"); /* can't occur */
                        }

                        if (Right[i] == 0 && Left[i] > 0)
                        {
                            P[i].DM_Hsml *= 1.26;
                        }

                        if (Right[i] > 0 && Left[i] == 0)
                        {
                            P[i].DM_Hsml /= 1.26;
                        }
                    }
                }
                else
                {
                    dens_eval_arg.todo[i] = 0;
                }
            }
        }

        sumup_large_ints(1, &npleft, &ntot);

        t1 = second();

        if (ntot > 0)
        {
            iter++;

            if (iter > 0 && ThisTask == 0)
            {
                VERBOSE_ALL(2, "ngb iteration %d: need to repeat for %d%09d particles. (took %g sec)\n", iter,
                            (int)(ntot / 1000000000), (int)(ntot % 1000000000), timediff(t0, t1));
            }

            ENDRUN_IF(iter > MAXITER, "failed to converge in neighbour iteration in density()");
        }
    } while (ntot > 0);

#ifdef SUBFIND_RESHUFFLE_CATALOGUE
    double atime = -1;
    if (All.ComovingIntegrationOn)
    {
        atime = All.Time;
    }
    else
    {
        atime = 1;
    }
    double vel_to_phys = 1.0 / atime;
#endif // SUBFIND_RESHUFFLE_CATALOGUE

#ifdef DENSITY_SPLIT_BY_TYPE
    VERBOSE(2, "final operations ... ");
#endif // DENSITY_SPLIT_BY_TYPE

    for (int i = 0; i < NumPart; i++)
#ifdef DENSITY_SPLIT_BY_TYPE
        if (P[i].Type == j_target && j_in >= 0)
#else // DENSITY_SPLIT_BY_TYPE
        if (((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif // DENSITY_SPLIT_BY_TYPE
        {
#ifdef DENSITY_SPLIT_BY_TYPE
            ENDRUN_IF(P[i].DM_NumNgb == 0, "Number of neighbours is zero");
#endif // DENSITY_SPLIT_BY_TYPE
#ifdef SUBFIND_RESHUFFLE_CATALOGUE
            DM_Vx[i] /= P[i].DM_NumNgb;
            DM_Vy[i] /= P[i].DM_NumNgb;
            DM_Vz[i] /= P[i].DM_NumNgb;
            P[i].v.DM_VelDisp /= P[i].DM_NumNgb;

            P[i].v.DM_VelDisp =
                vel_to_phys * sqrt(P[i].v.DM_VelDisp - DM_Vx[i] * DM_Vx[i] - DM_Vy[i] * DM_Vy[i] - DM_Vz[i] * DM_Vz[i]);
#endif // SUBFIND_RESHUFFLE_CATALOGUE
        }
#ifdef SUBFIND_RESHUFFLE_CATALOGUE
    myfree(DM_Vz);
    myfree(DM_Vy);
    myfree(DM_Vx);
#endif // SUBFIND_RESHUFFLE_CATALOGUE

    myfree(DataNodeList);
    myfree(DataIndexTable);

    myfree(dens_eval_arg.todo);
    myfree(Right);
    myfree(Left);

    myfree(Dist2list);
    myfree(Ngblist);
}

/**
 * @brief Evaluates density and related quantities for a target particle.
 *
 * This function computes the density and related quantities for a target particle,
 * including the number of neighbors (`DM_NumNgb`), density (`u.DM_Density`), and
 * velocity dispersion (`v.DM_VelDisp`). It iterates over neighboring particles within
 * the smoothing length (`DM_Hsml`) and calculates the contributions to density using
 * a kernel function.
 *
 * If density splitting by particle type is enabled (`DENSITY_SPLIT_BY_TYPE`), the function
 * may operate differently depending on the mode and target particle type.
 *
 * If DENSITY_SPLIT_BY_TYPE:
 * @param target The index of the target particle for which to evaluate density.
 * @param mode   The mode of operation: 0 for local particle, 1 for imported particle.
 * @param exportflag       Pointer to export flags for communication.
 * @param exportnodecount  Pointer to export node counts for communication.
 * @param exportindex      Pointer to export indices for communication.
 * @param ngblist          Pointer to the list of neighboring particles.
 * @param dist2list        Pointer to the list of squared distances to neighboring particles.
 *
 * Else:
 * @param dist2list        Pointer to the list of squared distances to neighboring particles.
 *
 * @return Returns 0 upon successful completion, or -1 if an error occurs.
 */
#ifdef DENSITY_SPLIT_BY_TYPE
int subfind_density_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                             int *ngblist, MyFloat *dist2list, int tp)
#else // DENSITY_SPLIT_BY_TYPE
int subfind_density_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex,
                             int *ngblist, MyFloat *dist2list)
#endif // DENSITY_SPLIT_BY_TYPE
{

    MyFloat h;
    MyLongDouble *pos;
    if (mode == 0)
    {
        pos = P[target].Pos;
        h = P[target].DM_Hsml;
    }
    else
    {
        pos = DensDataGet[target].Pos;
        h = DensDataGet[target].Hsml;
    }

#ifdef DENSITY_SPLIT_BY_TYPE
    ENDRUN_IF(h == 0, "h is zero");
#endif // DENSITY_SPLIT_BY_TYPE

    int startnode = 0;
    if (mode == 0)
    {
        startnode = All.MaxPart; /* root node */
    }
    else
    {
        startnode = DensDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode; /* open it */
    }

    int numngb = 0;
    double rho = 0;
    int numngb_inbox, listindex = 0;
    double vx = 0, vy = 0, vz = 0, v2 = 0;
    while (startnode >= 0)
    {
        while (startnode >= 0)
        {
            double hmax;
#ifdef DENSITY_SPLIT_BY_TYPE
            if (tp < 0)
            {
                numngb_inbox =
                    subfind_ngb_treefind_linkpairs_threads(pos, h, target, &startnode, mode, &hmax, exportflag,
                                                           exportnodecount, exportindex, ngblist, dist2list);
            }
            else // DENSITY_SPLIT_BY_TYPE
            {
                numngb_inbox = subfind_ngb_treefind_linkngb_threads(pos, h, target, &startnode, mode, &hmax, exportflag,
                                                                    exportnodecount, exportindex, ngblist, dist2list);
            }
#else // DENSITY_SPLIT_BY_TYPE
            numngb_inbox = subfind_ngb_treefind_linkngb_threads(pos, h, target, &startnode, mode, &hmax, exportflag,
                                                                exportnodecount, exportindex, ngblist, dist2list);
#endif // DENSITY_SPLIT_BY_TYPE

            if (numngb_inbox > NumPart / DENSITY_LESS_NGB_FACTOR_SUB)
            {
                ENDRUN(0, "numngb_inbox>NumPart/DENSITY_LESS_NGB_FACTOR_SUB => %d>%d/%d ", numngb_inbox, NumPart,
                       DENSITY_LESS_NGB_FACTOR_SUB);
            }

            if (numngb_inbox < 0)
            {
                return -1;
            }

#ifdef DENSITY_SPLIT_BY_TYPE
            if (tp >= 0)
#endif // DENSITY_SPLIT_BY_TYPE
                if (mode == 0 && hmax > 0)
                {
                    P[target].DM_Hsml = hmax;
                    h = hmax;

                    ENDRUN_IF(numngb_inbox != All.DesNumNgb, "Number of neighbours is not equal to desired number");
                }

            numngb += numngb_inbox;

            for (int n = 0; n < numngb_inbox; n++)
            {
                int j = ngblist[n];
                double r2 = dist2list[n];

#ifdef DENSITY_SPLIT_BY_TYPE
                if (tp < 0)
                {
                    h = P[j].DM_Hsml;
                    ENDRUN_IF(h == 0, "h is zero");
                }
#endif // DENSITY_SPLIT_BY_TYPE

                double h2 = h * h;

                if (r2 < h2)
                {
                    MyAtLeastDouble r = sqrt(r2);
                    MyAtLeastDouble hinv, hinv3, hinv4, wk, dwk;

                    kernel_hinv(h, &hinv, &hinv3, &hinv4);

                    MyAtLeastDouble u = r * hinv;

                    kernel_main(u, hinv3, hinv4, &wk, &dwk, -1);

                    double mass_j = P[j].Mass;

                    rho += (mass_j * wk);
                }

                vx += P[j].Vel[0];
                vy += P[j].Vel[1];
                vz += P[j].Vel[2];

                v2 += P[j].Vel[0] * P[j].Vel[0] + P[j].Vel[1] * P[j].Vel[1] + P[j].Vel[2] * P[j].Vel[2];
            }
        }

        if (mode == 1)
        {
            listindex++;
            if (listindex < NODELISTLENGTH)
            {
                startnode = DensDataGet[target].NodeList[listindex];
                if (startnode >= 0)
                {
                    startnode = Nodes[startnode].u.d.nextnode; /* open it */
                }
            }
        }
    }

    if (mode == 0)
    {
        P[target].DM_NumNgb = numngb;
        P[target].u.DM_Density = rho;
#ifdef DENSITY_SPLIT_BY_TYPE
        if (tp >= 0) /* sum up VelDisp only for own species */
#endif // DENSITY_SPLIT_BY_TYPE
            P[target].v.DM_VelDisp = v2;
#ifdef SUBFIND_RESHUFFLE_CATALOGUE
        DM_Vx[target] = vx;
        DM_Vy[target] = vy;
        DM_Vz[target] = vz;
#endif // SUBFIND_RESHUFFLE_CATALOGUE
    }
    else
    {
        DensDataResult[target].Ngb = numngb;
        DensDataResult[target].Rho = rho;
        DensDataResult[target].VelDisp = v2;
        DensDataResult[target].Vx = vx;
        DensDataResult[target].Vy = vy;
        DensDataResult[target].Vz = vz;
    }

    return 0;
}

/**
 * @brief Primary function for evaluating density and related quantities in parallel.
 *
 * This function is responsible for evaluating density and related quantities for particles
 * in parallel. It iterates over particles marked for evaluation (according to the `todo` array)
 * and calls the `subfind_density_evaluate` function for each particle.
 *
 * If density splitting by particle type is enabled (`DENSITY_SPLIT_BY_TYPE`), the function
 * may only evaluate particles of a specific type (`j_target`), as specified by the input parameter.
 *
 *
 * @param thread_id      The thread ID of the calling thread.
 * @param todo           Pointer to the array indicating particles to be evaluated.
 * Additonal parameters if DENSITY_SPLIT_BY_TYPE:
 * @param j_in           The input type of particle being evaluated.
 * @param j_target       The target type of particle for evaluation (relevant if density splitting by type is enabled).
 *
 * @return Returns NULL upon successful completion.
 */
#ifdef DENSITY_SPLIT_BY_TYPE
void *subfind_density_evaluate_primary(int thread_id, char *todo, int j_in, int j_target)
#else
void *subfind_density_evaluate_primary(int thread_id, char *todo)
#endif
{
    int i;
    int *exportflag, *exportnodecount, *exportindex, *ngblist;
    MyFloat *dist2list;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));
    dist2list = Dist2list + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

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
        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            if (BufferFullFlag != 0 || NextParticle >= NumPart)
            {
                exitFlag = 1;
            }
            else
            {
#ifdef _OPENMP
                do
                {
                    i = NextParticle;
                    NextParticle += 1;
                    if (NextParticle >= NumPart)
                    {
                        break;
                    }
                    ProcessedFlag[i] = 1;
                }
#ifdef DENSITY_SPLIT_BY_TYPE
                while (!todo[i] || P[i].Type != j_target);
#else // DENSITY_SPLIT_BY_TYPE
                while (!todo[i] || !((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)));
#endif // DENSITY_SPLIT_BY_TYPE
                ProcessedFlag[i] = 0;
#else // _OPENMP
                i = NextParticle;
                ProcessedFlag[NextParticle] = 0;
                NextParticle += 1;
#endif // _OPENMP
            }
        }
        UNLOCK_NEXPORT;
        if (exitFlag)
        {
            break;
        }

        if (todo[i])
        {
#ifdef DENSITY_SPLIT_BY_TYPE
            if (P[i].Type == j_target)
            {
                if (subfind_density_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist, dist2list, j_in) <
                    0)
                    break; /* export buffer has filled up */
            }
#else // DENSITY_SPLIT_BY_TYPE
            if (((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))

            {
                if (subfind_density_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist, dist2list) < 0)
                    break; /* export buffer has filled up */
            }
#endif // DENSITY_SPLIT_BY_TYPE
        }
        ProcessedFlag[i] = 1; /* particle successfully finished */
    }
    return NULL;
}

/**
 * @brief Secondary function for evaluating density and related quantities in parallel.
 *
 * This function is a secondary function for evaluating density and related quantities for particles
 * in parallel. It iterates over imported particle data (from other tasks) and calls the
 * `subfind_density_evaluate` function for each imported particle.
 *
 * If density splitting by particle type is enabled (`DENSITY_SPLIT_BY_TYPE`), the function
 * may only evaluate particles of a specific type (`j_in`), as specified by the input parameter.
 *
 * @param thread_id      The thread ID of the calling thread.
 * Additional parameter if not DENSITY_SPLIT_BY_TYPE:
 * @param j_in           The input type of particle being evaluated (relevant if density splitting by type is enabled).
 *
 * @return Returns NULL upon successful completion.
 */
#ifdef DENSITY_SPLIT_BY_TYPE
void *subfind_density_evaluate_secondary(int thread_id, int j_in)
#else // DENSITY_SPLIT_BY_TYPE
void *subfind_density_evaluate_secondary(int thread_id)
#endif // DENSITY_SPLIT_BY_TYPE
{
    int j, dummy, *ngblist;
    MyFloat *dist2list;

    ngblist = Ngblist + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));
    dist2list = Dist2list + thread_id * ((int)(NumPart / DENSITY_LESS_NGB_FACTOR_SUB));

    while (1)
    {
        LOCK_NEXPORT;
#ifdef _OPENMP
#pragma omp critical(_nexport_)
#endif // _OPENMP
        {
            j = NextJ;
            NextJ++;
        }
        UNLOCK_NEXPORT;

        if (j >= Nimport)
        {
            break;
        }

#ifdef DENSITY_SPLIT_BY_TYPE
        subfind_density_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist, dist2list, j_in);
#else // DENSITY_SPLIT_BY_TYPE
        subfind_density_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist, dist2list);
#endif // DENSITY_SPLIT_BY_TYPE
    }
    return NULL;
}

/**
 * @brief Sets up the smoothing lengths for particles for density calculation.
 *
 * This function sets up the smoothing lengths (`DM_Hsml`) for particles to be used in density calculation.
 * It iterates over all particles and calculates the appropriate smoothing length based on the desired number
 * of neighbors (`DesNumNgb`) and the mass distribution in the simulation volume.
 *
 * If density splitting by particle type is enabled (`DENSITY_SPLIT_BY_TYPE`), the function may only set up
 * smoothing lengths for particles of a specific type (`j`), as specified by the input parameter.
 *
 * @param j (Optional) The particle type for which smoothing lengths should be set up (relevant if density
 *          splitting by type is enabled).
 */
#ifdef DENSITY_SPLIT_BY_TYPE
void subfind_setup_smoothinglengths(int j)
#else // DENSITY_SPLIT_BY_TYPE
void subfind_setup_smoothinglengths(void)
#endif // DENSITY_SPLIT_BY_TYPE
{
    int no, p;

    for (int i = 0; i < NumPart; i++)
    {
#ifdef DENSITY_SPLIT_BY_TYPE
        if (P[i].Type == j)
#else // DENSITY_SPLIT_BY_TYPE
        if (((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)) || ((1 << P[i].Type) & (FOF_SECONDARY_LINK_TYPES)))
#endif // DENSITY_SPLIT_BY_TYPE
        {
            no = Father[i];

            /* Not a good guess for gas/stars component, need more thought ! */
            while (10 * All.DesNumNgb * P[i].Mass > Nodes[no].u.d.mass)
            {
                p = Nodes[no].u.d.father;

                if (p < 0)
                {
                    break;
                }

                no = p;
            }

#ifdef DONOT_KEEP_HSML_AS_GUESS
            P[i].DM_Hsml =
                pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
#else // DONOT_KEEP_HSML_AS_GUESS
#ifdef DENSITY_SPLIT_BY_TYPE
            if (P[i].Type == 0)
            {
                P[i].DM_Hsml = P[i].Hsml;
            }
            else
            {
                P[i].DM_Hsml =
                    pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
            }
#else // DENSITY_SPLIT_BY_TYPE
            P[i].DM_Hsml =
                pow(3.0 / (4 * M_PI) * All.DesNumNgb * P[i].Mass / Nodes[no].u.d.mass, 1.0 / 3) * Nodes[no].len;
#endif // DENSITY_SPLIT_BY_TYPE
#endif // DONOT_KEEP_HSML_AS_GUESS
        }
    }
}

static int Nhsml;
/**
 * @brief Structure to store smoothing length (Hsml), density, velocity dispersion, and ID for particles.
 */
static struct hsml_data
{
    float Hsml;
    float Density;
    float VelDisp;
    MyIDType ID;
} *Hsml_list;

/**
 * @brief Compares two instances of `hsml_data` based on their IDs.
 *
 * This function is used as a comparison function for sorting an array of `hsml_data` structures. It compares
 * two instances based on their particle IDs (`ID` member).
 *
 * @param a Pointer to the first `hsml_data` structure.
 * @param b Pointer to the second `hsml_data` structure.
 *
 * @return An integer less than, equal to, or greater than zero if the ID of the first structure is less than,
 *         equal to, or greater than the ID of the second structure, respectively.
 */
int subfind_compare_hsml_data(const void *a, const void *b)
{
    if (((struct hsml_data *)a)->ID < ((struct hsml_data *)b)->ID)
    {
        return -1;
    }

    if (((struct hsml_data *)a)->ID > ((struct hsml_data *)b)->ID)
    {
        return +1;
    }

    return 0;
}

/**
 * @brief Saves smoothing lengths and densities of particles.
 *
 * This function saves the smoothing lengths and densities of particles to files. It gathers the necessary data from all
 * MPI ranks, sorts it based on particle IDs, and then writes it to output files in a parallelized manner.
 *
 * @param num The identifier for the current iteration or snapshot.
 */
void subfind_save_densities(int num)
{
    TIMER_PUSH(3, "subfind_save_density");
    int nprocgroup, masterTask;
    char buf[1000];
    double t0, t1;

    VERBOSE(2, "start saving smoothing lengths and densities.");
    Nhsml = 0;
    for (int i = 0; i < NumPart; i++)
#ifdef DENSITY_SPLIT_BY_TYPE
        if (((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE)))
#else // DENSITY_SPLIT_BY_TYPE
        if (((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif // DENSITY_SPLIT_BY_TYPE
            Nhsml++;

    MPI_Allgather(&Nhsml, 1, MPI_INT, Send_count, 1, MPI_INT, MYMPI_COMM_WORLD);
    Send_offset[0] = 0;
    for (int i = 1; i < NTask; i++)
        Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];

    sumup_large_ints(1, &Nhsml, &Ntotal);

    Hsml_list = (struct hsml_data *)mymalloc("Hsml_list", Nhsml * sizeof(struct hsml_data));
    Nhsml = 0;
    for (int i = 0; i < NumPart; i++)
#ifdef DENSITY_SPLIT_BY_TYPE
        if (((1 << P[i].Type) & (DENSITY_SPLIT_BY_TYPE)))
#else // DENSITY_SPLIT_BY_TYPE
        if (((1 << P[i].Type) & (FOF_PRIMARY_LINK_TYPES)))
#endif // DENSITY_SPLIT_BY_TYPE
        {
            Hsml_list[Nhsml].Hsml = P[i].DM_Hsml;
            Hsml_list[Nhsml].Density = P[i].u.DM_Density;
            Hsml_list[Nhsml].VelDisp = P[i].v.DM_VelDisp;
#ifdef SAVE_HSML_IN_IC_ORDER
            Hsml_list[Nhsml].ID = P[i].ID_ic_order;
#else // SAVE_HSML_IN_IC_ORDER
            Hsml_list[Nhsml].ID = P[i].ID;
#endif // SAVE_HSML_IN_IC_ORDER
            Nhsml++;
        }

    t0 = second();
    parallel_sort(Hsml_list, Nhsml, sizeof(struct hsml_data), subfind_compare_hsml_data);
    t1 = second();

    VERBOSE(2, "Sorting of densities in ID sequence took = %g sec", timediff(t0, t1));

    if (ThisTask == 0)
    {
#ifdef LONG_OUTPUT_NUMBERS
        sprintf(buf, "%s/hsmldir_%04d", All.OutputDir, num);
#else // LONG_OUTPUT_NUMBERS
        sprintf(buf, "%s/hsmldir_%03d", All.OutputDir, num);
#endif // LONG_OUTPUT_NUMBERS
        mkdir(buf, 02755);
    }
    MPI_Barrier(MYMPI_COMM_WORLD);

    ENDRUN_IF(NTask < All.NumFilesWrittenInParallel,
              "Fatal error.\nNumber of processors must be a smaller or equal than `NumFilesWrittenInParallel'.");

    nprocgroup = NTask / All.NumFilesWrittenInParallel;
    if ((NTask % All.NumFilesWrittenInParallel))
    {
        nprocgroup++;
    }
    masterTask = (ThisTask / nprocgroup) * nprocgroup;
    for (int groupTask = 0; groupTask < nprocgroup; groupTask++)
    {
        if (ThisTask == (masterTask + groupTask)) /* ok, it's this processor's turn */
        {
            subfind_save_local_densities(num);
        }
        MPI_Barrier(MYMPI_COMM_WORLD); /* wait inside the group */
    }

    myfree(Hsml_list);
    TIMER_POP(3, "subfind_save_density");
}

/**
 * @brief Saves local densities of particles to a file.
 *
 * This function saves the local densities of particles to a file. It writes the smoothing lengths, densities, and
 * velocity dispersions of particles to the specified output file in a binary format.
 *
 * @param num The identifier for the current iteration or snapshot.
 */
void subfind_save_local_densities(int num)
{
    char fname[1000];
    float *tmp;
    FILE *fd;

#ifdef LONG_OUTPUT_NUMBERS
    sprintf(fname, "%s/hsmldir_%04d/%s_%04d.%d", All.OutputDir, num, "hsml", num, ThisTask);
#else // LONG_OUTPUT_NUMBERS
    sprintf(fname, "%s/hsmldir_%03d/%s_%03d.%d", All.OutputDir, num, "hsml", num, ThisTask);
#endif // LONG_OUTPUT_NUMBERS
    if (!(fd = fopen(fname, "w")))
    {
        ENDRUN("can't open file `%s`", fname);
    }

    my_fwrite(&Nhsml, sizeof(int), 1, fd);
    my_fwrite(&Send_offset[ThisTask], sizeof(int), 1, fd); /* this is the number of IDs in previous files */
    my_fwrite(&Ntotal, sizeof(long long), 1, fd);
    my_fwrite(&NTask, sizeof(int), 1, fd);

    tmp = (float *)mymalloc("tmp", Nhsml * sizeof(float));

    for (int i = 0; i < Nhsml; i++)
    {
        tmp[i] = Hsml_list[i].Hsml;
    }
    my_fwrite(tmp, sizeof(float), Nhsml, fd);

    for (int i = 0; i < Nhsml; i++)
    {
        tmp[i] = Hsml_list[i].Density;
    }
    my_fwrite(tmp, sizeof(float), Nhsml, fd);

    for (int i = 0; i < Nhsml; i++)
    {
        tmp[i] = Hsml_list[i].VelDisp;
    }
    my_fwrite(tmp, sizeof(float), Nhsml, fd);

    myfree(tmp);

    fclose(fd);
}
#endif // SUBFIND
