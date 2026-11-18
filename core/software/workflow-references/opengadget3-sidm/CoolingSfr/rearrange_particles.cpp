/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file contains a function ot rearrange particles.
 *
 * @author OpenGadget3-development team, in particular Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include "../CodeBase/allvars.hpp"
#include "../CodeBase/proto.hpp"

#if defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)
/* Rearange the sequence of particles specifically for cases where stars have been converted from gas particles. */
void rearrange_particle_sequence(void)
{
    int flag = 0;

#if defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM) || defined(PB_INITIALIZE_BH_PART)
    if (Particles_converted)
    {
        N_gas -= Particles_converted;
        Particles_converted = 0;

        for (int i = 0; i < N_gas; i++)
        {
            if (P[i].Type != 0)
            {
                int j;
                for (j = N_gas; j < NumPart; j++)
                {
                    if (P[j].Type == 0)
                    {
                        break;
                    }
                }

                ENDRUN_IF(j >= NumPart, "Particle that should be converted has index beyond total particle number. "
                                        "This should not happen.");

#ifdef SFR
                if (P[i].Type == 4)
                {
                    MetP[P[i].pt.MetID].PID = j;
                }
#endif // SFR

#ifdef BLACK_HOLES
                if (P[i].Type == 5)
                {
                    BHP[P[i].pt.BHID].PID = j;
                }
#endif // BLACK_HOLES

                struct particle_data psave = P[i];
                P[i] = P[j];
                SphP[i] = SphP[j];
                P[j] = psave;
            }
        }
        flag = 1;
    }
#endif // defined(SFR) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)

#ifdef BLACK_HOLES
    int count_elim = 0;
    int count_gaselim = 0;

    for (int i = 0; i < NumPart; i++)
    {
        if (P[i].Mass == 0)
        {
            TimeBinCount[P[i].TimeBin]--;

            if (TimeBinActive[P[i].TimeBin])
            {
                NumForceUpdate--;
            }

            if (P[i].Type == 0)
            {
                TimeBinCountSph[P[i].TimeBin]--;

                P[i] = P[N_gas - 1];
                SphP[i] = SphP[N_gas - 1];

                P[N_gas - 1] = P[NumPart - 1];
#ifdef SFR
                if (P[N_gas - 1].Type == 4)
                    MetP[P[N_gas - 1].pt.MetID].PID = N_gas - 1;
#endif // SFR
                if (P[N_gas - 1].Type == 5)
                {
                    BHP[P[N_gas - 1].pt.BHID].PID = N_gas - 1;
                }
                N_gas--;

                count_gaselim++;
            }
#if defined(GM_MUPPI) || defined(SFR)
            else if (P[i].Type == 4)
            {
                int j = P[i].pt.MetID;
                MetP[j] = MetP[N_stars - 1];
                P[MetP[j].PID].pt.MetID = j;
                N_stars--;

                P[i] = P[NumPart - 1];

                if (P[i].Type == 4)
                    MetP[P[i].pt.MetID].PID = i;

                if (P[i].Type == 5)
                    BHP[P[i].pt.BHID].PID = i;
            }
#endif // defined(GM_MUPPI) || defined(SFR)
            else if (P[i].Type == 5)
            {
                int j = P[i].pt.BHID;
                BHP[j] = BHP[N_BHs - 1];
                P[BHP[j].PID].pt.BHID = j;
                N_BHs--;

                P[i] = P[NumPart - 1];

#if defined(GM_MUPPI) || defined(SFR)
                if (P[i].Type == 4)
                    MetP[P[i].pt.MetID].PID = i;
#endif // defined(GM_MUPPI) || defined(SFR)

                if (P[i].Type == 5)
                {
                    BHP[P[i].pt.BHID].PID = i;
                }
            }
            else
            {
                P[i] = P[NumPart - 1];

#if defined(GM_MUPPI) || defined(SFR)
                if (P[i].Type == 4)
                    MetP[P[i].pt.MetID].PID = i;

                if (P[i].Type == 5)
                    BHP[P[i].pt.BHID].PID = i;
#endif // defined(GM_MUPPI) || defined(SFR)
            }

            NumPart--;
            i--;

            count_elim++;
        }
    }

    int tot_elim, tot_gaselim;
    MPI_Allreduce(&count_elim, &tot_elim, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);
    MPI_Allreduce(&count_gaselim, &tot_gaselim, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    if (count_elim)
    {
        flag = 1;
    }

    VERBOSE(2, "Blackholes: Eliminated %d gas particles and merged away %d black holes.", tot_gaselim,
            tot_elim - tot_gaselim);

    All.TotNumPart -= tot_elim;
    All.TotN_gas -= tot_gaselim;
    All.TotBHs -= tot_elim - tot_gaselim;
#endif // BLACK_HOLES

#if defined(BLACK_HOLES) && (defined(GM_MUPPI) || defined(LT_STELLAREVOLUTION))
#if defined(DO_SPLIT_OVERMASSIVE_GAS)
    int NumGasSplitted = 0;
    MyAtLeastDouble massGasPart = All.massDMpart * All.OmegaBaryon / (All.Omega0 - All.OmegaBaryon);
    for (int i = 0; i < N_gas; i++)
    {
        assert(P[i].Type == 0);
        assert(P[i].Mass > 0);

        if (P[i].Mass > massGasPart * 2)
        {
            assert(P[N_gas + NumGasSplitted].Type > 0);

            P[NumPart + NumGasSplitted] = P[N_gas + NumGasSplitted];

            if (P[NumPart + NumGasSplitted].Type == 4)
                MetP[P[NumPart + NumGasSplitted].pt.MetID].PID = NumPart + NumGasSplitted;

            if (P[NumPart + NumGasSplitted].Type == 5)
                BHP[P[NumPart + NumGasSplitted].pt.BHID].PID = NumPart + NumGasSplitted;

            P[i].Mass /= 2;
            SphP[i].MassRes /= 2;
            SphP[i].EgyRes /= 2;
            for (int k = 0; k < LT_NMetP; k++)
                SphP[i].Metals[k] /= 2;

            P[N_gas + NumGasSplitted] = P[i];
            SphP[N_gas + NumGasSplitted] = SphP[i];
            P[N_gas + NumGasSplitted].ID += All.TotNumPart;
            SPHP(N_gas + NumGasSplitted, wakeup) = TIMEBINS;
            NumGasSplitted++;
        }
    }
    NumPart += NumGasSplitted;
    N_gas += NumGasSplitted;

    int tot_split;
    MPI_Allreduce(&NumGasSplitted, &tot_split, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    VERBOSE(2, "Split Gas in re-arrange: Splitted %d gas particles ...", tot_split);

    All.TotNumPart += tot_split;
    All.TotN_gas += tot_split;

    if (NumGasSplitted > 0)
    {
        flag = 1;
    }

#endif // DO_SPLIT_OVERMASSIVE_GAS
#endif // BLACK_HOLES) && (GM_MUPPI || LT_STELLAREVOLUTION)

    int flag_sum;
    MPI_Allreduce(&flag, &flag_sum, 1, MPI_INT, MPI_SUM, MYMPI_COMM_WORLD);

    if (flag_sum)
    {
        reconstruct_timebins();
    }
}
#endif // defined(SFR) || defined(BLACK_HOLES) || defined(FG_MAKE_OUTSIDE_HIGHRES_DM) || defined(FG_MAKE_MAXVEL_DM)
