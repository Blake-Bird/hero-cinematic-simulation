/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Routines for periodic PM-force computation.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "pm.hpp"

#ifdef PMGRID
#ifdef PERIODIC

#define BINS_PS 2000 /* number of bins for power spectrum computation */
#define POWERSPEC_FOLDFAC 32

static char power_spec_fname[500];
static long long power_spec_totnumpart;

#ifdef LATER_LATER
static long long CountModes[2][BINS_PS];
static double SumPower[2][BINS_PS];
static double SumPowerUncorrected[2][BINS_PS]; /* without binning correction (as for shot noise) */
static double Power[2][BINS_PS];
static double PowerUncorrected[2][BINS_PS]; /* without binning correction */
static double Delta[2][BINS_PS];
static double DeltaUncorrected[2][BINS_PS]; /* without binning correction */
static double ShotLimit[2][BINS_PS];
static double Kbin[BINS_PS];
static double K0, K1;
static double binfac;
static double power_spec_totmass;
#endif // LATER_LATER

/**
 * @brief Calculate the power spectra of particles.
 *
 * This function calculates the power spectra of particles for all particle types
 * and optionally for each individual particle type separately. It calls the `pmforce`
 * function to compute the power spectra.
 *
 * @param num The index of the output.
 * @param ntot_type_all An array containing the total number of particles for each type.
 */
void calculate_power_spectra(int num, long long *ntot_type_all)
{
    TIMER_PUSH(1, "POWERSPEC");
    int typeflag[6];

    power_spec_totnumpart = 0;

    for (int i = 0; i < 6; i++)
    {
        typeflag[i] = 1;
        power_spec_totnumpart += ntot_type_all[i];
    }

    sprintf(power_spec_fname, "%s/powerspec_%03d.txt", All.OutputDir, num);

    pmforce(2); //, 1, typeflag); /* calculate power spectrum for all particle types */

#ifdef POWERSPEC_ON_OUTPUT_EACH_TYPE
    if (ntot_type_all)
    {
        for (int i = 0; i < 6; i++)
        {
            if (ntot_type_all[i] > 0)
            {
                for (int j = 0; j < 6; j++)
                {
                    typeflag[j] = 0;
                }

                typeflag[i] = 1;
                power_spec_totnumpart = ntot_type_all[i];
                sprintf(power_spec_fname, "%s/powerspec_type%d_%03d.txt", All.OutputDir, i, num);
                pmforce(2); //, 1, typeflag);       /* calculate power spectrum for type i */
            }
        }
    }
#endif // POWERSPEC_ON_OUTPUT_EACH_TYPE
    TIMER_POP(1, "POWERSPEC");
}

#ifdef LATER_LATER
/**
 * @brief Calculate the power spectrum.
 *
 * This function computes the power spectrum of particles in Fourier space.
 *
 * @param flag Flag indicating the purpose of the calculation.
 * @param typeflag Array indicating which particle types to include in the calculation.
 */
void powerspec(int flag, int *typeflag)
{
    double *powerbuf;
    long long *countbuf;

    VERBOSE(2, "begin power spectrum. (step=%d)  POWERSPEC_FOLDFAC=%d ", flag, POWERSPEC_FOLDFAC);

    TIMER_PUSH(5, "Power_spectrum");
    // Compute the total mass of particles
    double mass = 0;
    for (int i = 0; i < NumPart; i++)
    {
        if (typeflag[P[i].Type])
        {
            mass += P[i].Mass;
        }
    }

    MPI_Allreduce(&mass, &power_spec_totmass, 1, MPI_DOUBLE, MPI_SUM, MYMPI_COMM_WORLD);

    // Compute the normalization factor
    double fac = 1.0 / power_spec_totmass;

    K0 = 2 * M_PI / All.BoxSize; /* minimum k */
    K1 = K0 * All.BoxSize / All.SofteningTable[1]; /* maximum k */
    binfac = BINS_PS / (log(K1) - log(K0));

    if (flag == 0)
    {
        // Initialize arrays for the power spectrum
        for (int rep = 0; rep < 2; rep++)
        {
            for (int i = 0; i < BINS_PS; i++)
            {
                SumPower[rep][i] = 0;
                SumPowerUncorrected[rep][i] = 0;
                CountModes[rep][i] = 0;
            }
        }
    }
    // Compute power spectrum for each Fourier mode
    for (int y = slabstart_y; y < slabstart_y + nslab_y; y++)
    {
        for (int x = 0; x < PMGRID; x++)
        {
            for (int z = 0; z < PMGRID; z++)
            {
                int kx, ky, kz;
                int zz = z;
                if (z >= PMGRID / 2 + 1)
                {
                    zz = PMGRID - z;
                }

                if (x > PMGRID / 2)
                {
                    kx = x - PMGRID;
                }
                else
                    kx = x;
                if (y > PMGRID / 2)
                {
                    ky = y - PMGRID;
                }
                else
                {
                    ky = y;
                }
                if (z > PMGRID / 2)
                {
                    kz = z - PMGRID;
                }
                else
                {
                    kz = z;
                }

                double k2 = kx * kx + ky * ky + kz * kz;

                if (k2 > 0)
                {
                    if (k2 < (PMGRID / 2.0) * (PMGRID / 2.0))
                    {
                        /* do deconvolution */

                        double fx = 1;
                        double fy = 1;
                        double fz = 1;
                        if (kx != 0)
                        {
                            fx = (M_PI * kx) / PMGRID;
                            fx = sin(fx) / fx;
                        }
                        if (ky != 0)
                        {
                            fy = (M_PI * ky) / PMGRID;
                            fy = sin(fy) / fy;
                        }
                        if (kz != 0)
                        {
                            fz = (M_PI * kz) / PMGRID;
                            fz = sin(fz) / fz;
                        }
                        double ff = 1 / (fx * fy * fz);
                        double smth = ff * ff * ff * ff;

                        /* end deconvolution */

                        int ip = PMGRID * (PMGRID / 2 + 1) * (y - slabstart_y) + (PMGRID / 2 + 1) * x + zz;
                        double po = (fft_of_rhogrid[ip][0] * fft_of_rhogrid[ip][0] +
                                     fft_of_rhogrid[ip][1] * fft_of_rhogrid[ip][1]);
                        po *= fac * fac * smth;
                        double k = sqrt(k2) * 2 * M_PI / All.BoxSize;

                        if (flag == 0)
                        {
                            k *= POWERSPEC_FOLDFAC;
                        }

                        if (k >= K0 && k < K1)
                        {
                            int bin = log(k / K0) * binfac;

                            double ponorm = po / PowerSpec_Efstathiou(k);

                            SumPower[flag][bin] += ponorm;
                            SumPowerUncorrected[flag][bin] += po;
                            CountModes[flag][bin] += 1;

#ifdef RAYLEIGH_BINS
                            if (flag == 1) /* only for course grid */
                            {
                                ratio = sqrt(ponorm * RayleighFactor[bin]);

                                if (ratio > RayleighMax[bin])
                                {
                                    RayleighMax[bin] = ratio;
                                }

                                if (ratio >= 10.0)
                                {
                                    RayleighCountAbove[bin]++;
                                }
                                else
                                {
                                    i = RAYLEIGH_BINS * ratio / 10.0;
                                    RayleighCountModes[bin][i]++;
                                }
                            }
#endif // RAYLEIGH_BINS
                        }
                    }
                }
            }
        }
    }

    // Gather power spectrum data from all tasks
    countbuf = (long long int *)mymalloc("countbuf", NTask * BINS_PS * sizeof(long long));
    powerbuf = (double *)mymalloc("powerbuf", NTask * BINS_PS * sizeof(double));

    MPI_Allgather(CountModes[flag], BINS_PS * sizeof(long long), MPI_BYTE, countbuf, BINS_PS * sizeof(long long),
                  MPI_BYTE, MYMPI_COMM_WORLD);

    for (int i = 0; i < BINS_PS; i++)
    {
        CountModes[flag][i] = 0;
        for (int n = 0; n < NTask; n++)
        {
            CountModes[flag][i] += countbuf[n * BINS_PS + i];
        }
    }

    MPI_Allgather(SumPower[flag], BINS_PS * sizeof(double), MPI_BYTE, powerbuf, BINS_PS * sizeof(double), MPI_BYTE,
                  MYMPI_COMM_WORLD);

    for (int i = 0; i < BINS_PS; i++)
    {
        SumPower[flag][i] = 0;
        for (int n = 0; n < NTask; n++)
        {
            SumPower[flag][i] += powerbuf[n * BINS_PS + i];
        }
    }

    MPI_Allgather(SumPowerUncorrected[flag], BINS_PS * sizeof(double), MPI_BYTE, powerbuf, BINS_PS * sizeof(double),
                  MPI_BYTE, MYMPI_COMM_WORLD);

    for (int i = 0; i < BINS_PS; i++)
    {
        SumPowerUncorrected[flag][i] = 0;
        for (int n = 0; n < NTask; n++)
        {
            SumPowerUncorrected[flag][i] += powerbuf[n * BINS_PS + i];
        }
    }

    myfree(powerbuf);
    myfree(countbuf);

    for (int i = 0; i < BINS_PS; i++)
    {
        Kbin[i] = exp((i + 0.5) / binfac + log(K0));

        if (CountModes[flag][i] > 0)
        {
            Power[flag][i] = PowerSpec_Efstathiou(Kbin[i]) * SumPower[flag][i] / CountModes[flag][i];
            PowerUncorrected[flag][i] = SumPowerUncorrected[flag][i] / CountModes[flag][i];
        }
        else
        {
            Power[flag][i] = 0;
            PowerUncorrected[flag][i] = 0;
        }

        Delta[flag][i] = 4 * M_PI * pow(Kbin[i], 3) / pow(2 * M_PI / All.BoxSize, 3) * Power[flag][i];

        DeltaUncorrected[flag][i] =
            4 * M_PI * pow(Kbin[i], 3) / pow(2 * M_PI / All.BoxSize, 3) * PowerUncorrected[flag][i];

        ShotLimit[flag][i] =
            4 * M_PI * pow(Kbin[i], 3) / pow(2 * M_PI / All.BoxSize, 3) * (1.0 / power_spec_totnumpart);
    }
    if (flag == 1)
    {
        {
            powerspec_save();
#ifdef RAYLEIGH_BINS
            rayleigh_save();
#endif // RAYLEIGH_BINS
        }
    }
    TIMER_POP(5, "Power_spectrum");
}

/**
 * @brief Compute the Efstathiou power spectrum.
 *
 * This function calculates the power spectrum using the Efstathiou model.
 *
 * @param k Wave number.
 * @return Efstathiou power spectrum value at the given wave number.
 */
double PowerSpec_Efstathiou(double k)
{
    double ShapeGamma = 0.21;
    double AA = 6.4 / ShapeGamma * (3.085678e24 / All.UnitLength_in_cm);
    double BB = 3.0 / ShapeGamma * (3.085678e24 / All.UnitLength_in_cm);
    double CC = 1.7 / ShapeGamma * (3.085678e24 / All.UnitLength_in_cm);
    double nu = 1.13;

    return k / pow(1 + pow(AA * k + pow(BB * k, 1.5) + CC * CC * k * k, nu), 2 / nu);
}

/**
 * @brief Save the power spectrum data to file.
 *
 * This function saves the calculated power spectrum data to the powerspec save file.
 */
void powerspec_save(void)
{
    FILE *fd;
    char buf[500];

    if (ThisTask == 0)
    {
        if (!(fd = fopen(power_spec_fname, "w")))
        {
            sprintf(buf, "can't open file `%s`\n", power_spec_fname);
            ENDRUN_MAIN(buf);
        }

        for (int flag = 0; flag < 2; flag++)
        {
            fprintf(fd, "%g\n", All.Time);
            // i = BINS_PS;
            fprintf(fd, "%d\n", BINS_PS);
            fprintf(fd, "%g\n", power_spec_totmass);
            fprintf(fd, "%d%09d\n", (int)(power_spec_totnumpart / 1000000000),
                    (int)(power_spec_totnumpart % 1000000000));

            for (int i = 0; i < BINS_PS; i++)
            {
                fprintf(fd, "%g %g %g %g %g %g %g %g %g %g\n", Kbin[i], Delta[flag][i], ShotLimit[flag][i],
                        Power[flag][i], (double)CountModes[flag][i], DeltaUncorrected[flag][i],
                        PowerUncorrected[flag][i], PowerSpec_Efstathiou(Kbin[i]), SumPower[flag][i],
                        4 * M_PI * pow(Kbin[i], 3) / pow(2 * M_PI / All.BoxSize, 3));
            }
        }
        fclose(fd);
    }
}

/**
 * @brief Folds particle positions onto the computational grid for power spectrum estimation.
 *
 * This function takes the positions and masses of particles and distributes them onto a density grid.
 * The density grid is then Fourier transformed to prepare for estimating the power spectrum.
 *
 * @param typeflag Pointer to an array indicating which particle types to include in the calculation.
 *                 If a particle type is flagged (typeflag[i] = 1), its contribution will be included.
 *                 Otherwise, it will be excluded.
 */
void foldonitself(int *typeflag)
{
    int n, rest, iter = 0;
    int slab_x, slab_xx, slab_y, slab_yy, slab_z, slab_zz;
    int count, buf_capacity;
    double to_slab_fac_folded, dx, dy, dz;
    MyLongDouble pp[3];
    MyFloat *pos_sendbuf, *pos_recvbuf, *pos;
    MPI_Status status;

    VERBOSE(2, "begin folding for power spectrum estimation... ");

    TIMER_PUSH(3, 'folding_density_assembly');

    /* allocate density field */
    rhogrid = (fftw_real *)mymalloc_movable(&rhogrid, "rhogrid", maxfftsize * sizeof(fftw_real));
    forcegrid = (fftw_real *)mymalloc("forcegrid", maxfftsize * sizeof(fftw_real));

    fft_of_rhogrid = (FFTW(complex) *)&rhogrid[0];

    buf_capacity = (maxfftsize * sizeof(fftw_real)) / (4 * sizeof(MyFloat));
    buf_capacity /= 2;

    pos_sendbuf = (MyFloat *)forcegrid;
    pos_recvbuf = pos_sendbuf + 4 * buf_capacity;

    to_slab_fac_folded = to_slab_fac * POWERSPEC_FOLDFAC;

    memset(rhogrid, 0, fftsize * sizeof(fftw_real));

    int istart = 0;
    do
    {
        TIMER_PUSH(4, "Fill_buffer");

        for (int i = 0; i < NTask; i++)
        {
            Send_count[i] = 0;
        }
        int nbuf = 0;
        for (int i = istart; i < NumPart; i++)
        {
            if (typeflag[P[i].Type] == 0)
            {
                continue;
            }

            if (nbuf + 1 >= buf_capacity)
            {
                break;
            }

            /* make sure that particles are properly box-wrapped */
            pp[0] = P[i].Pos[0];

            while (pp[0] < 0)
            {
                pp[0] += All.BoxSize;
            }

            while (pp[0] >= All.BoxSize)
            {
                pp[0] -= All.BoxSize;
            }

            slab_x = (int)(to_slab_fac_folded * pp[0]);
            slab_xx = slab_x + 1;
            slab_x %= PMGRID;
            slab_xx %= PMGRID;

            Send_count[slab_to_task[slab_x]]++;
            nbuf++;

            if (slab_to_task[slab_x] != slab_to_task[slab_xx])
            {
                Send_count[slab_to_task[slab_xx]]++;
                nbuf++;
            }
        }
        Send_offset[0] = 0;
        for (int i = 1; i < NTask; i++)
        {
            Send_offset[i] = Send_offset[i - 1] + Send_count[i - 1];
        }

        for (int i = 0; i < NTask; i++)
        {
            Send_count[i] = 0;
        }

        nbuf = 0;
        for (int i = istart; i < NumPart; i++)
        {
            if (typeflag[P[i].Type] == 0)
            {
                continue;
            }

            if (nbuf + 1 >= buf_capacity)
            {
                break;
            }

            /* make sure that particles are properly box-wrapped */
            pp[0] = P[i].Pos[0];

            while (pp[0] < 0)
            {
                pp[0] += All.BoxSize;
            }

            while (pp[0] >= All.BoxSize)
            {
                pp[0] -= All.BoxSize;
            }

            slab_x = (int)(to_slab_fac_folded * pp[0]);
            slab_xx = slab_x + 1;
            slab_x %= PMGRID;
            slab_xx %= PMGRID;

            for (int j = 0; j < 3; j++)
            {
                pos_sendbuf[4 * (Send_offset[slab_to_task[slab_x]] + Send_count[slab_to_task[slab_x]]) + j] =
                    P[i].Pos[j];
            }

            pos_sendbuf[4 * (Send_offset[slab_to_task[slab_x]] + Send_count[slab_to_task[slab_x]]) + 3] = P[i].Mass;

            Send_count[slab_to_task[slab_x]]++;
            nbuf++;

            if (slab_to_task[slab_x] != slab_to_task[slab_xx])
            {
                for (int j = 0; j < 3; j++)
                {
                    pos_sendbuf[4 * (Send_offset[slab_to_task[slab_xx]] + Send_count[slab_to_task[slab_xx]]) + j] =
                        P[i].Pos[j];
                }

                pos_sendbuf[4 * (Send_offset[slab_to_task[slab_xx]] + Send_count[slab_to_task[slab_xx]]) + 3] =
                    P[i].Mass;

                Send_count[slab_to_task[slab_xx]]++;
                nbuf++;
            }
        }

        istart = i;

        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MYMPI_COMM_WORLD);

        TIMER_POPPUSH(4, "Fill_buffer", "particles_exchange_and_binning");

        for (int level = 0; level < (1 << PTask); level++) /* note: for level=0, target is the same task */
        {
            int sendTask = ThisTask;
            int recvTask = ThisTask ^ level;

            if (recvTask < NTask)
            {
                if (recvTask != sendTask)
                {
                    MPI_Sendrecv(&pos_sendbuf[4 * Send_offset[recvTask]], 4 * Send_count[recvTask] * sizeof(MyFloat),
                                 MPI_BYTE, recvTask, TAG_PM_FOLD, &pos_recvbuf[0],
                                 4 * Recv_count[recvTask] * sizeof(MyFloat), MPI_BYTE, recvTask, TAG_PM_FOLD,
                                 MYMPI_COMM_WORLD, &status);

                    pos = &pos_recvbuf[0];
                    count = Recv_count[recvTask];
                }
                else
                {
                    pos = &pos_sendbuf[4 * Send_offset[ThisTask]];
                    count = Send_count[ThisTask];
                }

                for (n = 0; n < count; n++, pos += 4)
                {
                    /* make sure that particles are properly box-wrapped */
                    for (int j = 0; j < 3; j++)
                    {
                        pp[j] = pos[j];

                        while (pp[j] < 0)
                        {
                            pp[j] += All.BoxSize;
                        }

                        while (pp[j] >= All.BoxSize)
                        {
                            pp[j] -= All.BoxSize;
                        }
                    }

                    slab_x = (int)(to_slab_fac_folded * pp[0]);
                    dx = to_slab_fac_folded * pp[0] - slab_x;
                    slab_xx = slab_x + 1;
                    slab_x %= PMGRID;
                    slab_xx %= PMGRID;

                    slab_y = (int)(to_slab_fac_folded * pp[1]);
                    dy = to_slab_fac_folded * pp[1] - slab_y;
                    slab_yy = slab_y + 1;
                    slab_y %= PMGRID;
                    slab_yy %= PMGRID;

                    slab_z = (int)(to_slab_fac_folded * pp[2]);
                    dz = to_slab_fac_folded * pp[2] - slab_z;
                    slab_zz = slab_z + 1;
                    slab_z %= PMGRID;
                    slab_zz %= PMGRID;

                    float mass = pos[3];

                    if (slab_to_task[slab_x] == ThisTask)
                    {
                        slab_x -= first_slab_of_task[ThisTask];

                        rhogrid[(slab_x * PMGRID + slab_y) * PMGRID2 + slab_z] +=
                            mass * (1.0 - dx) * (1.0 - dy) * (1.0 - dz);
                        rhogrid[(slab_x * PMGRID + slab_yy) * PMGRID2 + slab_z] += mass * (1.0 - dx) * dy * (1.0 - dz);
                        rhogrid[(slab_x * PMGRID + slab_y) * PMGRID2 + slab_zz] += mass * (1.0 - dx) * (1.0 - dy) * dz;
                        rhogrid[(slab_x * PMGRID + slab_yy) * PMGRID2 + slab_zz] += mass * (1.0 - dx) * dy * dz;
                    }

                    if (slab_to_task[slab_xx] == ThisTask)
                    {
                        slab_xx -= first_slab_of_task[ThisTask];

                        rhogrid[(slab_xx * PMGRID + slab_y) * PMGRID2 + slab_z] +=
                            mass * (dx) * (1.0 - dy) * (1.0 - dz);
                        rhogrid[(slab_xx * PMGRID + slab_yy) * PMGRID2 + slab_z] += mass * (dx)*dy * (1.0 - dz);
                        rhogrid[(slab_xx * PMGRID + slab_y) * PMGRID2 + slab_zz] += mass * (dx) * (1.0 - dy) * dz;
                        rhogrid[(slab_xx * PMGRID + slab_yy) * PMGRID2 + slab_zz] += mass * (dx)*dy * dz;
                    }
                }
            }
        }

        count = NumPart - istart; /* local remaining particles */
        MPI_Allreduce(&count, &rest, 1, MPI_INT, MPI_MAX, MYMPI_COMM_WORLD);
        iter++;

        TIMER_POP(4, "particles_exchange_and_binning");
    } while (rest > 0);

    TIMER_POPPUSH(3, 'folding_density_assembly', 'FFT_for_folded_density');

    /* Do the FFT of the self-folded density field */
#ifdef _OPENMP
    FFTW(plan_with_nthreads)(omp_get_num_threads());
#endif // _OPENMP
    fft_forward_plan = FFTW(mpi_plan_dft_r2c_3d)(PMGRID, PMGRID, PMGRID, rhogrid, fft_of_rhogrid, MYMPI_COMM_WORLD,
                                                 FFTW_ESTIMATE | FFTW_DESTROY_INPUT | FFTW_MPI_TRANSPOSED_OUT);
    FFTW(execute)(fft_forward_plan);
    FFTW(destroy_plan)(fft_forward_plan);

    TIMER_POP(3, 'FFT_for_folded_density');

    powerspec(0, typeflag);

    myfree(forcegrid);
    myfree_movable(rhogrid);
}
#endif // LATER_LATER

#endif // PERIODIC
#endif // PMGRID
