/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Analytic potential class and its definitions.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer and David Klemmer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "../Gravity/analytic_potentials.hpp"
#include "../CodeBase/allvars.hpp"

/**
 * @brief Constructor for the ANALYTIC_POTENTIAL class.
 *
 * Initializes the ANALYTIC_POTENTIAL object with parameters for the galactic potential components.
 * These parameters are retrieved from the global configuration settings.
 */
ANALYTIC_POTENTIAL::ANALYTIC_POTENTIAL()
    : do_halo(All.AP_doHalo), do_bulge(All.AP_doBulge), do_thick_disk(All.AP_doThickDisk),
      do_thin_disk(All.AP_doThinDisk), do_disk1(All.AP_doDisk1), do_disk2(All.AP_doDisk2), do_king(All.AP_doKing),
      halo_rho_s(All.AP_HaloRho), halo_r_s(All.AP_HaloR), bulge_M_h(All.AP_BulgeM), bulge_a_h(All.AP_BulgeA),
      thick_disk_M_d(All.AP_ThickDiskM), thick_disk_a_d(All.AP_ThickDiskA), thick_disk_b_d(All.AP_ThickDiskB),
      thin_disk_M_d(All.AP_ThinDiskM), thin_disk_a_d(All.AP_ThinDiskA), thin_disk_b_d(All.AP_ThinDiskB),
      disk1_M_d(All.AP_Disk1M), disk1_a_d(All.AP_Disk1A), disk1_b_d(All.AP_Disk1B), disk2_M_d(All.AP_Disk2M),
      disk2_a_d(All.AP_Disk2A), disk2_b_d(All.AP_Disk2B), king_rho_0(All.AP_KingRho0), king_r_c(All.AP_KingRc)
#ifdef ANALYTIC_POTENTIALS_VELOCITIES
      ,
      SampleVelocity(50, All.AP_HaloRho, All.AP_HaloR)
#endif // ANALYTIC_POTENTIALS_VELOCITIES
{
}

/**
 * @brief Destructor of the ANALYTIC_POTENTIAL class.
 */
ANALYTIC_POTENTIAL::~ANALYTIC_POTENTIAL()
{
}

/**
 * @brief This method is called in begrun() and initializes all relevant things for the analytic potentials.
 */
void ANALYTIC_POTENTIAL::init()
{
#ifdef ANALYTIC_POTENTIALS_VELOCITIES
    SampleVelocity.init();
#endif // ANALYTIC_POTENTIALS_VELOCITIES
}

/**
 * @brief Apply the gravitational accelerations from different components of the galactic potential.
 *
 * This function computes the gravitational accelerations due to different components of the galactic potential,
 * including the dark matter halo, bulge, thick disk, and thin disk, if enabled.
 */
void ANALYTIC_POTENTIAL::apply_accelerations() const
{
    if (do_halo)
    {
        apply_nfw(halo_rho_s, halo_r_s);
    }
    if (do_bulge)
    {
        apply_hern(bulge_M_h, bulge_a_h);
    }
    if (do_thick_disk)
    {
        apply_disk(thick_disk_M_d, thick_disk_a_d, thick_disk_b_d);
    }
    if (do_thin_disk)
    {
        apply_disk(thin_disk_M_d, thin_disk_a_d, thin_disk_b_d);
    }
    if (do_disk1)
    {
        apply_disk(disk1_M_d, disk1_a_d, disk1_b_d);
    }
    if (do_disk2)
    {
        apply_disk(disk2_M_d, disk2_a_d, disk2_b_d);
    }
    if (do_king)
    {
        apply_king(king_rho_0, king_r_c);
    }
}

#ifdef ANALYTIC_POTENTIALS_VELOCITIES
/**
 * @brief Sampels velocity vectors from an NFW profile.
 *
 * @param Vel array to return the sampled velocity vector.
 * @param r radius at which the velocity vector is sampled.
 */
void ANALYTIC_POTENTIAL::get_vel(MyFloat (&Vel)[3], const MyAtLeastDouble r)
{
    SampleVelocity.get_vel(Vel, r);
}

/**
 * @brief Computed the density of an NFW halo.
 *
 * @param r radius at which the density is computed.
 */
MyAtLeastDouble ANALYTIC_POTENTIAL::rho(const MyAtLeastDouble r) const
{
    return SampleVelocity.rho(r);
}
#endif // ANALYTIC_POTENTIALS_VELOCITIES

/**
 * @brief Apply the gravitational potential of a Navarro-Frenk-White (NFW) profile to the particle positions.
 *
 * This function computes the gravitational potential contribution of an NFW profile to the particle positions.
 * The NFW potential is computed based on its scale density and scale radius.
 *
 * @param rho_s Scale density of the NFW profile.
 * @param r_s Scale radius of the NFW profile.
 */
void ANALYTIC_POTENTIAL::apply_nfw(const MyAtLeastDouble rho_s, const MyAtLeastDouble r_s) const
{
    ENDRUN_IF(rho_s <= 0.0 || r_s <= 0.0, "Parameter for NFW potential must be positive.");

    const MyAtLeastDouble fact0 = 4.0 * M_PI * All.GravityConstantInternal * rho_s * r_s * r_s * r_s;

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        MyLongDouble &dx = P[i].Pos[0];
        MyLongDouble &dy = P[i].Pos[1];
        MyLongDouble &dz = P[i].Pos[2];

        const MyLongDouble r = sqrt(dx * dx + dy * dy + dz * dz);
        const MyAtLeastDouble fact1 = (r / (r + r_s) - log((r + r_s) / r_s)) / (r * r * r);
        const MyAtLeastDouble fact = fact0 * fact1;

        P[i].GravAccel[0] += fact * dx;
        P[i].GravAccel[1] += fact * dy;
        P[i].GravAccel[2] += fact * dz;
    }
}

/**
 * @brief Apply the gravitational potential of a Hernquist profile to the particle positions.
 *
 * This function computes the gravitational potential contribution of a Hernquist profile to the particle positions.
 * The Hernquist potential is computed based on its mass and scale radius.
 *
 * @param M_h Mass of the Hernquist profile.
 * @param a_h Scale radius of the Hernquist profile.
 */
void ANALYTIC_POTENTIAL::apply_hern(const MyAtLeastDouble M_h, const MyAtLeastDouble a_h) const
{
    ENDRUN_IF(M_h <= 0.0 || a_h <= 0.0, "Parameter for Hernquist potential must be positive.");

    const MyAtLeastDouble fact0 = -All.GravityConstantInternal * M_h;

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        MyLongDouble &dx = P[i].Pos[0];
        MyLongDouble &dy = P[i].Pos[1];
        MyLongDouble &dz = P[i].Pos[2];

        const MyLongDouble r = sqrt(dx * dx + dy * dy + dz * dz);
        const MyAtLeastDouble fact1 = a_h + r;
        const MyAtLeastDouble fact = fact0 / (r * fact1 * fact1);

        P[i].GravAccel[0] += fact * dx;
        P[i].GravAccel[1] += fact * dy;
        P[i].GravAccel[2] += fact * dz;
    }
}

/**
 * @brief Apply the gravitational potential of a disk to the particle positions.
 *
 * This function computes the gravitational potential contribution of a disk to the particle positions.
 * The disk potential is computed based on its mass and dimensions.
 *
 * @param M_d Mass of the disk.
 * @param a_d Semi-major axis of the disk.
 * @param b_d Semi-minor axis of the disk.
 */
void ANALYTIC_POTENTIAL::apply_disk(const MyAtLeastDouble M_d, const MyAtLeastDouble a_d,
                                    const MyAtLeastDouble b_d) const
{
    ENDRUN_IF(M_d <= 0.0 || a_d <= 0.0 || b_d <= 0.0, "Parameter for disk potential must be positive.");

    const MyAtLeastDouble fact0 = -All.GravityConstantInternal * M_d;

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        MyLongDouble &dx = P[i].Pos[0];
        MyLongDouble &dy = P[i].Pos[1];
        MyLongDouble &dz = P[i].Pos[2];

        const MyLongDouble R2 = dx * dx + dy * dy;
        const MyAtLeastDouble bz = sqrt(b_d * b_d + dz * dz);
        const MyAtLeastDouble fact1 = a_d + bz;
        const MyAtLeastDouble factz = fact1 / bz;
        const MyAtLeastDouble alpha = pow(fact1 * fact1 + R2, 1.5);
        const MyAtLeastDouble fact = fact0 / alpha;

        P[i].GravAccel[0] += fact * dx;
        P[i].GravAccel[1] += fact * dy;
        P[i].GravAccel[2] += fact * factz * dz;
    }
}

/**
 * @brief Apply the gravitational potential of a King profile to the particle positions.
 *
 * This function computes the gravitational potential contribution of a King profile to the particle positions.
 * The King model is computed based on its density parameter and scale radius.
 *
 * @param rho_0 Density parameter of the King model.
 * @param r_c Scale radius of the King model.
 */
void ANALYTIC_POTENTIAL::apply_king(const MyAtLeastDouble rho_0, const MyAtLeastDouble r_c) const
{
    ENDRUN_IF(rho_0 <= 0.0 || r_c <= 0.0, "Parameter for King potential must be positive.");

    const MyAtLeastDouble r_c2 = r_c * r_c;
    const MyAtLeastDouble fact0 = All.GravityConstantInternal * 4.0 * M_PI * rho_0 * r_c * r_c2;

    for (int i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        MyLongDouble &dx = P[i].Pos[0];
        MyLongDouble &dy = P[i].Pos[1];
        MyLongDouble &dz = P[i].Pos[2];

        const MyLongDouble r2 = dx * dx + dy * dy + dz * dz;
        const MyLongDouble r = sqrt(r2);
        const MyAtLeastDouble fact1 = 1.0 / (r2 * r);
        const MyAtLeastDouble fact2 = r / sqrt(r2 + r_c2) - asinh(r / r_c);
        const MyAtLeastDouble fact = fact0 * fact1 * fact2;

        P[i].GravAccel[0] += fact * dx;
        P[i].GravAccel[1] += fact * dy;
        P[i].GravAccel[2] += fact * dz;
    }
}

#ifdef ANALYTIC_POTENTIALS_VELOCITIES
/**
 * @brief Constructor for the EddingtonInversion class.
 */
EddingtonInversion::EddingtonInversion(const int NGRIDR_in, const MyAtLeastDouble &rho, const MyAtLeastDouble &r)
    : halo_rho_s(rho), halo_r_s(r), Gridr(nullptr), logr(nullptr), logPot(nullptr), fE(nullptr), NGRIDR(NGRIDR_in),
      FACTORINNER(1e-3), FACTOROUTER(1e8), TOLLININT(1e-10)
{
    seed = 20 + ThisTask;
    generator.seed(seed);

    omp_init_lock(&lock_generator);

    Gridr = new double[NGRIDR];
    logr = new double[NGRIDR];
    logPot = new double[NGRIDR];
    fE = new double[NGRIDR];
}

/**
 * @brief Destructor of the EddingtonInversion class.
 */
EddingtonInversion::~EddingtonInversion()
{
    omp_destroy_lock(&lock_generator);

    if (Gridr != nullptr)
    {
        delete[] Gridr;
    }
    if (logr != nullptr)
    {
        delete[] logr;
    }
    if (logPot != nullptr)
    {
        delete[] logPot;
    }
    if (fE != nullptr)
    {
        delete[] fE;
    }
}

/**
 * @brief This method initializes all relevant things for the Eddington inversion.
 */
void EddingtonInversion::init()
{
    rinner = FACTORINNER * halo_r_s;

    grid();
    pot();
    fe();
}

/**
 * @brief Public method to sample the velocity vector.
 *
 * @param Vel array to return the sampled velocity vector.
 * @param r radius at which the velocity vector is sampled.
 */
void EddingtonInversion::get_vel(MyFloat (&Vel)[3], const MyAtLeastDouble r)
{
    const double Potr = Potential(r);
    const double vesc = vescape(r);
    const double fEmax = f2(r);

    double vrand = 0;
    double fErand = 0;
    double fEcheck = 1;

    while (fEcheck > fErand)
    {
        vrand = pow(rand01(), 1.0 / 3.0) * vesc;
        const double Erand = 0.5 * vrand * vrand + Potr;
        fErand = f1(Erand);
        fEcheck = rand01() * fEmax;
    }

    const double theta = acos(2.0 * rand01() - 1.0);
    const double phi = rand01() * 2.0 * M_PI;
    Vel[0] = vrand * sin(theta) * cos(phi);
    Vel[1] = vrand * sin(theta) * sin(phi);
    Vel[2] = vrand * cos(theta);
}

/**
 * @brief Method that computes the density.
 *
 * @param r radius at which the density is evaluated.
 */
MyAtLeastDouble EddingtonInversion::rho(const MyAtLeastDouble r) const
{
    return halo_rho_s / tau(r);
}

/*
 * @brief Method to draw random numbers.
 */
double EddingtonInversion::rand01()
{
    return (((double)std::rand()) / ((double)(RAND_MAX + 1.0)));
}

/**
 * @brief Method that evaluates the potential.
 *
 * @param r radius at which the potential is computed.
 */
double EddingtonInversion::Potential(const double r) const
{
    return ((-4.0 * M_PI * All.GravityConstantInternal * halo_rho_s * halo_r_s * halo_r_s * halo_r_s) / r) *
           log(1.0 + r / halo_r_s);
}

/**
 * @brief Method for trapez computation used in integration.
 */
double EddingtonInversion::trapez(const double k, const int n) const
{
    int N[n + 1];

    const double Philower = Potential(1e3 * halo_r_s + 0.001 * halo_r_s);
    const double Phiupper = Potential(k);
    const double rupper = k;
    const double xlower = asin(sqrt(Philower / Phiupper));
    const double xupper = 0.5 * M_PI;
    const double rlower = 1e3 * halo_r_s + 0.001 * halo_r_s;

    if (n == 0)
    {
        return (sqrt(-0.5 * Phiupper) / (M_PI * M_PI)) * 0.5 * (xupper - xlower) *
               (d2rhodPhi2(rlower) * sin(xlower) + d2rhodPhi2(rupper) * sin(xupper));
    }
    else
    {
        for (int i = 0; i < (n + 1); ++i)
        {
            N[i] = pow(2, i);
        }
        double sum = 0.5 * (d2rhodPhi2(rlower) * sin(xlower) + d2rhodPhi2(rupper) * sin(xupper));
        for (int i = 1; i < (n + 1); ++i)
        {
            const double deltax = (xupper - xlower) / N[i];
            double sumN = 0;
            for (int j = 1; j < N[i - 1] + 1; ++j)
            {
                const double x = xlower + (2.0 * j - 1.0) * deltax;
                const double Phi = Phiupper * sin(x) * sin(x);
                const double r = exp(lininterpolation(NGRIDR, logPot, logr, log(-Phi)));
                sumN += d2rhodPhi2(r) * sin(x);
            }
            sum = sum + sumN;
        }
        return ((sqrt(-0.5 * Phiupper) / (M_PI * M_PI)) * sum * (xupper - xlower) / N[n]);
    }
}

/**
 * @brief Method for integration.
 */
double EddingtonInversion::integral(const double j) const
{
    const double inttol = 1e-4;
    double T[3];

    for (int loopc = 0; loopc < 3; ++loopc)
    {
        T[loopc] = trapez(j, loopc);
    }
    double SA = (4.0 * T[1] - T[0]) / 3.0;
    double SB = (4.0 * T[2] - T[1]) / 3.0;
    for (int i = 3; i < (38 + 1); ++i)
    {
        T[0] = T[1];
        T[1] = T[2];
        T[2] = trapez(j, i);
        SA = SB;
        SB = (4.0 * T[2] - T[1]) / 3.0;
        if (i > 8)
        {
            if ((abs(SB - SA) < inttol * abs(SB)) || (SA == 0.0 && SB == 0.0))
            {
                return SB;
            }
        }
    }

    std::cerr << "Warning!" << std::endl;
    std::cerr << "Too many steps in function integraldf!" << std::endl;
    std::cerr << "E = " << Potential(j) << " LU^2 Gyr^-2" << std::endl;
    std::cerr << "r = " << j << " LU" << std::endl;
    std::cerr << "f(E) = " << SB << " MU Gyr^3 LU^-6" << std::endl;
    std::cerr << "f(E) not converged within tolerance of " << inttol << std::endl;
    std::cerr << "Abort tolerance was " << abs((SA - SB) / SB) << std::endl;
    if (SB != SB)
    {
        std::cerr << "f(E) value set to " << 1e20 << " MU Gyr^3 LU^-6" << std::endl;
        return 1e20;
    }
    else
    {
        return SB;
    }
}

/**
 * @brief Computes grid point and prints them.
 */
void EddingtonInversion::grid()
{
    double router = halo_r_s;
    while (rho(halo_r_s) / rho(router) < FACTOROUTER)
    {
        router = router * 10.0;
    }

    const double logrinner = log(rinner);
    const double dlogr = (log(router) - logrinner) / (NGRIDR - 1);

    std::cout << "rinner = " << rinner << std::endl;
    std::cout << "logrinner = " << logrinner << std::endl;
    std::cout << "dlogr = " << dlogr << std::endl;

    for (int i = 0; i < NGRIDR; ++i)
    {
        logr[i] = logrinner + i * dlogr;
        Gridr[i] = exp(logr[i]);
        std::cout << "Gridr = " << Gridr[i] << std::endl;
    }
}

/**
 * @brief Computes potential at grid points.
 */
void EddingtonInversion::pot()
{
    for (int i = 0; i < NGRIDR; ++i)
    {
        logPot[i] = log(-Potential(Gridr[i]));
    }
}

/**
 * @brief Computes distribution function for grid points.
 */
void EddingtonInversion::fe()
{
    for (int i = 0; i < NGRIDR; ++i)
    {
        fE[i] = E_distribution(Gridr[i]);
    }
}

/**
 * @brief Computes distribution function.
 */
double EddingtonInversion::E_distribution(const double r)
{
    const double fe = integral(r);

    return fe;
}

/**
 * @brief Method for normalised density profile.
 *
 * @params r radius at which the profile is evaluated.
 */
double EddingtonInversion::tau(const double r) const
{
    double alpha = 1, beta = 3, gamma = 1;

    const double &exp1 = gamma;
    const double &exp2 = alpha;
    const double &exp3 = (beta - gamma) / alpha;

    const double fac1 = pow((r / halo_r_s), exp1);
    const double fac2 = 1.0 + pow((r / halo_r_s), exp2);
    const double fac3 = pow(fac2, exp3);

    return fac1 * fac3;
}

/**
 * @brief Computes eta.
 */
double EddingtonInversion::eta(const double r) const
{
    double alpha = 1, beta = 3, gamma = 1;
    const double rDr_s = r / halo_r_s;
    const double fac1 = (beta - gamma) / halo_r_s;
    const double fac2 = pow(r / halo_r_s, alpha - 1.0);
    const double fac3 = 1.0 + fac2 * rDr_s;

    return ((gamma / r) + (fac1 * fac2 / fac3));
}

/**
 * @brief Computes derivative of eta.
 */
double EddingtonInversion::detadr(const double r) const
{
    double alpha = 1, beta = 3, gamma = 1;

    const double fac1 = (beta - gamma) / (halo_r_s * halo_r_s);
    const double fac2 = pow((r / halo_r_s), (alpha - 2.0));
    const double fac3 = (alpha - 1.0) * (1.0 + pow((r / halo_r_s), alpha)) - alpha * pow((r / halo_r_s), alpha);
    const double fac4 = pow((1.0 + pow((r / halo_r_s), alpha)), 2);

    return (-gamma / (r * r) + fac1 * fac2 * fac3 / fac4);
}

/**
 * @brief Computes derivative of density.
 */
double EddingtonInversion::drhodr(const double r) const
{
    return -rho(r) * eta(r);
}

/**
 * @brief Computes second-order derviative of density.
 */
double EddingtonInversion::d2rhodr2(const double r) const
{
    const double etar = eta(r);
    return rho(r) * (etar * etar - detadr(r));
}

/**
 * @brief Computes second-order derivative of density with respect to the potential.
 */
double EddingtonInversion::d2rhodPhi2(const double r) const
{
    const double DMencr = 1.0 / NFW_M(r);
    const double rhor = rho(r);
    const double r2 = r * r;

    const double fac1 = r2 * DMencr * DMencr / (All.GravityConstantInternal * All.GravityConstantInternal);
    const double fac2 = 2.0 * r - 4.0 * M_PI * r2 * r2 * rhor * DMencr;

    return fac1 * (r2 * d2rhodr2(r) + fac2 * drhodr(r));
}

/**
 * @brief Computes enclosed mass.
 */
double EddingtonInversion::NFW_M(const double r) const
{
    const double rpr_s = r + halo_r_s;
    return 4.0 * M_PI * halo_rho_s * halo_r_s * halo_r_s * halo_r_s * (log(rpr_s / halo_r_s) - r / rpr_s);
}

/**
 * @brief Computes escape velocity.
 */
double EddingtonInversion::vescape(const double r) const
{
    return sqrt(2.0 * abs(Potential(r) - Potential(1e5 * halo_r_s)));
}

/**
 * @brief Finds grid point.
 */
int EddingtonInversion::locate(int n, const double grid[], const double x) const
{
    int jl = -1;
    int ju = n;
    int ascend = (grid[n - 1] >= grid[0]);
    while ((ju - jl) > 1)
    {
        int jm = (ju + jl) / 2;
        if ((x >= grid[jm]) == ascend)
        {
            jl = jm;
        }
        else
        {
            ju = jm;
        }
    }
    if (x == grid[0])
    {
        return 0;
    }
    else if (x == grid[n - 1])
    {
        return (n - 2);
    }
    else
    {
        return jl;
    }
}

/**
 * @brief Computes linear interpolation.
 */
double EddingtonInversion::lininterpolation(int n, const double gridx[], const double gridy[], const double x) const
{
    int i = locate(n, gridx, x);
    if (i < 0)
    {
        std::cerr << "Warning!" << std::endl;
        std::cerr << " x = " << x << " was below range of array! Index i = " << i << " / n = " << n << "." << std::endl;
        std::cerr << "Array between " << gridx[0] << " and " << gridx[n - 1] << "." << std::endl;
        double RE = abs((x - gridx[0]) / gridx[0]);
        if (RE < TOLLININT)
        {
            std::cerr << "Relative error (= " << RE << ") at lower boundary was within tolerance of " << TOLLININT
                      << std::endl;
            std::cerr << "Index set to i = 0" << std::endl;
            i = 0;
        }
    }
    else if (i > n - 1)
    {
        double RE = abs((x - gridx[n - 1]) / gridx[n - 1]);
        if (RE > TOLLININT)
        {
            std::cerr << "ERROR!" << std::endl;
            std::cerr << "x = " << x << " was above range of array! Index i = " << i << " / n = " << n << "."
                      << std::endl;
            std::cerr << "Array between " << gridx[0] << " and " << gridx[n - 1] << "." << std::endl;
            std::cerr << "Relative error (= " << RE << ") at upper boundary was greater than the tolerance value of "
                      << TOLLININT << std::endl;
        }
        else
            i = n - 1;
    }

    const double dgy = gridy[i + 1] - gridy[i];
    const double dgx = gridx[i + 1] - gridx[i];
    const double m = dgy / dgx;
    const double dx = x - gridx[i];
    const double y = gridy[i] + dx * m;

    return y;
}

/**
 * @brief Interpolates distribution function at specific energy.
 */
double EddingtonInversion::f1(const double E) const
{
    return (lininterpolation(NGRIDR, logPot, fE, log(-E)));
}

/**
 * @brief Interpolates distribution function at specific radius.
 */
double EddingtonInversion::f2(const double r) const
{
    return (lininterpolation(NGRIDR, Gridr, fE, r));
}
#endif // ANALYTIC_POTENTIALS_VELOCITIES
