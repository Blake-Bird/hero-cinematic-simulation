/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Kernel functions for SPH calculations.
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag, Moritz S. Fischer and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef KERNEL_HPP
#define KERNEL_HPP

#if GADGET_HYDRO == HYDRO_MFM
#include "../Mfm/Vector.hpp"
#include "../Mfm/FluidVector.hpp"
using namespace OpenGadget3;
#endif // GADGET_HYDRO == HYDRO_MFM

/* fall back to standard kernel */
#if !defined(QUINTIC_KERNEL) && !defined(WENDLAND_C2_KERNEL) && !defined(WENDLAND_C4_KERNEL) &&                        \
    !defined(WENDLAND_C6_KERNEL) && !defined(WENDLAND_C8_KERNEL)
#define STANDARD_KERNEL
#endif // !defined(QUINTIC_KERNEL) && !defined(WENDLAND_C2_KERNEL) && !defined(WENDLAND_C4_KERNEL) &&
       // !defined(WENDLAND_C6_KERNEL) && !defined(WENDLAND_C8_KERNEL)

/* fall back to three dimensions */
#if !defined(TWODIMS) && !defined(ONEDIM)
#define THREEDIMS
#endif // !defined(TWODIMS) && !defined(ONEDIM)

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the inverse smoothing length and its powers.
 *
 * This function calculates the inverse smoothing length and its powers (3rd and 4th powers) based on the input
 * smoothing length (h).
 *
 * @param h Smoothing length.
 * @param[out] hinv Pointer to store the computed inverse smoothing length.
 * @param[out] hinv3 Pointer to store the computed inverse cube smoothing length.
 * @param[out] hinv4 Pointer to store the computed inverse fourth power of the smoothing length.
 */
static inline void kernel_hinv(MyFloat h, MyAtLeastDouble *hinv, MyAtLeastDouble *hinv3, MyAtLeastDouble *hinv4)
{
    *hinv = 1.0 / h;

#ifdef THREEDIMS
    *hinv3 = *hinv * *hinv * *hinv;
#endif // THREEDIMS

#ifdef TWODIMS
#ifdef PERIODIC
    *hinv3 = *hinv * *hinv / boxSize_Z;
#else // PERIODIC
    *hinv3 = *hinv * *hinv;
#endif // PERIODIC
#endif // TWODIMS

#ifdef ONEDIM
#ifdef PERIODIC
    *hinv3 = *hinv / boxSize_Y / boxSize_Z;
#else // PERIODIC
    *hinv3 = *hinv;
#endif // PERIODIC
#endif // ONEDIM

    *hinv4 = *hinv3 * *hinv;

    return;
}

/**
 * @brief Struct for storing data related to the Hydra kernel computation.
 *
 * This struct encapsulates various properties and parameters used in the computation of the Hydra kernel.
 */
struct kernel_hydra
{
    MyAtLeastDouble dx, dy, dz;
    MyAtLeastDouble r2, r, vsig, sound_i, sound_j;
    MyAtLeastDouble dvx, dvy, dvz, vdotr;
    MyAtLeastDouble wk_i, wk_j, dwk_i, dwk_j;
    MyAtLeastDouble h_i, h_j, dwk_ij, rho_ij_inv;
    /* hinv_i/hinv3_i/hinv4_i/inv_density_i only depend on target particle, which is fixed for the whole
     * neighbour loop. precomputed once in set_additional_variables_and_kernel_i()
     * and reused for every neighbour j instead of being recomputed per pair. */
    MyAtLeastDouble hinv_i, hinv3_i, hinv4_i;
    MyAtLeastDouble inv_density_i;
#ifdef MAGNETIC
    MyAtLeastDouble mj_r;
    MyAtLeastDouble mf_i, mf_j;
    MyAtLeastDouble alfven2_i, alfven2_j;
    MyAtLeastDouble b2_i, b2_j;
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION)
    MyAtLeastDouble mf_dissInd;
#endif // if defined(ARTIFICIAL_MAGNETIC_DISSIPATION)  || defined(MAGNETIC_DIFFUSION)
#if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    MyAtLeastDouble mf_dissEnt;
#endif // if defined(ARTIFICIAL_MAGNETIC_DISSIPATION) || defined(MAGNETIC_DIFFUSION_HEAT)
    MyAtLeastDouble mf_Ind;
    MyAtLeastDouble dBx, dBy, dBz;
#endif // MAGNETIC
#ifdef GT_DUST
    MyAtLeastDouble dvx_mixture, dvy_mixture, dvz_mixture, vsig_dust;
    std::array<MyAtLeastDouble, GT_NUM_DUST> DeltaVeldotr2, vsig_DeltaVel;
#endif // GT_DUST
#ifdef DISK_SLICE
    MyAtLeastDouble dv_phi_disk;
#endif // DISK_SLICE
#if GADGET_HYDRO == HYDRO_MFM
    OpenGadget3::Vector<NUMDIMS, MyAtLeastDouble> dr;
#endif // GADGET_HYDRO == HYDRO_MFM
};

// define useful aliases for MFM
#if GADGET_HYDRO == HYDRO_MFM
#define kernel_flux kernel_hydra
#endif // GADGET_HYDRO == HYDRO_MFM

//////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OLD_KERNEL

enum kernel_mode
{
    calc_dwk_and_wk = 0,
    calc_wk = -1,
    calc_dwk = +1
};

template <typename CompDWK, typename CompWK>
inline static void kernel_select_mode(CompDWK comp_dwk, CompWK comp_wk, kernel_mode mode);

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the Top Hat kernel and its derivative.
 *
 * This function calculates the Top Hat kernel and its derivative based on the distance parameter (u)
 * and inverse cube smoothing length (hinv3).
 *
 * @param u Distance parameter within the range [0, 1].
 * @param hinv3 Inverse cube smoothing length.
 * @param[out] wk Pointer to store the computed kernel function value.
 * @param[out] dwk Pointer to store the computed derivative of the kernel function.
 * @param mode Computation mode specifying whether to calculate the kernel function, derivative, or both.
 */
inline static void kernel_top_hat(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble *wk, MyAtLeastDouble *dwk,
                                  kernel_mode mode)
{
#ifdef THREEDIMS
    constexpr MyAtLeastDouble norm = 3.0 / (4.0 * M_PI);
#elif defined(TWODIMS)
    constexpr MyAtLeastDouble norm = 0.5 / M_PI;
#elif defined(ONEDIM)
    constexpr MyAtLeastDouble norm = 0.5;
#endif // THREEDIMS

    auto comp_wk = [&wk, &u, &hinv3]() -> void { *wk = (u <= 1) * norm * hinv3; };

    auto comp_dwk = [&dwk]() -> void { *dwk = 0.0; };

    kernel_select_mode(comp_dwk, comp_wk, mode);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the Cubic Spline kernel and its derivative.
 *
 * This function calculates the Cubic Spline kernel and its derivative based on the distance parameter (u),
 * inverse cube smoothing length (hinv3), and inverse quartic smoothing length (hinv4).
 *
 * @param u Distance parameter within the range [0, 1].
 * @param hinv3 Inverse cube smoothing length.
 * @param hinv4 Inverse quartic smoothing length.
 * @param[out] wk Pointer to store the computed kernel function value.
 * @param[out] dwk Pointer to store the computed derivative of the kernel function.
 * @param mode Computation mode specifying whether to calculate the kernel function, derivative, or both.
 */
inline static void kernel_cubic_spline(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4,
                                       MyAtLeastDouble *wk, MyAtLeastDouble *dwk, kernel_mode mode)
{
#ifdef THREEDIMS
    constexpr MyAtLeastDouble norm = 8.0 / M_PI;
#elif defined(TWODIMS)
    constexpr MyAtLeastDouble norm = 40.0 / (7.0 * M_PI);
#elif defined(ONEDIM)
    constexpr MyAtLeastDouble norm = 4.0 / 3.0;
#endif // THREEDIMS

    if (u < 0.5)
    {
        auto comp_wk_inner = [&wk, &u, &hinv3]() -> void { *wk = (1.0 + 6.0 * (u - 1.0) * u * u) * norm * hinv3; };
        auto comp_dwk_inner = [&dwk, &u, &hinv4]() -> void { *dwk = (u * (18.0 * u - 12.0)) * norm * hinv4; };
        kernel_select_mode(comp_dwk_inner, comp_wk_inner, mode);
    }
    else
    {
        MyAtLeastDouble t1 = 1.0 - u;
        MyAtLeastDouble t2 = t1 * t1;
        auto comp_wk_outer = [&wk, &t1, &t2, &hinv3]() -> void { *wk = (2.0 * t1 * t2) * norm * hinv3; };
        auto comp_dwk_outer = [&dwk, &t2, &hinv4]() -> void { *dwk = (-6.0 * t2) * norm * hinv4; };
        kernel_select_mode(comp_dwk_outer, comp_wk_outer, mode);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the Quintic kernel and its derivative.
 *
 * This function calculates the Quintic kernel and its derivative based on the distance parameter (u),
 * inverse cube smoothing length (hinv3), and inverse quartic smoothing length (hinv4).
 *
 * @param u Distance parameter within the range [0, 1].
 * @param hinv3 Inverse cube smoothing length.
 * @param hinv4 Inverse quartic smoothing length.
 * @param[out] wk Pointer to store the computed kernel function value.
 * @param[out] dwk Pointer to store the computed derivative of the kernel function.
 * @param mode Computation mode specifying whether to calculate the kernel function, derivative, or both.
 */
inline static void kernel_quintic(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4, MyAtLeastDouble *wk,
                                  MyAtLeastDouble *dwk, kernel_mode mode)
{
    MyAtLeastDouble d1, d4, b1, b4;
    MyAtLeastDouble t1 = 1.0 - u;
    MyAtLeastDouble t4 = t1 * t1;
    t4 *= t4;

#ifdef THREEDIMS
    constexpr MyAtLeastDouble norm = 2187.0 / (40.0 * M_PI);
#elif defined(TWODIMS)
    constexpr MyAtLeastDouble norm = 15309.0 / (478.0 * M_PI);
#elif defined(ONEDIM)
    constexpr MyAtLeastDouble norm = 243.0 / 40.0;
#endif // THREEDIMS

    auto comp_wk_inner = [&wk, &t4, &t1, &d4, &d1, &b4, &b1, &hinv3]() -> void {
        *wk = (t4 * t1 - 6.0 * d4 * d1 + 15.0 * b4 * b1) * norm * hinv3;
    };
    auto comp_wk_middle = [&wk, &t4, &t1, &d4, &d1, &hinv3]() -> void {
        *wk = (t4 * t1 - 6.0 * d4 * d1) * norm * hinv3;
    };
    auto comp_wk_outer = [&wk, &t4, &t1, &hinv3]() -> void { *wk = (t4 * t1) * norm * hinv3; };
    auto comp_dwk_inner = [&dwk, &t4, &d4, &b4, &hinv4]() -> void {
        *dwk = (-5.0 * t4 + 30.0 * d4 - 75.0 * b4) * norm * hinv4;
    };
    auto comp_dwk_middle = [&dwk, &t4, &d4, &hinv4]() -> void { *dwk = (-5.0 * t4 + 30.0 * d4) * norm * hinv4; };
    auto comp_dwk_outer = [&dwk, &t4, &hinv4]() -> void { *dwk = (-5.0 * t4) * norm * hinv4; };

    if (u >= 2.0 / 3.0)
    {
        kernel_select_mode(comp_dwk_outer, comp_wk_outer, mode);
    }
    else if (u >= 1.0 / 3.0)
    {
        d1 = t1 - 1.0 / 3.0;
        d4 = d1 * d1;
        d4 *= d4;
        kernel_select_mode(comp_dwk_middle, comp_wk_middle, mode);
    }
    else
    {
        d1 = t1 - 1.0 / 3.0, b1 = d1 - 1.0 / 3.0;
        d4 = d1 * d1, b4 = b1 * b1;
        d4 *= d4, b4 *= b4;
        kernel_select_mode(comp_dwk_inner, comp_wk_inner, mode);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the Wendland C2 kernel and its derivative.
 *
 * This function calculates the Wendland C2 kernel and its derivative based on the distance parameter (u),
 * inverse cube smoothing length (hinv3), and inverse quartic smoothing length (hinv4).
 * See Dehnen & Aly 2012 for more details.
 *
 * @param u Distance parameter within the range [0, 1].
 * @param hinv3 Inverse cube smoothing length.
 * @param hinv4 Inverse quartic smoothing length.
 * @param[out] wk Pointer to store the computed kernel function value.
 * @param[out] dwk Pointer to store the computed derivative of the kernel function.
 * @param mode Computation mode specifying whether to calculate the kernel function, derivative, or both.
 */
inline static void kernel_wendland_c2(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4,
                                      MyAtLeastDouble *wk, MyAtLeastDouble *dwk, kernel_mode mode)
{
    MyAtLeastDouble t1, tp;

#ifdef THREEDIMS
    constexpr MyAtLeastDouble norm = 21.0 / (2.0 * M_PI);
#elif defined(TWODIMS)
    constexpr MyAtLeastDouble norm = 7.0 / M_PI;
#elif defined(ONEDIM)
    constexpr MyAtLeastDouble norm = 5.0 / 4.0;
#endif // THREEDIMS

#if defined(THREEDIMS) || defined(TWODIMS)
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void { *wk = (t1 * tp * (1.0 + u * 4.0)) * norm * hinv3; };
    auto comp_dwk = [&dwk, &u, &tp, &hinv4]() -> void { *dwk = (-20.0 * u * tp) * norm * hinv4; };

    t1 = 1.0 - u, tp = t1 * t1 * t1;

#else // defined(THREEDIMS) || defined(TWODIMS)
    // ONEDIM
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void { *wk = (t1 * tp * (1.0 + u * 3.0)) * norm * hinv3; };
    auto comp_dwk = [&dwk, &u, &hinv4]() -> void { *dwk = (-12.0 * u * (1.0 - u * u)) * norm * hinv4; };

    t1 = 1.0 - u, tp = t1 * t1;
#endif // defined(THREEDIMS) || defined(TWODIMS)
    kernel_select_mode(comp_dwk, comp_wk, mode);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the Wendland C4 kernel and its derivative.
 *
 * This function calculates the Wendland C4 kernel and its derivative based on the distance parameter (u),
 * inverse cube smoothing length (hinv3), and inverse quartic smoothing length (hinv4).
 * See Dehnen & Aly 2012 for more details.
 *
 * @param u Distance parameter within the range [0, 1].
 * @param hinv3 Inverse cube smoothing length.
 * @param hinv4 Inverse quartic smoothing length.
 * @param[out] wk Pointer to store the computed kernel function value.
 * @param[out] dwk Pointer to store the computed derivative of the kernel function.
 * @param mode Computation mode specifying whether to calculate the kernel function, derivative, or both.
 */
inline static void kernel_wendland_c4(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4,
                                      MyAtLeastDouble *wk, MyAtLeastDouble *dwk, kernel_mode mode)
{
    MyAtLeastDouble t1, tp;

#ifdef THREEDIMS
    constexpr MyAtLeastDouble norm = 495.0 / (32.0 * M_PI);
#elif defined(TWODIMS)
    constexpr MyAtLeastDouble norm = 9.0 / M_PI;
#elif defined(ONEDIM)
    constexpr MyAtLeastDouble norm = 3.0 / 2.0;
#endif // THREEDIMS

#if defined(THREEDIMS) || defined(TWODIMS)
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void {
        *wk = (t1 * tp * (1.0 + u * (6.0 + 35.0 / 3.0 * u))) * norm * hinv3;
    };
    auto comp_dwk = [&dwk, &u, &tp, &hinv4]() -> void {
        *dwk = (u * (-280.0 / 3.0 * tp * u - 56.0 / 3.0 * tp)) * norm * hinv4;
    };

    t1 = 1.0 - u, tp = t1 * t1;
    tp *= tp;
    tp *= t1;

#else // defined(THREEDIMS) || defined(TWODIMS)
    // ONEDIM
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void {
        *wk = (t1 * tp * (1.0 + u * (5.0 + 8.0 * u))) * norm * hinv3;
    };
    auto comp_dwk = [&dwk, &u, &tp, &hinv4]() -> void { *dwk = (u * (-14.0 * tp - 56.0 * u * tp)) * norm * hinv4; };

    t1 = 1.0 - u, tp = t1 * t1;
    tp *= tp;
#endif // defined(THREEDIMS) || defined(TWODIMS)
    kernel_select_mode(comp_dwk, comp_wk, mode);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the Wendland C6 kernel and its derivative.
 *
 * This function calculates the Wendland C6 kernel and its derivative based on the distance parameter (u),
 * inverse cube smoothing length (hinv3), and inverse quartic smoothing length (hinv4).
 * See Dehnen & Aly 2012 for more details.
 *
 * @param u Distance parameter within the range [0, 1].
 * @param hinv3 Inverse cube smoothing length.
 * @param hinv4 Inverse quartic smoothing length.
 * @param[out] wk Pointer to store the computed kernel function value.
 * @param[out] dwk Pointer to store the computed derivative of the kernel function.
 * @param mode Computation mode specifying whether to calculate the kernel function, derivative, or both.
 */
inline static void kernel_wendland_c6(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4,
                                      MyAtLeastDouble *wk, MyAtLeastDouble *dwk, kernel_mode mode)
{
    MyAtLeastDouble t1, tp;

#ifdef THREEDIMS
    constexpr MyAtLeastDouble norm = 1365.0 / (64.0 * M_PI);
#elif defined(TWODIMS)
    constexpr MyAtLeastDouble norm = 78.0 / (7.0 * M_PI);
#elif defined(ONEDIM)
    constexpr MyAtLeastDouble norm = 55.0 / 32.0;
#endif // THREEDIMS

#if defined(THREEDIMS) || defined(TWODIMS)
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void {
        *wk = (t1 * tp * (1.0 + u * (8.0 + u * (25.0 + 32.0 * u)))) * norm * hinv3;
    };
    auto comp_dwk = [&dwk, &tp, &u, &hinv4]() -> void {
        *dwk = (-22.0 * tp * u * (1.0 + u * (7.0 + 16.0 * u))) * norm * hinv4;
    };

    t1 = 1.0 - u, tp = t1 * t1;
    tp *= tp * tp;
    tp *= t1;

#else // defined(THREEDIMS) || defined(TWODIMS)
    // ONEDIM
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void {
        *wk = (t1 * tp * (1.0 + u * (7.0 + u * (19.0 + u * 21.0)))) * norm * hinv3;
    };
    auto comp_dwk = [&dwk, &tp, &u, &hinv4]() -> void {
        *dwk = (-6.0 * tp * u * (3.0 + u * (18.0 + u * 35.0))) * norm * hinv4;
    };

    t1 = 1.0 - u, tp = t1 * t1;
    tp *= tp * tp;
#endif // defined(THREEDIMS) || defined(TWODIMS)
    kernel_select_mode(comp_dwk, comp_wk, mode);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute the Wendland C8 kernel and its derivative.
 *
 * This function calculates the Wendland C8 kernel and its derivative based on the distance parameter (u),
 * inverse cube smoothing length (hinv3), and inverse quartic smoothing length (hinv4).
 * See Kummer et al. 2019 for details.
 *
 * @param u Distance parameter within the range [0, 1].
 * @param hinv3 Inverse cube smoothing length.
 * @param hinv4 Inverse quartic smoothing length.
 * @param[out] wk Pointer to store the computed kernel function value.
 * @param[out] dwk Pointer to store the computed derivative of the kernel function.
 * @param mode Computation mode specifying whether to calculate the kernel function, derivative, or both.
 */
inline static void kernel_wendland_c8(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4,
                                      MyAtLeastDouble *wk, MyAtLeastDouble *dwk, kernel_mode mode)
{
    MyAtLeastDouble t1, tp;

#ifdef THREEDIMS
    constexpr MyAtLeastDouble norm = 357.0 / (64.0 * M_PI);
#elif defined(TWODIMS)
    constexpr MyAtLeastDouble norm = 8.0 / (3.0 * M_PI);
#elif defined(ONEDIM)
    constexpr MyAtLeastDouble norm = -1.0; // this still has to be calculated!
#ifdef WENDLAND_C8_KERNEL
#error "ONEDIM not implemented for WENDLAND_C8_KERNEL"
#endif
#endif // THREEDIMS

#if defined(THREEDIMS) || defined(TWODIMS)
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void {
        *wk = (t1 * tp * (5.0 + u * (50.0 + u * (210.0 + u * (450.0 + u * 429.0))))) * norm * hinv3;
    };
    auto comp_dwk = [&dwk, &tp, &u, &hinv4]() -> void {
        *dwk = (-26.0 * tp * u * (5.0 + u * (45.0 + u * (159.0 + u * 231.0)))) * norm * hinv4;
    };

    t1 = 1.0 - u, tp = t1 * t1;
    tp *= tp;
    tp *= tp;
    tp *= t1;

#else // defined(THREEDIMS) || defined(TWODIMS)
    // ONEDIM
    auto comp_wk = [&wk, &t1, &tp, &u, &hinv3]() -> void {
        *wk = (t1 * tp * (5.0 + u * (50.0 + u * (210.0 + u * (450.0 + u * 429.0))))) * norm *
              hinv3; // this should still be properly calculated!
    };
    auto comp_dwk = [&dwk, &tp, &u, &hinv4]() -> void {
        *dwk = (-26.0 * tp * u * (5.0 + u * (45.0 + u * (159.0 + u * 231.0)))) * norm *
               hinv4; // this should still be properly calculated!
    };

    t1 = 1.0 - u, tp = t1 * t1;
    tp *= tp;
    tp *= tp;
    tp *= t1;
#endif // defined(THREEDIMS) || defined(TWODIMS)
    kernel_select_mode(comp_dwk, comp_wk, mode);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Select and execute computation functions based on the specified kernel mode.
 *
 * This function takes two callable objects representing functions for computing the derivative of the kernel function
 * (comp_dwk) and the kernel function itself (comp_wk), and executes them based on the specified kernel_mode.
 *
 * @tparam CompDWK Type of the callable object for computing the derivative of the kernel function.
 * @tparam CompWK Type of the callable object for computing the kernel function itself.
 * @param comp_dwk Callable object for computing the derivative of the kernel function.
 * @param comp_wk Callable object for computing the kernel function itself.
 * @param mode The mode indicating the type of computation:
 *             - calc_dwk: Compute only the derivative of the kernel function.
 *             - calc_wk: Compute only the kernel function.
 *             - calc_dwk_and_wk: Compute both the derivative and the kernel function.
 */
template <typename CompDWK, typename CompWK>
inline static void kernel_select_mode(CompDWK comp_dwk, CompWK comp_wk, kernel_mode mode)
{
    if (mode == calc_wk)
    {
        comp_wk();
    }
    else if (mode == calc_dwk_and_wk)
    {
        comp_wk();
        comp_dwk();
    }
    else
    {
        comp_dwk();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Main function for kernel calculation.
 *
 * This function calculates the kernel function and its derivative based on the specified kernel mode.
 *
 * @param u The normalized distance.
 * @param hinv3 The inverse of the smoothing length cubed.
 * @param hinv4 The inverse of the smoothing length raised to the power of 4.
 * @param wk Pointer to store the kernel function value.
 * @param dwk Pointer to store the derivative of the kernel function value.
 * @param mode The mode indicating the type of calculation:
 *
 * @note Attention: Here we assume that kernel is only called
 * with range 0..1 for u as done in hydra or density !!
 */
inline static void kernel_main(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4, MyAtLeastDouble *wk,
                               MyAtLeastDouble *dwk, kernel_mode mode)
{
#ifdef STANDARD_KERNEL
    kernel_cubic_spline(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(QUINTIC_KERNEL)
    kernel_quintic(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(WENDLAND_C2_KERNEL)
    kernel_wendland_c2(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(WENDLAND_C4_KERNEL)
    kernel_wendland_c4(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(WENDLAND_C6_KERNEL)
    kernel_wendland_c6(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(WENDLAND_C8_KERNEL)
    kernel_wendland_c8(u, hinv3, hinv4, wk, dwk, mode);
#else // STANDARD_KERNEL
#error Type for main kernel unkown
#endif // STANDARD_KERNEL
}

/**
 * @brief Converts the mode integer to a form suitable for kernel calculations.
 *
 * This function converts the mode integer to a form suitable for kernel calculations. It maps the input mode to the
 * appropriate value for the kernel function computation.
 *
 * @param mode Flag indicating the calculation mode:
 *             - 0: Calculate both the kernel function and its derivative.
 *             - -1: Calculate only the kernel function.
 *             - +1: Calculate only the derivative of the kernel function.
 *
 * @return The converted mode value suitable for kernel calculations.
 */
inline static void kernel_main(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4, MyAtLeastDouble *wk,
                               MyAtLeastDouble *dwk, int mode)
{
    kernel_main(u, hinv3, hinv4, wk, dwk, kernel_mode(mode));
}

#endif // !OLD_KERNEL

//////////////////////////////////////////////////////////////////////////////////////////////////

/* Norms */
#ifdef STANDARD_KERNEL
#ifdef THREEDIMS
#define NORM 8.0 / M_PI /*!< For 3D-normalized kernel */
#endif // THREEDIMS

#ifdef TWODIMS
#define NORM 40.0 / (7.0 * M_PI) /*!< For 2D-normalized kernel */
#endif // TWODIMS

#ifdef ONEDIM
#define NORM 4.0 / 3.0 /*!< For 1D-normalized kernel */
#endif // ONEDIM
#endif // STANDARD_KERNEL

#ifdef QUINTIC_KERNEL
#ifdef THREEDIMS
#define NORM 2187.0 / (40.0 * M_PI) /*!< For 3D-normalized kernel */
#endif // THREEDIMS

#ifdef TWODIMS
#define NORM 15309.0 / (478.0 * M_PI) /*!< For 2D-normalized kernel */
#endif // TWODIMS

#ifdef ONEDIM
#define NORM 243.0 / 40.0 /*!< For 1D-normalized kernel */
#endif // ONEDIM
#endif // QUINTIC_KERNEL

#ifdef WENDLAND_C2_KERNEL
#ifdef THREEDIMS
#define NORM 21.0 / (2 * M_PI) /*!< For 3D-normalized kernel */
#endif // THREEDIMS

#ifdef TWODIMS
#define NORM 7.0 / M_PI /*!< For 2D-normalized kernel */
#endif // TWODIMS

#ifdef ONEDIM
#define NORM 5.0 / 4.0 /*!< For 1D-normalized kernel */
#endif // ONEDIM
#endif // WENDLAND_C2_KERNEL

#ifdef WENDLAND_C4_KERNEL
#ifdef THREEDIMS
#define NORM 495.0 / (32 * M_PI) /*!< For 3D-normalized kernel */
#endif // THREEDIMS

#ifdef TWODIMS
#define NORM 9.0 / M_PI /*!< For 2D-normalized kernel */
#endif // TWODIMS

#ifdef ONEDIM
#define NORM 3.0 / 2.0 /*!< For 1D-normalized kernel */
#endif // ONEDIM
#endif // WENDLAND_C4_KERNEL

#ifdef WENDLAND_C6_KERNEL
#ifdef THREEDIMS
#define NORM 1365.0 / (64 * M_PI) /*!< For 3D-normalized kernel */
#endif // THREEDIMS

#ifdef TWODIMS
#define NORM 78.0 / (7 * M_PI) /*!< For 2D-normalized kernel */
#endif

#ifdef ONEDIM
#define NORM 55.0 / 32.0 /*!< For 1D-normalized kernel */
#endif // ONEDIM
#endif // WENDLAND_C6_KERNEL

#ifdef WENDLAND_C8_KERNEL
#ifdef THREEDIMS
#define NORM 357.0 / (64 * M_PI) /*!< For 3D-normalized kernel */
#endif // THREEDIMS

#ifdef TWODIMS
#define NORM 8.0 / (3 * M_PI) /*!< For 2D-normalized kernel */
#endif // TWODIMS

#ifdef ONEDIM
#error "ONEDIM not implemented for WENDLAND_C8_KERNEL"
#endif // ONEDIM

#endif // WENDLAND_C8_KERNEL

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief This struct contains the kernel properties for the density computation.
 */
struct kernel_density
{
    MyAtLeastDouble dx, dy, dz;
    MyAtLeastDouble r, r2;
    MyAtLeastDouble dvx, dvy, dvz;
    MyAtLeastDouble wk, dwk;
    MyAtLeastDouble hinv, hinv3, hinv4;
    MyAtLeastDouble mj_wk, mj_dwk_r;
#if GADGET_HYDRO == HYDRO_MFM
    WFluidVector<mfm_numdims> Wj;
#endif // GADGET_HYDRO == HYDRO_MFM
#ifdef DISK_SLICE
    MyAtLeastDouble dv_phi_disk;
#endif // DISK_SLICE
#ifdef GT_DUST
    MyAtLeastDouble dvx_mixture, dvy_mixture, dvz_mixture;
#endif // GT_DUST
#ifdef TIME_DEP_ART_COND
    /* local->EntropyPred * pow(local->Density, GAMMA_MINUS1) (or the GT_DUST equivalent), only depends on the
     * target particle, which is fixed for the whole neighbour loop. Precomputed once in gradients_evaluate()
     * and reused for every neighbour j instead of being recomputed per pair. */
    MyAtLeastDouble local_entropy_density_term;
#endif // TIME_DEP_ART_COND
#ifdef OUTPUT_MAGNETIC_CURVATURE
    /* only depends on local->BPred, which is fixed for the whole neighbour loop. Precomputed once in
     * gradients_evaluate() and reused for every neighbour j instead of being recomputed per pair. */
    MyAtLeastDouble Bnorm_local_inv2;
#endif // OUTPUT_MAGNETIC_CURVATURE
};

#ifdef OLD_KERNEL
/**
 * @brief Computes the kernel function and its derivative.
 *
 * This function calculates the kernel function and its derivative for a given interpolation parameter `u` and kernel
 * properties `hinv3` and `hinv4`. The computation is performed according to the selected kernel type, such as cubic
 * spline, quintic, Wendland C2, C4, C6, or C8 kernels. The result is stored in the provided variables `wk` for the
 * kernel function and `dwk` for its derivative.
 *
 * @param u Interpolation parameter within the range [0, 1] representing the distance relative to the kernel width.
 * @param hinv3 Inverse of the smoothing length raised to the power of 3.
 * @param hinv4 Inverse of the smoothing length raised to the power of 4.
 * @param wk Pointer to a variable where the computed kernel function value will be stored.
 * @param dwk Pointer to a variable where the computed derivative of the kernel function will be stored.
 * @param mode Flag indicating the calculation mode:
 *             - 0: Calculate both the kernel function and its derivative.
 *             - -1: Calculate only the kernel function.
 *             - +1: Calculate only the derivative of the kernel function.
 *
 * @note with range 0..1 for u as done in hydra or density !!
 * Call with mode 0 to calculate dwk and wk
 * Call with mode -1 to calculate only wk
 * Call with mode +1 to calculate only dwk
 */
inline static void kernel_main(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4, MyAtLeastDouble *wk,
                               MyAtLeastDouble *dwk, int mode)
{
#ifdef STANDARD_KERNEL /* cubic spline */
    if (u < 0.5)
    {
        if (mode >= 0)
#ifdef ACC
        // for some reason PGI 19 crashes otherwise
        {
            MyAtLeastDouble v;
            v = (u * (18.0 * u - 12.0));
            *dwk = v;
        }
#else // ACC
            *dwk = u * (18.0 * u - 12.0);
#endif // ACC

        if (mode <= 0)
#ifdef ACC
        // for some reason PGI 19 crashes otherwise
        {
            MyAtLeastDouble v;
            v = (1.0 + 6.0 * (u - 1.0) * u * u);
            *wk = v;
        }
#else // ACC
        {
            *wk = (1.0 + 6.0 * (u - 1.0) * u * u);
        }
#endif // ACC
    }
    else
    {
        MyAtLeastDouble t1 = (1.0 - u);
        MyAtLeastDouble t2 = t1 * t1;
        if (mode >= 0)
#ifdef ACC
        {
            MyAtLeastDouble v;
            v = -6.0 * t2;
            *dwk = v;
        }
#else // ACC
        {
            *dwk = -6.0 * t2;
        }
#endif // ACC
        if (mode <= 0)
#ifdef ACC
        {
            MyAtLeastDouble v;
            v = 2.0 * t2 * t1;

            *wk = v;
        }
#else // ACC
        {
            *wk = 2.0 * t2 * t1;
        }
#endif // ACC
    }
#endif // STANDARD_KERNEL

#ifdef QUINTIC_KERNEL
    MyAtLeastDouble t1 = (1.0 - u);
    MyAtLeastDouble t2 = t1 * t1;
    MyAtLeastDouble t4 = t2 * t2;

    if (mode >= 0)
    {
        *dwk = -5.0 * t4;
    }
    if (mode <= 0)
    {
        *wk = t4 * t1;
    }

    if (u < 2.0 / 3.0)
    {
        t1 = (2.0 / 3.0 - u);
        t2 = t1 * t1;
        t4 = t2 * t2;
        if (mode >= 0)
        {
            *dwk += 30.0 * t4;
        }
        if (mode <= 0)
        {
            *wk -= 6.0 * t4 * t1;
        }
    }
    if (u < 1.0 / 3.0)
    {
        t1 = (1.0 / 3.0 - u);
        t2 = t1 * t1;
        t4 = t2 * t2;
        if (mode >= 0)
        {
            *dwk -= 75.0 * t4;
        }
        if (mode <= 0)
        {
            *wk += 15.0 * t4 * t1;
        }
    }
#endif // QUINTIC_KERNEL

#ifdef WENDLAND_C2_KERNEL /* Dehnen & Aly 2012 */
#if defined(THREEDIMS) || defined(TWODIMS)
    MyAtLeastDouble t1 = (1 - u), t3 = t1 * t1 * t1, u2 = u * u;

    if (mode >= 0)
    {
        *dwk = -20.0 * u * t3;
    }
    if (mode <= 0)
    {
        *wk = t1 * t3 * (1.0 + 4.0 * u);
    }
#else /* ONEDIM */
    MyAtLeastDouble t1 = (1 - u), t2 = t1 * t1;

    if (mode >= 0)
    {
        *dwk = -12.0 * u * (1.0 - u2);
    }
    if (mode <= 0)
    {
        *wk = t1 * t2 * (1.0 + 3.0 * u);
    }
#endif /* ONEDIM */

#endif /* WENDLAND_C2_KERNEL */

#ifdef WENDLAND_C4_KERNEL /* Dehnen & Aly 2012 */
#if defined(THREEDIMS) || defined(TWODIMS)
    MyAtLeastDouble t1 = (1 - u);
    MyAtLeastDouble t1sq = t1 * t1, u2 = u * u;
    MyAtLeastDouble t5 = t1sq * t1sq * t1; /* t1^5 via squaring, cheaper than 5 sequential multiplies */

    if (mode >= 0)
    {
        *dwk = -280.0 / 3.0 * t5 * u2 - 56.0 / 3.0 * u * t5;
    }
    if (mode <= 0)
    {
        *wk = t1 * t5 * (1.0 + 6.0 * u + 35.0 / 3.0 * u2);
    }
#else // defined(THREEDIMS) || defined(TWODIMS)
    /* ONEDIM */
    MyAtLeastDouble t1 = (1 - u), t1sq = t1 * t1, u2 = u * u;
    MyAtLeastDouble t4 = t1sq * t1sq; /* t1^4 via squaring */

    if (mode >= 0)
    {
        *dwk = -14.0 * u * t4 - 56.0 * u2 * t4;
    }
    if (mode <= 0)
    {
        *wk = t1 * t4 * (1.0 + 5.0 * u + 8.0 * u2);
    }
#endif // defined(THREEDIMS) || defined(TWODIMS)

#endif // WENDLAND_C4_KERNEL

#ifdef WENDLAND_C6_KERNEL /* Dehnen & Aly 2012 */
#if defined(THREEDIMS) || defined(TWODIMS)
    MyAtLeastDouble t1 = (1 - u), t1sq = t1 * t1, u2 = u * u;
    MyAtLeastDouble t7 = t1sq * t1sq * t1sq * t1; /* t1^7 via squaring, cheaper than 7 sequential multiplies */

    if (mode >= 0)
    {
        *dwk = -22 * t7 * u * (16 * u2 + 7 * u + 1);
    }
    if (mode <= 0)
    {
        *wk = t1 * t7 * (1 + 8 * u + 25 * u2 + 32 * u2 * u);
    }
#else // defined(THREEDIMS) || defined(TWODIMS)
    /* ONEDIM */
    MyAtLeastDouble t1 = (1 - u), t1sq = t1 * t1, u2 = u * u;
    MyAtLeastDouble t6 = t1sq * t1sq * t1sq; /* t1^6 via squaring */

    if (mode >= 0)
    {
        *dwk = -6 * t6 * u * (35 * u2 + 18 * u + 3);
    }
    if (mode <= 0)
    {
        *wk = t6 * t1 * (1 + 7 * u + 19 * u2 + 21 * u2 * u);
    }
#endif // defined(THREEDIMS) || defined(TWODIMS)

#endif // WENDLAND_C6_KERNEL

#ifdef WENDLAND_C8_KERNEL /* Kummer et al. 2019 */
#if defined(THREEDIMS) || defined(TWODIMS)
    double t1 = (1 - u), t1sq = t1 * t1, u2 = u * u;
    double t1p4 = t1sq * t1sq;
    double t9 = t1p4 * t1p4 * t1; /* t1^9 via squaring, cheaper than 9 sequential multiplies */

    if (mode >= 0)
    {
        *dwk = -26 * t9 * u * (231 * u2 * u + 159 * u2 + 45 * u + 5);
    }
    if (mode <= 0)
    {
        *wk = t1 * t9 * (5 + 50 * u + 210 * u2 + 450 * u2 * u + 429 * u2 * u2);
    }
#else // defined(THREEDIMS) || defined(TWODIMS)
#error "ONEDIM not implemented for WENDLAND_C8_KERNEL"
#endif // defined(THREEDIMS) || defined(TWODIMS)

#endif /* WENDLAND_C8_KERNEL */

    if (mode >= 0)
    {
        *dwk *= NORM * hinv4;
    }
    if (mode <= 0)
    {
        *wk *= NORM * hinv3;
    }

    return;
}
#endif // OLD_KERNEL

//////////////////////////////////////////////////////////////////////////////////////////////////
#endif // KERNEL_HPP
