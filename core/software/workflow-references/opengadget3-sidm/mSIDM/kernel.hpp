/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief This file defines the kernel that is used for the interactions.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_KERNEL_HPP
#define MSIDM_KERNEL_HPP

#include "../CodeBase/precision.hpp"
#include <cmath>
#include "../Hydro/kernel.hpp"
#include <string>

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Function overloading for the top-hat kernel.
 */
inline void kernel_top_hat(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4, MyAtLeastDouble *wk,
                           MyAtLeastDouble *dwk, kernel_mode mode)
{
    kernel_top_hat(u, hinv3, wk, dwk, mode);
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Kernel for SIDM.
 */
class kernel_msidm
{
  public:
    /** name of kernel */
#ifdef mSIDM_KERNEL_TOPHAT
    inline static const std::string name = "top_hat";
#elif defined(mSIDM_KERNEL_SPLINE)
    inline static const std::string name = "cubic_spline";
#elif defined(mSIDM_KERNEL_QUINTIC)
    inline static const std::string name = "quintic";
#elif defined(mSIDM_KERNEL_WENDLAND_C4)
    inline static const std::string name = "wendland_c4";
#elif defined(mSIDM_KERNEL_WENDLAND_C6)
    inline static const std::string name = "wendland_c6";
#else // default kernel
    inline static const std::string name = "cubic_spline";
#endif

    /** overloading of kernel_msidm to have different arguments and return value */
    static MyAtLeastDouble evaluate(MyAtLeastDouble h, MyAtLeastDouble r)
    {
        const MyAtLeastDouble Dh = 1.0 / h;
        const MyAtLeastDouble Dh2 = Dh * Dh;
        MyAtLeastDouble result;
        MyAtLeastDouble *ptr = nullptr;
        evaluate(r * Dh, Dh2 * Dh, Dh2 * Dh2, &result, ptr, calc_wk);
        return result;
    };

  private:
    static void evaluate(MyAtLeastDouble u, MyAtLeastDouble hinv3, MyAtLeastDouble hinv4, MyAtLeastDouble *wk,
                         MyAtLeastDouble *dwk, kernel_mode mode)
    {
#ifdef mSIDM_KERNEL_TOPHAT
        kernel_top_hat(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(mSIDM_KERNEL_SPLINE)
        kernel_cubic_spline(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(mSIDM_KERNEL_QUINTIC)
        kernel_quintic(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(mSIDM_KERNEL_WENDLAND_C4)
        kernel_wendland_c4(u, hinv3, hinv4, wk, dwk, mode);
#elif defined(mSIDM_KERNEL_WENDLAND_C6)
        kernel_wendland_c6(u, hinv3, hinv4, wk, dwk, mode);
#else
        kernel_cubic_spline(u, hinv3, hinv4, wk, dwk, mode);
#endif
    };
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_KERNEL_HPP */
