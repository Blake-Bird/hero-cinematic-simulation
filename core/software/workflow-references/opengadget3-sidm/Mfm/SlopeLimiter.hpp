/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares the SlopeLimiter classes.
 *
 * This file declares a parent SlopeLimiter class and derived classes for the
 * different limiters implemented.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET_SLOPE_LIMITER_HPP
#define _OPENGADGET_SLOPE_LIMITER_HPP

#include "../CodeBase/allvars.hpp"
#include "FluidVector.hpp"
#include "../Hydro/hydra_structs.hpp"
#include "mfm_limiters.hpp"
#include <tuple>
#include <algorithm>

namespace OpenGadget3
{

/**
 * @brief   Slope Limiter Base class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class SlopeLimiter
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    SlopeLimiter(){};

    /**
     * @brief Limit the slopes according to pre-computed AlphaSlope.
     */
    void limit_slopes(
        Vector<orig_ndim> *gradW_orig, MyFloat *AlphaSlope, /**< [in] original gradient, slope limiter */
        Vector<orig_ndim, MyAtLeastDouble> drFace, /**< [in] face vector */
        WFluidVector<ndim, MyAtLeastDouble> &W, /**< [inout] primitive fluid vector (passed as reference) */
        Vector<orig_ndim, MyAtLeastDouble> *gradW) /**< [out] gradient */
    {
        WFluidVector<ndim, MyAtLeastDouble> dW;
        for (int var = 0; var < nvar; var++)
        {
            gradW[var] = gradW_orig[var];
            gradW[var] *= AlphaSlope[var];
            dW[var] = Vector<orig_ndim>::DotProduct(gradW[var], drFace);
            W[var] += dW[var];
        }
    };

    /**
     * @brief Compute the slope limiter.
     */
    virtual void compute_slope_limiter(Vector<orig_ndim, MyAtLeastDouble> &dr, struct slopedata_in &local, int j,
                                       struct slopedata_out &out) = 0;

    /**
     * @brief Perform final operations for slope limiter calculation.
     */
    virtual void slope_limiter_final(int i) = 0;
};

/**
 * @brief   Zero slopes Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class SlopeLimiter_ZeroSlopes final : public SlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    SlopeLimiter_ZeroSlopes() : SlopeLimiter<ndim>(){};

    /**
     * @brief Compute the slope limiter.
     *
     * Nothing to be done here for this limter
     */
    void compute_slope_limiter(Vector<orig_ndim, MyAtLeastDouble> &dr, struct slopedata_in &local, int j,
                               struct slopedata_out &out){
        // nothing to be done here
    };

    /**
     * @brief Perform final operations for slope limiter calculation.
     */
    void slope_limiter_final(int i)
    {
        for (int var = 0; var < nvar; var++)
        {
            SphP[i].AlphaSlope[var] = (MyFloat)0.0;
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                SphP[i].AlphaSlope_dust[idust][var] = (MyFloat)0.0;
            }
#endif // GT_DUST
        }
    };
};

/**
 * @brief   Null Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class SlopeLimiter_Null final : public SlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    SlopeLimiter_Null() : SlopeLimiter<ndim>(){};

    /**
     * @brief Compute the slope limiter.
     *
     * Nothing to be done here for this limiter.
     */
    void compute_slope_limiter(Vector<orig_ndim, MyAtLeastDouble> &dr, struct slopedata_in &local, int j,
                               struct slopedata_out &out){
        // nothing to be done here
    };

    /**
     * @brief Perform final operations for slope limiter calculation.
     */
    void slope_limiter_final(int i)
    {
        for (int var = 0; var < nvar; var++)
        {
            SphP[i].AlphaSlope[var] = (MyFloat)1.0;
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                SphP[i].AlphaSlope_dust[idust][var] = (MyFloat)1.0;
            }
#endif // GT_DUST
        }
    };
};

/**
 * @brief   TVD Scalar Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class SlopeLimiter_TVDScalar final : public SlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    SlopeLimiter_TVDScalar() : SlopeLimiter<ndim>(){};

    /**
     * @brief Compute the slope limiter.
     */
    void compute_slope_limiter(Vector<orig_ndim, MyAtLeastDouble> &dr, struct slopedata_in &local, int j,
                               struct slopedata_out &out)
    {
        for (int var = 0; var < nvar; var++)
        {
            const MyAtLeastDouble dW = Vector<orig_ndim>::DotProduct(local.gradW[var], dr);
            const MyAtLeastDouble dWcell = SphP[j].Wprim[var] - local.Wprim[var];
            if (dW != 0.0)
            {
                out.AlphaSlope[var] = std::min(out.AlphaSlope[var],
                                               std::max((MyFloat)0.0, std::min((MyFloat)1.0, (MyFloat)(dWcell / dW))));
            }
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                const MyAtLeastDouble dW_dust = Vector<orig_ndim>::DotProduct(local.gradW_dust[idust][var], dr);
                const MyAtLeastDouble dWcell_dust = SphP[j].Wprim_dust[idust][var] - local.Wprim_dust[idust][var];
                if (dW_dust != 0.0)
                {
                    out.AlphaSlope_dust[idust][var] =
                        std::min(out.AlphaSlope_dust[idust][var],
                                 std::max((MyFloat)0.0, std::min((MyFloat)1.0, (MyFloat)(dWcell_dust / dW_dust))));
                }
            }
#endif // GT_DUST
        }
    };

    /**
     * @brief Perform final operations for slope limiter calculation.
     *
     * Nothing ot be done here for this limiter.
     */
    void slope_limiter_final(int i){
        // nothing to be done here
    };
};

/**
 * @brief   Scalar Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class SlopeLimiter_Scalar final : public SlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    SlopeLimiter_Scalar() : SlopeLimiter<ndim>(){};

    /**
     * @brief Compute the slope limiter.
     *
     * Nothing to be done here for this limiter.
     */
    void compute_slope_limiter(Vector<orig_ndim, MyAtLeastDouble> &dr, struct slopedata_in &local, int j,
                               struct slopedata_out &out){
        // nothing to be done here
    };

    /**
     * @brief Perform final operations for slope limiter calculation.
     */
    void slope_limiter_final(int i)
    {
        // TODO : Factor of 1.0 refers to extend of M4 kernel: replace with whatever is needed
        const MyAtLeastDouble drMax =
            std::max((MyAtLeastDouble)sqrt(SphP[i].DistNgbSqdMax), (MyAtLeastDouble)P[i].Hsml);
        for (int var = 0; var < nvar; var++)
        {
            const MyAtLeastDouble gradW = SphP[i].gradW[var].GetMagnitude();
            const MyAtLeastDouble dW = drMax * gradW;
            const MyAtLeastDouble dWmax = SphP[i].Wmax[var] - SphP[i].Wprim[var];
            const MyAtLeastDouble dWmin = SphP[i].Wprim[var] - SphP[i].Wmin[var];
            SphP[i].AlphaSlope[var] = MyFloat(1.0);
            if (dW != 0.0)
            {
                SphP[i].AlphaSlope[var] =
                    std::max(MyAtLeastDouble(0.0), std::min(MyAtLeastDouble(1.0), std::min(dWmax / dW, dWmin / dW)));
            } // if (dW != 0.0)
#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                const MyAtLeastDouble gradW_dust = SphP[i].gradW_dust[idust][var].GetMagnitude();
                const MyAtLeastDouble dW_dust = drMax * gradW_dust;
                const MyAtLeastDouble dWmax_dust = SphP[i].Wmax_dust[idust][var] - SphP[i].Wprim_dust[idust][var];
                const MyAtLeastDouble dWmin_dust = SphP[i].Wprim_dust[idust][var] - SphP[i].Wmin_dust[idust][var];
                SphP[i].AlphaSlope_dust[idust][var] = MyFloat(1.0);
                if (dW != 0.0)
                {
                    SphP[i].AlphaSlope_dust[idust][var] =
                        std::max(MyAtLeastDouble(0.0),
                                 std::min(MyAtLeastDouble(1.0), std::min(dWmax_dust / dW_dust, dWmin_dust / dW_dust)));
                } // if (dW != 0.0)
            }
#endif // GT_DUST
        } // for (int var = 0; var < nvar; var++)
    };
};

#if MFM_SLOPE_LIMITER == LIMITER_AREPO
/**
 * @brief   Springel 2009/10 (Arepo) Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class SlopeLimiter_Arepo final : public SlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    SlopeLimiter_Arepo() : SlopeLimiter<ndim>(){};

    /**
     * @brief Compute the slope limiter.
     */
    void compute_slope_limiter(Vector<orig_ndim, MyAtLeastDouble> &dr, struct slopedata_in &local, int j,
                               struct slopedata_out &out)
    {
        for (int var = 0; var < nvar; var++)
        {
            MyAtLeastDouble dW = Vector<orig_ndim>::DotProduct(local.gradW[var], dr);
            if (dW > 0.0)
            {
                out.AlphaSlope[var] = std::min(out.AlphaSlope[var], MyFloat(local.dWmax[var] / dW));
            }
            else if (dW < 0.0)
            {
                out.AlphaSlope[var] = std::min(out.AlphaSlope[var], MyFloat(local.dWmin[var] / dW));
            }

#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                dW = Vector<orig_ndim>::DotProduct(local.gradW_dust[idust][var], dr);
                if (dW > 0.0)
                {
                    out.AlphaSlope_dust[idust][var] =
                        std::min(out.AlphaSlope_dust[idust][var], MyFloat(local.dWmax_dust[idust][var] / dW));
                }
                else if (dW < 0.0)
                {
                    out.AlphaSlope_dust[idust][var] =
                        std::min(out.AlphaSlope_dust[idust][var], MyFloat(local.dWmin_dust[idust][var] / dW));
                }
            }
#endif // GT_DUST
        }
    };

    /**
     * @brief Perform final operations for slope limiter calculation.
     *
     * Nothing to be done here for this limiter.
     */
    void slope_limiter_final(int i){
        // nothing to be done here
    };
};
#endif // MFM_SLOPE_LIMITER == LIMITER_AREPO

/**
 * @brief   Gizmo Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim, int slope_limiter_tolerance> class SlopeLimiter_Gizmo final : public SlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    SlopeLimiter_Gizmo() : SlopeLimiter<ndim>(){};

    //---------------------------------------------------------------------------------------------

    /**
     * @brief Compute the slope limiter.
     *
     * Nothing to be done here for this limiter
     */
    void compute_slope_limiter(Vector<orig_ndim, MyAtLeastDouble> &dr, struct slopedata_in &local, int j,
                               struct slopedata_out &out){
        // nothing to be done here
    };

    /**
     * @brief Perform final operations for slope limiter calculation.
     */
    void slope_limiter_final(int i)
    {
        MyAtLeastDouble a_limiter = 0.25;
        // if(SphP[i].ConditionNumber > 100) a_limiter=std::min(0.5, 0.25 + 0.25*(SphP[i].ConditionNumber-100)/100);
        MyAtLeastDouble shoot_tol = 0.0;
#if defined(NOGRAVITY) && !defined(MFM_MAGNETIC)
        shoot_tol = 0.0 /*0.1 */;
#endif // defined(NOGRAVITY) && !defined(MFM_MAGNETIC)
        if constexpr (slope_limiter_tolerance == 2)
        {
            a_limiter *= 2.0;
            shoot_tol = 0.125;
        }
        else if constexpr (slope_limiter_tolerance == 0)
        {
            a_limiter *= 0.5;
            shoot_tol = 0.0;
        }
        MyAtLeastDouble h0 = MyFloat(1.0) / (a_limiter * P[i].Hsml);
        MyAtLeastDouble drMax = std::max(sqrt(SphP[i].DistNgbSqdMax), (MyFloat)P[i].Hsml);
        for (int var = 0; var < nvar; var++)
        {
            MyAtLeastDouble d_abs = SphP[i].gradW[var].GetMagnitude();
            SphP[i].AlphaSlope[var] = MyFloat(1.0);
            if (d_abs > 0)
            {
                MyAtLeastDouble this_shoot_tol = (var == ndim) ? 0.0 : /*0.0 */ shoot_tol;
                MyAtLeastDouble cfac = h0 / d_abs;
                const MyAtLeastDouble dWmax = SphP[i].Wmax[var] - SphP[i].Wprim[var];
                const MyAtLeastDouble dWmin = SphP[i].Wprim[var] - SphP[i].Wmin[var];
                MyAtLeastDouble dWabsmax, dWabsmin;
                std::tie(dWabsmin, dWabsmax) = std::minmax(dWmax, dWmin);
                MyAtLeastDouble f_corr_over = std::min(dWabsmin + this_shoot_tol * dWabsmax, dWabsmax);
                cfac *= f_corr_over;
                if ((var >= ndim) && (var < ndim + 2))
                { // positivity-preserving for pressure and density
                    MyAtLeastDouble fmin = std::min(
                        SphP[i].Wprim[var],
                        std::max(MyFloat(0.0), std::max(MyFloat(1e-20) * SphP[i].Wprim[var],
                                                        (MyFloat)std::min(MyFloat(0.5) * (SphP[i].Wprim[var] - dWmin),
                                                                          SphP[i].Wprim[var] - f_corr_over))));
                    cfac = std::min(cfac, (SphP[i].Wprim[var] - fmin) / drMax / d_abs);
                }
                SphP[i].AlphaSlope[var] = std::min(MyFloat(1.0), (MyFloat)cfac);
            } // if (d_abs > 0)

#ifdef GT_DUST
            for (int idust = 0; idust < GT_NUM_DUST; idust++)
            {
                d_abs = SphP[i].gradW_dust[idust][var].GetMagnitude();
                SphP[i].AlphaSlope_dust[idust][var] = MyFloat(1.0);
                if (d_abs > 0)
                {
                    MyAtLeastDouble this_shoot_tol = (var == ndim) ? 0.0 : /*0.0 */ shoot_tol;
                    MyAtLeastDouble cfac = h0 / d_abs;
                    const MyAtLeastDouble dWmax = SphP[i].Wmax_dust[idust][var] - SphP[i].Wprim_dust[idust][var];
                    const MyAtLeastDouble dWmin = SphP[i].Wprim_dust[idust][var] - SphP[i].Wmin_dust[idust][var];
                    MyAtLeastDouble dWabsmax, dWabsmin;
                    std::tie(dWabsmin, dWabsmax) = std::minmax(dWmax, dWmin);
                    MyAtLeastDouble f_corr_over = std::min(dWabsmin + this_shoot_tol * dWabsmax, dWabsmax);
                    cfac *= f_corr_over;
                    if ((var >= ndim) && (var < ndim + 2))
                    { // positivity-preserving for pressure and density
                        MyAtLeastDouble fmin = std::min(
                            SphP[i].Wprim_dust[idust][var],
                            std::max(MyFloat(0.0),
                                     std::max(MyFloat(1e-20) * SphP[i].Wprim_dust[idust][var],
                                              (MyFloat)std::min(MyFloat(0.5) * (SphP[i].Wprim_dust[idust][var] - dWmin),
                                                                SphP[i].Wprim_dust[idust][var] - f_corr_over))));
                        cfac = std::min(cfac, (SphP[i].Wprim_dust[idust][var] - fmin) / drMax / d_abs);
                    }
                    SphP[i].AlphaSlope_dust[idust][var] = std::min(MyFloat(1.0), (MyFloat)cfac);
                } // if (d_abs > 0)
            }
#endif // GT_DUST

        } // for (int var = 0; var < nvar; var++)
    };
};

/**
 * @brief   Pairwise Slope Limiter Base class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class PairwiseSlopeLimiter
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    PairwiseSlopeLimiter(){};

    /**
     * @brief Limit the slopes for pairwise interaction.
     */
    virtual void limit_slopes_pairwise(const WFluidVector<ndim, MyAtLeastDouble> &Wi_in,
                                       const WFluidVector<ndim, MyAtLeastDouble> &Wj_in,
                                       WFluidVector<ndim, MyAtLeastDouble> &Wi, WFluidVector<ndim, MyAtLeastDouble> &Wj,
                                       Vector<ndim, MyAtLeastDouble> &vface) = 0;
};

/**
 * @brief   Pairwise Null Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim> class PairwiseSlopeLimiter_Null final : public PairwiseSlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    PairwiseSlopeLimiter_Null() : PairwiseSlopeLimiter<ndim>(){};

    /**
     * @brief Limit the slopes for pairwise interaction.
     *
     * Nothing to be done here for this limiter.
     */
    void limit_slopes_pairwise(const WFluidVector<ndim, MyAtLeastDouble> &Wi_in,
                               const WFluidVector<ndim, MyAtLeastDouble> &Wj_in,
                               WFluidVector<ndim, MyAtLeastDouble> &Wi, WFluidVector<ndim, MyAtLeastDouble> &Wj,
                               Vector<ndim, MyAtLeastDouble> &vface){
        // nothing to be done here
    };
};

/**
 * @brief   Pairwise gizmo Slope Limiter class.
 * @author  F. Groth
 * @date    July 2023
 */
template <int ndim, int tolerance, int use_paper_version>
class PairwiseSlopeLimiter_Gizmo final : public PairwiseSlopeLimiter<ndim>
{
  public:
    static const int nvar = ndim + 2;
    static const int orig_ndim = NUMDIMS;

    /**
     * @brief constructor
     */
    PairwiseSlopeLimiter_Gizmo() : PairwiseSlopeLimiter<ndim>(){};

    /**
     * @brief Limit the slopes for pairwise interaction.
     */
    void limit_slopes_pairwise(const WFluidVector<ndim, MyAtLeastDouble> &Wi_in,
                               const WFluidVector<ndim, MyAtLeastDouble> &Wj_in,
                               WFluidVector<ndim, MyAtLeastDouble> &Wi, WFluidVector<ndim, MyAtLeastDouble> &Wj,
                               Vector<ndim, MyAtLeastDouble> &vface)
    {
        MyAtLeastDouble fac_minmax = 0.5;
        MyAtLeastDouble fac_meddev = 0.375;
        if constexpr (use_paper_version)
        {
            fac_minmax = 0.5;
            fac_meddev = 0.25;
        }
        else if constexpr (tolerance == 2)
        {
            fac_minmax = 0.75;
            fac_meddev = 0.375;
        }
        else if constexpr (tolerance == 1)
        {
            fac_minmax = 0.5;
            fac_meddev = 0.4;
        }
        else if constexpr (tolerance == 0)
        {
            fac_minmax = 0.0;
            fac_meddev = 0.0;
        }

        WFluidVector<ndim, MyAtLeastDouble> Wi_orig(Wi_in);
        WFluidVector<ndim, MyAtLeastDouble> Wj_orig(Wj_in);
        //  move into Lagrangian frame
        for (int k = 0; k < ndim; k++)
        {
            Wi.GetVelocity()[k] -= vface[k];
            Wj.GetVelocity()[k] -= vface[k];
            Wi_orig.GetVelocity()[k] -= vface[k];
            Wj_orig.GetVelocity()[k] -= vface[k];
        }

        for (int var = 0; var < nvar; var++)
        {
            MyAtLeastDouble phimin, phimax;
            MyAtLeastDouble phimed = 0.5 * (Wi_orig[var] + Wj_orig[var]);
            std::tie(phimin, phimax) = std::minmax(Wi_orig[var], Wj_orig[var]);
            MyAtLeastDouble fac = fac_minmax * (phimax - phimin);
            MyAtLeastDouble phimax_eff = phimax + fac;
            MyAtLeastDouble phimin_eff = phimin - fac;
            if ((phimax < 0) && (phimax_eff > 0))
            {
                phimax_eff = phimax * phimax / (phimax - (phimax_eff - phimax));
            }
            if ((phimin > 0) && (phimin_eff < 0))
            {
                phimin_eff = phimin * phimin / (phimin + (phimin - phimin_eff));
            }
            fac = fac_meddev * (phimax - phimin);
            MyAtLeastDouble phimed_max = phimed + fac;
            MyAtLeastDouble phimed_min = phimed - fac;
            phimed_max = std::min(phimed_max, phimax_eff);
            phimed_min = std::max(phimed_min, phimin_eff);
            if (Wi_orig[var] < Wj_orig[var])
            {
                Wj[var] = std::max(Wj[var], phimin_eff);
                Wj[var] = std::min(Wj[var], phimed_max);
                Wi[var] = std::min(Wi[var], phimax_eff);
                Wi[var] = std::max(Wi[var], phimed_min);
            }
            else
            {
                Wj[var] = std::min(Wj[var], phimax_eff);
                Wj[var] = std::max(Wj[var], phimed_min);
                Wi[var] = std::max(Wi[var], phimin_eff);
                Wi[var] = std::min(Wi[var], phimed_max);
            }
        }
        // move back to original lab frame
        for (int k = 0; k < ndim; k++)
        {
            Wi.GetVelocity()[k] += vface[k];
            Wj.GetVelocity()[k] += vface[k];
        }
    };
};

template class SlopeLimiter_Null<mfm_numdims>;
template class SlopeLimiter_ZeroSlopes<mfm_numdims>;
template class SlopeLimiter_TVDScalar<mfm_numdims>;
template class SlopeLimiter_Scalar<mfm_numdims>;
#if MFM_SLOPE_LIMITER == LIMITER_AREPO
template class SlopeLimiter_Arepo<mfm_numdims>;
#endif // #if MFM_SLOPE_LIMITER == LIMITER_AREPO
template class SlopeLimiter_Gizmo<mfm_numdims, SLOPE_LIMITER_TOLERANCE>;
template class PairwiseSlopeLimiter_Null<mfm_numdims>;
template class PairwiseSlopeLimiter_Gizmo<mfm_numdims, SLOPE_LIMITER_TOLERANCE, 0>;
template class PairwiseSlopeLimiter_Gizmo<mfm_numdims, SLOPE_LIMITER_TOLERANCE, 1>;

} // namespace OpenGadget3
#endif // _OPENGADGET_SLOPE_LIMITER_HPP
