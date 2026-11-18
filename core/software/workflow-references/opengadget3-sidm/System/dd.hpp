/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief
 *
 * @author OpenGadget3-development team, in particular Klaus Dolag and Volker Springel.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

/* I (Volker Springel) have written this class `dd' based in part on
 * code contained in the
 *  ------------------------------------------------------------------------
 *  | QUAD-DOUBLE/DOUBLE-DOUBLE COMPUTATION PACKAGE                        |
 *  |                                                                      |
 *  | Yozo Hida        U.C. Berkeley               yozo@cs.berkeley.edu    |
 *  | Xiaoye S. Li     Lawrence Berkeley Natl Lab  xiaoye@nersc.gov        |
 *  | David H. Bailey  Lawrence Berkeley Natl Lab  dhbailey@lbl.gov        |
 *  |                                                                      |
 *  | Revised  2005-03-12  Copyright (c) 2005                              |
 *  ------------------------------------------------------------------------
 *  (available at http://crd.lbl.gov/~dhbailey/mpdist/)
 */

#ifndef DD_HPP
#define DD_HPP

/**
 * @brief Class for representing a double-double precision floating point number.
 *
 * This class provides functionality for performing arithmetic operations with double-double precision,
 * which consists of two double-precision floating-point numbers `hi` and `lo`.
 */
class dd
{
  public:
    double hi, lo;

    double quick_two_sum(double a, double b, double &err)
    {
        double s = a + b;

        err = b - (s - a);
        return s;
    }

    double two_sum(double a, double b, double &err)
    {
        double s = a + b;
        double bb = s - a;

        err = (a - (s - bb)) + (b - bb);
        return s;
    }

    /* Self-Addition with a double */
    dd &operator+=(double a)
    {
        double s1, s2;

        s1 = two_sum(hi, a, s2);
        s2 += lo;
        hi = quick_two_sum(s1, s2, lo);
        return *this;
    };

    /* Self-Addition with a doubledouble */
    dd &operator+=(const dd &a)
    {
        double s1, s2, t1, t2;

        s1 = two_sum(hi, a.hi, s2);
        t1 = two_sum(lo, a.lo, t2);
        s2 += t1;
        s1 = quick_two_sum(s1, s2, s2);
        s2 += t2;
        hi = quick_two_sum(s1, s2, lo);
        return *this;
    };

    /* Assignment */
    dd &operator=(double a)
    {
        hi = a;
        lo = 0.0;
        return *this;
    };

    /* Cast */
    operator double() const
    {
        return hi;
    };
};

#endif // DD_HPP
