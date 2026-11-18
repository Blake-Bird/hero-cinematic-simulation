/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Header file with class declarations used to compute the kernel overlap.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef MSIDM_KERNEL_INTEGRATION_HPP
#define MSIDM_KERNEL_INTEGRATION_HPP

#include <iostream>
#include "kernel.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Declaration of the kernel integration class.
 */
template <typename KERNEL> class LambdaNumerical
{
  public:
    LambdaNumerical(std::string name);
    ~LambdaNumerical();
    void setup();
    MyAtLeastDouble operator()(const MyAtLeastDouble &h_i, const MyAtLeastDouble &h_j, const MyAtLeastDouble &distance);

  private:
    static constexpr int max_int_order = 20;
    const int matrix_size, matrix_size_m2;
    const MyAtLeastDouble h_p_max;
    MyAtLeastDouble h_step, Dh_step, d_step, Dd_step;
    MyAtLeastDouble **matrix;
    const std::string kernel_str;
    bool matrix_computed;
    void compute_matrix();

    std::string filename;
    void write_to_file(const std::string &filename) const;
    void read_from_file(const std::string &filename);

    template <typename KERNELB> friend std::ostream &operator<<(std::ostream &s, const LambdaNumerical<KERNELB> &ln);

    template <typename KERNELB> friend std::istream &operator>>(std::istream &s, LambdaNumerical<KERNELB> &ln);
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Class declaration for Gauss Legendre integration.
 */
class GaussLegendre
{
  public:
    GaussLegendre();
    ~GaussLegendre();

    int max_order;

    template <typename FUNC>
    MyAtLeastDouble operator()(FUNC func, MyAtLeastDouble lower_limit, MyAtLeastDouble upper_limit, int order);

    void check_order_avail(int order_in) const;

    /** table for gauss legendre integration */
    class IntTable
    {
      public:
        IntTable();
        ~IntTable();
        const MyAtLeastDouble *x;
        const MyAtLeastDouble *w;
        int order;
    } *table;
};
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

namespace scatter
{
/**
 * @brief Overloaded operator to do the Gauss Legendre integration.
 */
template <typename FUNC>
MyAtLeastDouble GaussLegendre::operator()(FUNC func, MyAtLeastDouble lower_limit, MyAtLeastDouble upper_limit,
                                          int order)
{
    MyAtLeastDouble result = 0.0;
    MyAtLeastDouble intdom = upper_limit - lower_limit;
    for (int loop = 0; loop < order; ++loop)
    {
        result += table[order].w[loop] * 0.5 * intdom *
                  func(0.5 * (table[order].x[loop] * intdom + lower_limit + upper_limit));
    }
    return result;
}
} // namespace scatter

//////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* MSIDM_KERNEL_INTEGRATION_HPP */
