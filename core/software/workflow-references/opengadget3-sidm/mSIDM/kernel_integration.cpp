/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Source file with class definitions used to compute the kernel overlap.
 *
 * @author OpenGadget3-development team, in particular Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#include "kernel_integration.hpp"

#include "tools.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of class for kernel overlap computations.
 */
template <typename KERNEL>
scatter::LambdaNumerical<KERNEL>::LambdaNumerical(std::string name)
    : matrix_size(3076), matrix_size_m2(matrix_size - 2), h_p_max(60), matrix(nullptr), matrix_computed(false),
      filename(name)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Destructor of class for kernel overlap computations.
 */
template <typename KERNEL> scatter::LambdaNumerical<KERNEL>::~LambdaNumerical()
{
    for (int loop = 0; loop < matrix_size; ++loop)
    {
        delete[] matrix[loop];
    }
    delete[] matrix;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor of class for kernel overlap computations.
 */
template <typename KERNEL> void scatter::LambdaNumerical<KERNEL>::setup()
{
    using namespace scatter::tools;

    h_step = (h_p_max - 1.0) / (matrix_size - 1);
    Dh_step = 1.0 / h_step;
    d_step = (1.0 + h_p_max) / (matrix_size - 1);
    Dd_step = 1.0 / d_step;

    matrix = new MyAtLeastDouble *[matrix_size];
    for (int loop = 0; loop < matrix_size; ++loop)
    {
        matrix[loop] = new MyAtLeastDouble[matrix_size];
    }

    coutr(info) << " Try to read table for numerical integration from file." << endl;
    read_from_file(filename);
    if (!matrix_computed)
    {
        coutr(info) << "  ... compute table" << endl;
        compute_matrix();
        write_to_file(filename);
        coutr(info) << " Table for numerical integration is computed and written to file." << endl;
    }
    else
    {
        coutr(info) << "  ... successful." << endl;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Compute look-up table for kernel overlap integral.
 */
template <typename KERNEL> void scatter::LambdaNumerical<KERNEL>::compute_matrix()
{
    const int &int_order = max_int_order;
    const MyAtLeastDouble *x = gauss_legendre.table[int_order].x;
    const MyAtLeastDouble *w = gauss_legendre.table[int_order].w;

#pragma omp parallel for
    for (int loopd = 0; loopd < matrix_size; ++loopd)
    {
        for (int looph = 0; looph < matrix_size; ++looph)
        {
            const MyAtLeastDouble d_p = loopd * d_step;
            const MyAtLeastDouble h_p = 1.0 + looph * d_step;
            const MyAtLeastDouble zl_limit = std::min(MyAtLeastDouble(1.0), std::max(MyAtLeastDouble(-1.0), d_p - h_p));
            const MyAtLeastDouble zu_limit = 1.0;
            const MyAtLeastDouble intdom = zu_limit - zl_limit;
            if (intdom > 0)
            {
                MyAtLeastDouble integral_inner[int_order];
                for (int loopi = 0; loopi < int_order; ++loopi)
                {
                    const MyAtLeastDouble z = 0.5 * (x[loopi] * intdom + zl_limit + zu_limit);
                    const MyAtLeastDouble z2 = z * z;
                    const MyAtLeastDouble zd = z - d_p;
                    const MyAtLeastDouble zd2 = zd * zd;
                    auto integrand1 = [&z2, &h_p, &zd2](MyAtLeastDouble r) {
                        const MyAtLeastDouble r2 = r * r;
                        return KERNEL::evaluate(1.0, sqrt(r2 + z2)) * KERNEL::evaluate(h_p, sqrt(r2 + zd2)) * r;
                    };
                    const MyAtLeastDouble ru_limit = std::min(sqrt(1.0 - z2), sqrt(h_p * h_p - zd2));
                    const MyAtLeastDouble rl_limit = 0.0;
                    integral_inner[loopi] = gauss_legendre(integrand1, rl_limit, ru_limit, int_order);
                }
                MyAtLeastDouble integral_outer = 0;
                for (int loop = 0; loop < int_order; ++loop) // solving outer integral
                {
                    integral_outer += w[loop] * 0.5 * intdom * integral_inner[loop];
                }
                matrix[loopd][looph] = 2.0 * M_PI * integral_outer;
            }
            else
            {
                matrix[loopd][looph] = 0.0;
            }
        }
    }
    matrix_computed = true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Overloaded operator to interpolate kerneloverlap from look-up table.
 *
 * Here Lambda, i.e. the kernel overlap is computed by interpolating from a table that has
 * been computed at the beginning of the simulation.
 */
template <typename KERNEL>
MyAtLeastDouble scatter::LambdaNumerical<KERNEL>::operator()(const MyAtLeastDouble &h_i, const MyAtLeastDouble &h_j,
                                                             const MyAtLeastDouble &distance)
{
    using namespace scatter::tools;

#ifdef SCATTERING_DEBUG
    Expects(matrix_computed, "matrix for LambdaNumerical has not yet been computed");
#endif // SCATTERING_DEBUG

    if (h_i + h_j > distance)
    {
        const MyAtLeastDouble b = (h_i < h_j) * (h_i - h_j);
        const MyAtLeastDouble h_min = b + h_j;
        const MyAtLeastDouble h_max = h_i - b;
        const MyAtLeastDouble Dh_min = 1.0 / h_min;
        const MyAtLeastDouble d_p = distance * Dh_min;
        const MyAtLeastDouble h_p = h_max * Dh_min;
        int loopd = d_p * Dd_step; // truncation is intended
        int looph;

        auto interpolate = [this, &loopd, &looph, &d_p, &Dh_min, &h_p]() -> MyAtLeastDouble {
            // interpolation along d, also extrapolation
            MyAtLeastDouble x1 = loopd * d_step;
            MyAtLeastDouble x2 = x1 + d_step;
            const MyAtLeastDouble ip1 = interpol(d_p, x1, x2, matrix[loopd][looph], matrix[loopd + 1][looph]);
            const MyAtLeastDouble ip2 = interpol(d_p, x1, x2, matrix[loopd][looph + 1], matrix[loopd + 1][looph + 1]);

            // interpolation along h
            x1 = 1.0 + looph * h_step;
            x2 = x1 + h_step;
            return std::max(MyAtLeastDouble(0.0), interpol(h_p, x1, x2, ip1, ip2)) * Dh_min * Dh_min * Dh_min;
            // it is possible that interpol gives a negative value, when matrix has negative entries
        };

        if (h_p < h_p_max) // do interpolation
        {
            looph = (h_p - 1.0) * Dh_step; // truncation is intended
            return interpolate();
        }
        else // do extrapolation
        {
            loopd = std::min(loopd, matrix_size_m2);
            looph = matrix_size_m2;
#ifdef SCATTERING_KERNEL_WARNING
            cout(warning) << "SCATTER Warning: h_p_max too small (calc_lambda), extrapolation is used; h_p = " << h_p
                          << ", h_i = " << h_i << ", h_j = " << h_j << ", d = " << distance << endl;
#ifndef SCATTERING_IGNOREWARNING
            Expects(false, "This can be resolved by increasing h_p_max in the constructor of\
                      LambdaNumerical in mSIDM/kernel_integration.cpp. To keep the same\
                      accuracy matrix_size should be increased by the same factor.\
                      Usually, the warning occurs for particles at the outskirts of halos\
                      and can be ignored.");
#endif // SCATTERING_IGNOREWARNING
#endif // SCATTERING_KERNEL_WARNING
            return interpolate();
        }
    }
    else
    {
        return 0;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Write look-up table to file.
 */
template <typename KERNEL> void scatter::LambdaNumerical<KERNEL>::write_to_file(const std::string &filename) const
{
    const int default_precision(std::cout.precision());
    constexpr int max_precision(std::numeric_limits<long double>::digits10 + 1);
    std::ofstream dat_out;
    dat_out.open(filename.c_str(), std::ios_base::out);
    if (dat_out)
    {
        dat_out << std::setprecision(max_precision);
        dat_out << *this;
        dat_out << std::setprecision(default_precision);
    }
    else
    {
        using namespace scatter::tools;
        coutr(error) << "SCATTER error: not able to write table for kernel integration (" << filename << ")" << endl;
        throw std::exception();
    }
    dat_out.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Read look-up table from file.
 */
template <typename KERNEL> void scatter::LambdaNumerical<KERNEL>::read_from_file(const std::string &filename)
{
    using namespace scatter::tools;

    std::ifstream dat_in;
    dat_in.open(filename.c_str(), std::ios_base::in);
    if (dat_in)
    {
        dat_in >> *this;
    }
    else
    {
        coutr(info) << "  ... file not found (" << filename << ")" << endl;
    }
    dat_in.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Ostream operator for look-up table class.
 */
template <typename KERNELB>
std::ostream &scatter::operator<<(std::ostream &s, const scatter::LambdaNumerical<KERNELB> &ln)
{
    s << "Kernel: ";
    s << KERNELB::name;
    s << ", IntOrder: " << ln.max_int_order;
    s << ", MatrixSize: " << ln.matrix_size;
    s << ", hpmax: " << ln.h_p_max;
    s << ";" << std::endl;

    for (int loopx = 0; loopx < ln.matrix_size; ++loopx)
    {
        for (int loopy = 0; loopy < ln.matrix_size - 1; ++loopy)
        {
            s << ln.matrix[loopx][loopy] << " ";
        }
        s << ln.matrix[loopx][ln.matrix_size - 1] << std::endl;
    }
    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Istream operator for look-up table class.
 */
template <typename KERNELB> std::istream &scatter::operator>>(std::istream &s, scatter::LambdaNumerical<KERNELB> &ln)
{
    using namespace scatter;
    using namespace scatter::tools;

    auto read = [&s]() -> std::string {
        std::string readstr;
        s >> readstr;
        readstr.pop_back();
        return readstr;
    };

    if (read().compare("Kernel") != 0)
    {
        return s;
    }
    if (read().compare(KERNELB::name) != 0)
    {
        return s;
    }
    if (read().compare("IntOrder") != 0)
    {
        return s;
    }
    if (convert_from_string<unsigned int>(read()) != ln.max_int_order)
    {
        return s;
    }
    if (read().compare("MatrixSize") != 0)
    {
        return s;
    }
    if (convert_from_string<int>(read()) != ln.matrix_size)
    {
        return s;
    }
    if (read().compare("hpmax") != 0)
    {
        return s;
    }
    if (convert_from_string<MyAtLeastDouble>(read()) != ln.h_p_max)
    {
        return s;
    }

    for (int loopx = 0; loopx < ln.matrix_size; ++loopx)
    {
        for (int loopy = 0; loopy < ln.matrix_size; ++loopy)
        {
            s >> ln.matrix[loopx][loopy];
        }
    }

    ln.matrix_computed = true;
    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor for Gauss Legendre integration class.
 *
 * Gauss Legendre integration - Here we do not use a library but have our own implementation,
 * the tables are given explicitly and can be completed or a function to compute them could
 * be implemented. However, important is that one can pass Lambdas for integration.
 */
scatter::GaussLegendre::GaussLegendre() : max_order(20)
{
    table = new IntTable[max_order + 1];
    if (max_order >= 1)
    {
        table[1].order = 1;
        table[1].x = new MyAtLeastDouble[1]{0.0};
        table[1].w = new MyAtLeastDouble[1]{2.0};
        if (max_order >= 2)
        {
            table[2].order = 2;
            table[2].x = new MyAtLeastDouble[2]{-sqrt(1.0 / 3.0), sqrt(1.0 / 3.0)};
            table[2].w = new MyAtLeastDouble[2]{1.0, 1.0};
            if (max_order >= 3)
            {
                table[3].order = 3;
                table[3].x = new MyAtLeastDouble[3]{-sqrt(3.0 / 5.0), 0, sqrt(3.0 / 5.0)};
                table[3].w = new MyAtLeastDouble[3]{5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
                if (max_order >= 4)
                {
                    table[4].order = 4;
                    table[4].x = new MyAtLeastDouble[4]{
                        -sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(6.0 / 5.0)), -sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(6.0 / 5.0)),
                        sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(6.0 / 5.0)), sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(6.0 / 5.0))};
                    table[4].w = new MyAtLeastDouble[4]{(18.0 - sqrt(30.0)) / 36.0, (18.0 + sqrt(30.0)) / 36.0,
                                                        (18.0 + sqrt(30.0)) / 36.0, (18.0 - sqrt(30.0)) / 36.0};
                    if (max_order >= 5)
                    {
                        table[5].order = 5;
                        table[5].x = new MyAtLeastDouble[5]{-1.0 / 3.0 * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0)),
                                                            -1.0 / 3.0 * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)), 0.0,
                                                            1.0 / 3.0 * sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)),
                                                            1.0 / 3.0 * sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0))};
                        table[5].w = new MyAtLeastDouble[5]{
                            (322.0 - 13.0 * sqrt(70.0)) / 900.0, (322.0 + 13.0 * sqrt(70.0)) / 900.0, 128.0 / 225.0,
                            (322.0 + 13.0 * sqrt(70.0)) / 900.0, (322.0 - 13.0 * sqrt(70.0)) / 900.0};
                        if (max_order >= 6)
                        {
                            table[6].order = 6;
                            table[6].x =
                                new MyAtLeastDouble[6]{0.6612093864662645, -0.6612093864662645, -0.2386191860831969,
                                                       0.2386191860831969, -0.9324695142031521, 0.9324695142031521};
                            table[6].w =
                                new MyAtLeastDouble[6]{0.3607615730481386, 0.3607615730481386, 0.4679139345726910,
                                                       0.4679139345726910, 0.1713244923791704, 0.1713244923791704};
                            if (max_order >= 10)
                            {
                                table[10].order = 10;
                                table[10].x = new MyAtLeastDouble[10]{
                                    -0.1488743389816312, 0.1488743389816312, -0.4333953941292472, 0.4333953941292472,
                                    -0.6794095682990244, 0.6794095682990244, -0.8650633666889845, 0.8650633666889845,
                                    -0.9739065285171717, 0.9739065285171717};
                                table[10].w = new MyAtLeastDouble[10]{
                                    0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2692667193099963,
                                    0.2190863625159820, 0.2190863625159820, 0.1494513491505806, 0.1494513491505806,
                                    0.0666713443086881, 0.0666713443086881};
                                if (max_order >= 20)
                                {
                                    table[20].order = 20;
                                    table[20].x = new MyAtLeastDouble[20]{
                                        -0.0765265211334973, 0.0765265211334973,  -0.2277858511416451,
                                        0.2277858511416451,  -0.3737060887154195, 0.3737060887154195,
                                        -0.5108670019508271, 0.5108670019508271,  -0.6360536807265150,
                                        0.6360536807265150,  -0.7463319064601508, 0.7463319064601508,
                                        -0.8391169718222188, 0.8391169718222188,  -0.9122344282513259,
                                        0.9122344282513259,  -0.9639719272779138, 0.9639719272779138,
                                        -0.9931285991850949, 0.9931285991850949};
                                    table[20].w = new MyAtLeastDouble[20]{
                                        0.1527533871307258, 0.1527533871307258, 0.1491729864726037, 0.1491729864726037,
                                        0.1420961093183820, 0.1420961093183820, 0.1316886384491766, 0.1316886384491766,
                                        0.1181945319615184, 0.1181945319615184, 0.1019301198172404, 0.1019301198172404,
                                        0.0832767415767048, 0.0832767415767048, 0.0626720483341091, 0.0626720483341091,
                                        0.0406014298003869, 0.0406014298003869, 0.0176140071391521, 0.0176140071391521};
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Destructor for Gauss Legendre integration class.
 */
scatter::GaussLegendre::~GaussLegendre()
{
    delete[] table;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Method to check if order for integration is available.
 */
void scatter::GaussLegendre::check_order_avail(int order_in) const
{
    using namespace scatter::tools;

    Expects(order_in <= max_order && order_in > 0);
    Expects(table[order_in].order > 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Constructor for class that contains the integration table.
 */
scatter::GaussLegendre::IntTable::IntTable() : x(nullptr), w(nullptr), order(0)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Destructor for class that contains the integration table.
 */
scatter::GaussLegendre::IntTable::~IntTable()
{
    if (order > 0)
    {
        delete[] x;
        delete[] w;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/** Template specialization for LambdaNumerical */
template class scatter::LambdaNumerical<scatter::kernel_msidm>;

//////////////////////////////////////////////////////////////////////////////////////////////////
