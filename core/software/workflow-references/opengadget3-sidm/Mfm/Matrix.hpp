/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief Declares the Matrix classes.
 *
 * This file declares a parent Matrix class with derived classes e.g. for
 * SquareMatrix and RotationMatrix.
 *
 * @author OpenGadget3-development team, in particular Frederick Groth.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#ifndef _OPENGADGET3_MATRIX_HPP_
#define _OPENGADGET3_MATRIX_HPP_

#include "../CodeBase/precision.hpp"
#include "../CodeBase/verbose.hpp"
#include "../System/assert.hpp"
#include "Eos.hpp"
#include "FluidVector.hpp"
#include "Vector.hpp"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <math.h>

//-------------------------------------------------------------------------------------------------
namespace OpenGadget3
{

/**
 * @brief   General matrix class for storing data in column-major order (similar to C/C++ arrays)
 * @author  S. Heigl & D. A. Hubber, updated by F. Groth
 * @date    13/02/2018
 */
template <int M, int N, typename T = MyFloat> class Matrix
{
  protected:
    T data[M][N]; /**< Matrix data */

  public:
    /**
     * @brief default constructor.
     *
     * Initialize all entries to zero.
     */
    Matrix()
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                data[i][j] = T(0.0);
            }
        }
    }

    /**
     * @brief constructor.
     *
     * Initialize all entries with given value.
     */
    Matrix(const T value)
    {
        assert(std::isfinite(value));
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                data[i][j] = value;
            }
        }
    }

    /**
     * @brief copy constructor.
     */
    template <typename U> Matrix(const Matrix<M, N, U> other)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                data[i][j] = other(i, j);
            }
        }
    }

    /**
     * @brief static function to generate random matrix.
     */
    static inline Matrix<M, N, T> GenerateRandomMatrix()
    {
        Matrix<M, N, T> randMatrix;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                randMatrix(i, j) = T(rand() % RAND_MAX) / T(RAND_MAX);
            }
        }
        return randMatrix;
    }

    /**
     * @brief operator to set all entries to given value.
     */
    Matrix<M, N, T> &operator=(const T value)
    {
        assert(std::isfinite(value));
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                data[i][j] = value;
            }
        }
        return *this;
    }

    /**
     * @brief matrix multiplication operator.
     */
    template <int P> Matrix<M, P, T> operator*(const Matrix<N, P, T> &other) const
    {
        Matrix<M, P, T> product;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < P; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    product(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return product;
    }

    /**
     * @brief operator to access element at given indices.
     */
    T &operator()(const int rindex, const int cindex)
    {
        assert(rindex >= 0 && rindex < M);
        assert(cindex >= 0 && cindex < N);
        return data[rindex][cindex];
    }

    /**
     * @brief const operator to access element at given indices.
     */
    const T &operator()(const int rindex, const int cindex) const
    {
        assert(rindex >= 0 && rindex < M);
        assert(cindex >= 0 && cindex < N);
        return data[rindex][cindex];
    }

    /**
     * @brief matrix addition operator.
     */
    Matrix<M, N, T> operator+(const Matrix<M, N, T> &other) const
    {
        Matrix<M, N, T> result;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                result(i, j) = this->data[i][j] + other(i, j);
            }
        }
        return result;
    }

    /**
     * @brief matrix substraction operator.
     */
    Matrix<M, N, T> operator-(const Matrix<M, N, T> &other) const
    {
        Matrix<M, N, T> result;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                result(i, j) = this->data[i][j] - other(i, j);
            }
        }
        return result;
    }

    /**
     * @brief scalar multiplication operator for matrix.
     */
    Matrix<M, N, T> operator*(const T scalar) const
    {
        Matrix<M, N, T> result;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                result(i, j) = this->data[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * @brief output operator, add all entries to output stream.
     */
    friend std::ostream &operator<<(std::ostream &out, const Matrix<M, N, T> &matrix)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                out << matrix(i, j) << " ";
            }
        }
        return out;
    }

    /**
     * @brief Get the sum of all entries
     */
    inline T GetSum() const
    {
        T sum = T(0.0);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                sum += data[i][j];
            }
        }
        return sum;
    }

    /**
     * @brief Return transpose of Matrix.
     */
    inline Matrix<N, M, T> GetTranspose() const
    {
        Matrix<N, M, T> transpose;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                transpose(j, i) = data[i][j];
            }
        }
        return transpose;
    }

    /**
     * @brief Return general determinant (not defined for general matrix class).
     */
    T GetDeterminant() const
    {
        ENDRUN("Can't get determinant for general Matrix object");
        return T(0.0);
    }

    /**
     * @brief Invert general matrix (not defined for general matrix class).
     */
    void InvertMatrix()
    {
        ENDRUN("Can't invert general Matrix object");
    }
};

/**
 * @brief   General matrix class for storing data in column-major order (similar to C/C++ arrays)
 * @author  S. Heigl & D. A. Hubber, updated by F. Groth
 * @date    12/09/2018
 */
template <int N, typename T = MyFloat> class SquareMatrix : public Matrix<N, N, T>
{
  protected:
    using Matrix<N, N, T>::data;

  public:
    /**
     * @brief default constructor.
     *
     * Initialize all entries to zero.
     */
    SquareMatrix() : Matrix<N, N, T>(){};
    /**
     * @brief constructor.
     *
     * Initialize all entries with given value.
     */
    SquareMatrix(const T value) : Matrix<N, N, T>(value){};
    /**
     * @brief copy constructor.
     */
    SquareMatrix(const Matrix<N, N, T> &other) : Matrix<N, N, T>(other){};

    /**
     * @brief matrix multiplication operator.
     */
    friend SquareMatrix<N, T> operator*(const SquareMatrix<N, T> &mat1, const SquareMatrix<N, T> &mat2)
    {
        SquareMatrix<N, T> product;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    product(i, j) += mat1(i, k) * mat2(k, j);
                }
            }
        }
        return product;
    }

    /**
     * @brief matrix with vector multiplication operator.
     */
    template <typename T1> friend Vector<N, T1> operator*(const SquareMatrix<N, T> &mat, const Vector<N, T1> &vec)
    {
        Vector<N, T1> result;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                result[i] += mat(i, j) * vec[j];
            }
        }
        return result;
    }

    /**
     * @brief Return determinant of square matrix.
     */
    T GetDeterminant() const
    {
        if constexpr (N == 1)
        {
            return data[0][0];
        }
        else if constexpr (N == 2)
        {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        }
        else if constexpr (N == 3)
        {
            return data[0][0] * (data[1][1] * data[2][2] - data[2][1] * data[1][2]) -
                   data[0][1] * (data[1][0] * data[2][2] - data[1][2] * data[2][0]) +
                   data[0][2] * (data[1][0] * data[2][1] - data[1][1] * data[2][0]);
        }
        return 0.0; // something went wrong in this case
    }

    /**
     * @brief Invert square matrix for given determinant.
     */
    template <typename T1> void InvertMatrix(const T1 determinant)
    {
        T invdet;
        if (abs(determinant) > small_float)
        {
            invdet = T(1.0) / determinant;
        }
        else
        {
            invdet = T(0.0);
#ifdef DEBUG
            VERBOSE_ALL(5, "Error!  Zero determinant for SquareMatrix (%g < %g)", determinant, small_float);
#endif // DEBUG
            assert(std::isnormal(determinant));
        }

        SquareMatrix<N, T> invMatrix;
        if constexpr (N == 1)
        {
            invMatrix(0, 0) = invdet;
        }
        else if constexpr (N == 2)
        {
            invMatrix(0, 0) = invdet * data[1][1];
            invMatrix(0, 1) = -T(1.0) * invdet * data[0][1];
            invMatrix(1, 0) = -T(1.0) * invdet * data[1][0];
            invMatrix(1, 1) = invdet * data[0][0];
        }
        else if constexpr (N == 3)
        {
            invMatrix(0, 0) = (data[1][1] * data[2][2] - data[2][1] * data[1][2]) * invdet;
            invMatrix(0, 1) = (data[0][2] * data[2][1] - data[0][1] * data[2][2]) * invdet;
            invMatrix(0, 2) = (data[0][1] * data[1][2] - data[0][2] * data[1][1]) * invdet;
            invMatrix(1, 0) = (data[1][2] * data[2][0] - data[1][0] * data[2][2]) * invdet;
            invMatrix(1, 1) = (data[0][0] * data[2][2] - data[0][2] * data[2][0]) * invdet;
            invMatrix(1, 2) = (data[1][0] * data[0][2] - data[0][0] * data[1][2]) * invdet;
            invMatrix(2, 0) = (data[1][0] * data[2][1] - data[2][0] * data[1][1]) * invdet;
            invMatrix(2, 1) = (data[2][0] * data[0][1] - data[0][0] * data[2][1]) * invdet;
            invMatrix(2, 2) = (data[0][0] * data[1][1] - data[1][0] * data[0][1]) * invdet;
        }
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                data[i][j] = invMatrix(i, j);
            }
        }
    }

    /**
     * @brief Invert square matrix.
     */
    void InvertMatrix()
    {
        const T determinant = SquareMatrix<N, T>::GetDeterminant();

        InvertMatrix(determinant);
    }

    /**
     * @brief Return inverse matrix.
     */
    inline SquareMatrix<N, T> GetInverseMatrix() const
    {
        SquareMatrix<N, T> inverseMatrix(*this);
        inverseMatrix.InvertMatrix();
        return inverseMatrix;
    }

    /**
     * @brief Return trace (sum of diagonal entries).
     */
    inline T GetTrace() const
    {
        T trace = T(0.0);
        for (int i = 0; i < N; i++)
        {
            trace += data[i][i];
        }
        return trace;
    }

    /**
     * @brief Return transpose of matrix.
     */
    inline SquareMatrix<N, T> GetTranspose() const
    {
        SquareMatrix<N, T> transpose;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                transpose(j, i) = data[i][j];
            }
        }
        return transpose;
    }

    /**
     * @brief Return inverse matrix for given square matrix.
     */
    inline static SquareMatrix<N, T> GetInverseMatrixFromMatrix(const SquareMatrix<N, T> &matrix)
    {
        SquareMatrix<N, T> inverseMatrix(matrix);
        inverseMatrix.InvertMatrix();
        return inverseMatrix;
    }

    /**
     * @brief Return identity matrix.
     */
    static inline SquareMatrix<N, T> GenerateIdentitySquareMatrix()
    {
        SquareMatrix<N, T> identityMatrix;
        for (int i = 0; i < N; i++)
        {
            identityMatrix(i, i) = T(1.0);
        }
        return identityMatrix;
    }

    /**
     * @brief static function to generate random matrix.
     */
    static inline SquareMatrix<N, T> GenerateRandomMatrix()
    {
        SquareMatrix<N, T> randMatrix;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                randMatrix(i, j) = T(rand() % RAND_MAX) / T(RAND_MAX);
            }
        }
        return randMatrix;
    }

    /**
     * @brief Return true if condition number is reasonable, invert Matrix in the process.
     */
    template <typename T1> bool CheckConditionNumberAndInvertMatrix(const T1 determinant)
    {
        MyAtLeastDouble sqdConditionNumber = GetConditionNumberAndInvertMatrix(determinant);

        return (sqdConditionNumber < 1.0e4 && abs(sqdConditionNumber) != 0);
    }

    /**
     * @brief Return condition number, invert Matrix in the process.
     */
    template <typename T1> MyAtLeastDouble GetConditionNumberAndInvertMatrix(const T1 determinant)
    {
        // save the original matrix for checking the accuracy later
        SquareMatrix MatrixCopy(*this);

        // The Matrix inversion checks if the determinant is a normal number and not too small
        // and set it to zero in case
        this->InvertMatrix(determinant);

        // check the accuracy of the integral gradients using the square condition number
        // as for MFM
        MyAtLeastDouble modMatrix = 0.0;
        MyAtLeastDouble modMatrixCopy = 0.0;
        for (int k = 0; k < NUMDIMS; ++k)
        {
            for (int kk = 0; kk < NUMDIMS; ++kk)
            {
                modMatrix += (*this)(k, kk) * (*this)(k, kk);
                modMatrixCopy += MatrixCopy(k, kk) * MatrixCopy(k, kk);
            }
        }
        MyAtLeastDouble sqdConditionNumber = modMatrix * modMatrixCopy / (MyAtLeastDouble)(NUMDIMS * NUMDIMS);

        return sqdConditionNumber;
    }
};

/**
 * @brief   Specialised child matrix class for rotation operations on vectors only.
 * @author  S. Heigl & D. A. Hubber, updated by F. Groth
 * @date    12/09/2018
 */
template <int ndim, typename T = MyFloat> class RotationMatrix : public SquareMatrix<ndim, T>
{
  protected:
    using Matrix<ndim, ndim, T>::data;

  public:
    /**
     * @brief default constructor.
     *
     * Initialize all entries to zero.
     */
    RotationMatrix() : SquareMatrix<ndim, T>(){};
    /**
     * @brief constructor.
     *
     * Initialize all entries with given value.
     */
    RotationMatrix(const T value) : SquareMatrix<ndim, T>(value){};
    /**
     * @brief copy constructor.
     */
    RotationMatrix(const SquareMatrix<ndim, T> &other) : SquareMatrix<ndim, T>(other){};

    /**
     * @brief matrix with fluid vector multiplication operator.
     */
    template <typename T1> FluidVector<ndim, T1> operator*(const FluidVector<ndim, T1> &fvOrig) const
    {
        FluidVector<ndim, T1> fv(fvOrig);
        for (int k = 0; k < ndim; k++)
        {
            fv[k] = T(0.0);
        }
        for (int i = 0; i < ndim; i++)
        {
            for (int j = 0; j < ndim; j++)
            {
                fv[i] += data[i][j] * fvOrig[j];
            }
        }
        return fv;
    }

    /**
     * @brief Return transpose of matrix.
     */
    inline RotationMatrix<ndim, T> GetTranspose() const
    {
        RotationMatrix<ndim, T> transpose;
        for (int i = 0; i < ndim; i++)
        {
            for (int j = 0; j < ndim; j++)
            {
                transpose(j, i) = data[i][j];
            }
        }
        return transpose;
    }

    /**
     * @brief Return inverse matrix.
     */
    inline RotationMatrix<ndim, T> GetInverseMatrix() const
    {
        return RotationMatrix<ndim, T>::GetTranspose();
    }

    /**
     * @brief Create rotation matrix to rotate in direction of given vector.
     */
    template <typename T2> inline static RotationMatrix<1, T> GetRotationMatrixFromVector(const Vector<1, T2> &v)
    {
        assert(std::isnormal(v.GetMagnitudeSquared()));
        RotationMatrix<1, T> mat;
        mat(0, 0) = v[0];
        return mat;
    }

    /**
     * @brief Create rotation matrix to rotate in direction of given vector.
     */
    template <typename T2> inline static RotationMatrix<2, T> GetRotationMatrixFromVector(const Vector<2, T2> &v)
    {
        assert(std::isnormal(v.GetMagnitudeSquared()));
        RotationMatrix<2, T> mat;
        mat(0, 0) = v[0];
        mat(0, 1) = -v[1];
        mat(1, 0) = v[1];
        mat(1, 1) = v[0];
        return mat;
    }

    /**
     * @brief Create rotation matrix to rotate in direction of given vector.
     */
    template <typename T2> inline static RotationMatrix<3, T> GetRotationMatrixFromVector(const Vector<3, T2> &v)
    {
        assert(std::isnormal(v.GetMagnitudeSquared()));
        RotationMatrix<3, T> mat;
        const T radphi =
            sqrt((MyAtLeastDouble)v[0] * (MyAtLeastDouble)v[0] + (MyAtLeastDouble)v[1] * (MyAtLeastDouble)v[1]);
        T cosphi;
        T sinphi;
        if (radphi == T(0.0))
        {
            cosphi = T(1.0);
            sinphi = T(0.0);
        }
        else
        {
            cosphi = v[0] / radphi;
            sinphi = v[1] / radphi;
        }
        mat(0, 0) = radphi * cosphi;
        mat(0, 1) = -sinphi;
        mat(0, 2) = -v[2] * cosphi;
        mat(1, 0) = radphi * sinphi;
        mat(1, 1) = cosphi;
        mat(1, 2) = -v[2] * sinphi;
        mat(2, 0) = v[2];
        mat(2, 1) = T(0.0);
        mat(2, 2) = radphi;
        return mat;
    }

    /**
     * @brief Create inverse rotation matrix to rotate in direction of given vector.
     */
    template <typename T2> inline static RotationMatrix<1, T> GetInverseRotationMatrixFromVector(const Vector<1, T2> &v)
    {
        assert(std::isnormal(v.GetMagnitudeSquared()));
        RotationMatrix<1, T> mat;
        mat(0, 0) = v[0];
        return mat;
    }

    /**
     * @brief Create inverse rotation matrix to rotate in direction of given vector.
     */
    template <typename T2> inline static RotationMatrix<2, T> GetInverseRotationMatrixFromVector(const Vector<2, T2> &v)
    {
        assert(std::isnormal(v.GetMagnitudeSquared()));
        RotationMatrix<2, T> mat;
        mat(0, 0) = v[0];
        mat(0, 1) = v[1];
        mat(1, 0) = -v[1];
        mat(1, 1) = v[0];
        return mat;
    }

    /**
     * @brief Create inverse rotation matrix to rotate in direction of given vector.
     */
    template <typename T2> inline static RotationMatrix<3, T> GetInverseRotationMatrixFromVector(const Vector<3, T2> &v)
    {
        assert(std::isnormal(v.GetMagnitudeSquared()));
        RotationMatrix<3, T> mat;
        const T radphi =
            sqrt((MyAtLeastDouble)v[0] * (MyAtLeastDouble)v[0] + (MyAtLeastDouble)v[1] * (MyAtLeastDouble)v[1]);
        T cosphi;
        T sinphi;
        if (radphi == T(0.0))
        {
            cosphi = T(1.0);
            sinphi = T(0.0);
        }
        else
        {
            cosphi = v[0] / radphi;
            sinphi = v[1] / radphi;
        }
        mat(0, 0) = radphi * cosphi;
        mat(0, 1) = radphi * sinphi;
        mat(0, 2) = v[2];
        mat(1, 0) = -sinphi;
        mat(1, 1) = cosphi;
        mat(1, 2) = T(0.0);
        mat(2, 0) = -v[2] * cosphi;
        mat(2, 1) = -v[2] * sinphi;
        mat(2, 2) = radphi;
        return mat;
    }
};

/**
 * @brief   Symmetric matrix class for storing data
 * @author  F. Groth
 * @date    02/2025
 */
template <int N, typename T = MyFloat>
class SymmetricMatrix // this is not a derived class, as we store the data in a different way!
{
  protected:
    T data[N * (N + 1) / 2]; ///< Matrix data, stored in compressed way to optimize memory
    // store the lower half of the matrix:
    // 0  X  X
    // 1  2  X
    // 3  4  5
    // N*(N+1)/2-1 X   ..  X   X
    // ...
    // N           N+1 .. 2N-2 X
    // 0           1   ..  N-2 N-1
    // The values indicated by X can be deduced from the lower half of the matrix.

  private:
    inline int get_index_smaller_first(int i, int j) const
    {
        return j * (j + 1) / 2 + i;
    }

    /**
     * @brief helper function to get the index.
     */
    inline int get_index(int i, int j) const
    {
        if (i <= j)
        {
            return get_index_smaller_first(i, j);
        }
        else
        {
            return get_index_smaller_first(j, i);
        }
    }

  public:
    /**
     * @brief default constructor.
     *
     * Initialize all entries to zero.
     */
    SymmetricMatrix()
    {
        for (int i = 0; i < N * (N + 1) / 2; ++i)
        {
            data[i] = T(0.0);
        }
    };

    /**
     * @brief constructor based on SquareMatrix.
     *
     * Check that SquareMatrix is indeed symmetric and then initialize.
     */
    template <typename T1> SymmetricMatrix(const SquareMatrix<N, T1> &other)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                // check that the other matrix is also a symmetric matrix:
                assert(other(i, j) == other(j, i));
                // now the value can be set:
                data[get_index_smaller_first(j, i)] = T(other(i, j));
            }
        }
    };

    /**
     * @brief constructor.
     *
     * Initialize all entries with given value.
     */
    template <typename T1> SymmetricMatrix(const T1 value)
    {
        for (int i = 0; i < N * (N + 1) / 2; ++i)
        {
            data[i] = T(value);
        }
    };

    /**
     * @brief static function to generate random matrix.
     */
    static inline SymmetricMatrix<N, T> GenerateRandomMatrix()
    {
        SymmetricMatrix<N, T> randMatrix;
        for (int i = 0; i < N * (N + 1) / 2; ++i)
        {
            randMatrix(i) = T(rand() % RAND_MAX) / T(RAND_MAX);
        }
        return randMatrix;
    }

    /**
     * @brief operator to set all entries to given value.
     */
    template <typename T1> SymmetricMatrix<N, T> &operator=(const T1 value)
    {
        assert(std::isfinite(value));
        for (int i = 0; i < N * (N + 1) / 2; ++i)
        {
            data[i] = T(value);
        }
        return *this;
    }

    // matrix multiplication operator no defined in this case as not required.
    // It would require to return a square matrix, as symmetric matrices do not generally commute.

    /**
     * @brief matrix with vector multiplication operator.
     */
    template <typename T2> Vector<N, T2> operator*(const Vector<N, T2> &vec) const
    {
        Vector<N, T2> result;
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                result[i] += data[get_index(i, j)] * vec[j];
            }
        }
        return result;
    }

    /**
     * @brief operator to set all entries to given value.
     */
    template <typename T1> SymmetricMatrix<N, T> &operator+=(const SymmetricMatrix<N, T1> &other)
    {
        for (int i = 0; i < N * (N + 1) / 2; ++i)
        {
            data[i] += T(other(i));
        }
        return *this;
    }

    /**
     * @brief operator to access element at given indices.
     */
    T &operator()(const int rindex, const int cindex)
    {
        assert(rindex >= 0 && rindex < N);
        assert(cindex >= 0 && cindex < N);
        return data[get_index(rindex, cindex)];
    }

    /**
     * @brief operator to access element at given indices.
     */
    const T &operator()(const int rindex, const int cindex) const
    {
        assert(rindex >= 0 && rindex < N);
        assert(cindex >= 0 && cindex < N);
        return data[get_index(rindex, cindex)];
    }

    /**
     * @brief operator to access element at given flattened (single) index.
     */
    T &operator()(const int index)
    {
        assert(index >= 0 && index < N * (N + 1) / 2);
        return data[index];
    }

    /**
     * @brief operator to access element at given flattened (single) index.
     */
    const T &operator()(const int index) const
    {
        assert(index >= 0 && index < N * (N + 1) / 2);
        return data[index];
    }

    /**
     * @brief Return determinant of SymmetricMatrix.
     */
    T GetDeterminant() const
    {
        if constexpr (N == 1)
        {
            return data[0];
        }
        else if constexpr (N == 2)
        {
            return data[0] * data[2] - data[1] * data[1];
        }
        else if constexpr (N == 3)
        {
            return data[0] * (data[2] * data[5] - data[4] * data[4]) -
                   data[1] * (data[1] * data[5] - 2 * data[4] * data[3]) +
                   data[3] * (/*first term moved inside previous bracket*/ -data[2] * data[3]);
        }
        return 0.0; // something went wrong in this case
    }

    /**
     * @brief Invert SymmetricMatrix for given determinant.
     */
    template <typename T1> void InvertMatrix(const T1 determinant)
    {
        T1 invdet;
        if (abs(determinant) > small_float)
        {
            invdet = T1(1.0) / determinant;
        }
        else
        {
            invdet = T(0.0);
#ifdef DEBUG
            VERBOSE_ALL(5, "Error!  Zero determinant for SymmetricMatrix (%g,%g)", determinant, small_float);
#endif // DEBUG
       // exit(0);
            assert(std::isnormal(determinant));
        }

        SymmetricMatrix<N, T> invMatrix;
        if constexpr (N == 1)
        {
            invMatrix(0) = invdet;
        }
        else if constexpr (N == 2)
        {
            invMatrix(0) = invdet * data[2];
            invMatrix(1 /*0,1*/) = -invdet * data[1];
            invMatrix(2 /*1,1*/) = invdet * data[0];
        }
        else if constexpr (N == 3)
        {
            invMatrix(0 /*0, 0*/) = (data[2] * data[5] - data[4] * data[4]) * invdet;
            invMatrix(1 /*0, 1*/) = (data[3] * data[4] - data[1] * data[5]) * invdet;
            invMatrix(2 /*1, 1*/) = (data[0] * data[5] - data[3] * data[3]) * invdet;
            invMatrix(3 /*0, 2*/) = (data[1] * data[4] - data[3] * data[2]) * invdet;
            invMatrix(4 /*1, 2*/) = (data[1] * data[3] - data[0] * data[4]) * invdet;
            invMatrix(5 /*2, 2*/) = (data[0] * data[2] - data[1] * data[1]) * invdet;
        }
        for (int i = 0; i < N * (N + 1) / 2; ++i)
        {
            data[i] = invMatrix(i);
        }
    }

    /**
     * @brief Invert SymmetricMatrix.
     */
    void InvertMatrix()
    {
        const T determinant = SymmetricMatrix<N, T>::GetDeterminant();

        InvertMatrix(determinant);
    }

    /**
     * @brief Return inverse matrix.
     */
    inline SymmetricMatrix<N, T> GetInverseMatrix() const
    {
        SymmetricMatrix<N, T> inverseMatrix(*this);
        inverseMatrix.InvertMatrix();
        return inverseMatrix;
    }

    /**
     * @brief Return trace (sum of diagonal entries).
     */
    inline T GetTrace() const
    {
        T trace = T(0.0);
        for (int i = 0; i < N; ++i)
        {
            trace += data[i * (i + 1) / 2];
        }
        return trace;
    }

    /**
     * @brief Return transpose of matrix, which in this case equals the matrix itself.
     */
    inline SymmetricMatrix<N, T> GetTranspose() const
    {
        return *this;
    }

    /**
     * @brief Return true if condition number is reasonable, invert Matrix in the process.
     */
    template <typename T1> bool CheckConditionNumberAndInvertMatrix(const T1 determinant)
    {
        MyAtLeastDouble sqdConditionNumber = GetConditionNumberAndInvertMatrix(determinant);

        return (sqdConditionNumber < 1.0e4 && abs(sqdConditionNumber) != 0);
    }

    /**
     * @brief Return condition number, invert Matrix in the process.
     */
    template <typename T1> MyAtLeastDouble GetConditionNumberAndInvertMatrix(const T1 determinant)
    {
        // save the original matrix for checking the accuracy later
        SymmetricMatrix MatrixCopy(*this);

        // The Matrix inversion checks if the determinant is a normal number and not too small
        // and set it to zero in case
        this->InvertMatrix(determinant);

        // check the accuracy of the integral gradients using the square condition number
        // as for MFM
        MyAtLeastDouble modMatrix = 0.0;
        MyAtLeastDouble modMatrixCopy = 0.0;
        for (int k = 0; k < NUMDIMS; ++k)
        {
            // on-center elements are counted only once
            modMatrix += (*this)(get_index_smaller_first(k, k)) * (*this)(get_index_smaller_first(k, k));
            modMatrixCopy += MatrixCopy(get_index_smaller_first(k, k)) * MatrixCopy(get_index_smaller_first(k, k));
            for (int kk = 0; kk < k; ++kk)
            {
                // off-diagonal elements are counted twice (lower and upper half of the matrix)
                modMatrix += 2 * (*this)(get_index_smaller_first(kk, k)) * (*this)(get_index_smaller_first(kk, k));
                modMatrixCopy +=
                    2 * MatrixCopy(get_index_smaller_first(kk, k)) * MatrixCopy(get_index_smaller_first(kk, k));
            }
        }
        MyAtLeastDouble sqdConditionNumber = modMatrix * modMatrixCopy / (MyAtLeastDouble)(NUMDIMS * NUMDIMS);

        return sqdConditionNumber;
    }
#ifdef ACC
    T *get_raw()
    {
        return data;
    }
#endif
};

#ifdef ACC
#define SYMMETRIC_GET_INDEX_SMALLER_FIRST(i, j) j *(j + 1) / 2 + i
#define SYMMETRIC_INDEX(i, j)                                                                                          \
    ((i <= j) ? SYMMETRIC_GET_INDEX_SMALLER_FIRST(i, j) : SYMMETRIC_GET_INDEX_SMALLER_FIRST(j, i))
#endif

} // namespace OpenGadget3

//-------------------------------------------------------------------------------------------------
#endif // _OPENGADGET3_MATRIX_HPP_
