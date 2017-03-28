/**
 * @file
 * @brief Generic matrix algorithms
 */

#pragma once

#include <cstddef>
#include <algorithm>

#include <lm/util/assert.h>
#include <lm/matrix/type_util.h>
#include <lm/matrix/traits.h>
#include <lm/matrix/permutation.h>

//! lm namespace
namespace lm {

/**
 * @brief Swap two rows `r1` and `r2` in matrix `m`.
 *
 * For example, if 3x3 matrix `m` is:
 *
 *  @f[
 *  \begin{pmatrix}
 *      \textbf{a} & \textbf{b} & \textbf{c} \\
 *      d & e & f \\
 *      \textbf{g} & \textbf{h} & \textbf{i}
 *  \end{pmatrix}
 *  @f]
 *
 * and `r1 == 0`, `r2 == 2` then after execution matrix `m` becomes:
 *
 *  @f[
 *  \begin{pmatrix}
 *      \textbf{g} & \textbf{h} & \textbf{i} \\
 *      d & e & f \\
 *      \textbf{a} & \textbf{b} & \textbf{c}
 *  \end{pmatrix}
 *  @f]
 *
 * @tparam M matrix type
 * @param m matrix in which rows must be swapped
 * @param r1 first row index
 * @param r2 second row index
 */
template <typename M>
void swap_row(M& m, size_t r1, size_t r2) {
    if (r1 == r2) {
        return;
    }
    for (size_t i = 0; i < m.cols(); i++) {
        std::swap(m.at(r1, i), m.at(r2, i));
    }
}

/**
 * @brief Swap two columns `c1` and `c2` in matrix `m`.
 *
 * For example, if 3x3 matrix `m` is:
 *
 *  @f[
 *  \begin{pmatrix}
 *      \textbf{a} & b & \textbf{c} \\
 *      \textbf{d} & e & \textbf{f} \\
 *      \textbf{g} & h & \textbf{i}
 *  \end{pmatrix}
 *  @f]
 *
 * and `c1 == 0`, `c2 == 2` then after execution matrix `m` becomes:
 *
 *  @f[
 *  \begin{pmatrix}
 *      \textbf{c} & b & \textbf{a} \\
 *      \textbf{f} & e & \textbf{d} \\
 *      \textbf{i} & h & \textbf{g}
 *  \end{pmatrix}
 *  @f]
 *
 * @tparam M matrix type
 * @param m matrix in which columns must be swapped
 * @param c1 first column index
 * @param c2 second column index
 */
template <typename M>
void swap_col(M& m, size_t c1, size_t c2) {
    if (c1 == c2) {
        return;
    }
    for (size_t i = 0; i < m.rows(); i++) {
        std::swap(m.at(i, c1), m.at(i, c2));
    }
}

/**
 * @brief Stores transposed matrix.
 *
 * Computes and stores transposed matrix `m` in `result` matrix, i.e.
 * @f[
 *  result = M^\top
 * @f]
 *
 * By default if type of matrix `m` is static then `result` is also static with swapped row and column counts.
 * However its possible to specify type `P` explicitly with dynamic, or bigger-sized static matrix.
 *
 * @tparam M matrix type
 * @tparam P transposed matrix type
 * @param m matrix to transpose
 * @param result transposed matrix
 */
template <typename M, typename P = typename matrix_transpose<M>::value_matrix_type>
void transpose(const M& m, P& result) {
    result.resize(m.cols(), m.rows());
    for (size_t i = 0; i < result.rows(); i++) {
        for (size_t j = 0; j < result.cols(); j++) {
            result(i, j) = m(j, i);
        }
    }
}

/**
 * @brief Computes transposed matrix.
 *
 * Returns transposed matrix `m`, i.e.
 * @f$ M^\top @f$
 *
 * By default if type of matrix `m` is static then `result` is also static with swapped row and column counts.
 * However its possible to specify type `P` explicitly with dynamic, or bigger-sized static matrix.
 *
 * @tparam M matrix type
 * @tparam P transposed matrix type
 * @param m matrix to transpose
 * @return transposed matrix
 */
template <typename M, typename P = typename matrix_transpose<M>::value_matrix_type>
P transpose(const M& m) {
    P result;
    transpose<M, P>(m, result);
    return result;
}

/**
 * @brief Computes product of matrix `m` and `n` and stores result in `result` matrix.
 *
 * Matrix multiplication produces a matrix of @f$ m.rows() \times n.cols() @f$ with an expectation that @f$ m.cols() = n.rows() @f$.
 *
 * By default if type of matrix `m` and `n` is static then `result` matrix is also static (actual type is same as matrix `m`)
 * with dimensions `m.rows()` and `n.cols()` respectively.
 * If matrixes are static and can't be multiplied (i.e. `m.cols() != n.rows()`) then compilation error is generated.
 *
 * Also its possible to specify type `P` explicitly with dynamic, or bigger-sized static matrix
 * (in this case no static checks are performed).
 *
 * @tparam M first matrix type
 * @tparam N second matrix type
 * @tparam P product matrix type
 * @param m first matrix
 * @param n second matrix
 * @param result matrix to store product of m*n
 */
template <typename M, typename N, typename P = typename matrix_product<M, N>::value_matrix_type>
void product(const M& m, const N& n, P& result) {
    lm_assert(m.cols() == n.rows(), m.cols() << " must be equal to " << n.rows() );

    result.resize(m.rows(), n.cols());

    for (size_t i = 0; i < result.rows(); i++) {
        for (size_t j = 0; j < result.cols(); j++) {
            typename M::value_type sum = 0;
            for (size_t k = 0; k < n.rows(); k++) {
                sum += m(i, k) * n(k, j);
            }
            result(i, j) = sum;
        }
    }
}


/**
 * @brief Computes and returns product of matrix `m` and `n`.
 *
 * Matrix multiplication produces a matrix of @f$ m.rows() \times n.cols() @f$ with an expectation that @f$ m.cols() = n.rows() @f$.
 *
 * By default if type of matrix `m` and `n` is static then `result` matrix is also static (actual type is same as matrix `m`).
 * with dimensions `m.rows()` and `n.cols()` respectively.
 * If both matricies are static and can't be multiplied (i.e. `m.cols() != n.rows()`) then compilation error is generated.
 *
 * Also its possible to specify type `P` explicitly with dynamic, or bigger-sized static matrix
 * (in this case no static checks are performed).
 *
 * @tparam M first matrix type
 * @tparam N second matrix type
 * @tparam P product matrix type
 * @param m first matrix
 * @param n second matrix
 * @return product of m*n
 */
template <typename M, typename N, typename P = typename matrix_product<M, N>::value_matrix_type>
P product(const M& m, const N& n) {
    P result;
    lm::product<M, N, P>(m, n, result);
    return result;
}

/**
 * @brief Finds the best pivoting row for more stable LU-factorization results.
 *
 * @param m matrix in which pivoting row must be found
 * @param n row index to begin search
 * @return index of best pivot row and a `bool` flag which indicates status of operation.
 *   `true` means success and pivoting row is found
 *   `false` means there is no suitable rows to perform LU-factorization and in general - that matrix `m` is singular.
 */
template <typename M>
std::pair<size_t, bool> find_lu_pivot(M& m, const size_t n) {

    size_t l = m.rows();

    std::pair<size_t, bool> result(0, false);
    for (size_t i = n; i < l; i++) {

        const typename M::value_type v = m(i, n);
        if (v == 0) {
            continue;
        }

        if (!result.second) {
            result.first = i;
            result.second = true;
            continue;
        }

        if (std::abs(v) < std::abs(m(result.first, n))) {
            result.first = i;
        }

    }
    return result;

}


/**
 * @brief Performs LU-factorization with row pivoting of a given matrix `m`.
 *
 * LU factorization is allowed only for square (`rows() == cols()`), non-singular (@f$ \det m \neq 0 @f$) matricies.
 *
 * Note that even if LU-factorization will fail (i.e. return `false`) the output matrix `m` may still be modified, so
 * its unsafe to perform some tests on original (non-copied) matricies.
 *
 * To save a bit of memory, lower matrix is stored below main diagonal, upper matrix - on and above main diagonal.
 *
 * For example a result will be stored as follows:
 *
 * @f[
 * \begin{pmatrix}
 * U_{1,1} & U_{1,2} & \cdots & U_{1,n-1} & U_{1,n} \\
 * L_{2,1} & U_{2,2} & \cdots & U_{2,n-1} & U_{2,n} \\
 * \vdots & \vdots & \ddots & \vdots & \vdots \\
 * L_{n-1,1} & L_{n-1,2} & \cdots & U_{n-1,n-1} & U_{n-1,n} \\
 * L_{n,1} & L_{n,2} & \cdots & L_{n,n-1} & U_{n,n} \\
 * \end{pmatrix}
 * @f]
 *
 * @param m matrix to perform LU-factorization
 * @return `true` if LU-factorization succeds, `false` if matrix is singular and LU-factorization can't be performed
 *
 */
template <typename M>
bool lu_decomposition(M& m) {

    const size_t l = m.rows();
    if (l != m.cols()) {
        throw std::invalid_argument("lu_decomposition(..) available only for square matricies");
    }

    for (size_t i = 0; i < l - 1; i++) {
        std::pair<size_t, bool> pivot = find_lu_pivot(m, i);
        if (!pivot.second) {
            return false;
        }
        m.swap_row(pivot.first, i);
        for (size_t k = i + 1; k < l; k++) {
            for (size_t j = i + 1; j < l; j++) {
                m(k, j) -= m(i, j) * m(k, i) / m(i, i);
            }
            m(k, i) /= m(i, i);
        }
    }
    return m(l - 1, l - 1) != 0;
}


template <typename M, typename R>
bool lu_substitute(const M& lu, R& r) {
    typedef typename M::value_type value_type;
    for (size_t j = 0; j < lu.cols(); j++) {
        if (lu(j, j) == 0) {
            return false;
        }
        for (size_t i = 1; i < lu.rows();i++) {
            for (size_t k = 0; k < i; k++) {
                r(i, j) -= lu(i, k) * r(k, j);
            }
        }
        for (size_t i = lu.rows() - 1; i != static_cast<size_t>(-1); i--) {
            for (size_t k = lu.cols() - 1; k > i; k--) {
                r(i, j) -= lu(i, k) * r(k, j);
            }
            r(i, j) /= lu(i, i);
        }
    }
    return true;
}


/**
 * @brief Make matrix `m` an identity matrix.
 * @param m output matrix
 */
template <typename M>
void make_identity(M& m) {
    for (size_t i = 0; i < m.rows(); i++) {
        for (size_t j = 0; j < m.cols(); j++) {
            m(i, j) = static_cast<typename M::value_type>(i == j ? 1 : 0);
        }
    }
}

/**
 * @brief Computes inversion matrix of `m` and stores result in matrix `r`.
 *
 * Inversion of matrix `m` is such matrix for which following condition apply:
 *
 * @f$ M^{-1} * M = M * M^{-1} = I @f$
 * where @f$ M^{-1} @f$ is an inverted matrix and `I` is identity matrix.
 *
 * Matrix inversion is allowed only for square (`rows() == cols()`), non-singular (@f$ \det m \neq 0 @f$) matricies.
 *
 *
 * @param m matrix to invert
 * @param r matrix to store inverted matrix
 * @return `true` if inversion succeds, `false` if matrix is singular and inverted matrix can't be computed.
 */
template <typename M, typename R>
bool invert_matrix(const M& m, R& r) {
    permutation_matrix<M> lu(m);
    if (!lu_decomposition(lu)) {
        return false;
    }
    for (size_t i = 0; i < r.rows(); i++) {
        for (size_t j = 0; j < r.cols(); j++) {
            r(i, lu.permutation_vec()[j]) = static_cast<typename M::value_type>(i == j ? 1 : 0);
        }
    }
    return lu_substitute(lu, r);
}

/**
 * @brief Computes determinant by the given LU-factorized matrix and row permutation count.
 *
 * Determinant of matrix which has all zeroes below main diagonal is a multiplication of its elements on main diagonal, i.e.:
 *
 * @f$ \det LU = (-1)^P \times \left(\displaystyle \prod_{i=1}^n m_{i,i}\right) @f$
 *
 * where `P` is count of performed row swaps.
 *
 * @param m LU-factorized matrix.
 * @param permutation_count count of row swaps performed in LU-factorization process.
 * @return determinant of factorized matrix `lu`
 */
template <typename M>
typename M::value_type lu_determinant(const M& lu, size_t permutation_count) {
    typename M::value_type det = static_cast<typename M::value_type>((permutation_count & 1) == 0 ? 1 : -1);
    for (size_t i = 0; i < std::min(lu.rows(), lu.cols()); i++) {
        det *= lu(i, i);
    }
    return det;
}

/**
 * @brief Computes determinant of a given matrix `m`.
 * @param m matrix to compute determinant
 * @return determinant of matrix `m`
 */
template <typename M>
typename M::value_type determinant(const M& m) {
    permutation_matrix<M> lu(m);
    if (!lu_decomposition(lu)) {
        return 0;
    }
    return lu_determinant(lu, lu.permutation_count());
}


}
