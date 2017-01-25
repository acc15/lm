#pragma once

#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <type_traits>

#include <lm/algorithm.h>
#include <lm/matrix/algorithm.h>
#include <lm/matrix/layout.h>
#include <lm/matrix/traits.h>

#include <lm/matrix/static.h>
#include <lm/matrix/dynamic.h>
#include <lm/matrix/transpose.h>
#include <lm/matrix/permutation.h>

namespace lm {

template <typename S>
class matrix: public S {
public:

    typedef typename S::value_type value_type;
    typedef typename S::value_matrix_type value_matrix_type;
    typedef typename S::reference_matrix_type reference_matrix_type;
    typedef matrix<S> matrix_type;

    using S::S;

    value_type& cell(size_t row, size_t col) {
        return S::at(row, col);
    }

    const value_type& cell(size_t row, size_t col) const {
        return const_cast<matrix_type*>(this)->at(row, col);
    }

    value_type& operator()(size_t row, size_t col) {
        return cell(row, col);
    }

    const value_type& operator()(size_t row, size_t col) const {
        return cell(row, col);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& assign(const T& other) {
        S::resize(Traits::rows(other), Traits::cols(other));
        apply<return_2nd, T, Traits>(other, return_2nd());
        return *this;
    }

    template <typename F, typename T, typename Traits = matrix_traits<T>>
    matrix_type& apply(const T& other, F func) {
        for (size_t i = 0; i < S::rows(); i++) {
            for (size_t j = 0; j < S::cols(); j++) {
                cell(i, j) = func(cell(i, j), Traits::cell(other, i, j));
            }
        }
        return *this;
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& add(const T& other) {
        return apply<std::plus<void>, T, Traits>(other, std::plus<void>());
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& subtract(const T& other) {
        return apply<std::minus<void>, T, Traits>(other, std::minus<void>());
    }

    template <typename T, typename Traits = matrix_traits<T>>
    bool equal(const T& other) {
        if (S::rows() != Traits::rows(other) || S::cols() != Traits::cols(other)) {
            return false;
        }
        for (size_t i = 0; i < S::rows(); i++) {
            for (size_t j = 0; j < S::cols(); j++) {
                if (cell(i, j) != Traits::cell(other, i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T, typename Traits = matrix_traits<T>, typename P = typename matrix_product<value_matrix_type, T, Traits>::value_matrix_type>
    void compute_product(const T& other, P& p) const {
        lm::product<matrix_type, T, Traits, P>(*this, other, p);
    }

    template <typename T, typename Traits = matrix_traits<T>, typename P = typename matrix_product<value_matrix_type, T, Traits>::value_matrix_type>
    P compute_product(const T& other) const {
        return lm::product<matrix_type, T, Traits, P>(*this, other);
    }

    template <typename T, typename Traits = matrix_traits<T>,
              typename P = typename matrix_product<matrix_type, T>::value_matrix_type,
              typename = typename std::enable_if<std::is_same<value_matrix_type, P>::value>::type>
    matrix_type& product(const T& other) {
        P p;
        compute_product<T, Traits, P>(other, p);
        return assign(p);
    }

    template <typename P = typename matrix_transpose<value_matrix_type>::value_matrix_type>
    void compute_transposed(P& p) const {
        return lm::transpose<value_matrix_type, P>(*this, p);
    }

    template <typename P = typename matrix_transpose<value_matrix_type>::value_matrix_type>
    P compute_transposed() const {
        return lm::transpose<value_matrix_type, P>(*this);
    }

    template <typename P = typename matrix_transpose<value_matrix_type>::value_matrix_type,
               typename = typename std::enable_if<std::is_same<value_matrix_type, P>::value>::type>
    matrix_type& transpose() {
        P p;
        compute_transposed(p);
        return assign(p);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& operator=(const T& other) {
        return assign<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& operator+=(const T& other) {
        return add<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    matrix_type& operator-=(const T& other) {
        return subtract<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    value_matrix_type operator+(const T& other) {
        return value_matrix_type(*this) += other;
    }

    template <typename T, typename Traits = matrix_traits<T>>
    value_matrix_type operator-(const T& other) {
        return value_matrix_type(*this) -= other;
    }

    template <typename T, typename Traits = matrix_traits<T>, typename P = typename matrix_product<value_matrix_type, T, Traits>::matrix_type>
    P operator*(const T& other) {
        return compute_product<T, Traits, P>(other);
    }

    template <typename T,
              typename Traits = matrix_traits<T>,
              typename P = typename matrix_product<value_matrix_type, T, Traits>::value_matrix_type,
              typename = typename std::enable_if<std::is_same<value_matrix_type, P>::value>::type>
    matrix_type& operator*=(const T& other) {
        return product<T, Traits, P>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    bool operator==(const T& other) {
        return equal<T, Traits>(other);
    }

    template <typename T, typename Traits = matrix_traits<T>>
    bool operator!=(const T& other) {
        return !equal<T, Traits>(other);
    }

    value_type determinant() const {
        return lm::determinant(*this);
    }

};



}
