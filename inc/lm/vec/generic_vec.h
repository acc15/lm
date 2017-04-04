#pragma once

#include <array>
#include <algorithm>
#include <functional>
#include <cmath>

#include <lm/util/functional.h>
#include <lm/util/range.h>
#include <lm/util/random_access_iterator.h>

namespace lm {

template <typename Impl, typename Base>
class generic_vec: public Base {
public:

    typedef Impl vec_type;
    typedef Base base_type;

    typedef typename base_type::value_type value_type;

    constexpr generic_vec() : base_type() {
    }

    template <typename Other>
    generic_vec(const Other& val) : base_type() {
        operator=(val);
    }

    template <typename Val>
    generic_vec(const std::initializer_list<Val>& val) : base_type() {
        operator=(val);
    }

    /**
     * @brief Computes squared vector length.
     *
     * This computes sum of squared elements:
     *  @f$\sum_{i=1}^n E_i^2@f$
     *
     * For example if `size()` is 2 (2d vector) this is same as:
     *  @f$x^2 + y^2@f$
     *
     * @sa length()
     * @return squared vector length
     */
    value_type length_square() const {
        value_type result = value_type();
        for (size_t i = 0; i < base_type::size(); i++) {
            const value_type& val = as_vec()[i];
            result += val * val;
        }
        return result;
    }

    /**
     * @brief Computes vector length.
     *
     * This computes square root of squared element sum:
     *  @f$\sqrt{\sum_{i=1}^n E_i^2}@f$
     *
     * For example if `size()` is 2 (2d vector) this is same as:
     *  @f$\sqrt{x^2 + y^2}@f$
     *
     * @sa length_square()
     * @return vector length
     */
    value_type length() const {
        return static_cast<value_type>(sqrt(length_square()));
    }


    /**
     * @brief Computes scalar product of two vectors.
     *
     * This computes products sum of elements:
     *  @f$\sum_{i=1}^{size()} (E_{this} * E_{other})@f$
     *
     * For example if `size()` is 2 (2d vectors) this is same as:
     *  @f$x_1 * x_2 + y_1 * y_2@f$
     *
     * @tparam Vec vector type
     * @param other product vector
     * @return scalar product of `this` and `other` vectors
     */
    template <typename Vec>
    value_type scalar_product(const Vec& other) {

		auto r = range(other, base_type::size());
		auto b = base_type::begin(), e = base_type::end();
		auto rb = r.begin(), re = r.end();

        value_type product = value_type();
        while (b != e && rb != re) {
            product += (*b) * (*rb);
            ++b;
            ++rb;
        }
        return product;
    }

    /**
     * @brief Assigns `this` vector to `other`.
     *
     * Note that if `other` vector is shorter than `this` then partial assign is performed.
     * For example if `this` vector has `size() == 4` and `other.size() == 2`
     * then only first 2 elements will be assigned - and the others 2 remains unchanged.
     *
     * @tparam Other type of vector to assign
     * @param other vector to assign
     * @return reference to `this` vector
     */
    template <typename Other>
    vec_type& operator=(const Other& other) {
        return transform(other, return_2nd());
    }

    /**
     * @brief Tests two vectors for equality.
     *
     * Vectors are considered equal if all following conditions apply:
     *  - vectors have same `size()`
     *  - all elements (comparing by element `operator==`) are equal
     *
     * @sa operator!=()
     * @tparam Other type of vector to test for equality
     * @param other vector to test for equality
     */
    template <typename Other>
    bool operator==(const Other& other) const {
        auto r = range(other, base_type::size());
        if (base_type::size() != r.size()) {
            return false;
        }
        return std::equal(base_type::begin(), base_type::end(), r.begin());
    }

    /**
     * @brief Tests two vectors for inequality.
     *
     * Vectors are considered inequal if at least one following conditions apply:
     *  - vectors have different `size()`
     *  - any `this` element is not equal to `other` element (comparing by element `operator==`)
     *
     * This is inversion of operator==().
     *
     * @see operator==()
     * @tparam Other type of vector to test for equality
     * @param other vector to test for equality
     */
    template <typename Other>
    bool operator!=(const Other& other) const {
        return !operator==(other);
    }

    /**
     * @brief Negates `this` vectors.
     *
     * Negates all elements of `this` vectors. This is the same as:
     *
     *  @f${*this} = \begin{pmatrix} -a_1 & -a_2 & -a_3 & \cdots & -a_n \end{pmatrix}@f$
     *
     * @sa operator-()
     * @return reference to `this` vector
     */
    vec_type& negate() {
        std::transform(base_type::begin(), base_type::end(), base_type::begin(), std::negate<void>());
        return as_vec();
    }

    /**
     * @brief Computes negated vector.
     *
     * Returns vector with all elements negated:
     *  @f$\begin{pmatrix} -a_1 & -a_2 & -a_3 & \cdots & -a_n \end{pmatrix}@f$
     *
     * @see negate()
     * @return negated vector
     */
    vec_type operator-() const {
        return vec_type(as_vec()).negate();
    }

    /**
     * @brief Adds `other` vector to `this`.
     *
     * Adds all elements of `other` vector to this.
     * If `other` vector is shorter then `this` then only `other.size()`
     * elements will be changed.
     *
     * @tparam Other vector type to add (its may be also a primitive such as `int`, `long`, `double`, etc.)
     * @param other vector to add
     * @return `*this`
     */
    template <typename Other>
    vec_type& operator+=(const Other& other) {
        return transform(other, std::plus<void>());
    }

    /**
     * @brief Subtract `other` vector from `this`.
     *
     * Subtracts all elements of `other` vector from this.
     * If `other` vector is shorter then `this` then only `other.size()`
     * elements will be changed.
     *
     * @tparam Other vector type to subtract (its may be also a primitive such as `int`, `long`, `double`, etc.)
     * @param other vector to subtract
     * @return `*this`
     */
    template <typename Other>
    vec_type& operator-=(const Other& other) {
        return transform(other, std::minus<void>());
    }

    /**
     * @brief Multiplies `this` vector on `other`.
     *
     * Multiplies each element of `this` on corresponding element of `other` vector.
     * If `other` vector is shorter then `this` then only `other.size()`
     * elements will be changed.
     *
     * @tparam Other multiplier vector type (its may be also a primitive such as `int`, `long`, `double`, etc.)
     * @param other multiplier vector
     * @return `*this`
     */
    template <typename Other>
    vec_type& operator*=(const Other& other) {
        return transform(other, std::multiplies<void>());
    }

    /**
     * @brief Divides `this` vector on `other`.
     *
     * Divides each element of `this` vector on corresponding element of `other` vector.
     * If at least one element of `other` vector is `zero` then behavior is undefined
     * (i.e. its platform-specific, for example - linux x64 it will be SIGFPE).
     * If `other` vector is shorter then `this` then only `other.size()`
     * elements will be changed.
     *
     * @tparam Other divider vector type (its may be also a primitive such as `int`, `long`, `double`, etc.)
     * @param other divider vector
     * @return `*this`
     */
    template <typename Other>
    vec_type& operator/=(const Other& other) {
        return transform(other, std::divides<void>());
    }

    template <typename Other>
    vec_type operator+(const Other& other) const {
        return vec_type(as_vec()) += other;
    }

    template <typename Other>
    vec_type operator-(const Other& other) const {
        return vec_type(as_vec()) -= other;
    }

    template <typename Other>
    vec_type operator*(const Other& other) const {
        return vec_type(as_vec()) *= other;
    }

    template <typename Other>
    vec_type operator/(const Other& other) const {
        return vec_type(as_vec()) /= other;
    }

protected:

    template <typename Other, typename Func>
    vec_type& transform(const Other& other, Func func) {
        typename base_type::iterator i = base_type::begin();
        auto r = range(other, base_type::size());
        lm::transform(i, base_type::end(), r.begin(), r.end(), i, func);
        return as_vec();
    }

private:

    const vec_type& as_vec() const {
        return *static_cast<const vec_type*>(this);
    }

    vec_type& as_vec() {
        return *static_cast<vec_type*>(this);
    }

};


}
