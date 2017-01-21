#pragma once

#include <array>
#include <algorithm>
#include <functional>
#include <cmath>

#include <lm/algorithm.h>
#include <lm/range.h>
#include <lm/random_access_iterator.h>

namespace lm {

template <typename Impl, typename Base>
class generic_vec: public Base {
public:

    typedef Impl vec_type;
    typedef Base base_type;

    typedef typename base_type::value_type value_type;

    generic_vec() : base_type() {
    }

    template <typename Other>
    generic_vec(const Other& val) : base_type() {
        operator=(val);
    }

    template <typename Val>
    generic_vec(const std::initializer_list<Val>& val) : base_type() {
        operator=(val);
    }

    value_type length_square() const {
        value_type result = value_type();
        for (size_t i = 0; i < base_type::size(); i++) {
            const value_type& val = as_vec()[i];
            result += val * val;
        }
        return result;
    }

    value_type length() const {
        return static_cast<value_type>(sqrt(length_square()));
    }

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

    template <typename Other>
    vec_type& operator=(const Other& other) {
        return transform(other, return_2nd());
    }

    template <typename Other>
    bool operator==(const Other& other) const {
        auto r = range(other, base_type::size());
        if (base_type::size() != r.size()) {
            return false;
        }
        return std::equal(base_type::begin(), base_type::end(), r.begin());
    }

    template <typename Other>
    bool operator!=(const Other& other) const {
        return !operator==(other);
    }

    vec_type& negate() {
        std::transform(base_type::begin(), base_type::end(), base_type::begin(), std::negate<void>());
        return as_vec();
    }

    vec_type operator-() const {
        return vec_type(as_vec()).negate();
    }

    template <typename Other>
    vec_type& operator+=(const Other& other) {
        return transform(other, std::plus<void>());
    }

    template <typename Other>
    vec_type& operator-=(const Other& other) {
        return transform(other, std::minus<void>());
    }

    template <typename Other>
    vec_type& operator*=(const Other& other) {
        return transform(other, std::multiplies<void>());
    }

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
        auto i = base_type::begin();
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
