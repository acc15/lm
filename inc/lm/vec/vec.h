#pragma once

#include <lm/vec/generic_vec.h>

namespace lm {


template <typename Element, size_t Size, typename Impl>
class vec_storage_base {
public:

    typedef Impl impl_type;
    typedef Element value_type;
    typedef random_access_iterator<impl_type, value_type> iterator;
    typedef random_access_iterator<const impl_type, const value_type> const_iterator;
    typedef random_access_iterator<impl_type, value_type, false> reverse_iterator;
    typedef random_access_iterator<const impl_type, const value_type, false> const_reverse_iterator;

    constexpr size_t size() const {
        return Size;
    }

    const value_type& operator[](size_t idx) const {
        return const_cast<impl_type&>(impl()).at(idx);
    }

    value_type& operator [](size_t idx) {
        return impl().at(idx);
    }

    constexpr bool empty() const {
        return Size == 0;
    }

    iterator begin() {
        return iterator(impl(), 0);
    }

    iterator end() {
        return iterator(impl(), size());
    }

    const_iterator cbegin() const {
        return const_iterator(impl(), 0);
    }

    const_iterator cend() const {
        return const_iterator(impl(), size());
    }

    const_iterator begin() const {
        return const_iterator(impl(), 0);
    }

    const_iterator end() const {
        return const_iterator(impl(), size());
    }

    reverse_iterator rbegin() {
        return reverse_iterator(impl(), size() - 1);
    }

    reverse_iterator rend() {
        return reverse_iterator(impl(), reverse_iterator::REND);
    }

    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(impl(), size() - 1);
    }

    const_reverse_iterator rend() const {
        return const_reverse_iterator(impl(), const_reverse_iterator::REND);
    }

    const_reverse_iterator crbegin() const {
        return const_reverse_iterator(impl(), size() - 1);
    }

    const_reverse_iterator crend() const {
        return const_reverse_iterator(impl(), const_reverse_iterator::REND);
    }

private:
    impl_type& impl() {
        return *static_cast<impl_type*>(this);
    }

    const impl_type& impl() const {
        return *static_cast<const impl_type*>(this);
    }


};

template <typename T, size_t Size>
struct vec_storage: public std::array<T, Size> {
};

template <typename T>
struct vec_storage<T, 1>: public vec_storage_base<T, 1, vec_storage<T, 1>> {

    T x;

    T& at(size_t idx) {
        switch (idx) {
        case 0: return x;
        default: throw std::out_of_range("vec index out of range");
        }
    }

};

template <typename T>
struct vec_storage<T, 2>: public vec_storage_base<T, 2, vec_storage<T, 2>> {

    T x;
    T y;

    T& at(size_t idx) {
        switch (idx) {
        case 0: return x;
        case 1: return y;
        default: throw std::out_of_range("vec index out of range");
        }
    }
};

template <typename T>
struct vec_storage<T, 3>: public vec_storage_base<T, 3, vec_storage<T, 3>> {

    T x;
    T y;
    T z;

    T& at(size_t idx) {
        switch (idx) {
        case 0: return x;
        case 1: return y;
        case 2: return z;
        default: throw std::out_of_range("vec index out of range");
        }
    }

};

template <typename T>
struct vec_storage<T, 4>: public vec_storage_base<T, 4, vec_storage<T, 4>> {

    T x;
    T y;
    T z;
    T w;

    T& at(size_t idx) {
        switch (idx) {
        case 0: return x;
        case 1: return y;
        case 2: return z;
        case 3: return w;
        default: throw std::out_of_range("vec index out of range");
        }
    }

};

template <typename Element, size_t Size>
class vec: public generic_vec<vec<Element, Size>, vec_storage<Element, Size>> {
public:
    using generic_vec<vec<Element, Size>, vec_storage<Element, Size>>::generic_vec;
};

}
