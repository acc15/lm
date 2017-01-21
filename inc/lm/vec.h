#pragma once

#include <lm/generic_vec.h>

namespace lm {


template <size_t Size, typename Element, typename Impl>
class vec_storage_base {
public:

    typedef Impl impl_type;
    typedef Element value_type;
    typedef random_access_iterator<impl_type, value_type> iterator;
    typedef random_access_iterator<const impl_type, const value_type> const_iterator;
    typedef random_access_iterator<impl_type, value_type, false> reverse_iterator;
    typedef random_access_iterator<const impl_type, const value_type, false> const_reverse_iterator;

    size_t size() const {
        return Size;
    }

    const value_type& operator[](size_t idx) const {
        return const_cast<impl_type&>(impl()).at(idx);
    }

    value_type& operator [](size_t idx) {
        return impl().at(idx);
    }

    bool empty() const {
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

template <size_t Size, typename T>
struct vec_storage: public std::array<T, Size> {
};

template <typename T>
struct vec_storage<1, T>: public vec_storage_base<1, T, vec_storage<1, T>> {

    T x;

    T& at(size_t idx) {
        switch (idx) {
        case 0: return x;
        default: throw std::out_of_range("vec index out of range");
        }
    }

};

template <typename T>
struct vec_storage<2, T>: public vec_storage_base<2, T, vec_storage<2, T>> {

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
struct vec_storage<3, T>: public vec_storage_base<3, T, vec_storage<3, T>> {

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
struct vec_storage<4, T>: public vec_storage_base<4, T, vec_storage<4, T>> {

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

template <size_t Size, typename Element>
class vec: public generic_vec<vec<Size, Element>, vec_storage<Size, Element>> {
public:
    using generic_vec<vec<Size, Element>, vec_storage<Size, Element>>::generic_vec;
};

}
