#pragma once

#include <type_traits>

#include <lm/random_access_iterator.h>

namespace lm {

template <typename T>
class ptr_range {
public:
    ptr_range(T ptr, size_t count): _ptr(ptr), _count(count) {
    }

    T begin() const {
        return _ptr;
    }

    T end() const {
        return _ptr + _count;
    }

    size_t size() const {
        return _count;
    }

    T& operator[](size_t idx) {
        return _ptr[idx];
    }

private:
    T _ptr;
    size_t _count;

};



template <typename T>
class value_range {
public:

    typedef T value_type;

    typedef random_access_iterator<value_range<T>, value_type> iterator;

    value_range(const T& val, size_t count) : _val(val), _count(count) {}

    size_t size() const { return _count; }
    T& operator[](size_t idx) { return _val; }

    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, _count); }

private:
    T _val;
    size_t _count;


};


template <typename T, typename = typename std::enable_if<std::is_pointer<T>::value>::type>
ptr_range<T> range(T& ptr, const size_t& size) {
    return ptr_range<T>(ptr, size);
}

template <typename T, typename = typename std::enable_if<std::is_array<T>::value>::type>
ptr_range<typename std::remove_extent<T>::type *> range(T& arr_ref, const size_t& size) {
    return ptr_range<typename std::remove_extent<T>::type *>(arr_ref, std::extent<T>::value);
}

template <typename T, typename = typename std::enable_if<std::is_class<T>::value>::type>
T& range(T& ref, const size_t& size) {
    return ref;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
value_range<T> range(const T& ref, const size_t& size) {
    return value_range<T>(ref, size);
}


}
