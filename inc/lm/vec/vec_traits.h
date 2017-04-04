#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <lm/util/assert.h>

namespace lm {

template <typename T, typename = void>
struct vec_traits {

    typedef T type;

    constexpr static size_t length = T().size();
    constexpr static bool resizable = false;

    static void resize(type& vec, size_t new_size) {
        lm_assert(length == new_size, "static vectors can't be resized");
    }

    constexpr static size_t size(const type& vec) {
        return length;
    }

};

template <typename T>
struct vec_traits<T, decltype( std::declval<T>().resize(10) ) > {

    typedef T type;

    constexpr static const size_t length = 0;
    constexpr static bool resizable = true;

    static void resize(type& vec, size_t new_size) {
        vec.resize(new_size);
    }

    static size_t size(const type& vec) {
        return vec.size();
    }

};

template <typename E, size_t S>
struct vec_traits<E[S]> {

    typedef E type[S];

    constexpr static const size_t length = S;
    constexpr static bool resizable = false;

    static void resize(type& vec, size_t new_size) {
        lm_assert( length == new_size, "arrays can't be resized" );
    }

    constexpr static size_t size(const type& vec) {
        return S;
    }

};

template <typename T, typename Enable> constexpr size_t vec_traits<T, Enable>::length;
template <typename T> constexpr size_t vec_traits<T, decltype( std::declval<T>().resize(10) ) >::length;

}
