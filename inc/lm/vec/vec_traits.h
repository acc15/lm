#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include <lm/util/assert.h>

namespace lm {

template <typename T, typename = void>
struct is_resizable {
    typedef T type;
    enum {
        value = false
    };
};

template <typename T>
struct is_resizable<T, decltype( std::declval<T>().resize(10) ) > {
    typedef T type;
    enum {
        value = true
    };
};

template <typename T>
struct vec_traits {

    typedef T type;

    template <typename U = T>
    static typename std::enable_if<is_resizable<U>::value>::type resize(U& vec, size_t new_size) {
        vec.resize(new_size);
    }

    template <typename U = T>
    static typename std::enable_if<!is_resizable<U>::value>::type resize(U& vec, size_t new_size) {
        lm_assert( vec.size() == new_size, "static containers can't be resized" );
    }

    static size_t size(const type& vec) {
        return vec.size();
    }

};

template <typename E, size_t Size>
struct vec_traits<E[Size]> {

    typedef E type[Size];

    static void resize(type& vec, size_t new_size) {
        lm_assert( Size == new_size, "arrays can't be resized" );
    }

    static size_t size(const type& vec) {
        return Size;
    }

};


}
