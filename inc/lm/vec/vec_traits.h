#pragma once

#include <cstddef>
#include <cassert>

namespace lm {

template <typename T>
struct vec_traits {

    typedef T type;

    static void resize(type& vec, size_t new_size) {
        vec.resize(new_size);
    }

    static size_t size(const type& vec) {
        return vec.size();
    }

};

template <typename E, size_t Size>
struct vec_traits<E[Size]> {

    typedef E type[Size];

    static void resize(type& vec, size_t new_size) {
        assert( Size == new_size );
    }

    static size_t size(const type& vec) {
        return Size;
    }

};


}
