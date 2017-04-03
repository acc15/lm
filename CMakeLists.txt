cmake_minimum_required(VERSION 3.6)
project(lm)

#set_property( GLOBAL PROPERTY USE_FOLDERS ON)

include_directories(inc)

set(SOURCE_FILES
    inc/lm/util/assert.h
    inc/lm/util/random_access_iterator.h
    inc/lm/util/range.h
    inc/lm/util/functional.h
    inc/lm/vec/generic_vec.h
    inc/lm/vec/vec.h
    inc/lm/vec/vec_traits.h
    inc/lm/matrix/layout.h
    inc/lm/matrix/traits.h
    inc/lm/matrix/fwd.h
    inc/lm/matrix/decorator.h
    inc/lm/matrix/permutation.h
    inc/lm/matrix/transpose.h
    inc/lm/matrix/static.h
    inc/lm/matrix/dynamic.h
    inc/lm/matrix/algorithm.h
    inc/lm/matrix/matrix.h
    inc/lm/matrix/type_util.h
    inc/lm/transform/transform.h)

set(TEST_FILES test/main.cpp
    test/lm/assert.cpp
    test/lm/range.cpp
    test/lm/vec.cpp
    test/lm/matrix.cpp
    test/lm/vec_traits.cpp)

add_executable(lm_test
    ${SOURCE_FILES}
    ${TEST_FILES})

target_include_directories(lm_test PUBLIC test deps/Catch/single_include)
