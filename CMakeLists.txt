cmake_minimum_required(VERSION 3.6)
project(lm)

#set_property( GLOBAL PROPERTY USE_FOLDERS ON)

include_directories(inc)

set(VECTOR_SOURCE_FILES inc/lm/random_access_iterator.h
    inc/lm/range.h
    inc/lm/algorithm.h
    inc/lm/generic_vec.h
    inc/lm/vec.h)

set(MATRIX_SOURCE_FILES inc/lm/matrix/layout.h
    inc/lm/matrix/traits.h
    inc/lm/matrix/fwd.h
    inc/lm/matrix/decorator.h
    inc/lm/matrix/permutation.h
    inc/lm/matrix/transpose.h
    inc/lm/matrix/static.h
    inc/lm/matrix/dynamic.h
    inc/lm/matrix/algorithm.h
    inc/lm/matrix/matrix.h
    inc/lm/matrix/type_util.h)

set(TEST_SOURCE_FILES test/main.cpp
    test/lm/range.cpp
    test/lm/vec.cpp
    test/lm/matrix.cpp)

add_executable(lm_test
    ${VECTOR_SOURCE_FILES}
	${MATRIX_SOURCE_FILES}
	${TEST_SOURCE_FILES})

target_include_directories(lm_test PUBLIC test deps/Catch/single_include)

source_group("lm" FILES ${VECTOR_SOURCE_FILES})
source_group("lm\\matrix" FILES ${MATRIX_SOURCE_FILES})
source_group("test" FILES ${TEST_SOURCE_FILES})