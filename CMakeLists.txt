cmake_minimum_required(VERSION 3.6)
project(lm)

include_directories(inc)

set(SOURCE_FILES
    inc/lm/random_access_iterator.h
    inc/lm/range.h
    inc/lm/algorithm.h
    inc/lm/generic_vec.h
    inc/lm/vec.h
    inc/lm/matrix.h
    )

add_executable(lm_test
    ${SOURCE_FILES}
    test/main.cpp
    test/lm/range.cpp
    test/lm/vec.cpp
    test/lm/matrix.cpp
    )

target_include_directories(lm_test PUBLIC test deps/Catch/single_include)

enable_testing()
add_test(NAME lm_test COMMAND lm_test)
