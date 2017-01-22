#include <catch.hpp>
#include <lm/matrix/matrix.h>

#include <lm/vec.h>

#include <vector>

using namespace lm;

TEST_CASE("array_ref_matrix_storage", "[matrix]") {

    float a[3][3] = {
        {  9,  1,  2 },
        {  3,  4,  5 },
        {  6,  7,  8 }
    };

    static_matrix<float(&)[3][3]> m(a);
    REQUIRE( m.cols() == 3 );
    REQUIRE( m.rows() == 3 );
    REQUIRE( m(0, 0) == 9 );
    REQUIRE( m(1, 1) == 4 );

}


TEST_CASE("array_matrix_storage", "[matrix]") {

    float a[3][3] = {
        {  9,  1,  2 },
        {  3,  4,  5 },
        {  6,  7,  8 }
    };

    static_matrix<float[3][3]> m(a);

    REQUIRE( m.cols() == 3 );
    REQUIRE( m.rows() == 3 );
    REQUIRE( m(0, 0) == 9 );
    REQUIRE( m(1, 1) == 4 );

}

TEST_CASE("initializer_list", "[matrix]") {

    array_matrix<float, 3, 3> m1 = {1,2,3,4,5,6,7,8,9};
    array_matrix<float, 3, 3> m2 = {{1,2,3},{4,5,6},{7,8,9}};
    // single initializer_list not supported as there is no info about row and col amounts
    vector_matrix<float> m3 = {{1,2,3},{4,5,6},{7,8,9}};

    REQUIRE( m1(0, 0) == 1 );
    REQUIRE( m1(0, 1) == 2 );
    REQUIRE( m1(0, 2) == 3 );
    REQUIRE( m1(1, 0) == 4 );
    REQUIRE( m1(1, 1) == 5 );
    REQUIRE( m1(1, 2) == 6 );
    REQUIRE( m1(2, 0) == 7 );
    REQUIRE( m1(2, 1) == 8 );
    REQUIRE( m1(2, 2) == 9 );

    REQUIRE( m1 == m2 );
    REQUIRE( m1 == m3 );

}

TEST_CASE("vector_matrix init size", "[matrix]") {
    vector_matrix<float> m3(3, 3);
    REQUIRE( m3.rows() == 3 );
    REQUIRE( m3.cols() == 3 );
}

TEST_CASE("add", "[matrix]") {

    array_matrix<float, 3, 3> m1 = {1,2,3,4,5,6,7,8,9};
    array_matrix<float, 3, 3> m2 = m1;

    array_matrix<float, 3, 3> a1 = m1 + m2;
    array_matrix<float, 3, 3> a2 = m1;
    a2 += m2;

    array_matrix<float, 3, 3> a3 = m1;
    a3.add(m2);

    array_matrix<float, 3, 3> e = {
        { 2, 4, 6 },
        { 8, 10, 12 },
        { 14, 16, 18 }
    };

    REQUIRE(a1 == e);
    REQUIRE(a2 == e);
    REQUIRE(a3 == e);
}

TEST_CASE("subtract", "[matrix]") {

    array_matrix<float, 3, 3> m1 = {2,4,6,8,10,12,14,16,18};
    array_matrix<float, 3, 3> m2 = {1,2,3,4,5,6,7,8,9};

    array_matrix<float, 3, 3> a1 = m1 - m2;
    array_matrix<float, 3, 3> a2 = m1;
    a2 -= m2;

    array_matrix<float, 3, 3> a3 = m1;
    a3.subtract(m2);

    array_matrix<float, 3, 3> e = {1,2,3,4,5,6,7,8,9};

    REQUIRE(a1 == e);
    REQUIRE(a2 == e);
    REQUIRE(a3 == e);
}

TEST_CASE("product", "[matrix]") {

    array_matrix<float, 3, 3> m1 = {1,2,3,4,5,6,7,8,9};
    array_matrix<float, 3, 3> m2 = m1;

    vector_matrix<float> v1 = m1;
    vector_matrix<float> v2 = m1;

    array_matrix<float, 3, 3> a1 = product(m1, m2);
    vector_matrix<float> a2 = product(m1, v1);
    vector_matrix<float> a3 = product(v1, v2);
    vector_matrix<float> a4 = product(v1, m1);

    array_matrix<float, 3, 3> e = {
        { 30, 36, 42 },
        { 66, 81, 96 },
        { 102, 126, 150 }
    };

    REQUIRE(a1 == e);
    REQUIRE(a2 == e);
    REQUIRE(a3 == e);
    REQUIRE(a4 == e);

}

TEST_CASE("transpose", "[matrix]") {

    array_matrix<float, 2, 3> m1 = {1,2,3,4,5,6};
    array_matrix<float, 3, 2> a = m1.compute_transposed();

    // 1 2 3
    // 4 5 6

    // 1 4
    // 2 5
    // 3 6
    array_matrix<float, 3, 2> e = {1,4,2,5,3,6};
    REQUIRE(a == e);

    array_matrix<float, 3, 3> m2 = {1,2,3,4,5,6,7,8,9};
    m2.transpose();

    // 1 2 3
    // 4 5 6
    // 7 8 9

    // 1 4 7
    // 2 5 8
    // 3 6 9
    array_matrix<float, 3, 3> e2 = {1,4,7,2,5,8,3,6,9};
    REQUIRE(m2 == e2);

}

TEST_CASE("flat_array_matrix::ref", "[matrix]") {

    float x[] = {1,2,3,4,5,6,7,8,9};
    typename flat_array_matrix<float, 3, 3>::ref m(x);

    float v[] = {3,2,1,3,2,1,3,2,1};
    m.assign<float[9],array_matrix_traits<float,3,3>>(v);

    REQUIRE( std::equal(x, x + 9, v, v + 9) );

    m.transpose();

    float e[] = {3,3,3,2,2,2,1,1,1};
    REQUIRE( std::equal(x, x + 9, e, e + 9) );

}

TEST_CASE("container_matrix", "[matrix]") {

    std::array<float, 9> x;

    typename container_matrix<std::array, float, 3, 3>::ref m(x);
    m.assign<std::initializer_list<float>, range_matrix_traits<std::initializer_list<float>, 3, 3>>({ 1, 2, 3, 4, 5, 6, 7, 8, 9});

    for (size_t i = 0; i < x.size(); i++) {
        REQUIRE(x[i] == i + 1);
    }
}

TEST_CASE("lu_decomposition", "[matrix]") {

    float test_matricies[][3][3] = {
        {
            {  9,  1,  2 },
            {  3,  4,  5 },
            {  6,  7,  8 }
        },
        {
            {  1, -2,  3 },
            {  4,  0,  6 },
            { -7,  8,  9 }
        },
        {
            {  3,  3, -1 },
            {  4,  1,  3 },
            {  1, -2, -2 }
        },
    };

    float expected_det[] = { -27.f, 204.f, 54.f };
    for (size_t i = 0; i < sizeof(expected_det) / sizeof(float); i++) {

        array_matrix<float, 3, 3> m(test_matricies[i]);
        permutation_matrix<decltype(m)> pm(m);

        REQUIRE( lu_decomposition(pm) );

        float det = pm.cell(0, 0) * pm.cell(1, 1) * pm.cell(2, 2);
        if ((pm.permutation_count() & 1) != 0) {
            det = -det;
        }

        REQUIRE( det == expected_det[i] );
    }

}
