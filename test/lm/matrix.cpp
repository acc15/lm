#include <catch.hpp>

#include <vector>

#include <lm/matrix/matrix.h>
#include <lm/vec/vec.h>

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

TEST_CASE("product_ref", "[matrix]") {

    float m[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

    array_matrix<float, 3, 3>::reference_matrix_type p1(m);
    array_matrix<float, 3, 3> p2 = {1,2,3,4,5,6,7,8,9};
    p1 *= p2;

    array_matrix<float, 3, 3> p3 = {30,36,42,66,81,96,102,126,150};
    REQUIRE( p1 == p3);

}


TEST_CASE("product_this", "[matrix]") {

    array_matrix<float, 3, 3> p1 = {1,2,3,4,5,6,7,8,9};
    array_matrix<float, 3, 3> p2 = {1,2,3,4,5,6,7,8,9};
    p1 *= p2;

    array_matrix<float, 3, 3> p3 = {30,36,42,66,81,96,102,126,150};
    REQUIRE( p1 == p3);

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

TEST_CASE("dynamic_transpose", "[matrix]") {

    vector_matrix<float> m = {{1,2,3},{4,5,6}};
    m.transpose();

    REQUIRE(m.rows() == 3);
    REQUIRE(m.cols() == 2);

    array_matrix<float, 3, 2> e = {1,4,2,5,3,6};
    REQUIRE(m == e);

}

TEST_CASE("flat_array_matrix::reference_matrix_type", "[matrix]") {

    float x[] = {1,2,3,4,5,6,7,8,9};
    typename flat_array_matrix<float, 3, 3>::reference_matrix_type m(x);

    float v[] = {3,2,1,3,2,1,3,2,1};
    m.assign<float[9],array_matrix_traits<float,3,3>>(v);

    REQUIRE( std::equal(x, x + 9, v, v + 9) );

    m.transpose();

    float e[] = {3,3,3,2,2,2,1,1,1};
    REQUIRE( std::equal(x, x + 9, e, e + 9) );

}

TEST_CASE("container_matrix", "[matrix]") {

    vec<float, 9> x;

    typename container_matrix<vec, float, 3, 3>::reference_matrix_type m(x);
    m.assign<std::initializer_list<float>, range_matrix_traits<std::initializer_list<float>, 3, 3>>({ 1, 2, 3, 4, 5, 6, 7, 8, 9});

    for (size_t i = 0; i < x.size(); i++) {
        REQUIRE(x[i] == i + 1);
    }
}

TEST_CASE("determinant", "[matrix]") {

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
        {
            {  1,  2,  3 },
            {  4,  5,  6 },
            {  7,  8,  9 }
        }
    };

    float expected_det[] = { -27.f, 204.f, 54.f, 0.f };
    for (size_t i = 0; i < sizeof(expected_det) / sizeof(float); i++) {
        array_matrix<float, 3, 3> m(test_matricies[i]);
        float det = determinant(m);
        REQUIRE( det == expected_det[i] );
    }

}

TEST_CASE("invert", "[matrix]") {

    float test_matricies[][3][3] = {
        {
            {  9,  1,  2 },
            {  3,  4,  5 },
            {  6,  7,  8 }
        },
        {
            {  9,  2,  3 },
            {  4,  5,  6 },
            {  7,  8,  9 }
        },
        {
            {  1, -2,  3 },
            {  4,  0,  6 },
            { -7,  8,  9 }
        }
    };

    float exp_matricies[][3][3] = {
        {
            { 1.f/9, -2.f/9, 1.f/9 },
            { -2.f/9, -20.f/9, 13.f/9 },
            { 1.f/9, 19.f/9, -11.f/9 },
        },
        {
            { 0.125f, -0.25f, 0.125f },
            { -0.25f, -2.5f, 1.75f },
            { 0.125f, 29.f/12, -37.f/24 }
        },
        {
            { -4.f/17, 7.f/34, -1.f/17 },
            { -13.f/34, 5.f/34, 1.f/34 },
            { 8.f/51, 1.f/34, 2.f/51 }
        }
    };

    for (size_t i = 0; i < std::extent<decltype(test_matricies), 0>::value; i++) {
        typename array_matrix<float, 3, 3>::reference_matrix_type m(test_matricies[i]);
        array_matrix<float, 3, 3> inv;
        REQUIRE( invert_matrix(m, inv) );

        typename array_matrix<float, 3, 3>::reference_matrix_type e(exp_matricies[i]);
        for (size_t i = 0; i < e.rows(); i++) {
            for (size_t j = 0; j < e.cols(); j++) {
                REQUIRE( inv(i, j) == Approx(e(i, j)) );
            }
        }
    }

}

TEST_CASE("lu_decomposition must return false if singular", "[matrix]") {
    array_matrix<float, 3, 3> m = {1,2,3,4,5,6,7,8,9};
    REQUIRE_FALSE(lu_decomposition(m));
}

TEST_CASE("invert fails if singular", "[matrix]") {
    array_matrix<float, 3, 3> m = {1,2,3,4,5,6,7,8,9};
    array_matrix<float, 3, 3> inv;
    REQUIRE_FALSE( invert_matrix(m, inv) );
}

TEST_CASE("multiply 2d vec on 3d matrix", "[matrix]") {

    array_matrix<float, 4, 4> m = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    vec<float, 2> v = {1,2};

    container_matrix<vec, float, 2, 1>::reference_matrix_type vec_m(v);

    /*
                     1  2  3  4
                     5  6  7  8
                     9  10 11 12
                     13 14 15 16
                    -------------
        1, 2, 0, 0 | 11 14 XX XX


                        1
                        2
                        0
                        1
         1  0  0  10    11
         0  1  0  10    12
         //0  0  1  0     0
         //0  0  0  1     1

    */

    vec_m = m * vec_m;

    REQUIRE( v[0] == 11 );
    REQUIRE( v[1] == 14 );

}
