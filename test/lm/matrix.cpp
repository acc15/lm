#include <catch.hpp>
#include <lm/matrix/matrix.h>

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
