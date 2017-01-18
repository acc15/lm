#include <catch.hpp>
#include <lm/matrix.h>

using namespace lm;

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

        permutation_matrix<float (&)[3][3], size_t[3]> m(test_matricies[i]);

        REQUIRE( lu_decomposition(m) );

        float det = m.cell(0, 0) * m.cell(1, 1) * m.cell(2, 2);
        if (m.permutation_count() & 1 != 0) {
            det = -det;
        }

        REQUIRE( det == expected_det[i] );

    }


}
