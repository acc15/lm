#include <catch.hpp>
#include <lm/matrix.h>

using namespace lm;

TEST_CASE("lu_decomposition", "[matrix]") {

    float m[3][3] = {
        {0, 1, 2},
        {3, 4, 5},
        {6, 7, 8}
    };

    float p[3];

    lu_decomposition(m, p);
}
