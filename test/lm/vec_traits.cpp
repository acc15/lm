#include <catch.hpp>
#include <lm/vec/vec_traits.h>

#include <array>
#include <vector>

TEST_CASE("array", "[vec_traits]") {
    typedef lm::vec_traits<int[32]> traits;

    int v[32];
    REQUIRE( traits::size(v) == 32 );
    REQUIRE_THROWS( traits::resize(v, 10) );
}

TEST_CASE("std::array", "[vec_traits]") {
    typedef lm::vec_traits<std::array<int, 32>> traits;

    std::array<int, 32> v;
    REQUIRE( traits::size(v) == 32 );
    REQUIRE_THROWS( traits::resize(v, 10) );
}

TEST_CASE("std::vector", "[vec_traits]") {
    typedef lm::vec_traits<std::vector<int>> traits;

    std::vector<int> v = {1,2,3};
    REQUIRE( traits::size(v) == 3 );
    traits::resize(v, 10);
    REQUIRE( traits::size(v) == 10 );
}
