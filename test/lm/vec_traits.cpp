#include <catch.hpp>
#include <lm/vec/vec_traits.h>

#include <array>
#include <exception>

TEST_CASE("vec_traits", "[array]") {

    typedef lm::vec_traits<int[32]> traits;

    int v[32];

    REQUIRE( traits::size(v) == 32 );

    REQUIRE_THROWS( traits::resize(v, 10) );
}
