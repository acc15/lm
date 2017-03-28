#include <catch.hpp>

#undef NDEBUG
#include <lm/util/assert.h>

#include <iostream>

TEST_CASE("pass", "[assert]") {
    lm_assert(1 == 1, "test");
}

TEST_CASE("fail", "[assert]") {
    REQUIRE_THROWS_AS( lm_assert( 1 == 2, "msg"), lm::assert_error );
}

TEST_CASE("msg", "[assert]") {
    int x = 12;
    int y = 10;

    try {
        lm_assert(x == y, x << " not eq to " << y);
    } catch (const lm::assert_error& e) {
        std::ostringstream ss;
        ss << e;

        std::string str = ss.str();
        REQUIRE( str.substr(0, 29) == "Assertion (x == y) failed in ");
        REQUIRE( str.substr(str.length() - 38) == "assert.cpp:21. Reason: 12 not eq to 10");
    }

}
