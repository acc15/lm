#include <array>
#include <vector>

#include <catch.hpp>
#include <lm/range.h>

using lm::range;

TEST_CASE("range must iterate over std::vector", "[range]") {
    const std::vector<int> vec = { 1, 2, 3 };
    auto rv = range(vec, 10);
    REQUIRE(std::vector<int>(rv.begin(), rv.end()) == vec);
}

TEST_CASE("range must iterate over std::array", "[range]") {
    const std::array<int, 3> sarr = {1, 2, 3 };
    auto rs = range(sarr, 15);
    REQUIRE(std::vector<int>(rs.begin(), rs.end()) == std::vector<int>({ 1, 2, 3 }));
}

TEST_CASE("range must iterate over C array", "[range]") {
    const int arr[] = { 1, 2, 3 };
    auto ra = range(arr, 5);
    REQUIRE(std::vector<int>(ra.begin(), ra.end()) == std::vector<int>({ 1, 2, 3 }));
}

TEST_CASE("range must iterate over pointer", "[range]") {
    const int a[] = { 1, 2, 3 };
    const int* ptr = a;

    auto rp = range(ptr, 2);
    REQUIRE(std::vector<int>(rp.begin(), rp.end()) == std::vector<int>({ 1, 2 }));
}

TEST_CASE("range must iterate over single number", "[range]") {
    auto rk = range(15, 4);
    REQUIRE(std::vector<int>(rk.begin(), rk.end()) == std::vector<int>({ 15, 15, 15, 15 }));
}
