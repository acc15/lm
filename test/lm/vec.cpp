#include <catch.hpp>
#include <lm/vec.h>

#include <array>

using lm::vec;

TEST_CASE("iterator", "[vec]") {
    vec<3, int> v = {1, 2, 3};
    REQUIRE( std::equal(v.begin(), v.end(), std::vector<int>({ 1, 2, 3}).begin()) );
}

TEST_CASE("reverse_iterator", "[vec]") {
    vec<3, int> v = {3, 2, 1};
    REQUIRE( std::equal(v.rbegin(), v.rend(), std::vector<int>({ 1, 2, 3}).begin()) );
}

TEST_CASE("empty()", "[vec]") {
    REQUIRE( (vec<0,int>()).empty() );
    REQUIRE_FALSE( (vec<1,int>()).empty() );
}

TEST_CASE("operator=", "[vec]") {
    vec<3, int> v1;
    v1 = {1, 2, 3};
    REQUIRE( v1 == (vec<3, int>({1, 2, 3})) );
    REQUIRE( v1.length_square() == 14 );
}

TEST_CASE("operator==", "[vec]") {
    vec<3, int> v1 = { 1, 2, 3 };
    std::vector<int> v2 = {1, 2, 3};
    REQUIRE( v1 == v2 );
}

TEST_CASE("negate()", "[vec]") {
    vec<3, int> v1 = { 1, 2, 3 };
    REQUIRE( v1.negate() == (vec<3, int>({ -1, -2, -3 })) );
}

TEST_CASE("-operator()", "[vec]") {
    vec<4, int> v = { 1, 2, 3, 4 };
    REQUIRE( (-v) == (vec<4, int>({ -1, -2, -3, -4 })));
}

TEST_CASE("operator+", "[vec]") {
    vec<3, int> v1 = { 1, 2, 3 };
    REQUIRE( (v1 + vec<3, float>({4, 5, 6})) == (vec<3, int>({5, 7, 9})) );
    REQUIRE( (v1 + std::initializer_list<float>({ 4.75, 5.32, 6.9 })) == (vec<3, int>({5, 7, 9})) );
    REQUIRE( (v1 + 4) == (vec<3, int>({5, 6, 7})) );
}

TEST_CASE("operator-", "[vec]") {
    vec<3, int> v1 = { 1, 2, 3 };
    REQUIRE( (v1 - vec<3, float>({6, 5, 4})) == (vec<3, int>({-5, -3, -1})) );
    REQUIRE( (v1 - std::initializer_list<float>({ 6.75, 5.32, 4.9 })) == (vec<3, int>({-5, -3, -1})) );
    REQUIRE( (v1 - 2) == (vec<3, int>({-1, 0, 1})) );
}

TEST_CASE("operator*", "[vec]") {
    vec<3, int> v1 = { 1, 2, 3 };
    REQUIRE( (v1 * vec<3, float>({3, 2, 1})) == (vec<3, int>({3, 4, 3})) );
    REQUIRE( (v1 * std::initializer_list<float>({ 3, 2, 1 })) == (vec<3, int>({3, 4, 3})) );
    REQUIRE( (v1 * 2) == (vec<3, int>({2, 4, 6})) );
}

TEST_CASE("operator/", "[vec]") {
    vec<3, int> v1 = { 4, 5, 6 };
    REQUIRE( (v1 / vec<3, float>({1, 2, 3})) == (vec<3, int>({4, 2, 2})) );
    REQUIRE( (v1 / std::initializer_list<float>({ 1, 2, 3 })) == (vec<3, int>({4, 2, 2})) );
    REQUIRE( (v1 / 2) == (vec<3, int>({2, 2, 3})) );
}

TEST_CASE("scalar_product()", "[vec]") {
    vec<3, int> v = { 1, 2, 3 };
    vec<3, int> v2 = { 4, 5, 6 };
    REQUIRE( v.scalar_product(v2) == 32 );
}

TEST_CASE("length()", "[vec]") {
    vec<3, int> v = { 3, 0, 4 };
    REQUIRE( v.length() == 5 );
}
