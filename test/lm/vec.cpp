#include <catch.hpp>
#include <lm/vec/vec.h>

#include <array>

using lm::vec;

TEST_CASE("iterator", "[vec]") {
    vec<int, 3> v = {1, 2, 3};
    REQUIRE( std::equal(v.begin(), v.end(), std::vector<int>({ 1, 2, 3}).begin()) );
}

TEST_CASE("reverse_iterator", "[vec]") {
    vec<int, 3> v = {3, 2, 1};
    REQUIRE( std::equal(v.rbegin(), v.rend(), std::vector<int>({ 1, 2, 3}).begin()) );
}

TEST_CASE("empty()", "[vec]") {
    REQUIRE( (vec<int, 0>()).empty() );
    REQUIRE_FALSE( (vec<int, 1>()).empty() );
}

TEST_CASE("operator=", "[vec]") {
    vec<int, 3> v1;
    v1 = {1, 2, 3};
    REQUIRE( v1 == (vec<int, 3>({1, 2, 3})) );
    REQUIRE( v1.length_square() == 14 );
}

TEST_CASE("operator==", "[vec]") {
    vec<int, 3> v1 = { 1, 2, 3 };
    std::vector<int> v2 = {1, 2, 3};
    REQUIRE( v1 == v2 );
}

TEST_CASE("negate()", "[vec]") {
    vec<int, 3> v1 = { 1, 2, 3 };
    REQUIRE( v1.negate() == (vec<int, 3>({ -1, -2, -3 })) );
}

TEST_CASE("-operator()", "[vec]") {
    vec<int, 4> v = { 1, 2, 3, 4 };
    REQUIRE( (-v) == (vec<int, 4>({ -1, -2, -3, -4 })));
}

TEST_CASE("operator+", "[vec]") {
    vec<int, 3> v1 = { 1, 2, 3 };
    REQUIRE( (v1 + vec<float, 3>({4, 5, 6})) == (vec<int, 3>({5, 7, 9})) );
    REQUIRE( (v1 + std::initializer_list<float>({ 4.75f, 5.32f, 6.9f })) == (vec<int, 3>({5, 7, 9})) );
    REQUIRE( (v1 + 4) == (vec<int, 3>({5, 6, 7})) );
}

TEST_CASE("operator-", "[vec]") {
    vec<int, 3> v1 = { 1, 2, 3 };
    REQUIRE( (v1 - vec<float, 3>({6, 5, 4})) == (vec<int, 3>({-5, -3, -1})) );
    REQUIRE( (v1 - std::initializer_list<float>({ 6.75f, 5.32f, 4.9f })) == (vec<int, 3>({-5, -3, -1})) );
    REQUIRE( (v1 - 2) == (vec<int, 3>({-1, 0, 1})) );
}

TEST_CASE("operator*", "[vec]") {
    vec<int, 3> v1 = { 1, 2, 3 };
    REQUIRE( (v1 * vec<float, 3>({3.f, 2.f, 1.f})) == (vec<int, 3>({3, 4, 3})) );
    REQUIRE( (v1 * std::initializer_list<float>({ 3.f, 2.f, 1.f })) == (vec<int, 3>({3, 4, 3})) );
    REQUIRE( (v1 * 2) == (vec<int, 3>({2, 4, 6})) );
}

TEST_CASE("operator/", "[vec]") {
    vec<int, 3> v1 = { 4, 5, 6 };
    REQUIRE( (v1 / vec<float, 3>({1.f, 2.f, 3.f})) == (vec<int, 3>({4, 2, 2})) );
    REQUIRE( (v1 / std::initializer_list<float>({ 1.f, 2.f, 3.f })) == (vec<int, 3>({4, 2, 2})) );
    REQUIRE( (v1 / 2) == (vec<int, 3>({2, 2, 3})) );
}

TEST_CASE("operator/ on zero", "[vec]") {

    int k = 3;
    int v = 0;

    k /= v;

//    vec<int, 3> v1 = {1,2,3};
//    vec<int, 3> v2 = {0,0,0};

//    v1 /= v2;


}

TEST_CASE("scalar_product()", "[vec]") {
    vec<int, 3> v = { 1, 2, 3 };
    vec<int, 3> v2 = { 4, 5, 6 };
    REQUIRE( v.scalar_product(v2) == 32 );
}

TEST_CASE("length()", "[vec]") {
    vec<int, 3> v = { 3, 0, 4 };
    REQUIRE( v.length() == 5 );
}
