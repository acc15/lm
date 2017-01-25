#pragma once

#include <type_traits>

namespace lm {

template <typename Iter1, typename Iter2, typename OIter, typename Func>
OIter transform(Iter1 begin1, Iter1 end1, Iter2 begin2, Iter2 end2, OIter out, Func func) {

    while (begin1 != end1 && begin2 != end2) {
        *out = func(*begin1, *begin2);
        ++begin1;
        ++begin2;
        ++out;
    }
    return out;
}

struct return_2nd {

    template <typename T1, typename T2>
    const T2& operator()(const T1&/* v1*/, const T2& v2) {
        return v2;
    }

};

}
