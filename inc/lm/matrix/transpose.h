#pragma once

#include <type_traits>

#include <lm/matrix/decorator.h>

namespace lm {

template <typename M>
class transpose_storage: public matrix_decorator<transpose_storage, M> {
public:
    typedef matrix_decorator<::lm::transpose_storage, M> base_type;
    typedef typename base_type::value_type value_type;
    typedef typename base_type::storage_type storage_type;

    transpose_storage() = default;

    template <typename T> transpose_storage(const std::initializer_list<T>& other) : _m(other) {}
    template <typename T> transpose_storage(const std::initializer_list<std::initializer_list<T>>& other) : _m(other) {}
    template <typename T> transpose_storage(const T& other) : _m(other) {}

    template <typename T = M, typename = typename std::enable_if<std::is_reference<T>::value && std::is_same<T, M>::value>::type>
    transpose_storage(M ref) : _m(ref) {}

    constexpr size_t rows() const { return _m.cols(); }
    constexpr size_t cols() const { return _m.rows(); }

    value_type& at(size_t row, size_t col) {
        return _m.at(col, row);
    }

    void resize(size_t rows, size_t cols) {
        _m.resize(cols, rows);
    }

    void swap_row(size_t r1, size_t r2) {
        _m.swap_col(r1, r2);
    }

    void swap_col(size_t c1, size_t c2) {
        _m.swap_row(c1, c2);
    }

    const storage_type& value() const { return _m; }
    storage_type& value() { return _m; }

private:
    M _m;

};

template <typename M> using transpose_matrix = typename transpose_storage<M>::value_matrix_type;


}
