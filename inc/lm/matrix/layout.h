#pragma once

#include <cstddef>

namespace lm {

struct col_major_layout {
    static size_t compute_flat_index(size_t row, size_t col, size_t rows, size_t cols) {
        return col * rows + row;
    }
};

struct row_major_layout {
    static size_t compute_flat_index(size_t row, size_t col, size_t rows, size_t cols) {
        return row * cols + col;
    }
};

}
