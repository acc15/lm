#pragma once

#include <sstream>
#ifndef NDEBUG

#define lm_assert(cond, msg) ((cond) \
    ? 0 \
    : throw lm::assert_error(static_cast<std::ostringstream&>(std::ostringstream() << msg).str(), #cond, __FILE__, __LINE__))

#else

#define lm_assert(cond, msg) (0)

#endif

namespace lm {

class assert_error {
public:
    assert_error(const std::string& what, const char* expr, const char* file, const size_t line) :
        _what(what), _expr(expr), _file(file), _line(line)
    {
    }

    const char* what() const {
        return _what.c_str();
    }

    const char* file() const {
        return _file;
    }

    const char* expr() const {
        return _expr;
    }

    const size_t line() const {
        return _line;
    }

private:
    size_t _line;
    const char* _expr;
    const char* _file;
    std::string _what;

};

std::ostream& operator <<(std::ostream& stream, const assert_error& err) {
    return stream << "Assertion (" << err.expr() << ") failed in " << err.file() << ":" << err.line() << ". Reason: " << err.what();
}

}
