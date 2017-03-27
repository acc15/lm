#pragma once

#include <cstddef>
#include <iterator>

namespace lm {

/**
 * @brief random access iterator implementation
 *
 * Holds a reference to container and retrieves elements by using container `operator[]`.
 * Requires that container supports both `T& operator[](size_t idx)` and `const T& operator[](size_t idx) const`
 *
 * @tparam Container container type
 * @tparam Element element type
 * @tparam Dir forward or reverse iterator
 */
template <typename Container, typename Element, bool Dir = true>
class random_access_iterator {
public:
    
    /**
     * @brief REND index of last element in reverse iterator
     */
	static const size_t REND = static_cast<size_t>(-1);
	
    /**
     * @brief container_type type of underlying container
     */
	typedef Container container_type;

    /**
     * @brief iterator type of this iterator
     */
    typedef random_access_iterator iterator;

    /**
     * @brief _Unchecked_type hack to remove MSVC unsafe warnings
     */
	typedef iterator _Unchecked_type;

    /**
     * @brief difference_type difference between two iterators
     */
    typedef size_t difference_type;

    /**
     * @brief value_type element type
     */
    typedef Element value_type;

    /**
     * @brief reference reference to element
     */
    typedef value_type& reference;

    /**
     * @brief pointer pointer to element
     */
    typedef value_type* pointer;

    /**
     * @brief iterator_category category tag
     */
    typedef std::random_access_iterator_tag iterator_category;

    /**
     * @brief size_type size_type definition required to be defined
     */
    typedef size_t size_type;

    /**
     * @brief random_access_iterator constructs random access iterator by container reference and initial index
     * @param container reference to container
     * @param index initial iterator position
     */
    random_access_iterator(container_type& container, size_type index) : _container(&container), _index(index) {}

    /**
     * @brief random_access_iterator constructs random access iterator from another random access iterator of same type
     * @param other another random access iterator
     */
    random_access_iterator(const iterator& other) : _container(other._container), _index(other._index) {}

    iterator& operator=(const iterator& other) { _container = other._container; _index = other._index; return *this; }
    bool operator==(const iterator& other) const { return _container == other._container && _index == other._index; }
    bool operator!=(const iterator& other) const { return !operator==(other); }
    bool operator<(const iterator& other) const { return Dir ? _index < other._index : _index != REND && _index > other._index; }
    bool operator>(const iterator& other) const { return Dir ? _index > other._index : _index == REND || _index < other._index; }
    bool operator<=(const iterator& other) const { return !operator>(other); }
    bool operator>=(const iterator& other) const { return !operator<(other); }

    iterator& operator++() { return move<true>(1); }
    iterator& operator--() { return move<false>(1); }
    iterator& operator+=(size_type sz) { return move<true>(sz); }
    iterator& operator-=(size_type sz) { return move<false>(sz); }

    iterator operator++(int) { return ++iterator(*this); }
    iterator operator--(int) { return --iterator(*this); }
    iterator operator+(size_type sz) const { return iterator(*this) += sz; }
    iterator operator-(size_type sz) const { return iterator(*this) -= sz; }

    /**
     * @brief operator - computes difference between two iterators (by subtracting `this.index` from `other.index`)
     * @param other other iterator
     * @return difference between two iterators
     */
    difference_type operator-(const iterator& other) const { return Dir ? _index - other._index : other._index - _index; }

    /**
     * @brief operator * using container `operator[]` returns reference to current element in container
     * @return reference to current element in container
     */
    reference operator*() const { return (*_container)[_index]; }

    /**
     * @brief operator -> using container `operator[]` returns pointer to current element in container
     * @return pointer to current element in container
     */
    pointer operator->() const { return &operator*(); }

    /**
     * @brief operator [] using container `operator[]` returns reference to (`current + offset`) element in container
     * @param offset offset
     * @return reference to (`current + offset`) element in container
     */
    reference operator[](size_type offset) const { return (*_container)[_index + offset]; }

private:

    template <bool Sign>
    iterator& move(size_t offset) {
        if (Sign == Dir) {
            _index += offset;
        } else {
            _index -= offset;
        }
        return *this;
    }


private:
    container_type* _container;
    size_type _index;
};


}
