#pragma once

#include <cstddef>
#include <iterator>

namespace lm {

template <typename Container, typename Element, bool Dir = true>
class random_access_iterator {
public:
    
	
	typedef Container container_type;
    typedef random_access_iterator iterator;

	typedef iterator _Unchecked_type;


    typedef size_t difference_type;
    typedef Element value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef std::random_access_iterator_tag iterator_category;
    typedef size_t size_type;

    random_access_iterator(container_type& container, size_t index) : _container(&container), _index(index) {}
    random_access_iterator(const iterator& other) : _container(other._container), _index(other._index) {}

    iterator& operator=(const iterator& other) { _container = other._container; _index = other._index; return *this; }
    bool operator==(const iterator& other) const { return _container == other._container && _index == other._index; }
    bool operator!=(const iterator& other) const { return !operator==(other); }
    bool operator<(const iterator& other) const { return Dir ? _index < other._index : _index > other._index; }
    bool operator>(const iterator& other) const { return Dir ? _index > other._index : _index < other._index; }
    bool operator<=(const iterator& other) const { return !operator>(other); }
    bool operator>=(const iterator& other) const { return !operator<(other); }

    iterator& operator++() { _index = diff<true>(1); return *this; }
    iterator& operator--() { _index = diff<false>(1); return *this; }
    iterator& operator+=(size_type sz) { _index = diff<true>(sz); return *this; }
    iterator& operator-=(size_type sz) { _index = diff<false>(sz); return *this; }

    iterator operator++(int) { return ++iterator(*this); }
    iterator operator--(int) { return --iterator(*this); }
    iterator operator+(size_type sz) const { return iterator(*this) += sz; }
    iterator operator-(size_type sz) const { return iterator(*this) -= sz; }
    difference_type operator-(const iterator& other) const { return Dir ? _index - other._index : other._index - _index; }

    reference operator*() const { return (*_container)[_index]; }
    pointer operator->() const { return &operator*(); }
    reference operator[](size_type offset) const { return (*_container)[_index + offset]; }

private:

    template <bool Sign>
    size_t diff(size_t offset) {
        return Sign == Dir ? _index + offset : _index - offset;
    }


private:
    container_type* _container;
    size_t _index;
};


}
