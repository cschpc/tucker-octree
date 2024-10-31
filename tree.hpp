#include <iostream>
#include <cassert>
#include <initializer_list>
#include <bitset>
#include <memory>
#include <array>
#include <cmath>

#include <stack>
#include <iterator> // For std::forward_iterator_tag
#include <cstddef>  // For std::ptrdiff_t

namespace tree {
template <typename T> struct leaf {
  T data;
  leaf<T> *children = NULL;
  size_t level = 0;
  int8_t coordinate = -1; 
  size_t n_children=0;
public:
  /* leaf(T data): data(data) {} */

  ~leaf(){
    data.~T();
    if (!(this->children == NULL)) {
      delete[] children;
    }
  }

  class Iterator;
  Iterator begin() { return Iterator(this); }
};

template <typename T>
struct leaf<T>::Iterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = leaf<T>;
  using pointer = leaf<T>*;
  using reference  = leaf<T>&;

  Iterator(pointer ptr) {
    if (ptr) this->stack.push(ptr);
    moveToNextValid();
  };

  /* reference operator*() const { return *ptr;} */
  
  /* pointer operator->() { return ptr; } */

  // prefix increment
  Iterator& operator++ () {
    moveToNextValid();
    return stack.top();
  }

  // postfix increment
  Iterator& operator++(int) {
    leaf<T>* tmp = stack.top();
    ++this;
    return tmp;
  }

private:
  std::stack<leaf<T>*> stack;

  void moveToNextValid() {
    while (!stack.empty) {
      leaf<T>* node = stack.top();

      if (!node->children) return; // No children, top of stack is good.

      stack.pop(); // Otherwise push its children to stack
      for (int i = 0; i < node->n_children; i++) stack.push(&(node->children[i]));
    }
  }
};

}
