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

/* This is N-tree leaf (or node!) */
template <typename T, size_t N> struct leaf {
  T data;
  leaf<T,N> *children = NULL;
  leaf<T,N> *parent = NULL;
  size_t level = 0;
  int8_t coordinate = -1; 
  const size_t n_children = pow(2,3);
public:
  /* leaf(T data): data(data) {} */

  leaf() {};
  leaf(T data) : data(data) {};

  ~leaf(){
    data.~T();
    if (!(this->children == NULL)) {
      delete[] children;
    }
  }

  class LeafIterator;
  LeafIterator begin() { return LeafIterator(this); }
  LeafIterator end() { return LeafIterator(nullptr); }

};

template <typename T, size_t N>
struct leaf<T,N>::LeafIterator {
public:
  using iterator_category = std::forward_iterator_tag;

  LeafIterator(leaf<T,N>* ptr) {
    this->curr = ptr;
    if (ptr) this->stack.push(ptr);
    this->moveToNextValid();
  };

  leaf<T,N>& operator*() { return *(this->curr); }

  // prefix increment
  leaf<T,N>* operator++ () {
    moveToNextValid();
    return this->stack.top();
  }

  bool operator==(const LeafIterator& R) {
    return this->curr == R.curr;
  }

  bool operator!=(const LeafIterator& R) {
    return this->curr != R.curr;
  }

private:
  std::stack<leaf<T,N>*> stack;
  leaf<T,N>* curr;

  virtual void moveToNextValid() {
    this->curr = nullptr;
    while (!stack.empty()) {
      leaf<T,N>* node = stack.top();

      stack.pop();
      if (!node->children) {curr = node; return;}

      for (int i = 0; i < node->n_children; i++) stack.push(&(node->children[i]));
    }
  }
};

}
