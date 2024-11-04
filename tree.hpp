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

  class LeafIterator;
  LeafIterator begin() { return LeafIterator(this); }
  LeafIterator end() { return LeafIterator(nullptr); }

  class NodeIterator;
  NodeIterator begin_nodes() { return NodeIterator(this); }
  NodeIterator end_nodes() { return NodeIterator(this); }
};

template <typename T>
struct leaf<T>::NodeIterator : leaf<T>::LeafIterator {

  using iterator_category = std::forward_iterator_tag;

private:

  void moveToNextValid() {

    this->curr = nullptr;

    if (!this->stack.empty()) {
      leaf<T>* node = this->stack.top();

      this->stack.pop();
      if (!node->children) {
        return;
      }

      for (int i = 0; i < node->n_children; i++) {
        this->stack.push(&(node->children[i]));
      }

    }
  }

};

template <typename T>
struct leaf<T>::LeafIterator {
public:
  using iterator_category = std::forward_iterator_tag;

  LeafIterator(leaf<T>* ptr) {
    this->curr = ptr;
    if (ptr) this->stack.push(ptr);
    this->moveToNextValid();
  };

  leaf<T>& operator*() { return *(this->curr); }
  
  // prefix increment
  leaf<T>* operator++ () {
    moveToNextValid();
    return this->stack.top();
  }

  bool atend() const {
    return this->stack.empty();
  }

  bool operator==(const LeafIterator& R) {
    return this->curr == R.curr;
  }

  bool operator!=(const LeafIterator& R) {
    return this->curr != R.curr;
  }

private:
  std::stack<leaf<T>*> stack;
  leaf<T>* curr;

  virtual void moveToNextValid() {
    this->curr = nullptr;
    while (!stack.empty()) {
      leaf<T>* node = stack.top();

      stack.pop();
      if (!node->children) {curr = node; return;}

      for (int i = 0; i < node->n_children; i++) stack.push(&(node->children[i]));
    }
  }
};
}
