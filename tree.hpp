/*
 * Tucker-Octree multiresolution voxel data compression library
 * Copyright (C) 2024 CSC - IT Center for Science Ltd
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see
 * <https://www.gnu.org/licenses/>.
 */

/* 
 * Author: Juhani Kataja 
 * Affiliation: CSC - IT Center for Science Ltd
 * Email: juhani.kataja@csc.fi
 */

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

  enum empty_t {};

/* This is N-tree leaf (or node!) */
template <typename T, size_t N> struct leaf {
  T data;
  leaf<T,N> *children = NULL;
  leaf<T,N> *parent = NULL;
  size_t level = 0;
  int8_t coordinate = -1;  // TODO: N <= 7
  const size_t n_children = pow(2,N);
public:
  /* leaf(T data): data(data) {} */

  leaf() {};
  leaf(T data) : data(data) {};
  leaf(empty_t x) {};

  ~leaf(){
    data.~T();
    if (!(this->children == NULL)) {
      delete[] children;
    }
  }

  bool isempty() {
    return (this->children == NULL) && (this->parent == NULL) && (this->coordinate == -1);
  }

  /* Iterate over leaves with for(auto&& leaf : tree) */ 
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
    return (this->curr);
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
      if (!node->children) {this->curr = node; return;}

      for (int i = 0; i < node->n_children; i++) stack.push(&(node->children[i]));
    } 
  }
};

}
