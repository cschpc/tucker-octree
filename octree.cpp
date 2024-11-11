#include <iostream>
#include <cassert>
#include <initializer_list>
#include <bitset>
#include <memory>
#include <array>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>

#include <chrono>

#include <zfp.hpp>
#include <zfp/constarray1.hpp>
#include <zfp/array1.hpp>

#include "tree.hpp"

extern "C" {
#include "octree_compressor.h"
}

#ifndef VDF_REAL_DTYPE
#define VDF_REAL_DTYPE float
#endif

#define MIN(a,b) a<b?a:b
#define MAX(a,b) a>b?a:b

#define assertm(expression, message) assert(((void)message, expression))

#define starttime  { auto start = chrono::high_resolution_clock::now(); 
#define endtime \
  auto stop = chrono::high_resolution_clock::now(); \
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start); \
  cout << duration.count(); }

namespace tree_compressor {

using namespace tree;

struct logging {
public:
  static int count;
};

int logging::count = 1;

void here() { 
#if BRIEF_DEBUG || VERBOSE_DEBUG
  std::cout << "Here (" << logging::count << ")\n"; 
  logging::count++;
#endif
}

/* Forward definitions */

template<size_t N>
struct OctreeCoordinates;

/* End of forward definitions */

template <typename T, size_t N>
class indexrange 
{
private: 
  using arr = std::array<T,N>;
  arr a;
  arr b;
  std::array<std::array<size_t, N>,N> Jk;

  void setJk() {
    for (size_t mode = 0; mode < N; mode++) {
      this->Jk[mode][0] = 0;
      for (int k = 1; k <= N; k++) {
        size_t acc = 1;
        for (int l = 1; l <= k-1; l++) {
          if (l-1 != mode) acc = acc * this->size(l-1);
        }
        this->Jk[mode][k-1] = acc;
      }
      this->Jk[mode][mode] = 0;
    }

#ifdef VERBOSE_DEBUG
    for (size_t mode = 0; mode < N; mode++) {
      for (int k = 0; k < N; k++) {
        std::cout << Jk[mode][k] << " ";
      }
      std::cout << std::endl;
    }
#endif

  }

public:

  indexrange<T,N>() {}
  indexrange<T,N>(arr &aa, arr &bb) : a(aa), b(bb) {this->setJk();}
  indexrange<T,N>(arr &&aa, arr &&bb) : a(aa), b(bb) {this->setJk();}

  std::tuple<T, T> operator()(size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }
  std::tuple<T, T> operator[](size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }

  template<typename M1, typename ...M>
    size_t get_J(uint8_t mode, uint8_t p, M1 head, M ...rest) const {
      return this->Jk[mode][p]*head - this->Jk[mode][p] + this->get_J(mode, p+1, rest...);
    }

  template<typename M0>
    size_t get_J(uint8_t mode, uint8_t p, M0 rest) const {
      return this->Jk[mode][p]*rest - this->Jk[mode][p];
    }

  size_t size(uint8_t dim) {return this->b[dim]-this->a[dim] + 1;}

  template<typename T_, size_t N_>
    friend std::ostream& operator <<(std::ostream &o, const indexrange<T_, N_> R);

  indexrange<T, N> divide(uint8_t subcube) {
    indexrange<T, N> D(this->a, this->b);
    T two(2);
    T one(1); // TODO: calculate correct remainder, if T is a floating point type, one should be 0!
    for(uint8_t dim=0; dim < 3; dim++) {

      if (this->size(dim) > 3) { // TODO: 3 is magic number related to core_size
        if (subcube & (1 << dim)) { 
          D.a[dim] = this->a[dim] + (this->b[dim]-this->a[dim])/two + one;
          D.b[dim] = this->b[dim];
        } else {
          D.a[dim] = this->a[dim];
          D.b[dim] = this->a[dim] + (this->b[dim]-this->a[dim])/two;
        }
      } else {
        D.a[dim] = this->a[dim];
        D.b[dim] = this->b[dim];
      }
    }
    D.setJk();
    return D;
  }

  const std::array<T,N>& getA() const {
    return this->a;
  }

  const std::array<T,N>& getB() const {
    return this->b;
  }

  indexrange<T,N> getsubrange(OctreeCoordinates<N> coords) {
    indexrange<T,N> acc = *this;
    leaf<indexrange<T,N>,N> node(*this);
    leaf<indexrange<T,N>,N>* p_node = &node;
    for(auto it = coords.getCoords().end(); it-- != coords.getCoords().begin(); ) {
      uint8_t child = (*it)[2]*4+(*it)[1]*2+(*it)[0];
      p_node = &divide_leaf(*p_node).children[child];
      acc = p_node->data;
    }
    return acc;
  }

};


template <typename T, size_t N> 
std::ostream& operator <<(std::ostream &o, const indexrange<T,N> R) 
{
  for (uint8_t dim=0; dim<3; dim++) {
     o << unsigned(R.a[dim]);
     o << ":";
     o << unsigned(R.b[dim]);
     if (dim < 2) o << "-";
  }
  return o;
}

template <typename T> 
leaf<indexrange<T,3>,3>& divide_leaf(leaf<indexrange<T,3>,3> &root) 
{

  if (!(root.children == NULL)) return root;

  root.children = new leaf<indexrange<T,3>,3>[root.n_children];
  /* root.n_children = 8; */
  for(uint8_t lnum = 0; lnum<8; lnum++) {
    root.children[lnum].data = root.data.divide(lnum);
    root.children[lnum].level = root.level + size_t(1);
    root.children[lnum].children = NULL;
    root.children[lnum].coordinate = lnum;
    root.children[lnum].parent = &root;
  }
  return root;
}

template <typename T>
std::ostream& operator <<(std::ostream &o, const leaf<indexrange<T,3>,3>& root) 
{
  using namespace std;

  for (size_t i = 0; i < root.level; i++) o << "    ";
  if (root.level == 0) {
    o << "*" << endl; 
  } 
  else {
    o << bitset<3>(root.coordinate) << endl;
  }

  if (!(root.children == NULL)) { 
    for (size_t lnum = 0; lnum < 8; lnum++)
    o << root.children[lnum];
  }

 return o;
}

void hline() {
  /* std::cerr << "#############################################" << std::endl; */
}


template <typename T, typename L, size_t N>
struct TensorView {

private:

  indexrange<L,N> I;
  Eigen::Tensor<T,N,Eigen::ColMajor>& datatensor;

  template<typename ...M>
  size_t get_J_(uint8_t mode, uint8_t p, M ...rest) const {
    return this->I.get_J(mode, p, rest...);
  }

  T residual = -1;

public:

  TensorView(Eigen::Tensor<T,N,Eigen::ColMajor> &datatensor) : datatensor(datatensor) {};
  TensorView(Eigen::Tensor<T,N,Eigen::ColMajor> &datatensor, indexrange<L, N> I) : datatensor(datatensor),  I(I) {};

  void setIndexrange(indexrange<L,N> I) { this->I = I;}

  indexrange<L,N>& getIndexrange() { return this->I; }

  // TODO: iterator over elements so we wouldn't need always do std::enable_if_t
  
  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
    void fill(T x) {
      for (size_t i3 = 0; i3 < this->size(2); i3++) 
      for (size_t i2 = 0; i2 < this->size(1); i2++) 
      for (size_t i1 = 0; i1 < this->size(0); i1++) 
        {
        (*this)(i1,i2,i3) = x;
        }
    }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
    T get_residual() {

      if (this->residual > 0) return this->residual;

      auto MAX = [](T a,T b) { 
        /* return a*a + b; */
        return a > b ? a : b; 
      };
      T acc = T(0);
      for (size_t i3 = 0; i3 < this->size(2); i3++) 
      for (size_t i2 = 0; i2 < this->size(1); i2++) 
      for (size_t i1 = 0; i1 < this->size(0); i1++) 
        {
        acc = MAX(abs((*this)(i1,i2,i3)), acc);
        }
    return acc;
    }

  template<typename ...M>
  size_t get_J(uint8_t mode, M ...rest) const {
    return 1 + this->get_J_(mode, 0, rest...);
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
  T sqnorm() {
    T acc = T(0);
    for (size_t i3 = 0; i3 < this->size(2); i3++) 
    for (size_t i2 = 0; i2 < this->size(1); i2++) 
    for (size_t i1 = 0; i1 < this->size(0); i1++) 
      {
      acc = acc +((*this)(i1,i2,i3))*((*this)(i1,i2,i3));
      }
  return acc;
  }

  const std::array<L,N>& getA() const {
    std::cerr << "we shouldn't be here (TensorView::getB)";
    exit(1);
    return this->I.getA();
  }

  const std::array<L,N>& getB() const {
    std::cerr << "we shouldn't be here (TensorView::getA)";
    exit(1);
    return this->I.getB();
  }

  template<typename T_, size_t N_>
  friend std::ostream& operator <<(std::ostream &o, const indexrange<T_, N_> R);

  template<size_t N_ = N, std::enable_if_t<N_==2,int> = 0, typename... Ts>
  T& operator() (Ts... K) {
#ifdef RANGE_CHECK // TODO: implement range check for indexrange
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);
    return this->datatensor(std::get<0>(this->I[0]) + std::get<0>(Ktpl),
                            std::get<0>(this->I[1]) + std::get<1>(Ktpl));
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
  T operator() (Ts... K) const {
#ifdef RANGE_CHECK
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);

    return (this->datatensor)(std::get<0>(this->I(0)) + std::get<0>(Ktpl),
                        std::get<0>(this->I(1)) + std::get<1>(Ktpl),
                        std::get<0>(this->I(2)) + std::get<2>(Ktpl));
  }
/* #endif */

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
  T& operator() (Ts... K) {
#ifdef RANGE_CHECK
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);

    return (this->datatensor)(std::get<0>(this->I(0)) + std::get<0>(Ktpl),
                        std::get<0>(this->I(1)) + std::get<1>(Ktpl),
                        std::get<0>(this->I(2)) + std::get<2>(Ktpl));
  }

  L size(const uint8_t dim) const {
    return std::get<1>(this->I[dim]) - std::get<0>(this->I[dim]) + 1;
  }

  size_t cosize(const uint8_t dim) const {
    size_t acc = L(1);
    for (size_t i = 0; i < N; i++) {
      acc *= this->size(i);
    };
    acc = acc / this->size(dim);
#if VERBOSE_DEBUG
    std::cout << "Requested cosize: " << acc << std::endl;
#endif
    return acc;
  }

  size_t datasize() const {
    size_t acc = L(1);
    for (size_t i = 0; i < N; i++) {
      acc *= this->size(i);
    };
    return acc;
  }

};

template <typename T, typename L, size_t N>
std::ostream& operator <<(std::ostream &o, const TensorView<T,L,N>& view) {
  using namespace std;
  switch(N) {
    case 3:
      for (int k=0; k<view.size(2); k++) {
        for (int i=0 ; i<view.size(0); i++) {
          for (int j=0; j<view.size(1); j++) {
            cout << view(i,j,k) << ", ";
          }
          cout << endl;
        } cout << endl;
      }
       break;
       }
  return o;
}

template <typename T, typename L, typename M>
void fold_tensor_to_normal_matrix(const TensorView<T,L,3>& A, uint8_t mode, M& R) {
#ifdef VERBOSE_DEBUG
  std::cout << "Attempting to resize at " << A.size(mode) <<std::endl;
#endif
 R.resize(A.size(mode), A.size(mode));

 assertm(mode >= 0 && mode <=2, "Invalid fold mode");

 for (L j = 0; j < A.size(mode); j++) for (L i = 0; i < A.size(mode); i++) R(i,j) = T(0);

 switch(mode){
   case 0:
     for (size_t p = 0; p < A.size(1); p++) {
       for (size_t q = 0; q < A.size(2); q++) {
         for (size_t j = 0; j < A.size(mode); j++) {
           for (size_t i = 0; i<A.size(mode); i++) {
         R(i,j) = R(i,j) + A(i,p,q)*A(j,p,q);
       }}}}
     break;
   case 1:
     for (size_t q = 0; q < A.size(2); q++) {
       for (size_t j = 0; j < A.size(mode); j++) {
         for (size_t i = 0; i<A.size(mode); i++) {
           for (size_t p = 0; p < A.size(0); p++) {
         R(i,j) = R(i,j) + A(p,i,q)*A(p,j,q);
       }}}}
     break;
   case 2:
     for (size_t q = 0; q < A.size(1); q++) {
       for (size_t j = 0; j < A.size(mode); j++) {
         for (size_t i = 0; i < A.size(mode); i++) {
           for (size_t p = 0; p < A.size(0); p++) {
         R(i,j) = R(i,j) + A(p,q,i)*A(p,q,j);
       }}}}
     break;
 }

}

template <typename T, typename L>
void fold_tensor_vector_prod(const TensorView<T,L,3>& A, uint8_t mode, 
                             const T* x_in, 
                             T* y_out)
{
 assertm(mode >= 0 && mode <=2, "Invalid fold mode");
 L len = A.size(mode);

 switch(mode) {
   case 0:
     for (size_t p = 0; p < A.size(1); p++) {
       for (size_t q = 0; q < A.size(2); q++) {
         for (size_t j = 0; j < A.size(0); j++) {
           for (size_t i = 0; i<A.size(0); i++) {
             y_out[i] = y_out[i] + A(i,p,q)*A(j,p,q)*x_in[j];
           }}}}
     break;
   case 1:
     for (size_t q = 0; q < A.size(2); q++) {
       for (size_t j = 0; j < A.size(1); j++) {
         for (size_t i = 0; i<A.size(1); i++) {
           for (size_t p = 0; p < A.size(0); p++) {
             y_out[i] = y_out[i] + A(p,i,q)*A(p,j,q)*x_in[j];
           }}}}
     break;
   case 2:
     for (size_t q = 0; q < A.size(1); q++) {
       for (size_t j = 0; j < A.size(2); j++) {
         for (size_t i = 0; i < A.size(2); i++) {
           for (size_t p = 0; p < A.size(0); p++) {
             y_out[i] = y_out[i] + A(p,q,i)*A(p,q,j)*x_in[j];
           }}}}
     break;
 }

}

template <typename T, typename L>
void fold_tensor_vector_prod(const TensorView<T,L,3>& A, uint8_t mode, 
                             Eigen::Vector<T, Eigen::Dynamic>& x_in, 
                             Eigen::Vector<T, Eigen::Dynamic>& y_out)
{
 assertm(mode >= 0 && mode <=2, "Invalid fold mode");
 L len = A.size(mode);

 switch(mode) {
   case 0:
     for (size_t p = 0; p < A.size(1); p++) {
       for (size_t q = 0; q < A.size(2); q++) {
         for (size_t j = 0; j < A.size(0); j++) {
           for (size_t i = 0; i<A.size(0); i++) {
         y_out(i) = y_out(i) + A(i,p,q)*A(j,p,q)*x_in(j);
           }}}}
     break;
   case 1:
     for (size_t q = 0; q < A.size(2); q++) {
       for (size_t j = 0; j < A.size(1); j++) {
         for (size_t i = 0; i<A.size(1); i++) {
           for (size_t p = 0; p < A.size(0); p++) {
         y_out(i) = y_out(i) + A(p,i,q)*A(p,j,q)*x_in(j);
       }}}}
     break;
   case 2:
     for (size_t q = 0; q < A.size(1); q++) {
       for (size_t j = 0; j < A.size(2); j++) {
         for (size_t i = 0; i < A.size(2); i++) {
           for (size_t p = 0; p < A.size(0); p++) {
         y_out(i) = y_out(i) + A(p,q,i)*A(p,q,j)*x_in(j);
       }}}}
     break;
 }
}

template <typename T, typename L>
void fold_tensor_vector_prod(const TensorView<T,L,3>& A, const uint8_t mode,
                             const T* x_in, T* y_out, T* work) {

  fold_tensor_vector_prod(A, mode, x_in, y_out, work, T(1));

}

template <typename T, typename L>
void fold_tensor_vector_prod(const TensorView<T,L,3>& A, const uint8_t mode,
                             const T* x_in, T* y_out, T* work, T scale)
{
  using namespace std;
  assertm(mode >= 0 && mode <=2, "Invalid fold mode");
  array<size_t,3> len = {A.size(0), A.size(1), A.size(2)};
  size_t colen = A.cosize(mode);

  for (size_t i = 0; i<colen; i++) work[i] = T(0);
  for (size_t i = 0; i<len[mode]; i++) y_out[i] = T(0);

  size_t J, I;

  switch(mode){
    case 0:
      for (size_t k = 0; k < len[2]; k++) { 
        for (size_t i = 0; i < len[0]; i++) {
            for (size_t j = 0; j < len[1]; j++) {
            J = A.get_J((uint8_t) mode, i+1,j+1,k+1)-1;
            I = i;
            work[J] = work[J] + A(i,j,k)*x_in[I]*scale;
          }}}
      break;

    case 1:
      for (size_t k = 0; k < len[2]; k++) { 
        for (size_t j = 0; j < len[1]; j++) {
          for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J((uint8_t) mode, i+1,j+1,k+1)-1;
            I = j;
            work[J] = work[J] + A(i,j,k)*x_in[I]*scale;
          }}}
      break;

    case 2:
      for (size_t k = 0; k < len[2]; k++) { 
        for (size_t j = 0; j < len[1]; j++) {
          for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J((uint8_t) mode, i+1,j+1,k+1)-1;
            I = k;
            work[J] = work[J] + A(i,j,k)*x_in[I]*scale;
          }}}
      break;
  }

  switch(mode) {
    case 0:
      for (size_t k = 0; k < len[2]; k++) { 
          for (size_t j = 0; j < len[1]; j++) {
            for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J(mode, i+1,j+1,k+1)-1;
            I = i; 
            y_out[I] = y_out[I] + A(i,j,k)*work[J]*scale;
          }}}
      break;

    case 1:
      for (size_t k = 0; k < len[2]; k++) { 
          for (size_t j = 0; j < len[1]; j++) {
        for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J(mode, i+1,j+1,k+1)-1;
            I = j; 
            y_out[I] = y_out[I] + A(i,j,k)*work[J]*scale;
          }}}
      break;

    case 2:
      for (size_t k = 0; k < len[2]; k++) { 
        for (size_t j = 0; j < len[1]; j++) {
          for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J(mode, i+1,j+1,k+1)-1;
            I = k; 
            y_out[I] = y_out[I] + A(i,j,k)*work[J]*scale;
          }}}
      break;

  }

}

template <typename T, typename L, size_t N>
class NormalFoldProd
{
private:
  TensorView<T,L,N> view;
  int mode=-1;
  T* work = NULL;
  T scale;
public:
  using Scalar = T;

  void setMode(uint8_t mode) {
    this->mode = mode; 

    if (this->work != NULL) delete[] this->work;

    this->work = new T[view.cosize(mode)];
#if VERBOSE_DEBUG
    std::cout << "Allocated work size " << view.cosize(mode) << std::endl;
#endif
  }
  uint8_t getMode() const {return this->mode;}

  NormalFoldProd(TensorView<T,L,N> view, T scale) : view(view), scale(scale) { };

  Eigen::Index rows() const {return this->view.size(this->mode);}
  Eigen::Index cols() const {return this->rows();}

  void perform_op(const T* x_in, T* y_out) const
    {
    for (size_t i = 0; i<this->rows(); i++) y_out[i] = T(0);
    /* fold_tensor_vector_prod(this->view, this->getMode(), x_in, y_out); */ 
    /* std::cout << "at perform op mode: " << int(this->getMode()) << std::endl; */
    fold_tensor_vector_prod(this->view, this->getMode(), x_in, y_out, this->work, this->scale); 
    here();
    }
  ~NormalFoldProd(){ if (work != NULL) delete[] work;};
};


template<size_t N>
struct OctreeCoordinates {
private:
  std::vector<std::bitset<N>> c = {};
  uint8_t level;
public:


  OctreeCoordinates() {};

  template<typename T>
    OctreeCoordinates(T x, size_t level) : level(level){

      T mask = 0;
      for (int n = 0; n < N; n++) mask += T(1)<<n;

      for (int n = 0; n < level; n++) {
        c.push_back(mask & (x >> n*N));
      }
    }

  const std::vector<std::bitset<N>>& getCoords() const { return c; }

  void addCoord(std::bitset<N> x) { c.push_back(x); }

  bool empty() { return this->c.size() == 0; }

  uint8_t getLevel() const {return this->level;}

  template<typename T_, size_t N_>
    friend OctreeCoordinates<N_> leaf_to_coordinates(const leaf<indexrange<T_,N_>,N_>& L);

  template<size_t N_>
  friend std::ostream& operator <<(std::ostream &o, const OctreeCoordinates<N_>& C);

  template<typename T>
    T toAtomic() const {
      T C = T(0);
      size_t acc = 0;
      for(auto it : c) {
        for(int n=0; n<N; n++) {
          C += it[n]*(T(1)<<(acc++));
        }
      }
      return C;
    }

  template<typename T>
    void fromAtomic(T x, size_t level) {
      const T mask = 0b111;
      c.clear();
      for (int n=0; n<level; n++) {
        assertm(false, "not implemented");
      }
    }

  template<typename T>
  indexrange<T,N> toIndexRange(indexrange<T,N> root) {
    auto acc = root;
    for (auto& it : this->c) {
      uint8_t subcube = T(0);
      for(int n=0; n<N; n++) subcube+=it[n]*(T(1)<<n);
      acc = acc.divide(subcube);
    }
  }
};


template<size_t N>
std::ostream& operator <<(std::ostream &o, const OctreeCoordinates<N>& C){
 for(auto it = C.c.end(); it-- != C.c.begin(); ) o << (*it) << " ";
 return o;
}

template<typename T, size_t N>
OctreeCoordinates<N> leaf_to_coordinates(const leaf<indexrange<T,N>,N>& L)
{
  using namespace std;
  OctreeCoordinates<N> oc;
  auto* acc = &L;
  while(acc->parent != NULL) {
    oc.c.push_back(static_cast<bitset<N>>(acc->coordinate));
    acc = acc->parent;
  }
  oc.level = L.level;
  return oc;
}

template<typename T, typename L, size_t core_rank, size_t N>
struct Tucker {
private:
  /* TensorView<T,L,N>& view; */
  std::unique_ptr<TensorView<T,L,N>> view_ptr; // TODO: move view_ptr out of class and give it as reference in various places
  T scale;
  OctreeCoordinates<N> coordinates;

public:

  /* TODO: the main data of Tucker is now in public visibility */
  Eigen::Tensor<T, N> core;
  std::array<Eigen::Matrix<T, Eigen::Dynamic,core_rank>, N> factors;

public:

  void setCoordinates(OctreeCoordinates<N> coordinates) {
    this->coordinates=coordinates;
  };

  const OctreeCoordinates<N>& getCoordinates() const {return this->coordinates;}


  T getScale() const { return this->scale; }

  Tucker(Eigen::Tensor<T,N> core, 
         std::array<Eigen::Matrix<T, Eigen::Dynamic, core_rank>, N> factors, 
         OctreeCoordinates<N> coordinates) : 
    core(core), factors(factors), coordinates(coordinates) {}

  Tucker(std::unique_ptr<TensorView<T,L,N>> view_ptr) : view_ptr(std::move(view_ptr)) {
    this->init(); 
  }

  Tucker(TensorView<T,L,N>& view) : view_ptr(&view) {
    this->init();
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0>
  void make_core() {
    this->core.resize(core_rank, core_rank, core_rank);

  TensorView<T,L,N>& view = *(this->view_ptr);
    for(size_t j1 = 0; j1<core_rank; j1++) {
      for(size_t j2 = 0; j2<core_rank; j2++) {
        for(size_t j3 = 0; j3<core_rank; j3++) {
          core(j1,j2,j3) = T(0);
        }}}

    for(size_t j1 = 0; j1<core_rank; j1++)
    for(size_t j2 = 0; j2<core_rank; j2++)
    for(size_t j3 = 0; j3<core_rank; j3++)
      {
      for(size_t i1 = 0; i1<view.size(0); i1++) // TODO: correct ranges
      for(size_t i2 = 0; i2<view.size(1); i2++)
      for(size_t i3 = 0; i3<view.size(2); i3++)
        {
        this->core(j1,j2,j3) = this->core(j1,j2,j3) + 
          (view(i1,i2,i3))*
          (this->factors[0](i1,j1))*
          (this->factors[1](i2,j2))*
          (this->factors[2](i3,j3));

        }
      }
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0>
  void fill_residual() {
    /* No scaling here, core is computed via projections with factors! */
    this->fill_residual(VDF_REAL_DTYPE(1), VDF_REAL_DTYPE(-1)); 
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0>
  void fill_residual(TensorView<T,L,N>& main_view, const VDF_REAL_DTYPE mult_orig, const VDF_REAL_DTYPE mult_corr) {
    auto view = main_view;
    view.setIndexrange(view.getIndexrange().getsubrange(this->coordinates));

    for(size_t i1 = 0; i1 < view.size(0); i1++)
    for(size_t i2 = 0; i2 < view.size(1); i2++)
    for(size_t i3 = 0; i3 < view.size(2); i3++) {
      T acc = T(0);
      for(size_t j1 = 0; j1 < core_rank; j1++)
      for(size_t j2 = 0; j2 < core_rank; j2++)
      for(size_t j3 = 0; j3 < core_rank; j3++) {
        acc = acc + 
          this->core(j1,j2,j3)*
          this->factors[0](i1,j1)*
          this->factors[1](i2,j2)*
          this->factors[2](i3,j3);
      }
    view(i1,i2,i3) = mult_orig*view(i1,i2,i3) + mult_corr*acc;
    }

  }

  /* Warning! Mutates the view contents! */
  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0>
  void fill_residual(const VDF_REAL_DTYPE mult_orig, const VDF_REAL_DTYPE mult_corr) {

  TensorView<T,L,N>& view = *(this->view_ptr);

    for(size_t i1 = 0; i1 < view.size(0); i1++)
    for(size_t i2 = 0; i2 < view.size(1); i2++)
    for(size_t i3 = 0; i3 < view.size(2); i3++) {
      T acc = T(0);
      for(size_t j1 = 0; j1 < core_rank; j1++)
      for(size_t j2 = 0; j2 < core_rank; j2++)
      for(size_t j3 = 0; j3 < core_rank; j3++) {
        acc = acc + 
          this->core(j1,j2,j3)*
          this->factors[0](i1,j1)*
          this->factors[1](i2,j2)*
          this->factors[2](i3,j3);
      }
    view(i1,i2,i3) = mult_orig*view(i1,i2,i3) + mult_corr*acc;
    }
  }

private: 
  void init() {
    auto& view = *(this->view_ptr);

    /* TODO: Spectra might struggle finding eigenspaces of big linear ops (300x300 is too big?!)*/
    using namespace Spectra;
    using namespace Eigen;

    for (int m=0;m<N; m++) {
      this->factors[m].resize(view.size(m),core_rank);
    }

    /* TODO: too smart scaling, die gracefully if res < epsilon */
    auto res = MAX(VDF_REAL_DTYPE(1e-16), view.get_residual());
    this->scale = 1/res;

    NormalFoldProd<T, L, N> op(view, this->scale);

    for (size_t mode=0; mode<N; mode++) {
      op.setMode(mode);
      SymEigsSolver<NormalFoldProd<T, L, N>> eigs(op, core_rank, MIN(core_rank+1, view.size(mode)) );
      eigs.init();

      int nconv = eigs.compute(SortRule::LargestMagn);
      auto eigenvalues = eigs.eigenvalues();

#if TUCKER_DEBUG
      std::cout << "Mode " << mode << " converged " << nconv << " eigenvalues:\n" << eigenvalues << std::endl << std::endl;
#endif

      auto eigenvectors = eigs.eigenvectors();
      this->factors[mode].resize(view.size(mode), core_rank);
      this->factors[mode] = eigenvectors;


      auto normi = eigenvectors.norm();
#if TUCKER_DEBUG
      std::cout << "sq-norm of eigenvectors " << normi*normi << std::endl;
#endif
    }
    
    this->make_core();

#if BRIEF_DEBUG
    for (size_t mode=0; mode<3; mode++) std::cout << this->factors[mode] << std::endl << std::endl;
#endif

  }

};

/* auto serialize_leaves */

/* expose this to vlasiator */

/* In testing */
/* void compress_with_octree_method(VDF_REAL_DTYPE* buffer, const size_t Nx, const size_t Ny, const size_t Nz, */ 
/*                                  VDF_REAL_DTYPE tolerance, double& compression_ratio){}; */

/* In production */
/* void compress_with_octree(double* input_buffer, size_t Nx, ..., char* compressed); */
/* void uncompress_with_octree(double* output_buffer, char* compressed); */

template<typename T, typename L, size_t core_rank, size_t N, typename atomic_coord_type>
struct SerialTucker {

private:
  using vecT = std::vector<T>;
  using vecS = std::vector<size_t>;
  using uptr = std::unique_ptr<Tucker<T,L,core_rank,N>>;
  using zfp_array = zfp::const_array1<T>;

  vecT serialized;
  size_t n_leaves;
  std::vector<atomic_coord_type> leaf_coordinates;
  std::vector<uint8_t> leaf_levels;
  size_t core_size;

  T core_scale = 0;
  size_t n_core;

  indexrange<L, N> root_range;

  uchar** serialized_bytes = nullptr;

  // TODO: 
  void make_serialized_bytes() {
    serialized_bytes = (uchar**)malloc(sizeof(uchar*));
    auto conf = zfp_config_accuracy(1e-8);
    
    zfp_array compressed(this->serialized.size(), conf, this->serialized.data());
    size_t storage_size = compressed.size_bytes();
    uchar* compressed_data = (uchar*) compressed.compressed_data();

    *serialized_bytes = (uint8_t*)malloc(sizeof(uint8_t)*storage_size);

    std::cout << "storage size: " << storage_size << " vs float data size " << sizeof(T)*this->serialized.size() << std::endl;

    // TODO: this is memcpy but compiler probably knows what to do..
    for(size_t k=0; k<storage_size; ++k) {
      (*serialized_bytes)[k] = compressed_data[k];
    }

  }

  void unmake_serialized_bytes() {

  }

public:

  const size_t& getCoreSizes() { return this->core_size; }
  const vecT& getSerialized() { return this->serialized; }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0>
  SerialTucker(std::vector<uptr>& tuckers, indexrange<L,N> root_range) : root_range(root_range) {
    using namespace std;
    /* auto this = this; */

    const size_t n_per_core = pow(core_rank,N);
    /* vector<size_t>& factor_lengths = this->factor_lengths; // Calculate factors lengths from leaf coordinates */

    size_t total_factorlengths = 0;

    for (auto& tuck : tuckers) {

      auto inds = this->root_range.getsubrange(tuck->getCoordinates());

      for (int dim = 0; dim < N; dim++){
        total_factorlengths += inds.size(dim)*core_rank;
      }
      const OctreeCoordinates<N>& coords = tuck->getCoordinates();

      atomic_coord_type atomic_coords = coords.template toAtomic<atomic_coord_type>();
      this->leaf_coordinates.push_back(atomic_coords);
      this->leaf_levels.push_back(coords.getLevel());
    }

    size_t ser_len = tuckers.size()*n_per_core + total_factorlengths;

    vector<T> &serialized = this->serialized;
    serialized.resize(ser_len);

    size_t acc = 0;
    size_t bias; 
    this->core_scale = T(0);
    for (auto& tuck : tuckers) {
      bias = n_per_core*acc;
      for (size_t k = 0; k<n_per_core; ++k) {
        T x = tuck->core.data()[k];
        this->core_scale = MAX(this->core_scale, abs(x));
        serialized[bias + k] = x; //tuck->core.data()[k];
      }
      ++acc;
    }

    bias = n_per_core*tuckers.size();
    /* this->core_size = acc*n_per_core; */

    /* for (size_t k = 0; k < this->core_size; ++k) serialized[k] = serialized[k] / this->core_scale; */

    acc = 0;
    for (auto &tuck : tuckers) {
      for(int n=0; n<N; ++n) {
        for(int m=0;m<tuck->factors[n].size(); ++m) {
          serialized[bias+acc] = tuck->factors[n].data()[m];
          ++acc;
        }
      }
    }
    /* this->make_serialized_bytes(); */
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0>
  /* void */ 
  std::vector<std::unique_ptr<Tucker<T,L,core_rank,N>>> Deserialize() {
    using namespace Eigen;
    using namespace std;

    const size_t n_per_core = this->core_size;
    size_t n_leaves = this->core_size / n_per_core; //this->leaf_coordinates.size();
    std::vector<TensorMap<Tensor<T, N>>> cores;
    std::vector<std::array<Matrix<T, Dynamic, core_rank>,N>> factorss;
    std::vector<OctreeCoordinates<N>> coordss;
    std::vector<unique_ptr<Tucker<T,L,core_rank,N>>> tuckers;

#if VERBOSE_DEBUG
    std::cout << "n_leaves: " << n_leaves << endl;
    std::cout << "n_per_core: " << n_per_core << endl;
#endif

    /* Rescale back */
    /* for (size_t k = 0; k < this->core_size; ++k) this->serialized[k] = this->serialized[k] * this->core_scale; */

    for (int m = 0; m < n_leaves; ++m) {
      T* data = this->serialized.data();
      cores.push_back(TensorMap<Tensor<T,N>>(this->serialized.data()+m*n_per_core, core_rank, core_rank, core_rank));
    }
    
    size_t bias = n_leaves*n_per_core;

    for (int m = 0; m < n_leaves; ++m) {
      T* data = this->serialized.data()+bias;

      auto coords = this->leaf_coordinates[m];

      auto o_coords = OctreeCoordinates<N>(coords, this->leaf_levels[m]);
      coordss.push_back(o_coords);

      indexrange<L, N> leaf_inds = this->root_range.getsubrange(o_coords);

      const std::array<size_t,3> FL = {
        leaf_inds.size(0),
        leaf_inds.size(1),
        leaf_inds.size(2)};

#if VERBOSE_DEBUG
      for (auto& it : FL) cout << "FL: " << it << endl;
#endif

      factorss.push_back({
                         Map<Matrix<T, Dynamic, core_rank>>(data                           , FL[0], core_rank),
                         Map<Matrix<T, Dynamic, core_rank>>(data + FL[0]*core_rank         , FL[1], core_rank),
                         Map<Matrix<T, Dynamic, core_rank>>(data + (FL[1]+FL[0])*core_rank , FL[2], core_rank)});
      bias += (FL[0] + FL[1] + FL[2])*core_rank;
    }

    // TODO: just one loop that does core, factors and coords
    for (int m = 0; m < n_leaves; ++m) {
      tuckers.push_back(unique_ptr<Tucker<T,L,core_rank,N>>(new Tucker<T,L,core_rank,N>(cores[m], factorss[m], coordss[m])));
    }

#if VERBOSE_DEBUG
    int m = 0;
    for (auto& it : factorss) {
      cout << "leaf: " << ++m << endl;
      for (int n = 0; n<3; n++){
        cout << "\tfactor: "<< n << endl;
        cout << it[n] << endl;
      }
    }
#endif

    return tuckers;

  }
  
  template<typename T_, typename L_, size_t core_rank_, size_t N_, typename atomic_coord_type_>
    friend std::ostream& operator <<(std::ostream &o, const SerialTucker<T_, L_, core_rank_, N_, atomic_coord_type_>& R);
};

template<typename T_, typename L_, size_t core_rank_, size_t N_, typename atomic_coord_type_>
std::ostream& operator <<(std::ostream &o, const SerialTucker<T_, L_, core_rank_, N_, atomic_coord_type_>& R) {

  using namespace std;
  string corefactor = "C";

  size_t acc = 1;
  for (auto& it : R.serialized) {
    o << corefactor << ", " << acc<< ", " << it << endl;
    if (acc++ == R.core_size) {
      corefactor = "F";
    }
  }
  return o;
}

}

extern "C" {
  /* TODO: deal with situation w/ too small gridsizes */
  void compress_with_octree_method(VDF_REAL_DTYPE* buffer, 
                                   const size_t Nx, const size_t Ny, const size_t Nz, 
                                   VDF_REAL_DTYPE tolerance, double& compression_ratio) {
    using namespace Eigen;
    using namespace tree_compressor;
    using namespace std;

    TensorMap<Tensor<VDF_REAL_DTYPE, 3,ColMajor>> datamap(buffer, Nx, Ny, Nz);
    Tensor<VDF_REAL_DTYPE, 3, ColMajor> datatensor(datamap);

    typedef size_t UI;

    indexrange<UI, 3> K({0,0,0},{UI(Nx-1),UI(Ny-1),UI(Nz-1)});

    auto view = TensorView<VDF_REAL_DTYPE, UI, 3>(datatensor, K);

    auto tree = leaf<indexrange<UI, 3>,3>();
    tree.data = K;

    auto res0 = view.get_residual();

    const int maxiter = 40;

    std::vector<unique_ptr<Tucker<VDF_REAL_DTYPE,UI,2,3>>> tuckers;

    VDF_REAL_DTYPE relres = 10.0;
    
    size_t iter = 0;

    while((iter < maxiter) && (relres > tolerance)) {
      ++iter;
      /* cout << "iter: " << iter << " sqnorm of view " << view.sqnorm() << "\t\t|\t"; */

      VDF_REAL_DTYPE residual = -1.0;

      // find worst leaf
      leaf<indexrange<UI,3>,3>* worst_leaf;
      /* for (auto it = tree.begin(); it != tree.end(); ++it) { */
      /* for (auto* it : tree) { */
      for (auto&& leaf : tree) {
        auto c_view = TensorView<VDF_REAL_DTYPE, UI, 3>(datatensor, leaf.data);
        VDF_REAL_DTYPE c_residual = c_view.get_residual();
        if (c_residual > residual) {
          residual = c_residual;
          worst_leaf = &leaf;
        }
      };

      // improve worst leaf
      relres = residual/res0;

      /* cout << "tol: " << tolerance << " worst leaf: " << worst_leaf->level << " : "<<  worst_leaf->data << " residual: " << relres << endl; */
      divide_leaf(*worst_leaf);

      std::unique_ptr<TensorView<VDF_REAL_DTYPE,UI,3>> c_view(new TensorView<VDF_REAL_DTYPE,UI,3>(datatensor, worst_leaf->data));
      std::unique_ptr<Tucker<VDF_REAL_DTYPE, UI, 2, 3>> tuck(new Tucker<VDF_REAL_DTYPE,UI,2,3>(std::move(c_view))); /* TODO: save tucker to leaf! */
      tuck->fill_residual();
      tuckers.push_back(std::move(tuck));
    }

    view.fill(VDF_REAL_DTYPE(0));
    for (auto& it : tuckers) {
      auto& tuck = *(it);
      tuck.fill_residual(VDF_REAL_DTYPE(1), VDF_REAL_DTYPE(1));
    }

  }

};
