
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <initializer_list>
#include <bitset>
#include <memory>
#include <array>
#include <cassert>
#include <cmath>
#include <Spectra/SymEigsSolver.h>
#include <chrono>
#include <argparse.hpp>

#define MIN(a,b) a<b?a:b

#define assertm(expression, message) assert(((void)message, expression))

#define starttime  { auto start = chrono::high_resolution_clock::now(); 
#define endtime \
  auto stop = chrono::high_resolution_clock::now(); \
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start); \
  cout << duration.count(); }

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

template <typename T> struct leaf {
  T data;
  leaf<T> *children = NULL;
  size_t level = 0;
  int8_t coordinate = -1; 
public:

  ~leaf(){
    data.~T();
    if (!(this->children == NULL)) {
      delete[] children;
    }
  }
};

template <typename T, size_t N> class indexrange 
{
private: 
  std::array<T,N> a;
  std::array<T,N> b;
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

  };

public:

  indexrange<T,N>() {};
  indexrange<T,N>(std::array<T,N> &aa, std::array<T,N> &bb) : a(aa), b(bb) {this->setJk();};
  indexrange<T,N>(std::array<T,N> &&aa, std::array<T,N> &&bb) : a(aa), b(bb) {this->setJk();};

  std::tuple<T, T> operator()(size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }
  std::tuple<T, T> operator[](size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }

  template<typename M1, typename ...M>
    size_t get_J(uint8_t mode, uint8_t p, M1 head, M ...rest) const {
      return this->Jk[mode][p]*head - this->Jk[mode][p] + this->get_J(mode, p+1, rest...);
    }

  template<typename M0>
    size_t get_J(uint8_t mode, uint8_t p, M0 rest) const{
      return this->Jk[mode][p]*rest - this->Jk[mode][p];
    }


  size_t size(uint8_t dim) {return this->b[dim]-this->a[dim] + 1;}

  template <typename S> Eigen::Tensor<S,N> getslice(Eigen::Tensor<S,N> &A){
    using namespace Eigen;
    Eigen::array<Index, N> offsets;
    Eigen::array<Index, N> extents;

    for (uint8_t dim = 0; dim<N; dim++) {
      offsets[dim] = this->a[dim];
      extents[dim] = this->b[dim] - this->a[dim] + 1;
    }

    return A.slice(offsets, extents); // TODO: this returns a copy
  }

  template<typename T_, size_t N_>
    friend std::ostream& operator <<(std::ostream &o, const indexrange<T_, N_> R);

  indexrange<T, N> divide(uint8_t subcube) {
    indexrange<T, N> D(this->a, this->b);
    T two(2);
    T one(1); // TODO: calculate correct remainder, if T is a floatingh point type, one should be 0!
    for(uint8_t dim=0; dim < 3; dim++) {
      if (subcube & (1 << dim)) {
        D.a[dim] = this->a[dim] + (this->b[dim]-this->a[dim])/two + one;
        D.b[dim] = this->b[dim];
      } else {
        D.a[dim] = this->a[dim];
        D.b[dim] = this->a[dim] + (this->b[dim]-this->a[dim])/two;
      }
    }
    return D;
  }


  const std::array<T,N>& getA() const {
    return this->a;
  }

  const std::array<T,N>& getB() const {
    return this->b;
  }

};

template <typename T, size_t N> std::ostream& operator <<(std::ostream &o, const indexrange<T,N> R) 
{
  for (uint8_t dim=0; dim<3; dim++) {
     o << unsigned(R.a[dim]);
     o << ":";
     o << unsigned(R.b[dim]);
     if (dim < 2) o << "-";
  }
  return o;
}

template <typename T> leaf<indexrange<T,3>>& divide(leaf<indexrange<T,3>> &root) 
{

  if (!(root.children == NULL)) return root;

  root.children = new leaf<indexrange<T,3>>[8];
  for(uint8_t lnum = 0; lnum<8; lnum++) {
    root.children[lnum].data = root.data.divide(lnum);
    root.children[lnum].level = root.level + size_t(1);
    root.children[lnum].children = NULL;
    root.children[lnum].coordinate = lnum;
  }
  return root;
}

template <typename T> std::ostream& operator <<(std::ostream &o, const leaf<indexrange<T,3>>& root) 
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

public:

  TensorView(Eigen::Tensor<T,N,Eigen::ColMajor> &datatensor) : datatensor(datatensor) {};
  TensorView(Eigen::Tensor<T,N,Eigen::ColMajor> &datatensor, indexrange<L, N> I) : datatensor(datatensor),  I(I) {};

  template<typename ...M>
  size_t get_J(uint8_t mode, M ...rest) const {
    return 1 + this->get_J_(mode, 0, rest...);
  }


  const std::array<L,N>& getA() const {
    return this->I.getA();
  }

  const std::array<L,N>& getB() const {
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
    return acc;
  }

};


template <typename T, typename L, size_t N>
std::ostream& operator <<(std::ostream &o, const TensorView<T,L,N>& view) {
  using namespace std;
  switch(N) {
    case 3:
      auto A = view.getA();
      auto B = view.getB();
      for (int k=A[2]; k<=B[2]; k++) {
        for (int i=A[0]; i<=B[0]; i++) {
          for (int j=A[1]; j<=B[1]; j++) {
            cout << view(i,j,k) << ", ";
          }
          cout << endl;
        } cout << endl;
      }
       break;
       }
  return o;
}

/* void fold_tensor_to_normal_matrix(const TensorView<T,L,3>& A, uint8_t mode, Eigen::Tensor<T,2,Eigen::ColMajor>& R) { */
/* void fold_tensor_to_normal_matrix(const TensorView<T,L,3>& A, uint8_t mode, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& R) { */

/* M should support resize(P,Q) and M(i,j) indexing. For example Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> */

/* This is very slow... */

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


/* TODO: 
 * - This becomes slow w/ big tensors
 * - Make a POD version of this */

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
                             const T* x_in, T* y_out, T* work)
{
  using namespace std;
  assertm(mode >= 0 && mode <=2, "Invalid fold mode");
  array<L,3> len = {A.size(0), A.size(1), A.size(2)};
  L colen = A.cosize(mode);

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
            work[J] = work[J] + A(i,j,k)*x_in[I];
          }}}
      break;

    case 1:
      for (size_t k = 0; k < len[2]; k++) { 
        for (size_t j = 0; j < len[1]; j++) {
          for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J((uint8_t) mode, i+1,j+1,k+1)-1;
            I = j;
            work[J] = work[J] + A(i,j,k)*x_in[I];
          }}}
      break;

    case 2:
      for (size_t k = 0; k < len[2]; k++) { 
        for (size_t j = 0; j < len[1]; j++) {
          for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J((uint8_t) mode, i+1,j+1,k+1)-1;
            I = k;
            work[J] = work[J] + A(i,j,k)*x_in[I];
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
            y_out[I] = y_out[I] + A(i,j,k)*work[J];
          }}}
      break;

    case 1:
      for (size_t k = 0; k < len[2]; k++) { 
          for (size_t j = 0; j < len[1]; j++) {
        for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J(mode, i+1,j+1,k+1)-1;
            I = j; 
            y_out[I] = y_out[I] + A(i,j,k)*work[J];
          }}}
      break;

    case 2:
      for (size_t k = 0; k < len[2]; k++) { 
        for (size_t j = 0; j < len[1]; j++) {
          for (size_t i = 0; i < len[0]; i++) {
            J = A.get_J(mode, i+1,j+1,k+1)-1;
            I = k; 
            y_out[I] = y_out[I] + A(i,j,k)*work[J];
          }}}
      break;

  }

  /*----------------------------*/
#if 0
  for (size_t i = 0; i < len[0]; i++) {
    for (size_t j = 0; j < len[1]; j++) {
      for (size_t k = 0; k < len[2]; k++) { 
        ijk = {i,j,k};
        J = A.get_J((uint8_t) mode, i+1,j+1,k+1)-1;
        I = ijk[mode];
        work[J] = work[J] + A(i,j,k)*x_in[I];
      }}}

  for (size_t i = 0; i < len[0]; i++) {
    for (size_t j = 0; j < len[1]; j++) {
      for (size_t k = 0; k < len[2]; k++) { 
        ijk = {i,j,k};
        size_t J = A.get_J(mode, i+1,j+1,k+1)-1;
        size_t I = ijk[mode];
        y_out[I] = y_out[I] + A(i,j,k)*work[J];
      }}}
#endif
}

template <typename T, typename L, size_t N>
class NormalFoldProd
{
private:
  TensorView<T,L,N> view;
  int mode=-1;
  T* work = NULL;
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

  NormalFoldProd(TensorView<T,L,N> view) : view(view) { };

  Eigen::Index rows() const {return this->view.size(this->mode);}
  Eigen::Index cols() const {return this->rows();}

  void perform_op(const T* x_in, T* y_out) const
    {
    for (size_t i = 0; i<this->rows(); i++) y_out[i] = T(0);
    /* fold_tensor_vector_prod(this->view, this->getMode(), x_in, y_out); */ 
    /* std::cout << "at perform op mode: " << int(this->getMode()) << std::endl; */
    fold_tensor_vector_prod(this->view, this->getMode(), x_in, y_out, this->work); 
    here();
    }
  ~NormalFoldProd(){ if (work != NULL) delete[] work;};
};


void test_normalprods(std::vector<uint16_t> sizes) {
  using namespace std;
  using namespace Eigen;
  auto datatensor = Tensor<double, 3, ColMajor>(sizes[0],sizes[1],sizes[2]);

  for (int i3 = 0; i3 < sizes[2]; i3++) for (int i2 = 0; i2 < sizes[1]; i2++) for (int i1 = 0; i1 < sizes[0]; i1++)
  datatensor(i1,i2,i3) = sin((M_PIf*(i1+i2+i3))/(sizes[1]+sizes[2]+sizes[0])/3.0);

  std::array<uint16_t,3> siz = {uint16_t(int(sizes[0])-1), uint16_t(int(sizes[1])-1), uint16_t(int(sizes[2])-1)};
  std::array<uint16_t,3> zer = {0,0,0};
  indexrange<uint16_t,3> K(zer, siz);

  /* std::cout << siz[0] << " " << siz[1] << " " << siz[2] << endl; */
  auto view = TensorView<double, uint16_t,3>(datatensor, K);
  NormalFoldProd<double,uint16_t, 3> op(view);

  const int modes[3] = {1,0,2};
  for (int n = 0; n < 3; n++) {
    const int mode = modes[n];
  op.setMode(mode);
  double* x = new double[op.rows()];
  double* y = new double[op.rows()];

  for(int i = 0; i<op.rows(); i++) x[i] = double(i);
  for(int i = 0; i<op.rows(); i++) y[i] = double(i);

  starttime
    op.perform_op(x,y);
  endtime
    cout << ", ";
  delete[] x; delete[] y;
  }
}

template<typename T, typename L, size_t core_rank, size_t N>
struct Tucker {
private:
  Eigen::Tensor<T, N> core;
  std::array<Eigen::Matrix<T, Eigen::Dynamic,core_rank>, N> factors;
  TensorView<T,L,N>& view;
public:

  Tucker(TensorView<T,L,N>& view) : view(view) {

    using namespace Spectra;
    using namespace Eigen;

    for (int m=0;m<N; m++) {
      this->factors[m].resize(this->view.size(m),core_rank);
    }
    
    NormalFoldProd<T,L,N> op(this->view);


    for (size_t mode=0; mode<3; mode++) {
      op.setMode(mode);
      SymEigsSolver<NormalFoldProd<T, uint16_t, 3>> eigs(op, core_rank, MIN(core_rank+4, view.size(mode)) );
      eigs.init();
      int nconv = eigs.compute(SortRule::LargestMagn);
      std::cout << "Mode " << mode << " converged " << nconv << " eigenvalues.\n";
      auto eigenvalues = eigs.eigenvalues();
      auto eigenvectors = eigs.eigenvectors();
      this->factors[mode].resize(view.size(mode), core_rank);
      this->factors[mode] = eigenvectors;
    }
    
    this->make_core();

#if BRIEF_DEBUG
    for (size_t mode=0; mode<3; mode++) std::cout << this->factors[mode] << std::endl << std::endl;
#endif

  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0>
  void make_core() {
    this->core.resize(core_rank, core_rank, core_rank);
    for(size_t i = 0; i<core_rank; i++)
    for(size_t j = 0; j<core_rank; j++)
    for(size_t k = 0; k<core_rank; k++)
      {
      // This calls for some thinking
      // this->core(i,j,k) = this->view(I,J,K) * this->factors ...
      }
  }

};

void test_tensorsizes(std::vector<uint16_t> tensorsizes) {
  using namespace std;
  using namespace Eigen;
  const uint16_t tensorsize = tensorsizes[0];

  auto datatensor = Tensor<double, 3, ColMajor>(tensorsize,tensorsize,tensorsize);

  for (int i3 = 0; i3 < tensorsize; i3++) for (int i2 = 0; i2 < tensorsize; i2++) for (int i1 = 0; i1 < tensorsize; i1++)
  datatensor(i1,i2,i3) = sin(M_PIf*(i1+i2+i3)/tensorsize);

  for (uint16_t tensorsize_m = tensorsizes[1]; tensorsize_m < tensorsizes[2]; tensorsize_m+= tensorsizes[3]){
    indexrange<uint16_t,3> K({0,0,0},{tensorsize_m,tensorsize_m,tensorsize_m});
    auto view = TensorView<double, uint16_t, 3>(datatensor, K);

    cout << tensorsize << ", " << tensorsize_m << ",";
    for (size_t mode=0;mode<3;mode++) {
      starttime
        using namespace Spectra;

      NormalFoldProd<double, uint16_t, 3> op(view);
      op.setMode(mode);
      SymEigsSolver<NormalFoldProd<double, uint16_t, 3>> eigs(op, 2, 6);

      eigs.init();
      int nconv = eigs.compute(SortRule::LargestMagn);
      Vector<double, Dynamic> evalues;

      if (eigs.info() == CompInfo::Successful) evalues = eigs.eigenvalues();
      endtime
        if (mode < 2) cout << ",";
    }
    cout << endl;
  }
}

void test_indexing() {
  using namespace Eigen;
  using namespace std;
  auto datatensor = Tensor<double, 3, ColMajor>(10,20,30);
  indexrange<uint16_t, 3> K({0,0,0},{9,19,29});
  auto view = TensorView<double, uint16_t, 3>(datatensor, K);

  for (int mode = 0; mode < 3; mode++){
  for (int i = 1; i<=10; i++) {
    for (int j = 1 ; j<=20;j++) {
      cout << view.get_J(mode, i,j,15) << '\t';
    }
    cout << endl;
  }
  cout << endl;
  }

}

void test_tucker() {
  using namespace Eigen;
  using namespace std;
  auto datatensor = Tensor<double, 3, ColMajor>(300,300,300);
  indexrange<uint16_t, 3> K({0,0,0},{299,299,299});

  auto view = TensorView<double, uint16_t, 3>(datatensor, K);

  for (int i = view.getA()[0]; i<=view.getB()[0]; i++) 
  for (int j = view.getA()[1]; j<=view.getB()[1]; j++) 
  for (int k = view.getA()[2]; k<=view.getB()[2]; k++) 
    {
    view(i,j,k) = sin(M_PIf64*(i+j+k)/300);
  }

  starttime
  auto tucker = Tucker<double, uint16_t, 2, 3>(view);
  endtime

}

/* TODO:
 * - [ ] timings w/ different tensorsize */

int main(int argc, const char** argv) {
  using namespace std;
  using namespace Eigen;

  struct Myargs : public argparse::Args{
    vector<uint16_t> &tensorsizes = kwarg("b,benchmark", "Size of big tensor, size of current view, max size of view, increment").set_default("0,0,0,2");
    bool &indextest = flag("i,indextest", "test indexing");
    bool &help = flag("h,help", "help");
    bool &tucker = flag("t,tucker", "Test tucker functionality");
    vector<uint16_t> &normaltest = kwarg("n,normalsizes", "Size of normal-vector product tensor").set_default("0,0,0");
  };


  /* parser.ignoreFirstArgument(true); */
  auto args = argparse::parse<Myargs>(argc, argv);
  if (args.help) args.print();

  auto tensorsizes = args.tensorsizes;
  auto normalsize = args.normaltest;

  if (tensorsizes[0] > 0) test_tensorsizes(tensorsizes);

  if (normalsize[0] > 0) test_normalprods(normalsize);

  if (args.indextest) test_indexing();

  if (args.tucker) test_tucker();



/* if (false){ */
/*   hline(); */
/*   TensorView<double, uint16_t, 3> view2 = TensorView<double, uint16_t, 3>(datatensor, K.divide(0).divide(0).divide(0).divide(7));//.divide(1).divide(0).divide(1)); */
/*   for (int i2=0; i2 < view2.size(1); i2++) { */
/*     for (int i1=0; i1 < view2.size(0); i1++) { */
/*       cerr << view2(i2,i1,0) << " "; */
/*     } */
/*     cerr << endl; */
/*   } */
/*   cerr << endl; */
/*   hline(); */
/* } */


  /* char ok; */
  /* cin >> ok; */
  /* cout << "press enter to exit"; */
  /* getchar(); */
}

