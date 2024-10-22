
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

#define assertm(expression, message) assert(((void)message, expression))

#define starttime  { auto start = chrono::high_resolution_clock::now(); 
#define endtime \
  auto stop = chrono::high_resolution_clock::now(); \
  auto duration = chrono::duration_cast<chrono::microseconds>(stop-start); \
  cout << duration.count(); }


template <typename T> struct leaf {
  T data;
  leaf<T> *children = NULL;
  size_t level = 0;
  int8_t coordinate = -1; 
public:

  ~leaf();
};

template <typename T> leaf<T>::~leaf() {
  data.~T();
  if (!(this->children == NULL)) {
    delete[] children;
  }
}

template <typename... T>
struct itrange{
public:
  std::tuple<T...> a;
  std::tuple<T...> b;
  itrange<T...>(std::tuple<T...> a, std::tuple<T...> b) : a(a), b(b) {};
};



template <typename T, size_t N> 
class indexrange {
private: 
  std::array<T,N> a;
  std::array<T,N> b;
  std::array<size_t, N> Jk;
public:
  
  indexrange<T,N>() {};
  indexrange<T,N>(const std::array<T,N> &aa, const std::array<T,N> &bb) : a(aa), b(bb) {};
  indexrange<T,N>(const std::array<T,N> &&aa, const std::array<T,N> &&bb) : a(aa), b(bb) {};

  std::tuple<T, T> operator()(size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }
  std::tuple<T, T> operator[](size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }

  template<typename M1, typename ...M>
  size_t get_J(uint8_t p, M1 head, M ...rest){
    return this->Jk[p]*head - this->Jk[p] + this->get_J(p+1, rest...);
  }

  template<typename M0>
  size_t get_J(uint8_t p, M0 rest){
    return this->Jk[p]*rest - this->Jk[p];
  }

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
};



template <typename T, size_t N>
std::ostream& operator <<(std::ostream &o, const indexrange<T,N> R) {
  for (uint8_t dim=0; dim<3; dim++) {
     o << unsigned(R.a[dim]);
     o << ":";
     o << unsigned(R.b[dim]);
     if (dim < 2) o << "-";
  }
  return o;
}

template <typename T>
leaf<indexrange<T,3>>& divide(leaf<indexrange<T,3>> &root) {

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

template <typename T>
std::ostream& operator <<(std::ostream &o, const leaf<indexrange<T,3>>& root) {
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
  /* std::shared_ptr<Eigen::Tensor<T,N,Eigen::ColMajor>> datatensor; */
  Eigen::Tensor<T,N,Eigen::ColMajor> datatensor;

public:

  TensorView(Eigen::Tensor<T,N,Eigen::ColMajor> datatensor) : datatensor(datatensor) {};
  TensorView(Eigen::Tensor<T,N,Eigen::ColMajor> datatensor, indexrange<L, N> I) : datatensor(datatensor),  I(I) {};
  /* TensorView(std::shared_ptr<Eigen::Tensor<T,N,Eigen::ColMajor>> datatensor) : datatensor(datatensor) {}; */
  /* TensorView(std::shared_ptr<Eigen::Tensor<T,N,Eigen::ColMajor>> datatensor, indexrange<L, N> I) : datatensor(datatensor),  I(I) {}; */


  template<typename M1, typename ...M>
  size_t get_J(uint8_t p, M1 head, M ...rest){
    return this->I.get_J(p, head, rest...);
  }

  template<typename M0>
  size_t get_J(uint8_t p, M0 rest) {
    return this->I.get_J(p, rest);
  }


  template<size_t N_ = N, std::enable_if_t<N_==2,int> = 0, typename... Ts>
  T &operator() (Ts... K) {
#ifdef RANGE_CHECK // TODO: implement range check for indexrange
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);
    return datatensor(std::get<0>(this->I[0]) + std::get<0>(Ktpl),
                      std::get<0>(this->I[1]) + std::get<1>(Ktpl));
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
  T operator() (Ts... K) const {
#ifdef RANGE_CHECK
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);

    return (datatensor)(std::get<0>(this->I(0)) + std::get<0>(Ktpl),
                        std::get<0>(this->I(1)) + std::get<1>(Ktpl),
                        std::get<0>(this->I(2)) + std::get<2>(Ktpl));
  }
/* #endif */

  L size(const uint8_t dim) const {
    return std::get<1>(this->I[dim]) - std::get<0>(this->I[dim]) + 1;
  }

};

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


template <typename T, typename L, size_t N>
class NormalFoldProd
{
private:
  TensorView<T,L,N> view;
  uint8_t mode=0;
public:
  using Scalar = T;

  void setMode(uint8_t mode) {this->mode = mode;}
  uint8_t getMode() const {return this->mode;}

  NormalFoldProd(TensorView<T,L,N> view) : view(view) { };

  Eigen::Index rows() const {return this->view.size(this->mode);}
  Eigen::Index cols() const {return this->rows();}

  void perform_op(const T* x_in, T* y_out) const
    {
    for (size_t i = 0; i<this->rows(); i++) y_out[i] = T(0);
    fold_tensor_vector_prod(this->view, this->getMode(), x_in, y_out); 
    }
  ~NormalFoldProd(){};
};


void test_tensorsizes(std::vector<uint16_t> tensorsizes) {
  using namespace std;
  using namespace Eigen;
  const uint16_t tensorsize = tensorsizes[0];

  auto datatensor = Tensor<double, 3, ColMajor>(tensorsize,tensorsize,tensorsize);

  for (int i3 = 0; i3 < tensorsize; i3++) for (int i2 = 0; i2 < tensorsize; i2++) for (int i1 = 0; i1 < tensorsize; i1++)
  datatensor(i1,i2,i3) = sin(M_PIf*(i1+i2+i3)/tensorsize);

  for (uint16_t tensorsize_m = tensorsizes[1]; tensorsize_m < tensorsizes[2]; tensorsize_m+= 2){
    indexrange<uint16_t,3> K({0,0,0},{tensorsize_m,tensorsize_m,tensorsize_m});
    auto view = TensorView<double, uint16_t, 3>(datatensor, K);


    cout << tensorsize << ", " << tensorsize_m << ",";
    for (size_t mode=0;mode<3;mode++){
      starttime
        using namespace Spectra;
      /* DenseSymMatProd<double> op_folded(normal_mat[0]); */
      NormalFoldProd<double, uint16_t, 3> op(view);
      op.setMode(mode);
      SymEigsSolver<NormalFoldProd<double, uint16_t, 3>> eigs(op, 2, 3);

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
  cout << view.get_J(0, 1,2,3);
}

/* TODO:
 * - [ ] timings w/ different tensorsize */

int main(int argc, const char** argv) {
  using namespace std;
  using namespace Eigen;

  struct Myargs : public argparse::Args{
    vector<uint16_t> &tensorsizes = kwarg("b,benchmark", "Size of big tensor, size of current view, max size of view").set_default("0,0,0");
    bool &indextest = flag("i,indextest", "test indexing");
    bool &help = flag("h,help", "help");
  };


  /* parser.ignoreFirstArgument(true); */
  auto args = argparse::parse<Myargs>(argc, argv);
  if (args.help) args.print();

  auto tensorsizes = args.tensorsizes;

  if (tensorsizes[0] > 0) test_tensorsizes(tensorsizes);

  if (args.indextest) test_indexing();



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

