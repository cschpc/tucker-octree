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

#define assertm(expression, message) assert(((void)message, expression))

template <typename T>
using tensor3 = Eigen::Tensor<T,3>;

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
public:
  
  indexrange<T,N>() {};
  indexrange<T,N>(const std::array<T,N> &aa, const std::array<T,N> &bb) : a(aa), b(bb) {};
  indexrange<T,N>(const std::array<T,N> &&aa, const std::array<T,N> &&bb) : a(aa), b(bb) {};

  /* indexrange<T,N>(const T aa[N], const T bb[N]) : a(aa), b(bb) {}; */
  /* indexrange<T,N> divide(uint8_t subcube); */

  std::tuple<T, T> operator()(size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }
  std::tuple<T, T> operator[](size_t i) const { return std::make_tuple(this->a[i], this->b[i]); }

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
  std::cout << "*************************************" << std::endl;
}


template <typename T, typename L, size_t N>
struct TensorView {

private:

  indexrange<L,N> I;
  std::shared_ptr<Eigen::Tensor<T,N,Eigen::ColMajor>> datatensor;

public:

  TensorView(std::shared_ptr<Eigen::Tensor<T,N,Eigen::ColMajor>> datatensor) : datatensor(datatensor) {};
  TensorView(std::shared_ptr<Eigen::Tensor<T,N,Eigen::ColMajor>> datatensor, indexrange<L, N> I) : datatensor(datatensor),  I(I) {};


  template<size_t N_ = N, std::enable_if_t<N_==2,int> = 0, typename... Ts>
  T &operator() (Ts... K) {
#ifdef RANGE_CHECK // TODO: implement range check for indexrange
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);
    return (*datatensor)(std::get<0>(this->I[0]) + std::get<0>(Ktpl),
                         std::get<0>(this->I[1]) + std::get<1>(Ktpl));
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
  T &operator() (Ts... K) {
#ifdef RANGE_CHECK // TODO: implement range check for indexrange
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);
    return (*datatensor)(std::get<0>(this->I[0]) + std::get<0>(Ktpl),
                         std::get<0>(this->I[1]) + std::get<1>(Ktpl),
                         std::get<0>(this->I[2]) + std::get<2>(Ktpl));
  }

  template<size_t N_ = N, std::enable_if_t<N_==3,int> = 0, typename... Ts>
  T operator() (Ts... K) const {
#ifdef RANGE_CHECK
    I.checkrange(i,j,k);
#endif
    auto Ktpl = std::make_tuple(K...);

#if 0
#ifdef VERBOSE_DEBUG
    std::cout << "Accessing " 
      << std::get<0>(this->I(0)) << "+" << std::get<0>(Ktpl) << " " 
      << std::get<0>(this->I(1)) << "+" << std::get<1>(Ktpl) << " "
      << std::get<0>(this->I(2)) << "+" << std::get<2>(Ktpl) << std::endl;
#endif
#endif

    return (*datatensor)(std::get<0>(this->I(0)) + std::get<0>(Ktpl),
                         std::get<0>(this->I(1)) + std::get<1>(Ktpl),
                         std::get<0>(this->I(2)) + std::get<2>(Ktpl));
  }

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

 std::array<uint8_t,2> dims;

 assertm(mode >= 0 && mode <=2, "Invalid fold mode");

 switch(mode){
   case 0:
     dims = {1,2};
     break;
   case 1:
     dims = {0,2};
     break;
   case 2:
     dims = {0,1};
     break;
 }

 for (L j = 0; j < A.size(mode); j++) for (L i = 0; i < A.size(mode); i++) R(i,j) = T(0);

 Eigen::Vector<L, 3> inds, indsT;

 for (size_t j = 0; j < A.size(mode); j++) {
   indsT(mode) = j;
   for (size_t i = 0; i<A.size(mode); i++) {
     inds(mode) = i;
     T acc = T(0);
     for (size_t k1 = 0; k1 < A.size(dims[0]); k1++) for (size_t k2 = 0; k2 < A.size(dims[1]); k2++) {
       inds(dims[0]) = k1; 
       indsT(dims[0]) = k1; 
       inds(dims[1]) = k2; 
       indsT(dims[1]) = k2;
       acc = acc + A(inds[0], inds[1], inds[2])*A(indsT[0], indsT[1], indsT[2]);
     }
     R(i,j) = R(i,j) + acc;
   }
 }
}


int main(int argc, char** argv) {
  using namespace std;
  using namespace Eigen;

  indexrange<uint16_t,3> R({0,0,0},{49,49,49});
  auto D = R.divide(uint16_t(1+(1<<2)));

  cout << D << endl;
  leaf<indexrange<uint16_t,3>> root({R, NULL});
  root = divide(root);

  root.children[1] = divide(root.children[1]);
  root.children[1].children[2] = divide(root.children[1].children[2]);
  cout << root;

  size_t data_dims[3] = {5,5,5};
  /* auto datatensor = new Tensor<double, 3, ColMajor>(300,300,300); */
  auto datatensor = std::shared_ptr<Tensor<double, 3, ColMajor>> (new Tensor<double, 3, ColMajor>(300,300,300));

  for (int i3 = 0; i3 < 300; i3++) for (int i2 = 0; i2 < 300; i2++) for (int i1 = 0; i1 < 300; i1++)
  (*datatensor)(i1,i2,i3) = sin(M_PIf*(i1+i2+i3)/300);
  /* (*datatensor)(i1,i2,i3) = i1+1000*i2+1e6*i3; //sin(M_PIf*(i1+i2+i3)/300); */

  indexrange<uint16_t,3> K({0,0,0},{299,299,299});
  hline();
  TensorView<double, uint16_t, 3> view = TensorView<double, uint16_t, 3>(datatensor, K.divide(0).divide(0));//.divide(1).divide(0).divide(1));
  hline();

  /* Tensor<double, 2> normal_0; */
  /* Tensor<double, 2> normal_1; */
  /* Tensor<double, 2> normal_2; */
  /* std::array<Tensor<double, 2>, 3> normal_mat; */

  std::array<Matrix<double, Dynamic, Dynamic>, 3> normal_mat;
  auto start = chrono::high_resolution_clock::now();

  cout << "folding to normal ...";
  fold_tensor_to_normal_matrix(view,0,normal_mat[0]);
  auto stop = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(stop-start);
  cout << "duration: " << duration.count() << endl;
  fold_tensor_to_normal_matrix(view,1,normal_mat[1]);
  fold_tensor_to_normal_matrix(view,2,normal_mat[2]);

    {
    using namespace Spectra;
    DenseSymMatProd<double> op(normal_mat[1]);
    SymEigsSolver<DenseSymMatProd<double>> eigs(op, 3, 6);

    eigs.init();
    int nconv = eigs.compute(SortRule::LargestMagn);
    Vector<double, Dynamic> evalues;
    /* Matrix<double, Dynamic, Dynamic> */
    if (eigs.info() == CompInfo::Successful) {
      evalues = eigs.eigenvalues();
    }
    hline();
    cout << "eigenvalues: ";
    cout << evalues[0] << " " << evalues[1] << " " << evalues[2] << endl;
    }



  hline();
  cout << normal_mat[0] << endl;
  hline();

  hline();
  TensorView<double, uint16_t, 3> view2 = TensorView<double, uint16_t, 3>(datatensor, K.divide(0).divide(0).divide(0).divide(7));//.divide(1).divide(0).divide(1));
  for (int i2=0; i2 < view2.size(1); i2++) {
  for (int i1=0; i1 < view2.size(0); i1++) {
    cout << view2(i2,i1,0) << " ";
    }
  cout << endl;
  }
  cout << endl;
  hline();


  /* char ok; */
  /* cin >> ok; */
  hline();
  itrange<int,int,int> newr(make_tuple(1,2,3),make_tuple(4,5,6));
  cout << get<1>(newr.a);

  cout << "press enter to exit";
  getchar();
}

