#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <initializer_list>
#include <bitset>
#include <memory>
#include <array>

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

template <typename T, size_t N>
struct indexrange {
private: 
  std::array<T,N> a; 
  std::array<T,N> b;
public:
  indexrange<T,N>() {};
  indexrange<T,N>(const std::array<T,N> &aa, const std::array<T,N> &bb) : a(aa), b(bb) {};
  indexrange<T,N>(const std::array<T,N> &&aa, const std::array<T,N> &&bb) : a(aa), b(bb) {};
  indexrange<T,N> divide(uint8_t subcube);
  template <typename S> Eigen::Tensor<S,N> getslice(Eigen::Tensor<S,N> &A){
    using namespace Eigen;
    Eigen::array<Index, N> offsets;
    Eigen::array<Index, N> extents;
    for (uint8_t dim = 0; dim<N; dim++) {
      offsets[dim] = this->a[dim];
      extents[dim] = this->b[dim] - this->a[dim] + 1;
    }
    return A.slice(offsets, extents);
  };

  template<typename T_, size_t N_>
  friend std::ostream& operator <<(std::ostream &o, const indexrange<T_,N_> R);
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

template <typename T, size_t N> indexrange<T,N> indexrange<T,N>::divide(uint8_t subcube) {
  indexrange<T,N> D(this->a, this->b);
  T two(2);
  T one(1); // TODO: calculate correct remainder, if T is a floatingh point type, one should be 0!
  for(uint8_t dim=0; dim < N; dim++) {
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




template <typename T> struct TensorView3 {

private:

  indexrange<size_t,3> I;
  std::shared_ptr<tensor3<T>> datatensor;

public:


  T &operator() (size_t i, size_t j, size_t k) {
#ifdef RANGE_CHECK
    I.checkrange(i,j,k);
#endif

  }
  TensorView3(std::shared_ptr<tensor3<T>> datatensor, indexrange<size_t, 3> I) : 
    I(I), datatensor(datatensor) { 

  }

  ~TensorView3() { /* Don't delete datatensor here */ };

};

int main(int argc, char** argv) {
  using namespace std;
  using namespace Eigen;

  indexrange<uint8_t,3> R({0,0,0},{49,49,49});
  auto D = R.divide(uint8_t(1+(1<<2)));

  cout << D << endl;
  leaf<indexrange<uint8_t,3>> root({R, NULL});
  root = divide(root);

  root.children[1] = divide(root.children[1]);
  root.children[1].children[2] = divide(root.children[1].children[2]);
  cout << root;

  size_t data_dims[3] = {5,5,5};
  Tensor<double, 3, ColMajor> datatensor(5,5,5);
  double k(0);
  for (int i1 = 0; i1 < 5; i1++) {
  for (int i2 = 0; i2 < 5; i2++) {
  for (int i3 = 0; i3 < 5; i3++) { datatensor(i1,i2,i3) = k; k += 1.0f; }}}

  hline();
  cout << datatensor << endl;

  hline();
  indexrange<uint8_t,3> R2({0,0,0},{4,4,4});
  indexrange<uint8_t,3> R3({0,0,1},{4,4,1});
  auto R22 = R2.divide(0);
  cout << R22 << endl;
  hline();
  auto dataslice = R22.getslice(datatensor);
  cout << dataslice << endl;
  hline();
  cout << R3.getslice(datatensor) << endl;
}

