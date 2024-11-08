#include "octree.cpp"
#include <string>
#include <fstream>
#include <iostream>

#include <argparse.hpp>

namespace octree_test {

  using namespace std;
  using namespace tree_compressor;

  namespace settings {
    string outfile;
  };

#define NOTEST

  void test_tensorsizes(std::vector<uint16_t> tensorsizes) {
#ifndef NOTEST
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

        NormalFoldProd<double, uint16_t, 3> op(view, double(1));
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
#endif
  }

  void test_indexing() {
#ifndef NOTEST
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
#endif
  }

  void test_tucker(int big_N) {
#ifndef NOTEST
    using namespace Eigen;
    using namespace std;
    /* const int big_N = 255; */
    typedef size_t UI;
    auto datatensor = Tensor<double, 3, ColMajor>(big_N,big_N,big_N);
    indexrange<UI, 3> K({0,0,0},{UI(big_N-1),UI(big_N-1),UI(big_N-1)});

    auto view = TensorView<double, UI, 3>(datatensor, K);

    auto f = [&](int I) {return M_PIf64*(I+1)/big_N;};

    for (int i = 0; i<view.size(0); i++) {
      for (int j = 0 ;j<view.size(1); j++) {
        for (int k = 0 ;k<view.size(2); k++) {
          view(i,j,k) = sin(f(i))*sin(f(j))*cos(f(k));
        }}}

    cout << "sqnorm of view " << view.sqnorm() << endl;

    starttime

      auto tucker = Tucker<double, UI, 2, 3>(view);

    tucker.make_core();
    for (int i3 = 0; i3 < 2; i3++) {
      for (int i1 = 0; i1 < 2; i1++) {
        for (int i2 = 0; i2 < 2; i2++) {
          cout << tucker.core(i1,i2,i3) << " ";
        }
        cout << endl;
      }
      cout << endl;
    }
    tucker.fill_residual();
    cout << "sqnorm of view " << view.sqnorm() << endl << endl;
    cout << "timing: ";
    endtime

#endif
  }

  template<typename UI>
    void test_tree(size_t maxiter, UI big_N) {
      using namespace Eigen;
      using namespace std;
      auto datatensor = Tensor<double, 3, ColMajor>(big_N,big_N,big_N);
      indexrange<UI, 3> K({0,0,0},{UI(big_N-1),UI(big_N-1),UI(big_N-1)});

      indexrange<UI,3> window({0,0,big_N/2},{UI(big_N-1),UI(big_N-1),UI(big_N/2)});

      auto tree = leaf<indexrange<UI,3>,3>();
      tree.data = K;

      /* divide_leaf(tree); */

      auto f = [&](int I) {return M_PIf64*(I+1)/big_N;};
      auto F = [&](int I1, int I2, int I3) {return exp(-pow((I1+I2+I3)/(0.1*big_N),2))*sin(M_PIf64*(I1+I2+I3+1)/(2*big_N));};

      auto view = TensorView<double,UI,3>(datatensor, K);
      for (int i1 = 0; i1<view.size(0); i1++) {
        for (int i2 = 0; i2<view.size(0); i2++) {
          for (int i3 = 0; i3<view.size(0); i3++) {
            view(i1,i2,i3) = F(i1,i2,i3);
          }}}

      std::stack<unique_ptr<Tucker<double,UI,2,3>>> tuck_stack;
      std::vector<unique_ptr<Tucker<double,UI,2,3>>> tuckers;

      for(int iter=0; iter<maxiter; iter++) {
        cout << "iter: " << iter << " sqnorm of view " << view.sqnorm() << "\t\t|  ";

        double residual = -1.0;

        // find worst leaf
        leaf<indexrange<UI,3>,3>* worst_leaf;
        for (auto it = tree.begin(); it != tree.end(); ++it) {
          auto& leaf = (*it);
          auto c_view = TensorView<double, UI, 3>(datatensor, leaf.data);
          double c_residual = c_view.get_residual();
          if (c_residual > residual) {
            residual = c_residual;
            worst_leaf = &leaf;
          }
        };

        // improve worst leaf
        cout << "worst leaf: " << worst_leaf->level << " : "<<  worst_leaf->data << " residual: " << residual;
        divide_leaf(*worst_leaf);

        std::unique_ptr<TensorView<double,UI,3>> c_view(new TensorView<double,UI,3>(datatensor, worst_leaf->data));
        std::unique_ptr<Tucker<double, UI, 2, 3>> tuck(new Tucker<double,UI,2,3>(std::move(c_view))); /* TODO: save tucker to leaf! */
        tuck->fill_residual();

        OctreeCoordinates<3> worst_coords = leaf_to_coordinates(*worst_leaf);
        cout << "\t|  worst_coords: " << worst_coords <<":"<<worst_coords.toAtomic<int>()<< ":" << OctreeCoordinates<3>(worst_coords.toAtomic<int>(), worst_leaf->level) << " inds: " << K.getsubrange(worst_coords) << endl;


        tuck->setCoordinates(worst_coords);
        /* tuck_stack.push(std::move(tuck)); */
        tuckers.push_back(std::move(tuck));
        /* delete c_view; */
      }

      cout << "residual sqnorm: " << view.sqnorm() << std::endl;

      view.fill(double(0));

      for (auto& it : tuckers) {
        auto& tuck = *(it);
        tuck.fill_residual(double(1), double(1));
      }

      auto serialized = SerialTucker<double, UI, 2, 3>(tuckers, K);

      size_t acc = 1;
      size_t counter = 1;

      if (settings::outfile.length() > 0) {
        std::ofstream(settings::outfile, ios::trunc) << serialized;
      } else {
        cout << serialized;
      }

      /* auto detuckers = */ 
        /* serialized.Deserialize(); */

      cout << "reconstructed residual sqnorm: " << view.sqnorm() << std::endl;
      for (int i1 = 0; i1<view.size(0); i1++) {
        for (int i2 = 0; i2<view.size(0); i2++) {
          for (int i3 = 0; i3<view.size(0); i3++) {
            view(i1,i2,i3) = view(i1,i2,i3) - F(i1,i2,i3);
          }}}
      cout << "final reconstructed sqnorm: " << view.sqnorm() << std::endl;

    }

  void test_normalprods(std::vector<uint16_t> sizes) {
#ifndef NOTEST
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
    auto res = view.get_residual();

    NormalFoldProd<double,uint16_t, 3> op(view, 1/res);

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
#endif
  }

}

int main(int argc, const char** argv) {
  using namespace std;
  using namespace Eigen;
  using namespace tree_compressor;
  using namespace octree_test;

  struct Myargs : public argparse::Args {
    vector<uint16_t> &tensorsizes = kwarg("b,benchmark", "Size of big tensor, size of current view, max size of view, increment").set_default("0,0,0,2");
    bool &indextest = flag("i,indextest", "test indexing");
    bool &help = flag("h,help", "help");
    vector<int> &tucker = kwarg("t,tucker", "Test tucker functionality with given tensor size.").set_default("-1");
    vector<uint16_t> &normaltest = kwarg("n,normalsizes", "Size of normal-vector product tensor").set_default("0,0,0");
    vector<size_t> &treetest = kwarg("T,tree", "Test octree + tucker. Param: maxiter, big_N ").set_default("0,4");
    string& outfile = kwarg("o,outfile", "Output structured data to this file.").set_default("");
  };


  /* parser.ignoreFirstArgument(true); */
  auto args = argparse::parse<Myargs>(argc, argv);
  if (args.help) args.print();

  auto tensorsizes = args.tensorsizes;
  auto normalsize = args.normaltest;
  auto tuckersize = args.tucker;
  auto treeparam = args.treetest;

  octree_test::settings::outfile = args.outfile;

  if (tensorsizes[0] > 0) test_tensorsizes(tensorsizes);

  if (normalsize[0] > 0) test_normalprods(normalsize);

  if (args.indextest) test_indexing();

  if (tuckersize[0] > 0) test_tucker(tuckersize[0]);

  cout << "\ntesting tree next\n";
  if (treeparam[0] > 0) test_tree(treeparam[0], treeparam[1]);

}
