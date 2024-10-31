#include "octree.cpp"

int main(int argc, const char** argv) {
  using namespace std;
  using namespace Eigen;
  using namespace tree_compressor;

  struct Myargs : public argparse::Args{
    vector<uint16_t> &tensorsizes = kwarg("b,benchmark", "Size of big tensor, size of current view, max size of view, increment").set_default("0,0,0,2");
    bool &indextest = flag("i,indextest", "test indexing");
    bool &help = flag("h,help", "help");
    vector<int> &tucker = kwarg("t,tucker", "Test tucker functionality with given tensor size.").set_default("-1");
    vector<uint16_t> &normaltest = kwarg("n,normalsizes", "Size of normal-vector product tensor").set_default("0,0,0");
    bool &treetest = flag("T,tree", "Test octree + tucker");
  };


  /* parser.ignoreFirstArgument(true); */
  auto args = argparse::parse<Myargs>(argc, argv);
  if (args.help) args.print();

  auto tensorsizes = args.tensorsizes;
  auto normalsize = args.normaltest;
  auto tuckersize = args.tucker;

  if (tensorsizes[0] > 0) test_tensorsizes(tensorsizes);

  if (normalsize[0] > 0) test_normalprods(normalsize);

  if (args.indextest) test_indexing();

  if (tuckersize[0] > 0) test_tucker(tuckersize[0]);

  cout << "\ntesting tree next\n";
  if (args.treetest) test_tree();

}
