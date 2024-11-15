# Octree based multiresolution Tucker decomposition of dense 3-arrays

## Installation

Typical cmake installation:

```bash
> mkdir build; cd build
> cmake .. -DEigen3_DIR="path/to/share/eigen3/cmake" -Dzfp_DIR="path/to/lib/cmake/zfp" -DCMAKE_INSTALL_PREFIX="path/to/vdf_compression"
> make install
```
After that add `-Lpath/to/vdf_compression/lib` to linker flags and
`-Ipath/to/vdf_compression/include` to compiler flags for vlasiator.
