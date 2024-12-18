# T-Octree voxel data compression library

This *Tucker-Octree* (T-Octree) library compresses 3-dimensional dense datasets
using octree multiresolution approach with Tucker decomposition. The collected
core and factor matrices are further compressed with the ZFP library.

## Installation

Typical cmake installation:

```bash
> mkdir build; cd build
> cmake .. -DCMAKE_BUILD_TYPE=Release -DEigen3_DIR="path/to/share/eigen3/cmake" -Dzfp_DIR="path/to/lib/cmake/zfp" -DCMAKE_INSTALL_PREFIX="<vlasiator-path>/vdf_compression"
> make install
```

After that add `-Lpath/to/vdf_compression/lib` to linker flags and
`-Ipath/to/vdf_compression/include` to compiler flags for vlasiator.

## Licensing

T-Octree is licensed under GPL2.0 or later (see `LICENSE` text).

## Dependencies

T-Octree utilizes the following codes:

- [zfp](https://computing.llnl.gov/projects/zfp)
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Spectra C++ Library For Large Scale Eigenvalue Problems](https://spectralib.org) (included in `contrib/include` w/ it's own license)
- [argparse](https://github.com/morrisfranken/argparse) (included in `contrib/include` w/ it's own license)
