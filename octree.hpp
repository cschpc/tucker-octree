#ifndef VDF_REAL_DTYPE
#define VDF_REAL_DTYPE float
#endif

extern "C" {
void compress_with_octree_method(VDF_REAL_DTYPE* buffer, const size_t Nx, const size_t Ny, const size_t Nz, 
                                 VDF_REAL_DTYPE tolerance, float& compression_ratio);
}
