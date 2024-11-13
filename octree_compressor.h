#include <stdlib.h>
#include <stdint.h>

#ifndef VDF_REAL_DTYPE
#define VDF_REAL_DTYPE float
#endif

extern "C" {

// 
typedef struct {
  uint8_t* packed_bytes;
  size_t packed_length;
  size_t serialized_length;
} compressed_octree_t;

void compress_with_octree_method_new(VDF_REAL_DTYPE* buffer, const size_t Nx, const size_t Ny, const size_t Nz, 
                                     VDF_REAL_DTYPE tolerance, uint8_t* compressed);

void compress_with_octree_method(VDF_REAL_DTYPE* buffer, 
                                 const size_t Nx, const size_t Ny, const size_t Nz, 
                                 VDF_REAL_DTYPE tolerance, double& compression_ratio);

void uncompress_with_octree_method(VDF_REAL_DTYPE* buffer, const size_t NX, const size_t Ny, const size_t Nz,
                                   uint8_t* compressed);

}
