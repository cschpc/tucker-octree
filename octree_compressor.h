#ifndef OCTREE_COMPRESSOR_H
#define OCTREE_COMPRESSOR_H

#include <stdlib.h>
#include <stdint.h>

#ifndef VDF_REAL_DTYPE
#define VDF_REAL_DTYPE float
#endif

#ifndef ATOMIC_OCTREE_COORDINATE_DTYPE
#define ATOMIC_OCTREE_COORDINATE_DTYPE uint32_t
#endif

#ifndef OCTREE_TUCKER_CORE_RANK
#define OCTREE_TUCKER_CORE_RANK 2
#endif

#ifndef MAX_ROOT_DIMS
#define MAX_ROOT_DIMS 12
#endif

#ifndef VDF_REAL_DTYPE
#define VDF_REAL_DTYPE float
#endif

#ifndef OCTREE_VIEW_INDEX_TYPE
#define OCTREE_VIEW_INDEX_TYPE uint32_t
#endif

extern "C" {

// 
typedef struct {
  uint32_t root_dims[MAX_ROOT_DIMS]; //
  uint8_t n_root_dims; //

  uint8_t* packed_bytes;  //
  uint64_t n_packed_bytes; //

  uint64_t n_serialized; //

  ATOMIC_OCTREE_COORDINATE_DTYPE* leaf_coordinates;
  uint32_t n_leaf_coordinates;

  uint8_t* leaf_levels;
  uint64_t core_size;
  uint8_t bytes_per_leaf_coordinate;
  VDF_REAL_DTYPE core_scale;
} compressed_octree_t;


void print_compressed_octree_t(compressed_octree_t pod);

void compressed_octree_t_to_bytes(compressed_octree_t pod, uint8_t **bytes, uint64_t* n_bytes);

compressed_octree_t bytes_to_compressed_octree_t(uint8_t* data, uint64_t n_packed);

void compress_with_octree_method_new(VDF_REAL_DTYPE* buffer, const size_t Nx, const size_t Ny, const size_t Nz, 
                                     VDF_REAL_DTYPE tolerance, uint8_t* compressed);

void compress_with_octree_method(VDF_REAL_DTYPE* buffer, 
                                 const size_t Nx, const size_t Ny, const size_t Nz, 
                                 VDF_REAL_DTYPE tolerance, 
                                 uint8_t** serialized_buffer, uint64_t* serialized_buffer_size);

void uncompress_with_octree_method(VDF_REAL_DTYPE* buffer, const size_t Nx, const size_t Ny, const size_t Nz,
                                   uint8_t* serialized_buffer, uint64_t serialized_buffer_size, bool clear_buffer);

}
#endif
