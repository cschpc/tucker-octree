/*
 * Tucker-Octree multiresolution voxel data compression library
 * Copyright (C) 2024 CSC - IT Center for Science Ltd
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see
 * <https://www.gnu.org/licenses/>.
 */

/* 
 * Author: Juhani Kataja 
 * Affiliation: CSC - IT Center for Science Ltd
 * Email: juhani.kataja@csc.fi
 */


#ifndef TOCTREE_COMPRESSOR_H
#define TOCTREE_COMPRESSOR_H

#include <stdlib.h>
#include <stdint.h>

#include "toctree_config.h"

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
#define MAX_ROOT_DIMS 3
#endif

#ifndef OCTREE_VIEW_INDEX_TYPE
#define OCTREE_VIEW_INDEX_TYPE uint32_t
#endif

#define TOCTREE_COMPRESS_STAT_SUCCESS 1 // Successfull
#define TOCTREE_COMPRESS_STAT_FAIL_TOL 2 // Failed to reach tolerance in given max iterations
#define TOCTREE_COMPRESS_STAT_MEMORY 3 // Not enough memory
#define TOCTREE_COMPRESS_STAT_ZERO_ARRAY 4 // Input array full of zeros

extern "C" {

typedef struct {
  uint32_t root_dims[MAX_ROOT_DIMS];
  uint8_t n_root_dims;

  uint8_t* packed_bytes; 
  uint64_t n_packed_bytes;

  uint64_t n_serialized;

  ATOMIC_OCTREE_COORDINATE_DTYPE* leaf_coordinates;
  uint32_t n_leaf_coordinates;

  uint8_t* leaf_levels;
  uint64_t core_size;
  uint8_t bytes_per_leaf_coordinate;
  VDF_REAL_DTYPE core_scale;
} compressed_toctree_t;

void print_compressed_toctree_t(compressed_toctree_t pod);

void compressed_toctree_t_to_bytes(compressed_toctree_t pod, uint8_t **bytes, uint64_t* n_bytes);

compressed_toctree_t bytes_to_compressed_toctree_t(uint8_t* data, uint64_t n_packed);

int compress_with_toctree_method(VDF_REAL_DTYPE* buffer, 
                                 const size_t Nx, const size_t Ny, const size_t Nz, 
                                 VDF_REAL_DTYPE tolerance, uint8_t** serialized_buffer, 
                                 uint64_t* serialized_buffer_size, uint64_t maxiter, uint64_t skip_levels);

void uncompress_with_toctree_method(VDF_REAL_DTYPE* buffer, const size_t Nx, const size_t Ny, const size_t Nz,
                                   uint8_t* serialized_buffer, uint64_t serialized_buffer_size);

}
#endif
