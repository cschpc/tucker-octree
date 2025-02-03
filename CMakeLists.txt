CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(VDF_OCTREE VERSION 1.0.0 LANGUAGES CXX C )

set(CMAKE_BUILD_TYPE "RelWIthDebInfo" CACHE STRING "")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(zfp REQUIRED)

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(VERBOSE_DEBUG "Output verbose debugging messages" OFF)
option(BRIEF_DEBUG "Output brief debugging messages" OFF)
option(OCTREE_RANGE_CHECK "Check out of bounds accesses" OFF)
option(BUILD_TEST_BIN "Build test binary" OFF)
option(TOCTREE_L2ERROR "Use l_2 norm in error calculation. Otherwise use max-norm." OFF)

add_library(toctree_compressor toctree.cpp)
target_link_libraries (toctree_compressor Eigen3::Eigen zfp::zfp)
set_property(TARGET toctree_compressor PROPERTY CXX_STANDARD 17)

if(BUILD_TEST_BIN)
  ADD_EXECUTABLE(toctree test.cpp)
  set_property(TARGET toctree PROPERTY CXX_STANDARD 17)
  target_link_libraries (toctree Eigen3::Eigen toctree_compressor)
endif()

set(VDF_REAL_DTYPE "float" CACHE STRING "Datatype of the dense input buffer")
set(OCTREE_TUCKER_CORE_RANK 2 CACHE STRING "Tucker core dimension in one dimension")
SET(TOCTREE_ZFP_STREAM_ACCURACY "1e-3" CACHE STRING "ZFP accuracy for serializing Tucker decompositions")
mark_as_advanced(TOCTREE_ZFP_STREAM_ACCURACY)

CONFIGURE_FILE(toctree_config.h.cmake toctree_config.h)

if(VERBOSE_DEBUG)
  add_compile_definitions(VERBOSE_DEBUG)
endif()

if(BRIEF_DEBUG)
  add_compile_definitions(BRIEF_DEBUG)
endif()

add_compile_definitions("VDF_REAL_DTYPE=${VDF_REAL_DTYPE}")

include_directories(${CMAKE_SOURCE_DIR}/contrib/include)
include_directories(${CMAKE_BINARY_DIR})

install(TARGETS toctree_compressor)
INSTALL(FILES ${CMAKE_SOURCE_DIR}/toctree_compressor.h ${CMAKE_BINARY_DIR}/toctree_config.h TYPE INCLUDE)
