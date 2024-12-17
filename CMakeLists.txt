CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(VDF_OCTREE CXX C)

set(CMAKE_BUILD_TYPE "RelWIthDebInfo" CACHE STRING "")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(zfp REQUIRED)

add_library(toctree_compressor toctree.cpp)
ADD_EXECUTABLE(toctree test.cpp)

target_link_libraries (toctree_compressor Eigen3::Eigen zfp::zfp)
target_link_libraries (toctree Eigen3::Eigen toctree_compressor)

set_property(TARGET toctree PROPERTY CXX_STANDARD 17)
set_property(TARGET toctree_compressor PROPERTY CXX_STANDARD 17)

option(VERBOSE_DEBUG "Output verbose debugging messages" OFF)
option(BRIEF_DEBUG "Output brief debugging messages" OFF)
option(OCTREE_RANGE_CHECK "Check out of bounds accesses" OFF)

set(VDF_REAL_DTYPE "float" CACHE STRING "Datatype of the dense input buffer")
set(OCTREE_TUCKER_CORE_RANK 2 CACHE STRING "Tucker core dimension in one dimension")

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
INSTALL(FILES ${CMAKE_SOURCE_DIR}/toctree_compressor.h ${CMAKE_BINARY_DIR}/config.h TYPE INCLUDE)
