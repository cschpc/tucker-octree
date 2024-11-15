CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(VDF_OCTREE CXX C)

set(Eigen3_DIR "/home/jkataja/opt/eigen/vlasiator-3.22.1/share/eigen3/cmake" CACHE PATH "")
set(hip_cpu_rt_DIR "/home/jkataja/opt/hipcpu/share/hip_cpu_rt/cmake" CACHE PATH "")
set(zfp_DIR "/home/jkataja/opt/zfp/1.0.1/lib/cmake/zfp" CACHE PATH "")


# 
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE CACHE BOOL "" FORCE)
# option(CMAKE_EXPORT_COMPILE_COMMANDS "export compile_commands.json" ON)

set(CMAKE_BUILD_TYPE "RelWIthDebInfo" CACHE STRING "")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(zfp REQUIRED)

option(WITH_HIP "Use HIP" OFF)

add_library(octree_compressor octree.cpp)
ADD_EXECUTABLE(octree test.cpp)

target_link_libraries (octree_compressor Eigen3::Eigen zfp::zfp)
target_link_libraries (octree Eigen3::Eigen octree_compressor)

set_property(TARGET octree PROPERTY CXX_STANDARD 17)
set_property(TARGET octree_compressor PROPERTY CXX_STANDARD 17)


if(WITH_HIP)
  option(USE_HIP_GPU "Use normal HIP" OFF)

  if(USE_HIP_GPU)
    option(USE_HIP_CPU "Use HIP-CPU" OFF)
    message(STATUS "Using normal HIP, disabling HIP-CPU")
  else()
    option(USE_HIP_CPU "Use HIP-CPU" ON)
    message(STATUS "Using HIP-CPU, disabling normal HIP")
  endif()

  if(USE_HIP_CPU)
    find_package(hip_cpu_rt)
    add_compile_definitions(WITH_HIP_CPU)
  else()
    find_package(hip) # TODO: how to include HIP
    add_compile_definitions(WITH_HIP_GPU)
  endif()
endif(WITH_HIP)

option(VERBOSE_DEBUG "Output verbose debugging messages" OFF)
option(BRIEF_DEBUG "Output brief debugging messages" OFF)

set(VDF_REAL_DTYPE "float" CACHE STRING "Datatype of the dense input buffer")
set(OCTREE_TUCKER_CORE_RANK 2 CACHE STRING "Tucker core dimension in one dimension")

CONFIGURE_FILE(config.h.cmake config.h)

if(VERBOSE_DEBUG)
  add_compile_definitions(VERBOSE_DEBUG)
endif()

if(BRIEF_DEBUG)
  add_compile_definitions(BRIEF_DEBUG)
endif()

add_compile_definitions("VDF_REAL_DTYPE=${VDF_REAL_DTYPE}")

include_directories(${CMAKE_SOURCE_DIR}/contrib/include)
include_directories(${CMAKE_BINARY_DIR})

install(TARGETS octree_compressor)
INSTALL(FILES ${CMAKE_SOURCE_DIR}/octree_compressor.h ${CMAKE_BINARY_DIR}/config.h TYPE INCLUDE)


# target_link_libraries(octree ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


# CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
# PROJECT(TuckerMPItest C CXX)

# SET(TUCKER_MPI_SERIAL_ROOT "$ENV{TUCKER_MPI_SERIAL_ROOT}" CACHE PATH "")

# SET(SOURCES tuckertest.cpp)

# FIND_PACKAGE(BLAS REQUIRED)
# FIND_PACKAGE(LAPACK REQUIRED)

# add_executable(tuckertest ${SOURCES})
# include_directories(${TUCKER_MPI_SERIAL_ROOT}/include)
# target_link_libraries(tuckertest ${TUCKER_MPI_SERIAL_ROOT}/lib/libserial_tucker.a ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

