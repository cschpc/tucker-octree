CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(VDF_OCTREE CXX C)

set(Eigen3_DIR "/home/jkataja/opt/eigen/vlasiator-3.22.1" CACHE PATH "")
set(hip_cpu_rt_DIR "/home/jkataja/opt/hipcpu/share/hip_cpu_rt/cmake" CACHE PATH "")
# set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE CACHE BOOL "")
option(CMAKE_EXPORT_COMPILE_COMMANDS "export compile_commands.json" ON)

# SET(TUCKER_MPI_SERIAL_ROOT "$ENV{TUCKER_MPI_SERIAL_ROOT}" CACHE PATH "")
SET(TUCKER_MPI_SERIAL_ROOT "$ENV{HOME}/opt/TuckerMPI/serial" CACHE PATH "")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
ADD_EXECUTABLE(octree octree.cpp)
target_link_libraries (octree Eigen3::Eigen)


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

option(VERBOSE_DEBUG "Output verbose debugging messages" OFF)

if(VERBOSE_DEBUG)
  add_compile_definitions(VERBOSE_DEBUG)
endif()

include_directories(${TUCKER_MPI_SERIAL_ROOT}/include ${CMAKE_SOURCE_DIR}/contrib/include)
target_link_libraries(octree ${TUCKER_MPI_SERIAL_ROOT}/lib/libserial_tucker.a ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})


# CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
# PROJECT(TuckerMPItest C CXX)

# SET(TUCKER_MPI_SERIAL_ROOT "$ENV{TUCKER_MPI_SERIAL_ROOT}" CACHE PATH "")

# SET(SOURCES tuckertest.cpp)

# FIND_PACKAGE(BLAS REQUIRED)
# FIND_PACKAGE(LAPACK REQUIRED)

# add_executable(tuckertest ${SOURCES})
# include_directories(${TUCKER_MPI_SERIAL_ROOT}/include)
# target_link_libraries(tuckertest ${TUCKER_MPI_SERIAL_ROOT}/lib/libserial_tucker.a ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

