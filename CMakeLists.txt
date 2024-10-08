CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(VDF_OCTREE CXX C)

set(Eigen3_DIR "/home/jkataja/opt/eigen/vlasiator-3.22.1" CACHE PATH "")
# SET(TUCKER_MPI_SERIAL_ROOT "$ENV{TUCKER_MPI_SERIAL_ROOT}" CACHE PATH "")
SET(TUCKER_MPI_SERIAL_ROOT "$ENV{HOME}/opt/TuckerMPI/serial" CACHE PATH "")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
ADD_EXECUTABLE(octree octree.cpp)
target_link_libraries (octree Eigen3::Eigen)

include_directories(${TUCKER_MPI_SERIAL_ROOT}/include)
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

