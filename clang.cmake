set(CMAKE_C_COMPILER "$ENV{HOME}/opt/clang/bin/clang" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "$ENV{HOME}/opt/clang/bin/clang++" CACHE FILEPATH "")

set(Eigen3_DIR "/home/jkataja/opt/eigen/vlasiator-3.22.1/share/eigen3/cmake" CACHE PATH "")
set(hip_cpu_rt_DIR "/home/jkataja/opt/hipcpu/share/hip_cpu_rt/cmake" CACHE PATH "")
set(zfp_DIR "/home/jkataja/opt/zfp/1.0.1/lib/cmake/zfp" CACHE PATH "")

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE CACHE BOOL "Generate compile_commands.json file")
