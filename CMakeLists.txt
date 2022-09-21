cmake_minimum_required(VERSION 3.17...3.24)

project(
  SotaOps
  VERSION 0.1
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# pytorch
find_package(Torch REQUIRED)
set(CMAKE_CXXFLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED PATHS "${TORCH_INSTALL_PREFIX}/lib")

# cuda
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(
  MODERNGPU
  GIT_REPOSITORY https://github.com/moderngpu/moderngpu.git
  GIT_TAG        d8cf03d33917cb8ee792a59be54c4026913730e4
)
FetchContent_MakeAvailable(MODERNGPU)

add_subdirectory(src)
