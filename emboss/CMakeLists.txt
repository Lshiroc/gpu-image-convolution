cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_C_COMPILER gcc-10)
set(CMAKE_CXX_COMPILER g++-10)
set(CMAKE_CUDA_HOST_COMPILER g++-10)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(emboss LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
	message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
else()
	message(FATAL_ERROR "CUDA Toolkit not found")
endif()

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
	message(FATAL_ERROR "OpenCV not found")
endif()

if(CMAKE_CUDA_COMPILER)
    # Pass -w flag to nvlink to suppress warnings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xnvlink -w")
    # Or more specifically:
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wnvlink,-w")
endif()

add_executable(emboss main.cu support.cu kernel.cu)
#set_property(TARGET emboss PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(emboss PUBLIC CUDA::cudart)
target_link_libraries(emboss PUBLIC ${OpenCV_LIBS})
