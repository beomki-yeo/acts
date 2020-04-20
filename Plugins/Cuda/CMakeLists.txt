### CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# nvcc compiler flags that covers pascal/volta/turing
set(ACTS_CUDA_FLAGS "")
set(ACTS_CUDA_FLAGS "${ACTS_CUDA_FLAGS} -arch=sm_60")
set(ACTS_CUDA_FLAGS "${ACTS_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(ACTS_CUDA_FLAGS "${ACTS_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61")
set(ACTS_CUDA_FLAGS "${ACTS_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")
set(ACTS_CUDA_FLAGS "${ACTS_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ACTS_CUDA_FLAGS}")

add_library(
  ActsCudaPlugin SHARED
  src/Seeding/SeedfinderCudaKernels.cu)
target_include_directories(
  ActsCudaPlugin
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
  ActsCudaPlugin
  PUBLIC ActsCore)

install(
  TARGETS ActsCudaPlugin
  EXPORT ActsCudaPluginTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(
  DIRECTORY include/Acts
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
