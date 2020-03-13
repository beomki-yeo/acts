add_executable(SeedfinderTest SeedfinderTest.cpp)
add_executable(SeedfinderCUDAValidate SeedfinderCUDAValidate.cpp)
target_link_libraries(SeedfinderTest PRIVATE cuda cudart ActsCore Boost::boost)
target_link_libraries(SeedfinderCUDAValidate PRIVATE cuda cudart ActsCore Boost::boost)
