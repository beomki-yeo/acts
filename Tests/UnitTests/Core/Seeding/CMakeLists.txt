add_executable(SeedfinderTest SeedfinderTest.cpp)
target_link_libraries(SeedfinderTest PRIVATE cuda cudart ActsCore Boost::boost)
