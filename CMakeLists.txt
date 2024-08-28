# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(HDX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory to the project
include_directories(deps/include)

# Add the library directory
link_directories(deps/lib)

# Add the subdirectories for the examples
add_subdirectory(samples/01_compute_particles)
add_subdirectory(samples/02_viking_room)
add_subdirectory(samples/03_texture)
add_subdirectory(samples/04_cubemap)
add_subdirectory(samples/05_normal_mapping)
add_subdirectory(samples/06_instanced)
add_subdirectory(samples/07_blinn_phong)
add_subdirectory(samples/08_reflect)
add_subdirectory(samples/09_pbr)
add_subdirectory(samples/10_shadow_mapping)
add_subdirectory(samples/11_hdr_cubemap)
add_subdirectory(samples/12_ibl)
