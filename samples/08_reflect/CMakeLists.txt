# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(08_reflect)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the include directory to the project
include_directories(../../deps/include)

# Add the library directory
link_directories(../../deps/lib)

# Organize source files into filters (virtual folders in Visual Studio)
source_group("Source Files" FILES
    src/main.cpp
    src/Application.cpp
)

source_group("Internal Sources" FILES
    ../../deps/source/internal/hdx.cpp
    ../../deps/source/internal/input.cpp
)

source_group("External Sources" FILES
    ../../deps/source/external/stb_image.cpp
    ../../deps/source/external/stb_image_write.cpp
    ../../deps/source/external/tiny_obj_loader.cpp
)

# Add the executable
add_executable(08_reflect
    src/main.cpp
    src/Application.cpp
    ../../deps/source/internal/hdx.cpp
    ../../deps/source/internal/input.cpp
    ../../deps/source/external/stb_image.cpp
    ../../deps/source/external/stb_image_write.cpp
    ../../deps/source/external/tiny_obj_loader.cpp
)

# Link the libraries needed 
target_link_libraries(08_reflect SDL2 SDL2main vulkan-1)
