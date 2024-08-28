# Set the project name for example 1
project(06_instanced)

# Organize source files into filters (virtual folders in Visual Studio)
source_group("Source Files" FILES
    src/main.cpp
    src/Application.cpp
)

source_group("Internal Sources" FILES
    ${PROJECT_SOURCE_DIR}/../../deps/source/internal/hdx.cpp
    ${PROJECT_SOURCE_DIR}/../../deps/source/internal/input.cpp
)

source_group("External Sources" FILES
    ${PROJECT_SOURCE_DIR}/../../deps/source/external/stb_image.cpp
    ${PROJECT_SOURCE_DIR}/../../deps/source/external/tiny_obj_loader.cpp
)

# Add the executable for example 1
add_executable(06_instanced
    src/main.cpp
    src/Application.cpp
    ${PROJECT_SOURCE_DIR}/../../deps/source/internal/hdx.cpp
    ${PROJECT_SOURCE_DIR}/../../deps/source/internal/input.cpp
    ${PROJECT_SOURCE_DIR}/../../deps/source/external/stb_image.cpp
    ${PROJECT_SOURCE_DIR}/../../deps/source/external/tiny_obj_loader.cpp
)

# Link the libraries needed 
target_link_libraries(06_instanced SDL2 SDL2main vulkan-1)
