cmake_minimum_required(VERSION 3.10)
project(AiPlane)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add LibTorch
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(matplotplusplus)

include_directories(
    ${CMAKE_SOURCE_DIR}
    /usr/include
    /usr/include/python3.10
    /usr/lib/python3/dist-packages/numpy/core/include
)

find_package(PythonLibs 3.10 REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Core files
file(GLOB CORE_HEADERS "${CMAKE_SOURCE_DIR}/*.h")
file(GLOB CORE_SOURCES "${CMAKE_SOURCE_DIR}/*.cpp")

# TagGame files
file(GLOB TAGGAME_HEADERS "${CMAKE_SOURCE_DIR}/taggame/*.h")
file(GLOB TAGGAME_SOURCES "${CMAKE_SOURCE_DIR}/taggame/*.cpp")

# Exercise files
file(GLOB EXERCISES_HEADERS 
    "${CMAKE_SOURCE_DIR}/barto_sutton_exercises/*/*.h"
)
file(GLOB EXERCISES_SOURCES
    "${CMAKE_SOURCE_DIR}/barto_sutton_exercises/*/*.cpp"
)

add_executable(output_executable 
    ${CORE_SOURCES} 
    ${CORE_HEADERS} 
    ${TAGGAME_HEADERS}
    ${TAGGAME_SOURCES}
    ${EXERCISES_HEADERS} 
    ${EXERCISES_SOURCES}
)

target_link_libraries(output_executable PUBLIC 
    matplot 
    nlohmann_json::nlohmann_json 
    ${PYTHON_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# Make sure the executable finds the Torch libraries at runtime
set_property(TARGET output_executable PROPERTY CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)