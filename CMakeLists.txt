cmake_minimum_required(VERSION 3.10)  # Specify CMake version

# Set the project name
project(MyProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
add_subdirectory(matplotplusplus)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}  # This is equivalent to "${workspaceFolder}/"
    /usr/include/python3.10
    /usr/lib/python3/dist-packages/numpy/core/include
)

find_package(PythonLibs 3.10 REQUIRED)

add_executable(output_executable main.cpp MDPCore.h MDPSolver.h PolicyIteration.h QValuePolicyIteration.h VValuePolicyIteration.h m_utils.h m_utils.cpp CarRentalEnvironment.h CarRentalEnvironment.cpp
        Policy.h
)

target_link_libraries(output_executable PUBLIC matplot ${PYTHON_LIBRARIES})
