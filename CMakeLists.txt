cmake_minimum_required(VERSION 3.10)
project(AiPlane)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_subdirectory(matplotplusplus)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    ${CMAKE_SOURCE_DIR}  # This is equivalent to "${workspaceFolder}/"
    /usr/include
    /usr/include/python3.10
    /usr/lib/python3/dist-packages/numpy/core/include
)

find_package(PythonLibs 3.10 REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(output_executable main.cpp MDPCore.h MDPSolver.h PolicyIteration.h QValuePolicyIteration.h QValuePolicyIteration.cpp VValuePolicyIteration.h VValuePolicyIteration.cpp m_utils.h m_utils.cpp CarRentalEnvironment.h CarRentalEnvironment.cpp
        Policy.h ValueIteration.h ValueIteration.cpp
)

target_link_libraries(output_executable PUBLIC matplot nlohmann_json::nlohmann_json ${PYTHON_LIBRARIES})
