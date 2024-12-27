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

set(CORE_HEADERS
    ${CMAKE_SOURCE_DIR}/MDPCore.h
    ${CMAKE_SOURCE_DIR}/MDPSolver.h
    ${CMAKE_SOURCE_DIR}/GPI.h
    ${CMAKE_SOURCE_DIR}/QValuePolicyIteration.h
    ${CMAKE_SOURCE_DIR}/VValuePolicyIteration.h
    ${CMAKE_SOURCE_DIR}/m_utils.h
    ${CMAKE_SOURCE_DIR}/Policy.h
    ${CMAKE_SOURCE_DIR}/ValueIteration.h
)

set(CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/QValuePolicyIteration.cpp
    ${CMAKE_SOURCE_DIR}/VValuePolicyIteration.cpp
    ${CMAKE_SOURCE_DIR}/m_utils.cpp
    ${CMAKE_SOURCE_DIR}/ValueIteration.cpp
    ${CMAKE_SOURCE_DIR}/main.cpp
)

set(EXERCISES_HEADERS
    ${CMAKE_SOURCE_DIR}/barto_sutton_exercises/4_7/CarRentalEnvironment.h
    ${CMAKE_SOURCE_DIR}/barto_sutton_exercises/4_9/GamblersProblemEnvironment.h
)

set(EXERCISES_SOURCES
    ${CMAKE_SOURCE_DIR}/barto_sutton_exercises/4_7/CarRentalEnvironment.cpp
    ${CMAKE_SOURCE_DIR}/barto_sutton_exercises/4_9/GamblersProblemEnvironment.cpp
)

add_executable(output_executable ${CORE_SOURCES} ${CORE_HEADERS} ${EXERCISES_HEADERS} ${EXERCISES_SOURCES})

target_link_libraries(output_executable PUBLIC matplot nlohmann_json::nlohmann_json ${PYTHON_LIBRARIES})
