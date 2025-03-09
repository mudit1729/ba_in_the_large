cmake_minimum_required(VERSION 3.10)
project(ba_in_the_large VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(Ceres REQUIRED)

# Use pybind11 from the Python package
if(NOT DEFINED PYBIND11_INCLUDE_DIR)
    message(FATAL_ERROR "PYBIND11_INCLUDE_DIR must be specified")
endif()

include_directories(${PYBIND11_INCLUDE_DIR})
message(STATUS "Using pybind11 include: ${PYBIND11_INCLUDE_DIR}")

# Add the C++ library
add_library(ba_core STATIC
    ba_core.cpp
    ba_core.h
)

# Link dependencies to the core library
target_link_libraries(ba_core
    PUBLIC Ceres::ceres
)

# Ensure Ceres headers are found
target_include_directories(ba_core
    PUBLIC ${CERES_INCLUDE_DIRS}
)

# Determine Python include directory - use the same Python that's running the build
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c" "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using Python include: ${PYTHON_INCLUDE}")

# Create the Python module
add_library(ba_cpp MODULE
    pybind_wrapper.cpp
)

# Set Python module properties
set_target_properties(ba_cpp PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

# Include Python headers
target_include_directories(ba_cpp PRIVATE 
    ${PYTHON_INCLUDE}
    ${PYBIND11_INCLUDE_DIR}
)

# Link the C++ library to the Python module
target_link_libraries(ba_cpp
    PRIVATE ba_core
)