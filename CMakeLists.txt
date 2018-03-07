###############################################################################
# GLFCV - Light field disparity estimation using a guided filter cost volume
#
# Copyright (C) 2017 Adam Stacey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

#Change this if you need to target a specific CMake version
cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# Enable release by default
#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
#endif()

# Enable C++14
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)
# Set up project
project(GLFCV)
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS_INIT} -fPIC")
# Python Bindings
set(PYTHON_OPTIONS "2.X" "3.X")
set(PYTHON_DESIRED_VERSION "3.X" CACHE STRING "Choose which python version to use, options are: ${PYTHON_OPTIONS}.")
set_property(CACHE PYTHON_DESIRED_VERSION PROPERTY STRINGS ${PYTHON_OPTIONS})

include("DetectPython")
if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    SET(PYTHON_INCLUDE_DIRS ${PYTHON2_INCLUDE_DIR} ${PYTHON2_INCLUDE_DIR2} ${PYTHON2_NUMPY_INCLUDE_DIRS})
    SET(PYTHON_LIBRARIES ${PYTHON2_LIBRARY})
    SET(PYTHON_EXECUTABLE ${PYTHON2_EXECUTABLE})
    SET(PYTHON_PACKAGES_PATH ${PYTHON2_PACKAGES_PATH})
    SET(ARCHIVE_OUTPUT_NAME glfcv2)
else ()
    SET(PYTHON_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR} ${PYTHON3_INCLUDE_DIR2} ${PYTHON3_NUMPY_INCLUDE_DIRS})
    SET(PYTHON_LIBRARIES ${PYTHON3_LIBRARY})
    SET(PYTHON_EXECUTABLE ${PYTHON3_EXECUTABLE})
    SET(PYTHON_PACKAGES_PATH ${PYTHON3_PACKAGES_PATH})
    SET(ARCHIVE_OUTPUT_NAME glfcv3)
endif ()

# Detect and add OpenCV
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
find_package(OpenCV REQUIRED HINTS /usr/share/opencv)

find_package(CUDA 8.0 REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.58 COMPONENTS python3 filesystem regex REQUIRED)

# Add files
file(GLOB SOURCES src/*.cpp src/cuda-kernels/*.cu)
file(GLOB HEADERS src/*.h src/cuda-kernels/*.h)
source_group("Header Files" FILES ${HEADERS})


# Define sources and executable
set(LIBRARY_NAME "GLFCV")
cuda_add_library(${LIBRARY_NAME} ${SOURCES} ${HEADERS})

# Link against OpenCV
include_directories(${OpenCV_INCLUDE_DIR})
target_link_libraries(${LIBRARY_NAME} ${OpenCV_LIBS})

# Link against CUDA
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} ${CUDA_LIBRARIES})

# Link against boost filesystem
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} ${Boost_LIBRARIES})

#============================
file(GLOB project_sources ${CMAKE_CURRENT_SOURCE_DIR}/pythonbinding/src/*.cpp)

add_library(${PROJECT_NAME}py SHARED ${project_sources}
  ${HEADERS}
  ${CMAKE_CURRENT_SOURCE_DIR}/pythonbinding/include/pyboostcvconverter/pyboostcvconverter.hpp)

target_include_directories(${PROJECT_NAME}py PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/pythonbinding/include"
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME}py
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${LIBRARY_NAME}
        )

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
        RESULT_VARIABLE PYTHON_${PROJECT_NAME}_PY_PROCESS
        OUTPUT_VARIABLE ${PROJECT_NAME}_PY_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)

set_target_properties(${PROJECT_NAME}py PROPERTIES
        ARCHIVE_OUTPUT_NAME ${ARCHIVE_OUTPUT_NAME}  # prevent name conflict for python2/3 outputs
        PREFIX ""
        OUTPUT_NAME GLFCV
        SUFFIX ${${PROJECT_NAME}_PY_SUFFIX})

if (MSVC AND NOT PYTHON_DEBUG_LIBRARIES)
    set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else ()
    set(PYTHON_INSTALL_CONFIGURATIONS "")
endif ()

if (WIN32)
    set(PYTHON_INSTALL_ARCHIVE "")
else ()
    set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python)
endif ()

install(TARGETS ${PROJECT_NAME}py
        ${PYTHON_INSTALL_CONFIGURATIONS}
        RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        ${PYTHON_INSTALL_ARCHIVE}
        )
