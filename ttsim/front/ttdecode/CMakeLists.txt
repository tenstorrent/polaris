cmake_minimum_required(VERSION 3.15)
project(ttdecode LANGUAGES CXX)

# Only set global defaults if we are the top-level project
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
  elseif (CMAKE_CXX_STANDARD LESS 17)
    message(FATAL_ERROR "This project requires C++17 or newer. "
                        "Set -DCMAKE_CXX_STANDARD=17 (or higher).")
  endif()
endif()

# Optional: put fetched deps in a centralized folder
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/__third_party")

# Prefer locally vendored packages under __third_party when resolving find_package
if(EXISTS "${CMAKE_SOURCE_DIR}/__third_party")
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/__third_party")
endif()

# Options to control optional components
option(BUILD_PYTHON_BINDINGS "Build Python bindings with nanobind" ON)
# Toggle for YAML-based features you’ll add later
option(ENABLE_YAML "Enable YAML-based features" ON)

# Provide the standard BUILD_TESTING option and CTest integration
include(CTest)

# --------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------

# nanobind: only if we’re building Python bindings
if (BUILD_PYTHON_BINDINGS)
  include("${CMAKE_SOURCE_DIR}/cmake/deps/nanobind.cmake")
endif()

# yaml-cpp: only if YAML features are enabled (safe to turn on later)
if (ENABLE_YAML)
  include("${CMAKE_SOURCE_DIR}/cmake/deps/yaml-cpp.cmake")
endif()

# googletest: only if tests are enabled
if (BUILD_TESTING)
  include("${CMAKE_SOURCE_DIR}/cmake/deps/googletest.cmake")
endif()

# --------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------

# Library sources
add_subdirectory(src)

# Examples
add_subdirectory(examples)

# Tests (only if enabled)
if (BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Python bindings (ensure nanobind is in scope before entering python/)
if (BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
