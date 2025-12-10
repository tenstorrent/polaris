# Ensure safe re-inclusion
include_guard(GLOBAL)

# Allow network fetching at configure time (disable in packaging/offline builds)
option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)

# By default, skip pip for Python ≥3.13 to avoid older wheels; override if you know pip is recent.
# You can enable pip via: -DUSE_PIP_NANOBIND=ON
set(_USE_PIP_DEFAULT OFF)
if(NOT DEFINED USE_PIP_NANOBIND)
  set(USE_PIP_NANOBIND ${_USE_PIP_DEFAULT} CACHE BOOL "Prefer pip-installed nanobind when available")
endif()

# Pin what to fetch. Default is the repo's default branch HEAD (non-reproducible).
# To pin to a release tag:  -DNANOBIND_GIT_TAG=v1.3.2
# To use a branch:          -DNANOBIND_GIT_TAG=main   (or master, depending on repo)
# To pin to a commit:       -DNANOBIND_GIT_TAG=deadbeefcafebabe1234
set(NANOBIND_GIT_TAG "HEAD" CACHE STRING "Git tag/branch/commit to fetch for nanobind")

# 1) Find Python (Interpreter + dev headers)
if(CMAKE_VERSION VERSION_LESS 3.18)
  set(_NB_DEV_COMPONENT Development)
else()
  set(_NB_DEV_COMPONENT Development.Module)
endif()
find_package(Python 3.8 COMPONENTS Interpreter ${_NB_DEV_COMPONENT} REQUIRED)
message(STATUS "Found Python interpreter: ${Python_EXECUTABLE} (version: ${Python_VERSION})")

# 2) Try a toolchain/system-provided nanobind first (prefer local third_party paths)
set(_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/__third_party")
if(EXISTS "${_THIRD_PARTY_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${_THIRD_PARTY_DIR}")
  file(GLOB _nb_prefixes
    "${_THIRD_PARTY_DIR}/nanobind*"
  )
  foreach(_p IN LISTS _nb_prefixes)
    if(IS_DIRECTORY "${_p}")
      list(APPEND CMAKE_PREFIX_PATH
        "${_p}"
        "${_p}/lib/cmake/nanobind"
        "${_p}/lib64/cmake/nanobind"
        "${_p}/cmake"
      )
    endif()
  endforeach()
endif()

find_package(nanobind CONFIG QUIET)

# 3) Optionally try pip-installed nanobind (same Python as above)
if(NOT nanobind_FOUND AND USE_PIP_NANOBIND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE _NB_CMAKE_DIR
    RESULT_VARIABLE _NB_RC
  )
  if(_NB_RC EQUAL 0 AND EXISTS "${_NB_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${_NB_CMAKE_DIR}")
    find_package(nanobind CONFIG QUIET)
    if(nanobind_FOUND)
      message(STATUS "Using pip-installed nanobind at ${_NB_CMAKE_DIR}")
    endif()
  endif()
endif()

# 4) Fallback: fetch from Git
if(NOT nanobind_FOUND)
  if(NOT FETCH_DEPS)
    message(FATAL_ERROR
      "nanobind not found. Set nanobind_DIR/CMAKE_PREFIX_PATH, enable USE_PIP_NANOBIND, "
      "or enable FETCH_DEPS to fetch from Git.")
  endif()

  include(FetchContent)

  # Optional: choose a shared place for fetched deps at the top-level before including this file
  # set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/third_party")

  FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        ${NANOBIND_GIT_TAG}  # defaults to latest default-branch HEAD
    # GIT_SHALLOW    TRUE                # uncomment to speed up downloads (no history)
  )
  FetchContent_MakeAvailable(nanobind)
  message(STATUS "Fetched nanobind from Git (tag/branch/commit: ${NANOBIND_GIT_TAG})")
endif()

# After this, nanobind_add_module(...) should be available.
