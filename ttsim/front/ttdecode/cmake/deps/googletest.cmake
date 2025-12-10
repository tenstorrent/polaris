# cmake/deps/googletest.cmake
# Resolve GoogleTest (gtest/gmock) from system/toolchain or fetch from Git.
# Exposes stable targets you can always link to:
#   deps::gtest, deps::gtest_main, deps::gmock, deps::gmock_main

include_guard(GLOBAL)

# Global knobs
option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)
option(USE_SYSTEM_GTEST "Prefer system/toolchain-provided GoogleTest package" ON)

# Minimum version for system/toolchain. Set empty to skip.
set(GTEST_MIN_VERSION "1.12" CACHE STRING "Minimum GoogleTest version to find")

# Git tag/branch/commit when fetching
# Examples:
#   -DGTEST_GIT_TAG=v1.15.2    # pin to a release
#   -DGTEST_GIT_TAG=main       # track branch (non-reproducible)
#   -DGTEST_GIT_TAG=<commit>   # exact commit
set(GTEST_GIT_TAG "HEAD" CACHE STRING "Git tag/branch/commit to fetch for googletest")

# 1) Try system/toolchain package first (prefer local third_party paths)
set(_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/__third_party")
if(EXISTS "${_THIRD_PARTY_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${_THIRD_PARTY_DIR}")
  file(GLOB _gtest_prefixes
    "${_THIRD_PARTY_DIR}/googletest*"
    "${_THIRD_PARTY_DIR}/gtest*"
  )
  foreach(_p IN LISTS _gtest_prefixes)
    if(IS_DIRECTORY "${_p}")
      list(APPEND CMAKE_PREFIX_PATH
        "${_p}"
        "${_p}/lib/cmake/GTest"
        "${_p}/lib64/cmake/GTest"
        "${_p}/cmake"
      )
    endif()
  endforeach()
endif()

if(USE_SYSTEM_GTEST)
  if(GTEST_MIN_VERSION)
    find_package(GTest ${GTEST_MIN_VERSION} CONFIG QUIET)
  else()
    find_package(GTest CONFIG QUIET)
  endif()

  if(NOT GTest_FOUND)
    find_package(GTest QUIET)
  endif()
endif()

# Detect if anything usable is already present
set(_GTEST_ANY FALSE)
foreach(_cand GTest::gtest gtest)
  if(TARGET ${_cand})
    set(_GTEST_ANY TRUE)
  endif()
endforeach()

# 2) Fetch from Git if not found
if(NOT _GTEST_ANY)
  if(NOT FETCH_DEPS)
    message(FATAL_ERROR
      "googletest not found. Provide it via your toolchain/package manager "
      "(set GTest_DIR or CMAKE_PREFIX_PATH), or enable FETCH_DEPS to fetch it.")
  endif()

  include(FetchContent)

  # Optional: centralize fetched sources
  # set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/third_party")

  # Trim extras
  set(INSTALL_GTEST         OFF CACHE BOOL "" FORCE)
  set(gtest_build_tests     OFF CACHE BOOL "" FORCE)
  set(gtest_build_samples   OFF CACHE BOOL "" FORCE)
  # set(gtest_disable_pthreads ON  CACHE BOOL "" FORCE)  # if you need single-threaded gtest
  # set(gtest_force_shared_crt ON  CACHE BOOL "" FORCE)  # MSVC: match shared runtime

  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        ${GTEST_GIT_TAG}
    # GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(googletest)
  message(STATUS "Fetched googletest from Git (tag/branch/commit: ${GTEST_GIT_TAG})")
else()
  message(STATUS "Using system/toolchain-provided googletest")
endif()

# Helper: make a forwarding INTERFACE target and a namespaced ALIAS to it.
# This avoids "ALIAS of ALIAS" errors and works with IMPORTED targets too.
function(_deps_forward alias)
  set(_resolved "")
  foreach(_cand IN LISTS ARGN)
    if(TARGET ${_cand})
      set(_t ${_cand})
      # Resolve alias chains to a real target
      while(TRUE)
        get_target_property(_aliased ${_t} ALIASED_TARGET)
        if(NOT _aliased)
          break()
        endif()
        set(_t ${_aliased})
      endwhile()
      set(_resolved ${_t})
      break()
    endif()
  endforeach()

  if(NOT _resolved)
    return()  # nothing to forward
  endif()

  set(_forward "deps_${alias}")
  if(NOT TARGET ${_forward})
    add_library(${_forward} INTERFACE)
    target_link_libraries(${_forward} INTERFACE ${_resolved})
  endif()

  if(NOT TARGET deps::${alias})
    add_library(deps::${alias} ALIAS ${_forward})
  endif()
endfunction()

# Create stable forwarding targets (only if their sources exist)
_deps_forward(gtest       GTest::gtest       gtest)
_deps_forward(gtest_main  GTest::gtest_main  gtest_main)
_deps_forward(gmock       GTest::gmock       gmock)
_deps_forward(gmock_main  GTest::gmock_main  gmock_main)

# Sanity: require at least gtest to exist
if(NOT TARGET deps::gtest)
  message(FATAL_ERROR "googletest was expected to be available, but no gtest target was found.")
endif()
