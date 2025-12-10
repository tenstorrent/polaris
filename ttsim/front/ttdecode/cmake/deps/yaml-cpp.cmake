# cmake/deps/yaml-cpp.cmake
# Resolve yaml-cpp either from the system/toolchain or by fetching from Git.
# Provides a stable alias target: deps::yaml-cpp

include_guard(GLOBAL)

# Global knobs (can be overridden on the CMake command line)
option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)
option(USE_SYSTEM_YAML_CPP "Prefer system/toolchain-provided yaml-cpp" ON)

# Minimum version to accept from system/toolchain
# Set to empty ("") to skip version enforcement.
set(YAML_CPP_MIN_VERSION "0.7" CACHE STRING "Minimum yaml-cpp version to find")

# Tag/branch/commit when fetching from Git
# Examples:
#   -DYAML_CPP_GIT_TAG=v0.8.0     # pin to release
#   -DYAML_CPP_GIT_TAG=main       # track branch (non-reproducible)
#   -DYAML_CPP_GIT_TAG=<commit>   # exact commit
set(YAML_CPP_GIT_TAG "v0.8.0" CACHE STRING "Git tag/branch/commit to fetch for yaml-cpp")

# 1) Try system/toolchain provided yaml-cpp (prefer local third_party paths first)
# Add common third_party hint paths to CMAKE_PREFIX_PATH to enable offline discovery.
set(_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/__third_party")
if(EXISTS "${_THIRD_PARTY_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${_THIRD_PARTY_DIR}")
  file(GLOB _yamlcpp_prefixes
    "${_THIRD_PARTY_DIR}/yaml-cpp*"
  )
  foreach(_p IN LISTS _yamlcpp_prefixes)
    if(IS_DIRECTORY "${_p}")
      list(APPEND CMAKE_PREFIX_PATH
        "${_p}"
        "${_p}/lib/cmake/yaml-cpp"
        "${_p}/lib64/cmake/yaml-cpp"
        "${_p}/cmake"
      )
    endif()
  endforeach()
endif()

if(USE_SYSTEM_YAML_CPP)
  if(YAML_CPP_MIN_VERSION)
    find_package(yaml-cpp ${YAML_CPP_MIN_VERSION} CONFIG QUIET)
  else()
    find_package(yaml-cpp CONFIG QUIET)
  endif()
endif()

# 2) Fallback: fetch from Git if not found
if(NOT TARGET yaml-cpp AND
   NOT TARGET YAML::yaml-cpp AND
   NOT TARGET yaml-cpp::yaml-cpp)

  if(NOT FETCH_DEPS)
    message(FATAL_ERROR
      "yaml-cpp not found. Provide yaml-cpp via your toolchain/package manager "
      "(set yaml-cpp_DIR or CMAKE_PREFIX_PATH), or enable FETCH_DEPS to fetch it.")
  endif()

  include(FetchContent)

  # Optional: choose a shared place for fetched sources at the top level:
  # set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/third_party")

  # Trim extras when building as a subproject
  set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
  set(YAML_CPP_INSTALL       OFF CACHE BOOL "" FORCE)
  # Some versions use YAML_BUILD_SHARED_LIBS; uncomment if you need to force static/shared:
  # set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        ${YAML_CPP_GIT_TAG}
    # GIT_SHALLOW    TRUE   # speed up download; disable if you need full history
  )
  FetchContent_MakeAvailable(yaml-cpp)
  message(STATUS "Fetched yaml-cpp from Git (tag/branch/commit: ${YAML_CPP_GIT_TAG})")
else()
  message(STATUS "Using system/toolchain-provided yaml-cpp")
endif()

# 3) Provide a stable alias for consumers: deps::yaml-cpp
set(_YAML_CPP_TARGET "")
foreach(_cand yaml-cpp YAML::yaml-cpp yaml-cpp::yaml-cpp)
  if(TARGET ${_cand})
    set(_YAML_CPP_TARGET ${_cand})
    break()
  endif()
endforeach()

if(_YAML_CPP_TARGET)
  add_library(deps::yaml-cpp ALIAS ${_YAML_CPP_TARGET})
else()
  message(FATAL_ERROR "yaml-cpp was expected to be available but no suitable CMake target was found.")
endif()
