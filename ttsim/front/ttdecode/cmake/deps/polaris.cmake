include_guard(GLOBAL)

option(FETCH_DEPS "Allow fetching dependencies at configure time" ON)
set(POLARIS_GIT_TAG "HEAD" CACHE STRING "Git tag/branch/commit to fetch for polaris")

set(_POLARIS_DIR "${CMAKE_SOURCE_DIR}/__third_party/polaris")

if(NOT FETCH_DEPS)
  if(NOT EXISTS "${_POLARIS_DIR}")
    message(FATAL_ERROR "polaris not available. Enable FETCH_DEPS or provide third_party/polaris.")
  endif()
  return()
endif()

include(ExternalProject)

ExternalProject_Add(polaris_project
  GIT_REPOSITORY https://github.com/tenstorrent/polaris.git
  GIT_TAG        ${POLARIS_GIT_TAG}
  GIT_SHALLOW    TRUE
  UPDATE_DISCONNECTED 1
  SOURCE_DIR     "${_POLARIS_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

add_custom_target(polaris_fetch ALL DEPENDS polaris_project)

