# python/CMakeLists.txt

# Make sure nanobind is available (top-level usually includes it already)
if(NOT COMMAND nanobind_add_module)
  include(${PROJECT_SOURCE_DIR}/cmake/deps/nanobind.cmake)
endif()

# One combined module with submodules elf and yaml
nanobind_add_module(ttdecode_core
  src/ttdecode/core/decode_bindings.cpp
  src/ttdecode/core/elf_bindings.cpp
  src/ttdecode/core/isa_bindings.cpp
  src/ttdecode/core/module.cpp
  # Conditionally add YAML bindings to the build if you guard at CMake level:
  # $<BOOL:${ENABLE_YAML}> is a generator expression that includes the file only when true
  # $<$<BOOL:${ENABLE_YAML}>:src/ttdecode/core/isa_bindings.cpp>
  # $<$<BOOL:${ENABLE_YAML}>:src/ttdecode/core/decode_bindings.cpp>
)

# Enforce C++17 and strict standard mode for this target
set_target_properties(ttdecode_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS OFF
  OUTPUT_NAME _core         # ensure the artifact is named _core.* for import
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/src/ttdecode/core"
)

# Link against your unified library (from src/CMakeLists.txt)
target_link_libraries(ttdecode_core PRIVATE ttdecode)
