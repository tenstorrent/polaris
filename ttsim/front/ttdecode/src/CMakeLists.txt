set(TTDECODE_SOURCES
  decode/decode.cpp
  decode/decoded_instruction.cpp
  decode/rv32.cpp
  decode/tensix.cpp
  decode/utils.cpp
  elf/elf_parser.cpp
  elf/elf_parsers.cpp
  isa/defaults.cpp
  isa/isa.cpp
  isa/yaml_parser.cpp
)

add_library(ttdecode
  ${TTDECODE_SOURCES}
)

add_library(ttdecode::ttdecode ALIAS ttdecode)

target_include_directories(ttdecode
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Resolve and link mandatory yaml-cpp dependency
if(TARGET deps::yaml-cpp)
  set(_YAML_TARGET deps::yaml-cpp)
elseif(TARGET yaml-cpp::yaml-cpp)
  set(_YAML_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET YAML::yaml-cpp)
  set(_YAML_TARGET YAML::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(_YAML_TARGET yaml-cpp)
else()
  message(FATAL_ERROR
    "Required dependency yaml-cpp not found. Include cmake/deps/yaml-cpp.cmake before add_subdirectory(src)."
  )
endif()
target_link_libraries(ttdecode PUBLIC ${_YAML_TARGET})
target_compile_definitions(ttdecode PUBLIC TTDECODE_ENABLE_YAML=1)

set_target_properties(ttdecode PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

if(APPLE)
  target_include_directories(ttdecode PRIVATE
    /opt/homebrew/include
    /opt/homebrew/include/libelf
  )
endif()

install(TARGETS ttdecode
  EXPORT TTDecodeTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION include
)
