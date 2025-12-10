if (BUILD_PYTHON_BINDINGS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME py_bindings_tests
    COMMAND ${Python3_EXECUTABLE} -m pytest -q tests/python -k bindings)
  set_tests_properties(py_bindings_tests PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ENVIRONMENT "CMAKE_SOURCE_DIR=${PROJECT_SOURCE_DIR};PYTHONPATH=${PROJECT_SOURCE_DIR}/python/src")
endif()

include(GoogleTest)
add_executable(cpp_tests
  cpp/test_decode.cpp
  cpp/test_decode_all_rv32.cpp
  cpp/test_tensix_swizzle.cpp
  cpp/test_elf_parser.cpp
  cpp/test_isa.cpp
  cpp/test_yaml_parser.cpp
)
target_link_libraries(cpp_tests PRIVATE
  ttdecode
  deps::gtest_main
  deps::gmock
)
target_compile_definitions(cpp_tests PRIVATE
  CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
gtest_discover_tests(cpp_tests DISCOVERY_MODE PRE_TEST)
