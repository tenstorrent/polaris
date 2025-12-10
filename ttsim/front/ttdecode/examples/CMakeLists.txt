# examples/CMakeLists.txt
# Consolidated definition of example executables to reduce repetition.

function(add_ttdecode_example name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE ttdecode)
  install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()

add_ttdecode_example(elf_example cpp/elf_example.cpp)
add_ttdecode_example(isa_example cpp/isa_example.cpp)
add_ttdecode_example(parse_elf   cpp/parse_elf.cpp)
