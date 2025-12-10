# binutils-cpp

## Overview

- C++ library and Python bindings for instruction set parsing/decoding and basic ELF utilities.
- Single CMake project: core library under `src/`, headers in `include/`, optional Python module in `python/` using nanobind, tests in `tests/`, and examples under `examples/`.

## Dependencies

- CMake 3.15+ and a C++17 compiler.
- Python 3.8+ if building bindings and running binding-backed tests.
- nanobind and googletest are fetched automatically by CMake; yaml-cpp is fetched if YAML features are enabled.
- `libelf` is required:
  - On macOS install via Homebrew: `brew install libelf`.
  - On Ubuntu install via `sudo apt install libelf-dev`

## Configure And Build

- C++ library only:
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_PYTHON_BINDINGS=OFF`
  - `cmake --build build --config Release -j`
- With Python bindings and binding-backed tests (default):
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_PYTHON_BINDINGS=ON`
  - Or simply: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` (testing and bindings are ON by default)
  - `cmake --build build --config Release -j`

## Running Tests

- `ctest --test-dir build --output-on-failure`.
- Some tests reference external YAML files and will `skip` if not present.

## Python Usage

- The extension module builds to `python/src/ttdecode/core/_core.*` and is importable via `PYTHONPATH`.
- Example (from repo root):
  - `export PYTHONPATH=$(pwd)/python/src`
  - `python examples/python/elf_parsers_example.py`

## Examples

- C++ examples: see `examples/cpp` and link with the `ttdecode` target.
- Python examples: see `examples/python/*.py`.

## Notes
