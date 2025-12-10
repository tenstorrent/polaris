#!/usr/bin/env python3
import sys

# If your package is installed, this just works:
from ttdecode import elf

# If developing without install, ensure the source package is on sys.path and the
# compiled extension is in ttdecode/core/_core*.so:
#   export PYTHONPATH=/path/to/repo/python/src
# or adjust sys.path here.

def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <elf_file_path>", file=sys.stderr)
        return 1

    elf_path = argv[1]

    try:
        parser = elf.Parser(elf_path)

        # Validate ELF, if the API provides is_valid()
        if hasattr(parser, "is_valid") and not parser.is_valid():
            print("Error: Not a valid ELF file.", file=sys.stderr)
            return 1

        print("--- ELF Header Info ---")
        print(f"Type:  {parser.get_type()}")
        print(f"Class: {parser.get_class()}")
        print(f"Data:  {parser.get_data()}")

        print("\n--- Functions in .text section ---")
        funcs = parser.get_functions(".text")
        for func in funcs:
            # Assuming func has attributes: name (str), value (int address), size (int bytes)
            name = getattr(func, "name", "<noname>")
            value = getattr(func, "value", 0)
            size = getattr(func, "size", 0)
            print(f"Name: {name:<30} Address: {value:#x} Size: {size} bytes")

        return 0

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())