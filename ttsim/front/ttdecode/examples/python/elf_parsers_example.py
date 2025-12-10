#!/usr/bin/env python3
import sys

from ttdecode import elf


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <elf_file1> [<elf_file2> ...]", file=sys.stderr)
        return 1

    paths = argv[1:]
    try:
        ps = elf.parsers(paths)
        print(f"num_elfs={len(ps)}")
        merged = ps.get_instruction_kinds("merged")
        common = ps.get_instruction_kinds("common")
        print("instruction_kinds merged:", sorted(str(k) for k in merged))
        print("instruction_kinds common:", sorted(str(k) for k in common))
        print("match_for_all:", ps.instruction_kinds_match_for_all_elfs())
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

