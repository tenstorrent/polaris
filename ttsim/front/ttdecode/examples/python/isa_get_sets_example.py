#!/usr/bin/env python3
import os
from ttdecode import isa

def main():
    kinds = [isa.instruction_kind.rv32, isa.instruction_kind.ttwh]
    base = os.environ.get("CMAKE_SOURCE_DIR", "")
    paths = {
        isa.instruction_kind.rv32: os.path.join(base, "third_party/polaris/ttsim/config/llk/instruction_sets/rv32/assembly.yaml"),
        isa.instruction_kind.ttwh: os.path.join(base, "third_party/polaris/ttsim/config/llk/instruction_sets/ttwh/assembly.yaml"),
    }
    print("tensix kinds:", isa.tensix_instruction_kinds())
    for k in kinds:
        p = paths[k]
        if not os.path.exists(p):
            print("skip kind", k, "missing path:", p)
            continue
        iset = isa.get_instruction_set(p, k)
        print(isa.to_string(k))
        print("count:", len(iset))

if __name__ == "__main__":
    main()
