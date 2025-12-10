import os
try:
    from ttdecode.core import _core as core
except Exception:
    raise SystemExit("_core module not available; build Python bindings")

def main():
    rv32_yaml = os.path.join(os.environ.get("CMAKE_SOURCE_DIR", ""), "third_party/polaris/ttsim/config/llk/instruction_sets/rv32/assembly.yaml")
    if not os.path.exists(rv32_yaml):
        print("rv32 assembly.yaml not found; example skipped")
        return
    uimm = 0x12345 & 0xFFFFF
    rd = 2
    opcode = 0x37
    word = (uimm << 12) | (rd << 7) | opcode
    sets = {core.isa.instruction_kind.rv32: core.isa.get_instruction_set(rv32_yaml, core.isa.instruction_kind.rv32)}
    kind = core.decode.get_instruction_kind(word, sets, True)
    di = core.decode.decode(word, kind, sets, True)
    print("mnemonic:", di.mnemonic)
    print("opcode:", di.opcode)

if __name__ == "__main__":
    main()
