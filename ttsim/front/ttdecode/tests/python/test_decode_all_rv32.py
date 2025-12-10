import os
import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")

def test_all_rv32_mnemonics_match():
    path = os.path.join(os.path.dirname(__file__), "../../../../config/llk/instruction_sets/rv32/assembly.yaml")
    if not os.path.exists(path):
        pytest.skip("rv32 assembly.yaml not found")

    iset = core.isa.get_instruction_set(path, core.isa.instruction_kind.rv32)

    def set_bits(word, start, size, value):
        if size == 32:
            mask = 0xFFFFFFFF
        else:
            mask = (1 << size) - 1
        mask <<= start
        word &= ~mask
        word |= ((value & (0xFFFFFFFF if size == 32 else ((1 << size) - 1))) << start)
        return word

    for mnemonic, ins in iset.items():
        if mnemonic in ["SLLI.R1", "SRLI.R1", "SRAI.R1"]:
            continue
        word = 0
        word = set_bits(word, 0, 7, int(ins.opcode))
        for enc in ins.encodings.values():
            word = set_bits(word, int(enc.start), int(enc.size), int(enc.value))

        di = core.decode.rv32_decode(word, path, True)
        assert di.mnemonic == mnemonic
        assert di.opcode == int(ins.opcode)

