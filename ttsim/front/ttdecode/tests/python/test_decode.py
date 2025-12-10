import os
import sys
import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")

def test_rv32_decode_basic():
    rv32 = core.decode
    path = os.path.join(os.path.dirname(__file__), "../../../../config/llk/instruction_sets/rv32/assembly.yaml")
    if not os.path.exists(path):
        pytest.skip("rv32 assembly.yaml not found")
    uimm = 0x12345 & 0xFFFFF
    rd = 2
    opcode = 0x37
    word = (uimm << 12) | (rd << 7) | opcode
    di = rv32.rv32_decode(word, path, True)
    assert di.mnemonic == "LUI"
    assert di.opcode == opcode

def test_tensix_decode_opcode_only():
    tensix = core.decode
    path = os.path.join(os.path.dirname(__file__), "../../../../config/llk/instruction_sets/ttwh/assembly.yaml")
    if not os.path.exists(path):
        pytest.skip("ttwh assembly.yaml not found")

    # Craft swizzled word such that opcode after rotl(6)&0xFF == 0xA0
    def make_swizzled(op):
        x = 0
        for i in range(8):
            src = (i + 26) & 31
            if (op >> i) & 1:
                x |= (1 << src)
        return x

    word = make_swizzled(0xA0)
    di = tensix.tensix_decode(word, tensix.instruction_kind.ttwh, path, True)
    assert di.opcode == 0xA0
    assert di.mnemonic is not None
