import os
import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")

def test_struct_bindings_present():
    r = core.decode.registers()
    r.integers = [1, 2]
    r.floats = [3, 4]
    op = core.decode.operands()
    op.all = {"rd": 2}
    op.attributes = {"x": 1}
    op.sources = r
    op.destinations = r
    op.immediates = [0x10]
    op.decoded_values = {"foo": ["bar"]}
    di = core.decode.decoded_instruction()
    di.word = 0
    # di.program_counter = None
    # di.kind = None
    # di.opcode = None
    # di.mnemonic = None
    di.operands = op
    assert di.operands is not None

def test_generic_decode_and_kind_rv32():
    path = os.path.join(os.path.dirname(__file__), "../../../../config/llk/instruction_sets/rv32/assembly.yaml")
    if not os.path.exists(path):
        pytest.skip("rv32 assembly.yaml not found")
    uimm = 0x12345 & 0xFFFFF
    rd = 2
    opcode = 0x37
    word = (uimm << 12) | (rd << 7) | opcode
    sets = {core.isa.instruction_kind.rv32: core.isa.get_instruction_set(path, core.isa.instruction_kind.rv32)}
    kind = core.decode.get_instruction_kind(word, sets, True)
    di = core.decode.decode(word, kind, sets, True)
    assert di.mnemonic == "LUI"
    assert di.opcode == opcode
