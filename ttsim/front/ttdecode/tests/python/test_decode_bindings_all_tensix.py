import os
import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")

@pytest.mark.parametrize("kind,path_rel", [
    (core.isa.instruction_kind.ttwh, "../../../../config/llk/instruction_sets/ttwh/assembly.yaml"),
    (core.isa.instruction_kind.ttbh, "../../../../config/llk/instruction_sets/ttbh/assembly.yaml"),
    (core.isa.instruction_kind.ttqs, "../../../../config/llk/instruction_sets/ttqs/assembly.yaml"),
])
def test_all_tensix_instructions_and_swizzle_parity(kind, path_rel):
    base = os.path.dirname(__file__)
    path = os.path.join(base, path_rel)
    if not os.path.exists(path):
        pytest.skip("assembly.yaml not found")

    iset = core.isa.get_instruction_set(path, kind)
    for mnemonic, ins in iset.items():
        op = int(ins.opcode)
        word = op << 24 
        d1 = core.decode.tensix_decode(word, kind, path, False)
        d2 = core.decode.tensix_decode(core.decode.swizzle(word), kind, path, True)

        assert d1.mnemonic == mnemonic
        assert d1.opcode == op
        assert d1.opcode == d2.opcode
        assert d1.mnemonic == d2.mnemonic

