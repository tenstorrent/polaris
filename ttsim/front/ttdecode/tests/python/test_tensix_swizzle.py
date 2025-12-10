import os
import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")

@pytest.mark.parametrize("kind,path_rel", [
    ("ttwh", os.path.join(os.path.dirname(__file__), "../../../../config/llk/instruction_sets/ttwh/assembly.yaml")),
    ("ttbh", os.path.join(os.path.dirname(__file__), "../../../../config/llk/instruction_sets/ttbh/assembly.yaml")),
    ("ttqs", os.path.join(os.path.dirname(__file__), "../../../../config/llk/instruction_sets/ttqs/assembly.yaml")),
])
def test_swizzled_and_unswizzled_equal(kind, path_rel):
    path: str = str(path_rel)
    if not os.path.exists(path):
        pytest.skip("assembly.yaml not found for kind")

    word = 0xA0
    ic = getattr(core.isa.instruction_kind, kind)

    di_s = core.decode.tensix_decode(core.decode.swizzle(word), ic, path, True)
    di_u = core.decode.tensix_decode(word, ic, path, False)

    assert di_s.opcode == di_u.opcode
    assert di_s.mnemonic == di_u.mnemonic
    if di_s.operands is not None and di_u.operands is not None:
        assert di_s.operands.all == di_u.operands.all
        assert di_s.operands.attributes == di_u.operands.attributes

