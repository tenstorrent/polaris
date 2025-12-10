import copy
import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")

def test_operands_persist_through_copy_and_deepcopy():
    op = core.decode.operands()
    op.set_integer_sources([1, 2])
    op.set_integer_destinations(3)
    op.set_immediates([10, -5])
    c = copy.copy(op)
    d = copy.deepcopy(op)
    assert c.sources.integers == [1, 2]
    assert c.destinations.integers == [3]
    assert c.immediates == [10, -5]
    assert d.sources.integers == [1, 2]
    assert d.destinations.integers == [3]
    assert d.immediates == [10, -5]

def test_instruction_operands_survive_copy():
    op = core.decode.operands()
    op.set_integer_sources([4])
    di = core.decode.decoded_instruction()
    di.set_operands(op)
    di2 = copy.copy(di)
    di3 = copy.deepcopy(di)
    assert di2.operands.sources.integers == [4]
    assert di3.operands.sources.integers == [4]

