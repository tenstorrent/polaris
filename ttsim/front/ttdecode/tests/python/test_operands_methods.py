import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")

def test_registers_setters():
    r = core.decode.registers()
    assert r.set_integers([1, 2]) is True
    assert r.integers == [1, 2]
    assert r.set_integers(3) is True
    assert r.integers == [3]
    assert r.set_floats([0, 4]) is True
    assert r.floats == [0, 4]
    assert r.set_floats(5) is True
    assert r.floats == [5]

def test_operands_setters():
    op = core.decode.operands()
    op.set_all({"rd": 2})
    assert op.all == {"rd": 2}
    op.set_immediates([10, -1])
    assert op.immediates == [10, -1]
    op.set_immediates(7)
    assert op.immediates == [7]
    rs = core.decode.registers()
    rd = core.decode.registers()
    rs.set_integers([1, 2])
    rd.set_integers([3])
    op.set_sources(rs)
    op.set_destinations(rd)
    assert op.sources.integers == [1, 2]
    assert op.destinations.integers == [3]
    op.set_integer_sources([4])
    op.set_integer_destinations([5])
    assert op.sources.integers == [4]
    assert op.destinations.integers == [5]
    op.set_float_sources([0, 1])
    op.set_float_destinations(2)
    assert op.sources.floats == [0, 1]
    assert op.destinations.floats == [2]
    op.set_attributes({"x": 1})
    assert op.attributes["x"] == 1

