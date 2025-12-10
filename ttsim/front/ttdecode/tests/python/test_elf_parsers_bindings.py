import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")


def test_parsers_empty_construct():
    ps = core.elf.parsers([])
    assert len(ps) == 0
    assert ps.size() == 0
    assert ps.instruction_kinds_match_for_all_elfs() is True
    assert ps.get_instruction_kinds() == set()
    assert ps.get_instruction_kinds("common") == set()

