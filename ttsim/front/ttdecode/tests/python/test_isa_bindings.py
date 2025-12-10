import os
import pytest

try:
    from ttdecode.core import _core as core
except Exception:
    core = None

pytestmark = pytest.mark.skipif(core is None, reason="_core module not available; build with Python bindings enabled")


def get_path(rel):
    return os.path.join(os.path.dirname(__file__), rel)


def test_enums_and_helpers():
    isa = core.isa
    assert isa.is_tensix(isa.instruction_kind.ttwh)
    assert not isa.is_tensix(isa.instruction_kind.rv32)
    assert isinstance(isa.opcode_start_bit(isa.instruction_kind.rv32), int)
    assert isa.to_string(isa.instruction_kind.ttwh) == "ttwh"
    kinds = isa.tensix_instruction_kinds()
    assert isa.instruction_kind.ttwh in kinds
    assert isa.instruction_kind.ttbh in kinds
    assert isa.instruction_kind.ttqs in kinds


def test_yaml_parse_and_get_instruction_set_from_file():
    isa = core.isa
    path = get_path("../../../../config/llk/instruction_sets/rv32/assembly.yaml")
    if not os.path.exists(path):
        pytest.skip("rv32 assembly.yaml not found")
    data = isa.parse_instruction_set_file(path)
    assert isinstance(data, dict)
    iset = isa.get_instruction_set(path, isa.instruction_kind.rv32)
    assert isinstance(iset, dict)
    assert len(iset) > 0


def test_get_instruction_sets_from_paths_map():
    isa = core.isa
    rv32 = get_path("../../../../config/llk/instruction_sets/rv32/assembly.yaml")
    if not os.path.exists(rv32):
        pytest.skip("assembly.yaml not found")
    mp = {isa.instruction_kind.rv32: rv32}
    sets = isa.get_instruction_sets(mp)
    assert isa.instruction_kind.rv32 in sets
    assert len(sets[isa.instruction_kind.rv32]) > 0
