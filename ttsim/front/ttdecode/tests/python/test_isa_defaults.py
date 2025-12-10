from ttdecode.core import isa

def test_global_defaults_singleton_update_paths():
    d1 = isa.global_defaults()
    rv32_path_before = d1.instruction_set_file_paths()[isa.instruction_kind.rv32]
    d1.update_instruction_set_path(isa.instruction_kind.rv32, "/tmp/rv32.yaml")
    d2 = isa.global_defaults()
    assert d2.instruction_set_file_paths()[isa.instruction_kind.rv32] == "/tmp/rv32.yaml"
    # restore
    d2.update_instruction_set_path(isa.instruction_kind.rv32, rv32_path_before)

def test_append_and_remove_riscv_attributes():
    d = isa.global_defaults()
    key = {isa.instruction_kind.rv32, isa.instruction_kind.ttwh}
    d.append_riscv_attribute(key, "riscv_test_attr")
    assert "riscv_test_attr" in d.riscv_attributes_instruction_kinds()[frozenset(key)]
    d.remove_riscv_attribute(key, "riscv_test_attr")
    assert "riscv_test_attr" not in d.riscv_attributes_instruction_kinds()[frozenset(key)]
