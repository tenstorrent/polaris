from ttdecode.core import isa

def main():
    d = isa.global_defaults()
    print("instruction_set_file_paths:")
    for k, v in d.instruction_set_file_paths().items():
        print(f"  {k}: {v}")

    print("adding custom path for rv32...")
    d.update_instruction_set_path(isa.instruction_kind.rv32, "/path/to/custom/rv32/assembly.yaml")
    print("rv32 ->", d.instruction_set_file_paths()[isa.instruction_kind.rv32])

    print("current riscv_attributes_instruction_kinds:")
    for kinds, attrs in d.riscv_attributes_instruction_kinds().items():
        kinds_str = ",".join(isa.to_string(k) for k in kinds)
        print(f"  {{{kinds_str}}}: {sorted(list(attrs))}")

    print("adding a custom attribute mapping for rv32+ttwh...")
    d.append_riscv_attribute({isa.instruction_kind.rv32, isa.instruction_kind.ttwh}, str("riscv_custom_arch_string"))
    print("current riscv_attributes_instruction_kinds:")
    for kinds, attrs in d.riscv_attributes_instruction_kinds().items():
        kinds_str = ",".join(isa.to_string(k) for k in kinds)
        print(f"  {{{kinds_str}}}: {sorted(list(attrs))}")
    d.append_riscv_attribute({isa.instruction_kind.rv32, isa.instruction_kind.ttwh}, {"abc", "def"})
    print("current riscv_attributes_instruction_kinds:")
    for kinds, attrs in d.riscv_attributes_instruction_kinds().items():
        kinds_str = ",".join(isa.to_string(k) for k in kinds)
        print(f"  {{{kinds_str}}}: {sorted(list(attrs))}")
    print("- reset attributes")
    d.reset_riscv_attributes_instruction_kinds()
    print("current riscv_attributes_instruction_kinds:")
    for kinds, attrs in d.riscv_attributes_instruction_kinds().items():
        kinds_str = ",".join(isa.to_string(k) for k in kinds)
        print(f"  {{{kinds_str}}}: {sorted(list(attrs))}")
    print("done.")

if __name__ == "__main__":
    main()
