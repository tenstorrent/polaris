#include <nanobind/nanobind.h>
namespace nb = nanobind;

// Declarations provided by your binding units
void bind_elf(nb::module_& m);
void bind_isa(nb::module_& m);
void bind_decode(nb::module_& m);

NB_MODULE(_core, m) {
    m.doc() = "ttdecode core bindings (submodules: elf, isa, decode)";

    // ELF submodule
    auto m_elf = m.def_submodule("elf", "ELF helpers");
    bind_elf(m_elf);

    // ISA submodule (guarded by YAML feature flag)
    #if TTDECODE_ENABLE_YAML
        auto m_isa = m.def_submodule("isa", "ISA helpers");
        bind_isa(m_isa);

        auto m_decode = m.def_submodule("decode", "Instruction decode");
        bind_decode(m_decode);

        // Expose enum alias under decode for convenience
        m_decode.attr("instruction_kind") = m_isa.attr("instruction_kind");
    #endif
}
