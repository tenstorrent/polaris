from . import _core as _c

# Submodules from the compiled extension
elf = _c.elf
isa = getattr(_c, "isa", None)      # None if ISA (YAML) was not built
decode = getattr(_c, "decode", None)  # None if ISA (YAML) was not built

__all__ = ["elf", "isa", "decode"]
