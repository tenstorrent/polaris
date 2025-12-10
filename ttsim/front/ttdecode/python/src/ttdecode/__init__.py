from . import core as core

# Re-export for a flat public API
elf = core.elf
isa = core.isa      # may be None if ISA (YAML) disabled
decode = core.decode  # may be None if ISA (YAML) disabled

__all__ = ["elf", "isa", "decode"]
