try:
    from ttdecode.core import _core as core
except Exception:
    raise SystemExit("_core module not available; build with Python bindings enabled")

def main():
    r = core.decode.registers()
    r.set_integers([1, 2])
    op = core.decode.operands()
    op.set_integer_sources([1, 2])
    op.set_integer_destinations(3)
    op.set_immediates([0x10, -4])
    op.set_attributes({"attr": 1})
    di = core.decode.decoded_instruction()
    di.word = 0
    di.mnemonic = "NOP"
    di.operands = op
    print("operands sources:", di.operands.sources.integers)
    print("operands destinations:", di.operands.destinations.integers)
    print("operands immediates:", di.operands.immediates)
    print("operands attributes:", di.operands.attributes)

if __name__ == "__main__":
    main()

