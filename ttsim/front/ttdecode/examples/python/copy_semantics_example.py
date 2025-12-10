import copy

try:
    from ttdecode.core import _core as core
except Exception:
    raise SystemExit("_core module not available; build with Python bindings enabled")

def main():
    op = core.decode.operands()
    op.set_integer_sources([1, 2])
    op.set_integer_destinations(3)
    di = core.decode.decoded_instruction()
    di.set_operands(op)
    di2 = copy.copy(di)
    di3 = copy.deepcopy(di)
    print("orig src:", di.operands.sources.integers)
    print("copy src:", di2.operands.sources.integers)
    print("deepcopy src:", di3.operands.sources.integers)

if __name__ == "__main__":
    main()

