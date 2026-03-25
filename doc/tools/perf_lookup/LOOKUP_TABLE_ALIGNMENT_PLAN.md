# Operator lookup table (superseded plan)

The simulator consumes **tt-perf master YAML** only. See [LOOKUP_TABLE_MASTER.md](LOOKUP_TABLE_MASTER.md) and [YAML_MASTER_FORMAT.md](../../YAML_MASTER_FORMAT.md).

The former list-of-rows format and `(optype, precision, shape0, shape1)` key are **removed**. Use `tt_perf_mapper` / Excel pipeline (or hand-authored master files) to produce lookup YAML.
