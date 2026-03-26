# Operator lookup table (superseded plan)

The simulator consumes **tt-perf master YAML** only. See [LOOKUP_TABLE_MASTER.md](LOOKUP_TABLE_MASTER.md) and [YAML_MASTER_FORMAT.md](../../YAML_MASTER_FORMAT.md).

The former list-of-rows format and `(optype, precision, shape0, shape1)` key are **removed**. Use `tt_perf_mapper` / Excel pipeline (or hand-authored master files) to produce lookup YAML.

**Validation:** Matching rows must supply resolvable **`matrix_pipe_util`** and **`vector_pipe_util`** (percentages **[0, 100]**); optional util keys are range-checked when present. Invalid rows raise **`OperatorPerfLUTValidationError`** and terminate the run (see **LOOKUP_TABLE_MASTER.md**).
