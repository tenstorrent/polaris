# Operator perf lookup: tt-perf master YAML

## Wire format

Normative schema: [YAML_MASTER_FORMAT.md](../../YAML_MASTER_FORMAT.md) (`schema_name: correqn.tt-perf-master`, `schema_version: 1`).

Loaded by [lookup_operator_perf.py](../../../tools/perf_lookup/lookup_operator_perf.py) via [`tools.perf_lookup.tt_perf_master_loader.load_existing_yaml`](../../../tools/perf_lookup/tt_perf_master_loader.py). Legacy top-level **list** YAML is **not** supported.

**Producer (Excel/CSV → master YAML):** [`tools/perf_lookup/tt_perf_mapper.py`](../../../tools/perf_lookup/tt_perf_mapper.py). From the repository root: `python -m tools.perf_lookup.tt_perf_mapper --help`. Pipeline and column semantics: source in that module and [YAML_MASTER_FORMAT.md](../../YAML_MASTER_FORMAT.md).

### Entry `value` shapes

- **`entry_type: single`:** flat mapping: `num_cores` plus all canonical stat keys (`msecs`, `memory_traffic`, `mem_util`, `noc_util`, `noc_multicast_util`, `npe_cong_impact_pct`, `vector_pipe_util`, `matrix_pipe_util`) on the **same** mapping as `entry_type`. There are **no** per-core integer buckets (e.g. `8: { ... }`).
- **`entry_type: curve`:** `curve_family` (`linear` or `power`) plus one sub-mapping per stat, each with `a`, `b`, `r2`, and `equation`.
- **`entry_type: hybrid`:** optional `single` (same flat shape as above, **without** an inner `entry_type`) and/or optional `curve` (same shape as a top-level curve). **`op_code: matmul`** must use `hybrid` (enforced by the loader).

### Runtime resolution (`OperatorPerfMap.lookup`)

Given simulator `core_count` (from `resolve_operator_lookup_core_count`):

| `entry_type` | Behavior |
|----------------|----------|
| **single** | Use the flat scalars as-is. If `core_count != num_cores`, a **debug** log notes the mismatch; values are still taken from the row. |
| **curve** | Evaluate each requested stat via `curve_family` and that stat’s `a` / `b` at `core_count`. |
| **hybrid** | If **`hybrid.curve`** is present **and** curve mode is enabled (`OperatorPerfMap(..., use_hybrid_curve=True)`, or package / CLI `operator_lookup_hybrid_curve`), evaluate resolved stats from `hybrid.curve` at `core_count`. Otherwise use **`hybrid.single`** like **single** above. |

### LUT validation (Polaris `OperatorPerfMap`)

On a **lookup hit** (key match and `msecs` resolves), the simulator **requires** finite **`matrix_pipe_util`** and **`vector_pipe_util`** (after single read or curve/hybrid evaluation). **0 is allowed.**

Utilization fields stored as **percentages on the wire** must be in **[0, 100] inclusive** when used by lookup:

- **Required (resolved):** `matrix_pipe_util`, `vector_pipe_util`
- **Optional (if present in YAML / curve):** `mem_util`, `noc_util`, `noc_multicast_util`, `npe_cong_impact_pct`

Violations raise **`OperatorPerfLUTValidationError`** with **`file=`** and **`key=`** in the message; **`Device`** logs and **re-raises**, terminating the run.

**Curve / `hybrid.curve`:** Include regression blocks for **both** `matrix_pipe_util` and `vector_pipe_util` so both resolve at runtime `core_count` (omit either → validation error on hit).

## Polaris key bridge

### One-input ops (8-tuple)

Use **only** `op_code` and `input_0_*` fields in the YAML `key` (no `input_1_*`). Same derivation as below for the first tensor.

### Two-input ops (15-tuple)

Full `KEY_TUPLE_YAML_KEYS` order (`op_code`, `input_0_*`, `input_1_*`):

| Field | Source |
|-------|--------|
| `op_code` | `op.optype` lowercased |
| `input_0_*_pad_logical` | `SimTensor.shape` (or, when enabled, tile-padded extents from `Tensor.padded_shape`) padded/truncated to rank 4 via `Shape.to_rank(4)` → `(w,z,y,x)` |
| `input_0_layout` | Tensor `layout` name if present; else `TILE` |
| `input_0_datatype` | Tensor `dtype` name when recognized; else `op.precision` mapped (`BF16` → `BFLOAT16`, …) |
| `input_0_memory` | Tensor memory config string if present; else `DEV_1_DRAM_INTERLEAVED` |
| `input_1_*` | Same for the second input (binary ops only) |

### Three-input ops (22-tuple)

Extended `KEY_TUPLE_YAML_KEYS` order (`op_code`, `input_0_*`, `input_1_*`, `input_2_*`):

| Field | Source |
|-------|--------|
| `op_code` | `op.optype` lowercased |
| `input_0_*` | Same derivation as 8/15 tuple |
| `input_1_*` | Same derivation as 15 tuple |
| `input_2_*` | Same derivation for third input tensor |

**Core count** for curve / hybrid curve evaluation: package `operator_lookup_core_count` if set, else compute IP group `num_units`, else `64`. See `resolve_operator_lookup_core_count`.

**Logical shapes only:** Master lookup keys always use each tensor’s logical rank-4 WZYX from `shape` (after promotion rules in `lookup_operator_perf`). Tile-padded extents (`padded_shape`) are not used for keys.

### Arity not in master format

Ops with **0** or **more than 3** graph inputs are **not** looked up (no 8/15/22 key). The simulator skips them **without** a warning; use DEBUG logs if needed.

A **miss** for unary/binary/ternary (key built but no matching row) logs a **WARNING** with op name, optype, key tuple, core count, and LUT path.

## Simulator output when `uses_perf_lookup`

| Column | Source |
|--------|--------|
| `msecs`, `cycles`, `ideal_*` | Master `msecs` (after guardband handling in `Device.get_exec_stats`) |
| `matrix_pipe_util`, `vector_pipe_util` | **Required** from master; YAML/curve values are **percentages [0, 100]**; stored in exec stats / CSV as **fractions [0, 1]** (`value / 100`) |
| `memory_traffic`, `mem_util` | Master when present; `mem_util` same **% → fraction** rule when present |
| `mem_rd_util`, `mem_wr_util` | Set to **0** on LUT hit (cycle-derived ratios are inconsistent with LUT-rescaled `ideal_cycles`; use `mem_util` from master when present) |
| `uses_perf_lookup` | `True` on hit |

Analytical pipe/mem utilization **> 1.0** checks are skipped when `uses_perf_lookup` (matrix/vector come from validated master percentages converted to fractions).

**Aggregating opstats by LUT hit:** `tools/profiling/sum_polaris_opstats_duration_by_operator.py` sums `msecs` per `optype` and, when the `uses_perf_lookup` column exists (and `--uses-lookup-column` is not empty), appends rollup rows **`LUT-Matches`** and **`LUT-Mismatches`**. **`total_entries`** / **`total_duration_ms`** count only real `optype` rows, not those two.

## Example file

Use the same path as `operator_lookup_file` in package YAML (e.g. [__ext/perf_lookup/whb0_n150_lut.yaml](../../../__ext/perf_lookup/whb0_n150_lut.yaml)): unary `single` rows, binary ops, and matmul as `hybrid` (`single` + `curve`) per the normative format. Those LUT YAML files are not in Git; fetch them from LFC with `lfc_downloader.sh` as described in [OPERATOR_LUT_LFC.md](OPERATOR_LUT_LFC.md).
