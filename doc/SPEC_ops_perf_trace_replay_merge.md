# SPEC: `ops_perf_trace_replay_merge.py`

Merge three TT-Metal-style ops perf CSVs from a **non-iterative trace+replay** capture (ViT, llama3 decode/prefill) into one output CSV.

This is the trace+replay sibling of [`ops_perf_three_csv_merge.py`](SPEC_ops_perf_three_csv_merge.md) (the iterative, VGG-style merge). It **reuses that tool's classification, header-validation, join, MemTraffic, duration-check, and output machinery verbatim** — the only difference is the **row-reduction** step. Everything in the iterative spec's *File discovery and classification*, *Join keys and sorting*, *Duration check*, *Aggregate kernel duration summary*, *MemTraffic*, and *Output columns* sections applies unchanged; only *Iteration filtering* is replaced by *Replay-session filtering* below.

- **Implementation:** stdlib **`csv`** only; **`loguru`** for warnings/info. Imports the shared helpers from `ops_perf_three_csv_merge.py` (dual import: package path under pytest/`mypy`, bare name when run as an on-device flat-bundle script).
- **Exit code:** `0` success, `1` on any validation error (message on stderr).

## When to use which tool

| Capture shape | Tool | Reduction |
|---|---|---|
| **Iterative** — `run_device_perf(num_iterations=N)` concatenates equal-size iterations (VGG-style) | `ops_perf_three_csv_merge.py` | select one **iteration** |
| **Trace+replay** — one compile/warmup pass + N replays of a captured trace (ViT, llama3) | `ops_perf_trace_replay_merge.py` | select one **replay session** |

The trace+replay tool errors (directing the user to the iterative tool) if the input has no populated `METAL TRACE REPLAY SESSION ID` — i.e. it is not a trace+replay capture. Until the unified pass-classifier is built, tool choice is explicit.

## Why the iterative reducer fails on trace+replay

A trace+replay capture is **not** equal-size iterations. TT-Metal dispatches a compile/warmup pass (each op built individually) and then replays a captured trace N times, tagging each op:

- `METAL TRACE REPLAY SESSION ID`:
  - `''` (blank) → the **compile/warmup** pass. Its op set overlaps the replay op set, so its rows share `(GLOBAL CALL COUNT, OP CODE, OP TYPE)` join keys with the replay rows.
  - `'1'`, `'2'`, … → successive **replays** of the captured trace. Within one replay session the join keys are unique, and they are identical across the raw/perf/trace variants.

The iterative tool's first-op marker heuristic mis-fires (the entry marker recurs intra-pass → unequal chunk sizes → the whole capture is wrongly treated as one iteration), and the downstream join then aborts on a **duplicate join key** (the same op present in the compile pass and each replay session).

## Replay-session filtering (replaces *Iteration filtering*)

1. **Detection column:** `METAL TRACE REPLAY SESSION ID`. Candidate sessions are the **distinct non-blank** ids, in first-appearance (capture) order. If none are present, error (not a trace+replay capture).
2. **Drop the compile/warmup pass:** rows with a blank session id are always removed (steady-state only; compile cost excluded).
3. **Select one session** on the **vanilla** CSV, then apply the same session id to all three variants (they share session layout):
   - `median` (default) — lower-middle by total `DEVICE KERNEL DURATION [ns]` (stable; matches the iterative tool's `median`).
   - `min` / `max` — least / greatest total kernel ns.
   - `first` / `last` — earliest / latest replay session in capture order.
4. **Result:** each variant is reduced to exactly the selected session's ops, whose join keys are unique — handed unchanged to the shared join.

### Composition with iteration reduction (trace+replay+iterative)

If a future capture is trace+replay **and** iterative (a replay session that itself contains several concatenated iterations), the selected session's join keys will **not** be unique. In that case the iterative reducer (`reduce_iterative`) is applied *within* the selected session, composing the two reductions. This is on by default; `--no-compose-iteration` turns it into a hard error instead. For the common one-inference-per-session case the keys are already unique, so no iteration reduction runs — and the iterative marker heuristic (and its warning) is never triggered.

## CLI

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `--input-dir` | yes | — | Root directory; recursively discovers exactly three `ops_perf_results_*.csv` files (falls back to `*.csv`). |
| `--dram-peak-bw-gbps` | yes | — | Peak DRAM bandwidth in **GB/s** (MemTraffic). |
| `--output` | no | `<input-dir>/merged_ops.csv` | Output CSV path. |
| `--duration-rel-tol` | no | `0.05` | Relative tolerance for fpu vs vanilla `DEVICE KERNEL DURATION [ns]`. |
| `--encoding` | no | `utf-8` | Input/output text encoding. |
| `--select-session` | no | `median` | Which replay session to keep: `median`, `min`, `max`, `first`, `last`. The compile/warmup pass is always dropped. |
| `--no-compose-iteration` | no | off | Disable the within-session iteration reducer; a session with duplicate keys becomes a hard error instead. |
| `--ops-per-iteration` | no | — | Forwarded to the composed iteration reducer (trace+replay+iterative only). |
| `--measured-iteration-indices` | no | — | Forwarded to the composed iteration reducer (trace+replay+iterative only). |
| `--select-iteration` | no | `median` | Forwarded to the composed iteration reducer (trace+replay+iterative only). |

## Shared behavior (unchanged from the iterative tool)

Classification (noctrace / fpu / vanilla), header subsequence/omission rules, join keys and sorting, per-row and aggregate duration checks, MemTraffic derivation, and the full output column order are exactly as specified in [`SPEC_ops_perf_three_csv_merge.md`](SPEC_ops_perf_three_csv_merge.md). The reduction runs **after** classification/header validation and **before** the join, so the join always receives one unique-keyed row set per variant.

**Signpost-row filtering** (see the iterative spec's *Input row filtering* section) is likewise inherited: `signpost` marker rows are dropped in the shared `run_merge` immediately after each CSV is read — **before** classification and the replay-session reduction — so a trace+replay capture whose passes are wrapped in Tracy `start`/`stop` signposts is handled identically to an iterative one.
