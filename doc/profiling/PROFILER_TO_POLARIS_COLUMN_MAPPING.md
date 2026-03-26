# TTNN Profiler to Polaris Column Mapping

**Document Purpose:** Define mappings between TTNN Profiler CSV columns and Polaris CSV columns for data conversion.

**Source Files:**
- Profiler CSV: `csvs/ops_perf_results_vitattention_2025_12_30_16_33_05.csv`
- Polaris CSV: `csvs/n150-TTNN-vit_attention-vit_b16_attention-b1-opstats.csv`

**Last Updated:** 2026-02-11

---

## 1. Direct Mappings

These columns have similar content and can be mapped directly (with possible format transformations).

| Profiler Column | Polaris Column | Mapping Type | Transformation | Notes |
|----------------|----------------|--------------|----------------|-------|
| `OP CODE` | `optype` | Direct | Apply optype_mapping | See `map_optype_to_polaris()` function |
| `GLOBAL CALL COUNT` | `opnum` | Direct | Use as-is or normalize | Profiler uses 1024, 2048... Polaris uses 0, 1, 2... |
| `ATTRIBUTES` | `attrs` | Direct | Parse YAML, convert to dict | Remove semicolons, handle nullopt |
| `DEVICE KERNEL DURATION [ns]` | `msecs` | Direct | Divide by 1,000,000 | Convert nanoseconds to milliseconds |
| `INPUT_0_DATATYPE` | `precision` | Direct | Format normalization | BFLOAT16 → BF16 |
| `MATH FIDELITY` | - | Reference | Store in attrs | HiFi2, HiFi4 (currently embedded in attributes) |
| `CORE COUNT` | - | Reference | - | Related to parallelization strategy |

---

## 2. Complex/Derived Mappings

These columns require combining multiple source columns or calculations.

### 2.1 Tensor Shape Mappings

| Profiler Columns | Polaris Column | Transformation | Notes |
|-----------------|----------------|----------------|-------|
| `INPUT_0_W_PAD[LOGICAL]`<br>`INPUT_0_Z_PAD[LOGICAL]`<br>`INPUT_0_Y_PAD[LOGICAL]`<br>`INPUT_0_X_PAD[LOGICAL]`<br>`INPUT_0_DATATYPE` | `input_tensors` | Format as: `input_0[WxZxYxX]:dtype`<br>Collapse W=1 dimension | Currently implemented in `format_tensors()` |
| `INPUT_1_W_PAD[LOGICAL]`<br>`INPUT_1_Z_PAD[LOGICAL]`<br>`INPUT_1_Y_PAD[LOGICAL]`<br>`INPUT_1_X_PAD[LOGICAL]`<br>`INPUT_1_DATATYPE` | `input_tensors` | Format as: `input_1[WxZxYxX]:dtype`<br>Append with semicolon separator | Currently implemented in `format_tensors()` |
| `OUTPUT_0_W_PAD[LOGICAL]`<br>`OUTPUT_0_Z_PAD[LOGICAL]`<br>`OUTPUT_0_Y_PAD[LOGICAL]`<br>`OUTPUT_0_X_PAD[LOGICAL]`<br>`OUTPUT_0_DATATYPE` | `output_tensors` | Format as: `output_0[WxZxYxX]:dtype`<br>Collapse W=1 dimension | Currently implemented in `format_tensors()` |

### 2.2 Data Volume Calculations

| Polaris Column | Source Calculation | Notes |
|----------------|-------------------|-------|
| `inElems` | Calculate from input tensor dimensions | Total input elements |
| `outElems` | Calculate from output tensor dimensions | Total output elements |
| `inBytes` | `inElems × bytes_per_element` | Depends on datatype (BF16 = 2 bytes) |
| `outBytes` | `outElems × bytes_per_element` | Depends on datatype (BF16 = 2 bytes) |

### 2.3 Operation Type Refinement

| Profiler Source | Polaris Column | Logic |
|----------------|----------------|-------|
| `OP CODE` + `ATTRIBUTES['binary_op_type']` | `optype` | If binary_op_type exists, extract ADD/MUL/etc.<br>Otherwise use OP CODE with mapping |

---

## 3. Profiler Columns → Polaris Mapping Table

| # | Profiler Column | Maps to Polaris | Status | Notes |
|---|----------------|-----------------|--------|-------|
| 1 | `OP CODE` | `optype` | ✅ Implemented | With mapping transformation |
| 2 | `OP TYPE` | - | ⚠️ Unused | Value: "tt_dnn_device" - could map to `domain` |
| 3 | `GLOBAL CALL COUNT` | `opnum` | ⚠️ Partial | Used but needs normalization |
| 4 | `DEVICE ID` | - | ❌ Unmapped | Device identifier (0, 1, etc.) |
| 5 | `ATTRIBUTES` | `attrs` | ✅ Implemented | YAML parsed to dict |
| 6 | `MATH FIDELITY` | - | ⚠️ Reference | HiFi2/HiFi4 - could map to pipe or attrs |
| 7 | `CORE COUNT` | - | ❌ Unmapped | Number of cores used |
| 8 | `PARALLELIZATION STRATEGY` | - | ❌ Unmapped | Parallel execution strategy |
| 9 | `HOST START TS` | - | ❌ Unmapped | Host timestamp start |
| 10 | `HOST END TS` | - | ❌ Unmapped | Host timestamp end |
| 11 | `HOST DURATION [ns]` | - | ❌ Unmapped | Host-side duration |
| 12 | `DEVICE FW START CYCLE` | - | ❌ Unmapped | Firmware start cycle |
| 13 | `DEVICE FW END CYCLE` | - | ❌ Unmapped | Firmware end cycle |
| 14 | `OP TO OP LATENCY [ns]` | - | ❌ Unmapped | Inter-op latency |
| 15 | `OP TO OP LATENCY BR/NRISC START [ns]` | - | ❌ Unmapped | Specific latency metric |
| 16 | `DEVICE FW DURATION [ns]` | - | ❌ Unmapped | Firmware duration |
| 17 | `DEVICE KERNEL DURATION [ns]` | `msecs` | ✅ Implemented | Converted to milliseconds |
| 18 | `DEVICE KERNEL DURATION DM START [ns]` | - | ❌ Unmapped | Data movement timing |
| 19 | `DEVICE KERNEL DURATION PER CORE MIN [ns]` | - | ❌ Unmapped | Per-core minimum |
| 20 | `DEVICE KERNEL DURATION PER CORE MAX [ns]` | - | ❌ Unmapped | Per-core maximum |
| 21 | `DEVICE KERNEL DURATION PER CORE AVG [ns]` | - | ❌ Unmapped | Per-core average |
| 22 | `DEVICE KERNEL FIRST TO LAST START [ns]` | - | ❌ Unmapped | Kernel start variance |
| 23 | `DEVICE BRISC KERNEL DURATION [ns]` | - | ❌ Unmapped | BRISC processor timing |
| 24 | `DEVICE NCRISC KERNEL DURATION [ns]` | - | ❌ Unmapped | NCRISC processor timing |
| 25 | `DEVICE TRISC0 KERNEL DURATION [ns]` | - | ❌ Unmapped | TRISC0 processor timing |
| 26 | `DEVICE TRISC1 KERNEL DURATION [ns]` | - | ❌ Unmapped | TRISC1 processor timing |
| 27 | `DEVICE TRISC2 KERNEL DURATION [ns]` | - | ❌ Unmapped | TRISC2 processor timing |
| 28 | `DEVICE ERISC KERNEL DURATION [ns]` | - | ❌ Unmapped | ERISC processor timing |
| 29 | `DEVICE COMPUTE CB WAIT FRONT [ns]` | - | ❌ Unmapped | Circular buffer wait time |
| 30 | `DEVICE COMPUTE CB RESERVE BACK [ns]` | - | ❌ Unmapped | Circular buffer reserve time |
| 31 | `DISPATCH TOTAL CQ CMD OP TIME [ns]` | - | ❌ Unmapped | Command queue timing |
| 32 | `DISPATCH GO SEND WAIT TIME [ns]` | - | ❌ Unmapped | Dispatch wait time |
| 33 | `INPUT_0_W_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 34 | `INPUT_0_Z_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 35 | `INPUT_0_Y_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 36 | `INPUT_0_X_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 37 | `INPUT_0_LAYOUT` | - | ❌ Unmapped | TILE/ROW_MAJOR layout |
| 38 | `INPUT_0_DATATYPE` | `precision` + `input_tensors` | ✅ Implemented | Used for both precision and tensor format |
| 39 | `INPUT_0_MEMORY` | - | ❌ Unmapped | Memory location (DRAM/L1) |
| 40 | `INPUT_1_W_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 41 | `INPUT_1_Z_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 42 | `INPUT_1_Y_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 43 | `INPUT_1_X_PAD[LOGICAL]` | `input_tensors` | ✅ Implemented | Combined into tensor format |
| 44 | `INPUT_1_LAYOUT` | - | ❌ Unmapped | TILE/ROW_MAJOR layout |
| 45 | `INPUT_1_DATATYPE` | `precision` + `input_tensors` | ✅ Implemented | Used for both precision and tensor format |
| 46 | `INPUT_1_MEMORY` | - | ❌ Unmapped | Memory location (DRAM/L1) |
| 47 | `OUTPUT_0_W_PAD[LOGICAL]` | `output_tensors` | ✅ Implemented | Combined into tensor format |
| 48 | `OUTPUT_0_Z_PAD[LOGICAL]` | `output_tensors` | ✅ Implemented | Combined into tensor format |
| 49 | `OUTPUT_0_Y_PAD[LOGICAL]` | `output_tensors` | ✅ Implemented | Combined into tensor format |
| 50 | `OUTPUT_0_X_PAD[LOGICAL]` | `output_tensors` | ✅ Implemented | Combined into tensor format |
| 51 | `OUTPUT_0_LAYOUT` | - | ❌ Unmapped | TILE/ROW_MAJOR layout |
| 52 | `OUTPUT_0_DATATYPE` | `precision` + `output_tensors` | ✅ Implemented | Used for both precision and tensor format |
| 53 | `OUTPUT_0_MEMORY` | - | ❌ Unmapped | Memory location (DRAM/L1) |
| 54 | `METAL TRACE ID` | - | ❌ Unmapped | Trace identifier |
| 55 | `METAL TRACE REPLAY SESSION ID` | - | ❌ Unmapped | Replay session ID |
| 56 | `COMPUTE KERNEL SOURCE` | - | ❌ Unmapped | Kernel source file paths |
| 57 | `COMPUTE KERNEL HASH` | - | ❌ Unmapped | Kernel hash values |
| 58 | `DATA MOVEMENT KERNEL SOURCE` | - | ❌ Unmapped | DM kernel source paths |
| 59 | `DATA MOVEMENT KERNEL HASH` | - | ❌ Unmapped | DM kernel hash values |
| 60 | `TENSIX DM 0 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Kernel size information |
| 61 | `TENSIX DM 1 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Kernel size information |
| 62 | `TENSIX COMPUTE 0 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Kernel size information |
| 63 | `TENSIX COMPUTE 1 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Kernel size information |
| 64 | `TENSIX COMPUTE 2 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Kernel size information |
| 65 | `ACTIVE ETH DM 0 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Ethernet kernel size |
| 66 | `ACTIVE ETH DM 1 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Ethernet kernel size |
| 67 | `IDLE ETH DM 0 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Ethernet kernel size |
| 68 | `IDLE ETH DM 1 MAX KERNEL SIZE [B]` | - | ❌ Unmapped | Ethernet kernel size |
| 69 | `PM IDEAL [ns]` | `ideal_cycles` or `ideal_msecs` | ⚠️ Potential | Performance model ideal time |
| 70 | `PM COMPUTE [ns]` | `compute_cycles` | ⚠️ Potential | Could map with conversion |
| 71 | `PM BANDWIDTH [ns]` | `mem_rd_cycles` or `mem_wr_cycles` | ⚠️ Potential | Memory bandwidth time |
| 72 | `PM REQ I BW` | - | ❌ Unmapped | Required input bandwidth |
| 73 | `PM REQ O BW` | - | ❌ Unmapped | Required output bandwidth |
| 74 | `PM FPU UTIL (%)` | `matrix_pipe_util` or `vector_pipe_util` | ⚠️ Potential | FPU utilization |
| 75 | `NOC UTIL (%)` | - | ❌ Unmapped | Network-on-chip utilization |
| 76 | `DRAM BW UTIL (%)` | `mem_rd_util` or `mem_wr_util` | ⚠️ Potential | DRAM bandwidth utilization |
| 77 | `NPE CONG IMPACT (%)` | - | ❌ Unmapped | NPE congestion impact |
| 78 | `SFPU Util Min (%)` | - | ❌ Unmapped | SFPU minimum utilization |
| 79 | `SFPU Util Median (%)` | - | ❌ Unmapped | SFPU median utilization |
| 80 | `SFPU Util Max (%)` | - | ❌ Unmapped | SFPU maximum utilization |
| 81 | `Avg SFPU util on full grid (%)` | - | ❌ Unmapped | SFPU average utilization |
| 82 | `FPU Util Min (%)` | - | ❌ Unmapped | FPU minimum utilization |
| 83 | `FPU Util Median (%)` | - | ❌ Unmapped | FPU median utilization |
| 84 | `FPU Util Max (%)` | - | ❌ Unmapped | FPU maximum utilization |
| 85 | `Avg FPU util on full grid (%)` | - | ❌ Unmapped | FPU average utilization |
| 86 | `MATH Util Min (%)` | - | ❌ Unmapped | Math minimum utilization |
| 87 | `MATH Util Median (%)` | - | ❌ Unmapped | Math median utilization |
| 88 | `MATH Util Max (%)` | - | ❌ Unmapped | Math maximum utilization |
| 89 | `Avg Math util on full grid (%)` | - | ❌ Unmapped | Math average utilization |

**Legend:**
- ✅ **Implemented**: Currently mapped in `csv_profiler_to_polaris.py`
- ⚠️ **Potential**: Could be mapped with additional logic
- ❌ **Unmapped**: No current mapping (profiler-specific or requires clarification)

---

## 4. Polaris Columns ← Source Mapping

| # | Polaris Column | Source | Status | Notes |
|---|----------------|--------|--------|-------|
| 1 | `archname` | - | ❌ Hardcode | "Wormhole" - from config |
| 2 | `devname` | - | ❌ Hardcode | "n150" - from config |
| 3 | `freq_MHz` | - | ❌ Hardcode | "1000.0" - from config |
| 4 | `pipe` | `MATH FIDELITY` or inferred | ❌ Needs logic | MATRIX/VECTOR - based on optype |
| 5 | `precision` | `INPUT_0_DATATYPE` | ✅ Implemented | BFLOAT16 → BF16 |
| 6 | `wlgroup` | - | ❌ Hardcode | "TTNN" - from config |
| 7 | `wlname` | - | ❌ Hardcode | "vit_attention" - from config |
| 8 | `wlinstance` | - | ❌ Hardcode | "vit_b16_attention" - from config |
| 9 | `batch` | - | ❌ Hardcode | "1" - from config |
| 10 | `opnum` | `GLOBAL CALL COUNT` | ⚠️ Partial | Needs normalization |
| 11 | `opname` | - | ❌ Generate | Generate from optype + counter |
| 12 | `is_input_node` | - | ❌ Needs analysis | Requires graph topology analysis |
| 13 | `is_output_node` | - | ❌ Needs analysis | Requires graph topology analysis |
| 14 | `optype` | `OP CODE` + `ATTRIBUTES` | ✅ Implemented | With mapping function |
| 15 | `op_rpt_count` | - | ❌ Hardcode | "1" - typically 1 |
| 16 | `attrs` | `ATTRIBUTES` | ✅ Implemented | YAML to dict conversion |
| 17 | `inList` | - | ❌ Needs analysis | Input tensor names list |
| 18 | `outList` | - | ❌ Needs analysis | Output tensor names list |
| 19 | `input_tensors` | `INPUT_{0,1}_{W,Z,Y,X}_PAD` + datatype | ✅ Implemented | Formatted as tensor strings |
| 20 | `output_tensors` | `OUTPUT_0_{W,Z,Y,X}_PAD` + datatype | ✅ Implemented | Formatted as tensor strings |
| 21 | `weight_tensors` | - | ❌ Empty | Usually empty |
| 22 | `domain` | `OP TYPE` | ⚠️ Potential | Could use "tt_dnn_device" or "None" |
| 23 | `opclass` | - | ❌ Hardcode | "None" |
| 24 | `removed` | - | ❌ Hardcode | "False" |
| 25 | `fused` | - | ❌ Needs analysis | Fusion analysis required |
| 26 | `fused_with_op` | - | ❌ Needs analysis | Fusion partner identification |
| 27 | `inElems` | Calculated from input_tensors | ⚠️ Needed | Total input elements |
| 28 | `outElems` | Calculated from output_tensors | ⚠️ Needed | Total output elements |
| 29 | `inBytes` | `inElems × dtype_size` | ⚠️ Needed | Total input bytes |
| 30 | `outBytes` | `outElems × dtype_size` | ⚠️ Needed | Total output bytes |
| 31 | `instrs` | - | ❌ Complex | Instruction type dictionary |
| 32 | `inParamCount` | - | ❌ Hardcode | "0" - parameter count |
| 33 | `inActCount` | `inElems` | ⚠️ Needed | Input activation count |
| 34 | `outActCount` | `outElems` | ⚠️ Needed | Output activation count |
| 35 | `instr_count` | - | ❌ Complex | Total instruction count |
| 36 | `compute_cycles` | `PM COMPUTE [ns]` | ⚠️ Potential | With conversion |
| 37 | `mem_rd_cycles` | - | ❌ Needed | Memory read cycles |
| 38 | `mem_wr_cycles` | - | ❌ Needed | Memory write cycles |
| 39 | `ramp_penalty` | - | ❌ Hardcode | "50.0" - typical value |
| 40 | `rsrc_bnck` | - | ❌ Needs logic | Resource bottleneck (COMP/MEM/NA) |
| 41 | `ideal_cycles` | `PM IDEAL [ns]` | ⚠️ Potential | With conversion |
| 42 | `ideal_msecs` | `PM IDEAL [ns]` | ⚠️ Potential | Divide by 1,000,000 |
| 43 | `cycles` | - | ❌ Needed | Total cycles estimate |
| 44 | `matrix_cycles` | `PM COMPUTE [ns]` | ⚠️ Potential | For MATRIX ops |
| 45 | `vector_cycles` | - | ❌ Needed | For VECTOR ops |
| 46 | `msecs` | `DEVICE KERNEL DURATION [ns]` | ✅ Implemented | Converted to ms |
| 47 | `matrix_pipe_util` | `PM FPU UTIL (%)` | ⚠️ Potential | For MATRIX ops |
| 48 | `vector_pipe_util` | - | ❌ Needed | For VECTOR ops |
| 49 | `mem_rd_util` | `DRAM BW UTIL (%)` | ⚠️ Potential | Memory read utilization |
| 50 | `mem_wr_util` | `DRAM BW UTIL (%)` | ⚠️ Potential | Memory write utilization |

**Simulator + tt-perf master lookup (`uses_perf_lookup`):** Master YAML stores pipe/DRAM/NOC util columns as **percentages [0, 100]**; Polaris requires **`matrix_pipe_util`** and **`vector_pipe_util`** to resolve on every LUT hit. Exported **`matrix_pipe_util`** / **`vector_pipe_util`** / **`mem_util`** in study JSON/CSV are **fractions (0–1)** (`percent / 100`). **`mem_rd_util`** and **`mem_wr_util`** are forced to **0** on a LUT hit (see **`doc/tools/perf_lookup/LOOKUP_TABLE_MASTER.md`**); use **`mem_util`** from the master row for DRAM utilization when present.

---

## 5. Operation Type Mapping Table

Current mapping implemented in `map_optype_to_polaris()`:

| Profiler OP CODE | Polaris optype | Notes |
|-----------------|----------------|-------|
| `Matmul` | `MatMul` | Case correction |
| `ADD` | `Add` | Case correction |
| `ReshapeDeviceOperation` | `Reshape` | Simplified |
| `TransposeDeviceOperation` | `Transpose` | Simplified |
| `MUL` | `Mul` | Case correction |
| `SoftmaxDeviceOperation` | `Softmax` | Simplified |
| `BinaryNgDeviceOperation` | *Extract from `binary_op_type`* | ADD, MUL, etc. from attributes |

**TODO:** Add more operation mappings as needed.

---

## 6. Data Type Mapping Table

| Profiler Type | Polaris Type | Bytes per Element |
|--------------|--------------|-------------------|
| `BFLOAT16` | `BF16` or `float16` | 2 |
| `FLOAT32` | `FP32` or `float32` | 4 |
| `INT32` | `INT32` | 4 |
| `UINT32` | `UINT32` | 4 |

**Note:** Polaris uses both short forms (BF16) in `precision` and long forms (float16) in `input_tensors`/`output_tensors`.
Use long form in tensor format strings 

---

## 7. Open Questions / TODOs

### 7.1 Configuration Values
- [ ] How to determine: `archname`, `devname`, `freq_MHz`? Leave these unmapped
- [ ] How to determine: `wlgroup`, `wlname`, `wlinstance`, `batch`? Leave these unmapped
- [ ] Should these come from command-line args or config file? freq_MHz should be a mandatory command line option; leave others blank

### 7.2 Graph Analysis
- [ ] How to identify `is_input_node` and `is_output_node`? is_input_node is true if its inputs are external, i.e. not outputs of other operations. is_output_node is true if the outputs are not inputs to any other operations.
- [ ] How to build `inList` and `outList` (tensor connectivity)? These track actual names of tensors. Since profiler output does not have actual tensor names, leave these as blank
- [ ] Can we infer fusion relationships for `fused` and `fused_with_op`?

### 7.3 Performance Metrics
- [ ] Are PM (performance model) metrics directly comparable to Polaris cycles? No
- [ ] How to calculate or estimate `cycles`, `matrix_cycles`, `vector_cycles`? cycles = DEVICE KERNEL DURATION [ns] converted to cycles using freq_MHz by formula `cycles = (DEVICE KERNEL DURATION [ns] / 1000) * freq_MHz`; matrix_cycles = fpu_utilization * cycles, vector_cycles = sfpu_utilization * cycles . Use Avg FPU Util on full grid and Avg SFPU util on full grid. Do not use PM Compute [ns]. Interpret utilization as percentage. If matrix utilization is either 45 or "45%" and cycles is 10000, then matrix cycles is 45/100*10000 = 4500. 
- [ ] How to determine `rsrc_bnck` (resource bottleneck)? We need to first find out memory cycles. Then if memory cycles > compute cycles, it is MEM, if compute_cycles > memory_cycles, it is COMP. Memory cycles is a sum of both read and write cycles. Estimate memory cycles from PM BANDWIDTH [ns] by `memory_cycles = (PM BANDWIDTH [ns] / 1000) * freq_MHz`. Since PM BANDWIDTH does not distinguish read and write, estimate this as total memory cycles. Do not use PM IDEAL CYCLES . Interpret PM BANDWIDTH as total memory transfer time. For sake of simplicity, mark rd_cycles and wr_cycles both as half of the memory cycles. Calculate compute_cycles as max(matrix_cycles, vector_cycles)
- [ ] Leave ideal cycles and ideal msecs as blank

### 7.4 Instruction Counting
- [ ] How to populate `instrs` dictionary (mac, add, mul, etc.)? Currently not possible
- [ ] How to calculate `instr_count` from profiler data? Currently not possible

### 7.5 Pipe Classification
- [ ] Logic for determining `pipe` (MATRIX vs VECTOR)? fpu is MATRIX and sfpu is VECTOR. Pipe column should be left blank, since an operation can use more than one pipe.
- [ ] Based on operation type? Core count? Math fidelity? No

### 7.6 Missing Calculations
- [ ] Implement `inElems`, `outElems`, `inBytes`, `outBytes` calculations. Each is the sum of input or output, elements or bytes. For broadcast, it is result of broadcast
- [ ] Add element and byte calculations to the conversion script

---

## 8. Implementation Priority

### Phase 1: Core Mappings (Currently Implemented)
- [x] Operation type mapping
- [x] Tensor shape formatting
- [x] Precision/datatype mapping
- [x] Basic timing (msecs)

### Phase 2: Data Volume Calculations
- [ ] Calculate `inElems` and `outElems` from tensor shapes
- [ ] Calculate `inBytes` and `outBytes` from elements × datatype size

### Phase 3: Configuration Integration
- [ ] Add command-line args for workload metadata
- [ ] Support config file for hardware/workload info

### Phase 4: Advanced Metrics
- [ ] Map performance model metrics (cycles, utilization)
- [ ] Implement bottleneck detection
- [ ] Add graph topology analysis

### Phase 5: Complete Coverage
- [ ] Instruction counting
- [ ] Fusion detection
- [ ] All remaining unmapped fields

---

## 8.1 Clarifications
4. opnum = GLOBAL_CALL_COUNT / 1024
9. Unmapped columns should be NA
10. f"{optype}_{opnum}"

---

## 9. Example Row Comparison

### Profiler Row (MatMul):
```
OP CODE: Matmul
GLOBAL CALL COUNT: 1024
INPUT_0: 1x8x224x768 BFLOAT16
INPUT_1: 1x1x768x768 BFLOAT16
OUTPUT_0: 1x8x224x768 BFLOAT16
DEVICE KERNEL DURATION: 411088 ns
CORE COUNT: 24
MATH FIDELITY: HiFi2
```

### Polaris Row (MatMul):
```
optype: MatMul
opnum: 0
input_tensors: input_0[8x224x768]:float16;input_1[768x768]:float16
output_tensors: output_0[8x224x768]:float16
precision: BF16
msecs: 0.061167
inElems: 1966080
outElems: 1376256
```

**Observations:**
- Dimensions collapse W=1 in Polaris
- Time units differ (ns vs ms)
- Polaris has additional computed fields
- Profiler has more detailed timing breakdowns

---

## 10. Notes

- Current implementation focuses on core operation data
- Many profiler columns are device/kernel-specific and may not have Polaris equivalents
- Polaris includes simulation/performance model data not present in profiler output
- Some Polaris fields require multi-row analysis (e.g., graph topology)
- Consider adding a validation mode to compare converted data against reference Polaris files

---

**End of Mapping Document**
