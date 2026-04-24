# TTNN Shim Functions for Simulation

This module provides drop-in replacements for ttnn operations that can either:
1. **Track operations** without executing them (for simulation and performance modeling)
2. **Actually execute operations** (for testing logic and verification)

These shims are designed for simulation, performance modeling, and logic testing purposes.

## Features

- **Complete API Compatibility**: All functions match the exact Python API signatures of their ttnn counterparts
- **Dual-Mode Operation**: Can track operations OR actually execute them (or both)
- **Operation Tracking**: Tracks all tilize, untilize, shard layout, and transformer head operations with detailed internal operation tracking
- **Duck-Compatible**: Works with duck-compatible objects (objects that have the same attributes/methods as ttnn.Tensor)
- **Pure Python**: No dependencies on ttnn or torch - completely self-contained
- **Validation**: Includes all validation logic and error handling matching the original functions
- **Logic Testing**: Can actually perform tilize/untilize operations to test correctness

## Functions Provided

### Core Layout Operations

- `to_layout(tensor, layout, dtype=None, memory_config=None, sub_core_grids=None)`
  - Converts tensor between ROW_MAJOR_LAYOUT and TILE_LAYOUT
  - Tracks tilize/untilize operations internally
  - Handles padding/unpadding automatically

- `tilize(input_tensor, memory_config=None, output_dtype=None, use_multicore=True, use_low_perf=False, sub_core_grids=None)`
  - Converts ROW_MAJOR_LAYOUT to TILE_LAYOUT
  - Validates input requirements
  - Tracks operation for simulation

- `untilize(input_tensor, memory_config=None, use_multicore=True, use_pack_untilize=True, sub_core_grids=None)`
  - Converts TILE_LAYOUT to ROW_MAJOR_LAYOUT
  - Validates input requirements
  - Tracks operation for simulation

### Padding Operations

- `tilize_with_val_padding(input_tensor, output_padded_shape, pad_value, memory_config=None, output_dtype=None, use_multicore=True, sub_core_grids=None)`
  - Tilizes with explicit padding to specified shape
  - Tracks operation with input/output shape information

- `untilize_with_unpadding(input_tensor, output_tensor_end, memory_config=None, use_multicore=True, use_pack_untilize=True, sub_core_grids=None)`
  - Untilizes and removes padding
  - Tracks operation with input/output shape information

- `tilize_with_zero_padding(input_tensor, memory_config=None, output_dtype=None, use_multicore=True, sub_core_grids=None)`
  - Convenience function that auto-calculates padded shape
  - Uses zero padding

### Helper Functions

- `reshape(tensor, shape, padded_shape=None, memory_config=None, pad_value=None, reshape_map_mode=None, sub_core_grids=None)`
  - Reshapes tensor to new shape
  - Tracks reshape operations

- `squeeze(tensor, dim)`
  - Removes dimension of size 1

- `pad(tensor, padding, pad_value, output_memory_config=None)`
  - Pads tensor with specified padding

### Shard Layout Operations

- `interleaved_to_sharded(input_tensor, memory_config=None, output_dtype=None)`
  - Converts tensor from interleaved to sharded memory layout
  - Tracks data movement for simulation

- `sharded_to_interleaved(input_tensor, memory_config=None, output_dtype=None)`
  - Converts tensor from sharded to interleaved memory layout
  - Tracks data movement for simulation

- `reshard(input_tensor, memory_config, output_tensor=None)`
  - Changes shard layout of an already-sharded tensor
  - Supports optional `output_tensor` parameter (per tt-metal nanobind)

### Transformer Head Operations

- `nlp_create_qkv_heads(input_tensor, kv_input_tensor=None, *, num_heads, num_kv_heads=None, transpose_k_heads=False, memory_config=None)`
  - Splits fused QKV tensor into separate Q, K, V head tensors
  - Returns `(Q, K, V)` tuple; single SimOp with 3 outputs
  - Also available as `ttnn.experimental.nlp_create_qkv_heads(...)`

- `nlp_concat_heads(input_tensor, memory_config=None)`
  - Merges attention heads: `[B, num_heads, S, head_dim]` -> `[B, S, num_heads*head_dim]`
  - Also available as `ttnn.experimental.nlp_concat_heads(...)`

### Device-graph operator APIs (SimOp-only)

These build `SimOp` nodes on the device graph (like `ttnn` front-end ops) but live only on `ttnn_shim` — not re-exported from `ttsim.front.ttnn`.

- `tilize_op`, `untilize_op`, `tilize_with_val_padding_op`, `untilize_with_unpadding_op` — layout conversions as first-class ops (see `ttsim/ops/desc/ttsim_layout.py`).
- `interleaved_to_sharded_op`, `sharded_to_interleaved_op`, `reshard_op` — shard layout conversions as SimOp graph nodes.
- `nlp_create_qkv_heads_op` — fused QKV head split; 1-2 inputs, 3 outputs (Q, K, V). SimOp type **NLPCreateQKVHeads**.
- `nlp_concat_heads_op` — head merge; SimOp type **NLPConcatHeads**.
- `permute_op(input_tensor, dims, memory_config=None)` — axis reorder with SimOp type **Permute** and attrs `perm`. Same mathematical role as `ttnn.permute(tensor, dims)`, but the generic `ttnn.permute` in `op.py` still records **Transpose** for other call sites.

## Classes

### TensorProxy

Duck-compatible proxy for ttnn.Tensor that tracks state:

```python
tensor = TensorProxy(
    shape=[8, 224, 768],
    dtype=DataType.BFLOAT16,
    layout=Layout.TILE_LAYOUT,
    memory_config=DRAM_MEMORY_CONFIG,
    device="device_0"
)
```

**Attributes:**
- `shape` / `logical_shape()`: Logical shape of tensor
- `padded_shape()`: Padded shape (for tile layout)
- `dtype()`: Data type
- `layout()`: Layout (ROW_MAJOR_LAYOUT or TILE_LAYOUT)
- `memory_config()`: Memory configuration
- `device()`: Device handle
- `is_sharded()`: Whether tensor is sharded

### OperationTracker

Tracks all operations for simulation:

```python
from ttnn_shims import get_tracker, reset_tracker

# Reset tracker
reset_tracker()

# ... perform operations ...

# Get summary
tracker = get_tracker()
summary = tracker.get_summary()
print(f"Tilize count: {summary['tilize_count']}")
print(f"Untilize count: {summary['untilize_count']}")
```

**Tracked Operations:**
- `tilize_count`: Number of tilize operations
- `untilize_count`: Number of untilize operations
- `tilize_with_val_padding_count`: Number of tilize_with_val_padding operations
- `untilize_with_unpadding_count`: Number of untilize_with_unpadding operations
- `to_layout_count`: Number of to_layout operations
- `reshape_count`: Number of reshape operations
- `pad_count`: Number of pad operations
- `memory_operations`: List of all operations with details
- `memory_reads` / `memory_writes` / `data_movements`: Detailed per-op data movement tracking (includes shard and NLP head ops)

## Constants

- `TILE_WIDTH = 32`
- `TILE_HEIGHT = 32`
- `TILE_HW = 1024`
- `Layout.ROW_MAJOR_LAYOUT` / `Layout.TILE_LAYOUT`
- `DataType.BFLOAT16`, `DataType.FLOAT32`, etc.
- `DRAM_MEMORY_CONFIG`, `L1_MEMORY_CONFIG`

## Execution Modes

The module supports three execution modes:

1. **TRACK_ONLY** (default): Only track operations, don't execute
2. **EXECUTE**: Actually perform operations, don't track
3. **EXECUTE_AND_TRACK**: Both execute and track operations

### Setting Execution Mode

```python
from ttnn_shims import set_execution_mode, ExecutionMode

# Set to track only (default, backward compatible)
set_execution_mode(ExecutionMode.TRACK_ONLY)

# Set to execute operations
set_execution_mode(ExecutionMode.EXECUTE)

# Set to both execute and track
set_execution_mode(ExecutionMode.EXECUTE_AND_TRACK)
```

## Usage Examples

### Example 1: Track Only Mode (Default)

```python
from ttnn_shims import (
    TensorProxy, Layout, DataType, DRAM_MEMORY_CONFIG,
    to_layout, get_tracker, reset_tracker, ExecutionMode, set_execution_mode
)

# Reset tracker and set mode
reset_tracker()
set_execution_mode(ExecutionMode.TRACK_ONLY)  # Default

# Create tensor (no data stored)
tensor = TensorProxy(
    shape=[8, 224, 768],
    dtype=DataType.BFLOAT16,
    layout=Layout.ROW_MAJOR_LAYOUT,
    device="device_0"
)

# Convert to tile layout (only tracked, not executed)
tiled = to_layout(tensor, Layout.TILE_LAYOUT)

# Get operation summary
tracker = get_tracker()
summary = tracker.get_summary()
print(f"Tilize count: {summary['tilize_count']}")
print(f"Total compute ops: {summary['total_compute_operations']}")
```

### Example 2: Execute Mode (Test Logic)

```python
from ttnn_shims import (
    TensorProxy, Layout, DataType,
    tilize, untilize, set_execution_mode, ExecutionMode
)

# Set to execute mode
set_execution_mode(ExecutionMode.EXECUTE)

# Create tensor with actual data
test_data = [float(i) for i in range(32 * 32)]
tensor = TensorProxy(
    shape=[1, 1, 32, 32],
    dtype=DataType.BFLOAT16,
    layout=Layout.ROW_MAJOR_LAYOUT,
    device="device_0",
    data=test_data
)

# Actually perform tilize
tiled = tilize(tensor)
print(f"Tiled data available: {tiled.has_data()}")
print(f"First value: {tiled.get_data()[0]}")

# Actually perform untilize
row_major = untilize(tiled)
print(f"Recovered data: {row_major.get_data()[:10]}")
```

### Example 3: Execute and Track Mode

```python
from ttnn_shims import (
    TensorProxy, Layout, DataType,
    to_layout, set_execution_mode, ExecutionMode,
    get_tracker
)

# Set to both execute and track
set_execution_mode(ExecutionMode.EXECUTE_AND_TRACK)

# Create tensor with data
tensor = TensorProxy(
    shape=[8, 224, 768],
    dtype=DataType.BFLOAT16,
    layout=Layout.ROW_MAJOR_LAYOUT,
    device="device_0",
    data=[float(i) for i in range(8 * 224 * 768)]
)

# Perform operation (both executed and tracked)
tiled = to_layout(tensor, Layout.TILE_LAYOUT)

# Access actual data
if tiled.has_data():
    print(f"Data length: {len(tiled.get_data())}")

# Access tracking information
tracker = get_tracker()
summary = tracker.get_summary()
print(f"Operations tracked: {summary['tilize_count']}")
```

## Validation

All functions include the same validation as the original ttnn functions:

- **tilize**: Validates device tensor, row-major layout, valid data types, tile alignment
- **untilize**: Validates device tensor, tile layout, tile alignment, pack untilize requirements
- **to_layout**: Validates layout compatibility, dtype restrictions, padding requirements

## Error Handling

Functions raise the same exceptions as the original ttnn functions:
- `RuntimeError` for validation failures
- `TypeError` for invalid argument types
- Warnings for ignored parameters (matching ttnn behavior)

## Integration

To use as a drop-in replacement:

```python
# Instead of: import ttnn
import ttnn_shims as ttnn

# Set execution mode (default is TRACK_ONLY)
ttnn.set_execution_mode(ttnn.ExecutionMode.TRACK_ONLY)

# All operations work the same way
tensor = ttnn.TensorProxy(...)
result = ttnn.to_layout(tensor, ttnn.Layout.TILE_LAYOUT)
```

## Data Storage

When in `EXECUTE` or `EXECUTE_AND_TRACK` mode, tensors can store actual data:

```python
# Create tensor with data
tensor = TensorProxy(
    shape=[8, 224, 768],
    dtype=DataType.BFLOAT16,
    layout=Layout.ROW_MAJOR_LAYOUT,
    data=[...]  # Actual data values
)

# Check if tensor has data
if tensor.has_data():
    data = tensor.get_data()
    print(f"Data: {data}")

# Set data on existing tensor
tensor.set_data([1.0, 2.0, 3.0, ...])
```

## Notes

- **Default behavior**: TRACK_ONLY mode (backward compatible)
- **Execution mode**: Can be changed at runtime without affecting function signatures
- **Data storage**: Only stores data when in EXECUTE or EXECUTE_AND_TRACK mode
- **Tensor state**: Shape, layout, dtype are always maintained accurately
- **Padding calculations**: Match ttnn behavior
- **Rank > 4**: Automatically handles squeezing/unsqueezing
- **Empty tensors**: Handled correctly in both modes
- **Sharded tensors**: Tracked but not validated in detail
- **Performance**: Pure Python execution is slower than device operations (acceptable for testing)
- **Precision**: Floating point operations may have slight differences from device operations
