# TTNN Shims - Internal Operation Tracking

## Answer to Your Question

**Yes, the tilize and untilize operations now track internal operations including computations and memory movements.**

## What is Tracked

### For Each Tilize Operation:

1. **Memory Read Operations**
   - Bytes read from input buffer (row-major format)
   - Number of tiles read
   - Tiles per row and column
   - Operation type: `tilize_read`

2. **Memory Write Operations**
   - Bytes written to output buffer (tile layout format)
   - Number of tiles written
   - Tiles per row and column
   - Operation type: `tilize_write`

3. **Compute Operations**
   - Number of compute operations (one per tile)
   - Operation type: `datacopy_tilize`
   - Whether multicore is used
   - Total count of compute operations

4. **Circular Buffer Operations**
   - Input CB read operations (number of tiles)
   - Output CB write operations (number of tiles)
   - Total circular buffer operations

5. **Kernel Invocations**
   - Number of kernel invocations (estimated based on multicore usage)
   - Tiles per core
   - Whether multicore is enabled

6. **Data Movement**
   - Source: row_major_buffer
   - Destination: tile_layout_buffer
   - Total bytes moved
   - Number of tiles

### For Each Untilize Operation:

1. **Memory Read Operations**
   - Bytes read from input buffer (tile layout format)
   - Number of tiles read
   - Tiles per row and column
   - Operation type: `untilize_read`

2. **Memory Write Operations**
   - Bytes written to output buffer (row-major format)
   - Number of tiles written
   - Tiles per row and column
   - Operation type: `untilize_write`

3. **Compute Operations**
   - Number of compute operations (one per tile)
   - Operation type: `datacopy_untilize`
   - Whether pack_untilize is used
   - Whether multicore is used
   - Total count of compute operations

4. **Circular Buffer Operations**
   - Input CB read operations (number of tiles)
   - Output CB write operations (number of tiles)
   - Whether pack_untilize is used
   - Total circular buffer operations

5. **Kernel Invocations**
   - Number of kernel invocations (estimated based on multicore usage)
   - Tiles per core
   - Whether multicore is enabled
   - Whether pack_untilize is enabled

6. **Data Movement**
   - Source: tile_layout_buffer
   - Destination: row_major_buffer
   - Total bytes moved
   - Number of tiles

## Accessing Tracked Data

### Basic Summary

```python
from ttnn_shims import get_tracker

tracker = get_tracker()
summary = tracker.get_summary()

print(f"Total memory reads: {summary['memory_read_count']}")
print(f"Total memory writes: {summary['memory_write_count']}")
print(f"Total compute operations: {summary['total_compute_operations']}")
print(f"Total kernel invocations: {summary['total_kernel_invocations']}")
print(f"Total memory read bytes: {summary['total_memory_read_bytes']}")
print(f"Total memory write bytes: {summary['total_memory_write_bytes']}")
print(f"Total circular buffer ops: {summary['total_circular_buffer_ops']}")
```

### Detailed Summary

```python
detailed = tracker.get_detailed_summary()

# Access individual memory reads
for read_op in detailed['memory_reads']:
    print(f"Read {read_op['bytes']} bytes, {read_op['num_tiles']} tiles")

# Access individual compute operations
for compute_op in detailed['compute_operations']:
    print(f"Compute: {compute_op['type']}, count: {compute_op['count']}")

# Access kernel invocations
for kernel in detailed['kernel_invocations']:
    print(f"Kernel: {kernel['op']}, cores: {kernel['count']}")

# Access data movements
for movement in detailed['data_movements']:
    print(f"Move {movement['bytes']} bytes from {movement['from']} to {movement['to']}")
```

## Example Output

After running tilize operations, the tracker will contain:

```python
{
    'tilize_count': 4,
    'untilize_count': 4,
    'total_memory_read_bytes': 524288,  # Total bytes read across all operations
    'total_memory_write_bytes': 524288,  # Total bytes written across all operations
    'total_compute_operations': 784,     # Total compute operations (tiles processed)
    'total_kernel_invocations': 128,     # Estimated kernel invocations
    'total_circular_buffer_ops': 1568,   # Total CB operations (reads + writes)
    'memory_read_count': 4,              # Number of memory read operations
    'memory_write_count': 4,             # Number of memory write operations
    'compute_operation_count': 4,        # Number of compute operation groups
    'kernel_invocation_count': 4,        # Number of kernel invocation groups
    'circular_buffer_op_count': 4,       # Number of CB operation groups
    'data_movement_count': 4              # Number of data movement operations
}
```

## Calculation Details

### Memory Operations
- **Tile size**: TILE_WIDTH × TILE_HEIGHT × element_size
- **Total bytes**: num_tiles × tile_size_bytes
- **Read bytes**: Input buffer size (row-major or tile layout)
- **Write bytes**: Output buffer size (tile layout or row-major)

### Compute Operations
- **One compute operation per tile**: Each tile requires a datacopy operation
- **Type**: `datacopy_tilize` or `datacopy_untilize`
- **Count**: Equal to number of tiles processed

### Kernel Invocations
- **Single core**: 1 kernel invocation
- **Multicore**: Estimated as min(num_tiles, 128) cores
- **Tiles per core**: num_tiles / num_cores

### Circular Buffer Operations
- **Input CB ops**: Number of tiles read from input circular buffer
- **Output CB ops**: Number of tiles written to output circular buffer
- **Total**: Input CB ops + Output CB ops

## Notes

1. **Estimations**: Some values (like kernel invocations for multicore) are estimates based on typical behavior. Actual values may vary based on hardware configuration.

2. **Element Size**: Automatically calculated from tensor dtype:
   - BFLOAT16, UINT16: 2 bytes
   - FLOAT32, UINT32, INT32: 4 bytes

3. **Multicore Estimation**: The multicore kernel count is estimated. For precise counts, you would need actual hardware configuration details.

4. **Pack Untilize**: The `use_pack_untilize` flag affects the compute operation type and is tracked in the operation details.

5. **All Operations Tracked**: Every tilize/untilize operation creates entries in all tracking categories (memory, compute, CB, kernels, data movement).
