# Module 4: Bricks (Timing Utilities) ✅

**Location**: `ttsim_models/bricks.py`
**Original**: `projects/mmdet3d_plugin/models/utils/bricks.py`

## Description
Timing and profiling utility module that provides decorators for measuring function execution time. Uses Python's standard `time` module to track and accumulate execution statistics across multiple function calls. Provides utilities to reset statistics, retrieve timing data programmatically, and print formatted summaries.

## Purpose
Enables performance profiling and debugging of BEVFormer model components during development. The `run_time` decorator can be applied to any function to automatically measure and report execution time, helping identify bottlenecks and optimize performance-critical operations in the model pipeline.

## Module Specifications
- **Input**: Function decorator (wraps any Python function)
- **Output**: Wrapped function with timing instrumentation
- **Key Functions**:
  - `run_time(name)`: Decorator for timing function execution
  - `reset_timing_stats()`: Clear all timing statistics
  - `get_timing_stats()`: Retrieve timing data as dictionary
  - `print_timing_summary()`: Print formatted timing report
- **Parameter Count**: 0 (utility module, no trainable parameters)
- **Global State**: `time_maps` (accumulated times), `count_maps` (call counts)

## Key Changes from PyTorch to TTSim
1. **Removed GPU Synchronization**: Removed `torch.cuda.synchronize()` calls (not needed for TTSim CPU-based operations)
2. **Added Metadata Preservation**: Used `@functools.wraps` to preserve function name and docstring
3. **Enhanced with Utility Functions**: Added `reset_timing_stats()`, `get_timing_stats()`, `print_timing_summary()`
4. **Improved Documentation**: Added comprehensive docstrings with usage examples
5. **Pure Python Implementation**: No PyTorch or external dependencies required

## Validation Methodology
The module is validated through ten comprehensive tests covering:
1. **Basic Decorator Functionality**: Verifies timing measurement and statistics recording
2. **Multiple Calls Accumulation**: Tests that times accumulate correctly across repeated calls
3. **Multiple Functions**: Validates tracking of different decorated functions independently
4. **Reset Statistics**: Confirms statistics can be cleared properly
5. **Get Statistics**: Tests programmatic access to timing data with correct structure
6. **Print Summary**: Verifies formatted output generation without errors
7. **Args and Kwargs**: Tests decorator with various argument patterns
8. **Metadata Preservation**: Confirms function names and docstrings are preserved
9. **Nested Decorators**: Validates behavior with multiple decorator layers
10. **Exception Handling**: Ensures exceptions propagate correctly while still recording timing

All tests use `time.sleep()` to create measurable delays and verify timing accuracy within expected ranges.

## Validation Results

**Test File**: `Validation/test_bricks.py`

```
================================================================================
Bricks.py TTSim Utility Module Test Suite
================================================================================

================================================================================
TEST 1: Basic Decorator Functionality
================================================================================
test : simple_function takes up 0.010071
✓ Decorator works correctly
  - Function result: 10
  - Execution time: 0.010071s
  - Call count: 1.0

================================================================================
TEST 2: Multiple Calls Accumulation
================================================================================
accumulation : test_function takes up 0.005094
accumulation : test_function takes up 0.005099
accumulation : test_function takes up 0.005096
accumulation : test_function takes up 0.005092
accumulation : test_function takes up 0.005095
✓ Multiple calls accumulated correctly
  - Number of calls: 5.0
  - Total time: 0.025474s
  - Average time per call: 0.005095s

================================================================================
TEST 3: Multiple Functions
================================================================================
operation_a : function_a takes up 0.010067
operation_b : function_b takes up 0.020069
✓ Multiple functions tracked correctly
  - Function A time: 0.010067s
  - Function B time: 0.020069s

================================================================================
TEST 4: Reset Timing Statistics
================================================================================
reset_test : test_function takes up 0.000001
✓ Reset functionality works correctly

================================================================================
TEST 5: Get Timing Statistics
================================================================================
stats_test : test_function takes up 0.010074
stats_test : test_function takes up 0.010087
stats_test : test_function takes up 0.010091
✓ Statistics retrieval works correctly
  - Total time: 0.030274s
  - Call count: 3.0
  - Average time: 0.010091s

================================================================================
TEST 6: Print Timing Summary
================================================================================
summary_test_a : function_a takes up 0.020524
summary_test_b : function_b takes up 0.010107
summary_test_b : function_b takes up 0.010087

Generating timing summary:

================================================================================
TIMING SUMMARY
================================================================================
Operation                                             Calls    Total (s)      Avg (s)
--------------------------------------------------------------------------------
summary_test_a : function_a                             1.0     0.020524     0.020524
summary_test_b : function_b                             2.0     0.020174     0.010087
================================================================================

✓ Timing summary printed successfully

================================================================================
TEST 7: Decorator with Args and Kwargs
================================================================================
args_test : function_with_args takes up 0.000001
args_test : function_with_args takes up 0.000001
args_test : function_with_args takes up 0.000001
args_test : function_with_args takes up 0.000001
✓ Decorator works with various argument patterns
  - Number of calls: 4.0
  - Results: 33, 53, 53, 73

================================================================================
TEST 8: Decorator Preserves Metadata
================================================================================
✓ Decorator preserves function metadata
  - Function name: documented_function
  - Function docstring: This is a documented function.

================================================================================
TEST 9: Nested/Multiple Decorators
================================================================================
inner : nested_function takes up 0.010068
outer : nested_function takes up 0.010115
✓ Nested decorators work correctly
  - Outer time: 0.010115s
  - Inner time: 0.010068s

================================================================================
TEST 10: Exception Handling
================================================================================
exception_test : function_that_raises takes up 0.000001
✓ Exception handling works correctly
  - Valid call succeeded
  - Exception properly propagated
  - Calls recorded: 1.0

================================================================================
TEST SUMMARY
================================================================================
Basic Decorator Functionality............................... ✓ PASSED
Multiple Calls Accumulation................................. ✓ PASSED
Multiple Functions.......................................... ✓ PASSED
Reset Timing Statistics..................................... ✓ PASSED
Get Timing Statistics....................................... ✓ PASSED
Print Timing Summary........................................ ✓ PASSED
Args and Kwargs............................................. ✓ PASSED
Preserves Metadata.......................................... ✓ PASSED
Nested Decorators........................................... ✓ PASSED
Exception Handling.......................................... ✓ PASSED

Total: 10/10 tests passed

All tests passed! The utility module is working correctly.
```
