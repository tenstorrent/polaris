# Module 3: BBox Utils ✅

**Location**: `ttsim_models/bbox_utils.py`
**Original**: `projects/mmdet3d_plugin/models/utils/util.py`

## Description
Utility functions for 3D bounding box normalization and denormalization in BEVFormer. Handles conversion between standard bounding box representation (center coordinates, dimensions, rotation) and normalized representation (log dimensions, sin/cos rotation encoding). Supports both 7D (without velocity) and 9D (with velocity) bounding box formats.

## Purpose
Provides essential transformations for 3D object detection in BEVFormer by converting bounding boxes to a normalized space that's easier for the network to learn. The log transformation on dimensions helps with scale invariance, while sin/cos encoding of rotation angles ensures continuity at angle boundaries (e.g., -π and π).

## Module Specifications
- **Input Formats**:
  - Standard: `[cx, cy, cz, w, l, h, rot]` or `[cx, cy, cz, w, l, h, rot, vx, vy]`
  - Normalized: `[cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot)]` or with velocity
- **Output**: Converted bounding boxes with same batch dimensions
- **Functions**:
  - `normalize_bbox()`: Standard → Normalized
  - `denormalize_bbox()`: Normalized → Standard
  - Simplified variants without reference boxes
- **Parameter Count**: 0 (utility functions, no trainable parameters)

## Implementation Notes
- Custom `atan2` implementation using TTSim operations with quadrant handling
- Handles edge cases like rot=π/2 where cos=0
- Epsilon-based safe division to avoid numerical instabilities
- Preserves center coordinates (cx, cy, cz) and velocity unchanged
- Enhanced TTSim operations: Added `Atan` and `Sign` operations with data propagation

## Validation Methodology
The module is validated through six comprehensive tests:
1. **Normalize Without Velocity**: Tests 7D bbox normalization with various rotation angles
2. **Normalize With Velocity**: Tests 9D bbox normalization preserving velocity components
3. **Denormalize Without Velocity**: Tests conversion from normalized back to standard format
4. **Round-Trip Consistency**: Validates normalize→denormalize recovers original values (tolerance: 1e-4)
5. **Batch Processing**: Tests multi-dimensional batching (2×2×7 tensor)
6. **Edge Cases**: Tests 11 different rotation angles including critical values (0, ±π/2, ±π, ±π/3, ±π/4, ±π/6)

All tests compare TTSim outputs against PyTorch reference implementation with numerical tolerance verification.

## Validation Results

**Test File**: `Validation/test_bbox_utils.py`

```
================================================================================
BBox Utils TTSim Module Test Suite
================================================================================

TEST 1: Normalize BBoxes (Without Velocity)
--------------------------------------------------------------------------------
Input shape: (3, 7)
Numerical accuracy:
  Max absolute difference: 0.0000000596
  Mean absolute difference: 0.0000000037
✓ Test PASSED (within tolerance 1e-05)

TEST 2: Normalize BBoxes (With Velocity)
--------------------------------------------------------------------------------
Input shape: (3, 9)
PyTorch output shape: (3, 10)
TTSim output shape: (3, 10)
Numerical accuracy:
  Max absolute difference: 0.0000000596
  Mean absolute difference: 0.0000000030
✓ Test PASSED (within tolerance 1e-05)

TEST 3: Denormalize BBoxes (Without Velocity)
--------------------------------------------------------------------------------
Input shape: (3, 8)
TTSim output shape: (3, 7)
Numerical accuracy:
  Max absolute difference: 0.0000002384
  Mean absolute difference: 0.0000000284
✓ Test PASSED (within tolerance 1e-05)

TEST 4: Round-Trip Consistency (Normalize -> Denormalize)
--------------------------------------------------------------------------------
Original bboxes shape: (4, 7)
Recovered shape: (4, 7)
Round-trip accuracy:
  Max absolute difference: 0.0000004768
  Mean absolute difference: 0.0000000213
Component-wise differences:
  Center (cx, cy, cz): 0.0000000000
  Dimensions (w, l, h): 0.0000004768
  Rotation: 0.0000000000
✓ Test PASSED (within tolerance 0.0001)

TEST 5: Batch Processing
--------------------------------------------------------------------------------
Input shape (batched): (2, 2, 7)
PyTorch output shape: (2, 2, 8)
TTSim output shape: (2, 2, 8)
Numerical accuracy:
  Max absolute difference: 0.0000001192
  Mean absolute difference: 0.0000000065
✓ Test PASSED (within tolerance 1e-05)

TEST 6: Edge Cases
--------------------------------------------------------------------------------
Testing 11 different rotation angles
Input shape: (11, 7)
Numerical accuracy:
  Max absolute difference: 0.0000000596
  Mean absolute difference: 0.0000000007
✓ Test PASSED (within tolerance 1e-05)

================================================================================
TEST SUMMARY
================================================================================
Normalize Without Velocity.................................. ✓ PASSED
Normalize With Velocity..................................... ✓ PASSED
Denormalize Without Velocity................................ ✓ PASSED
Round-Trip Consistency...................................... ✓ PASSED
Batch Processing............................................ ✓ PASSED
Edge Cases.................................................. ✓ PASSED

Total: 6/6 tests passed

All tests passed! The module is working correctly.
```

## PyTorch vs TTSim Comparison

| Test Case | Input Shape | PyTorch Output | TTSim Output | Max Diff | Match |
|-----------|-------------|----------------|--------------|----------|-------|
| Normalize (no vel) | (3, 7) | (3, 8) | (3, 8) | 5.96e-08 | ✅ |
| Normalize (with vel) | (3, 9) | (3, 10) | (3, 10) | 5.96e-08 | ✅ |
| Denormalize | (3, 8) | (3, 7) | (3, 7) | 2.38e-07 | ✅ |
| Round-trip | (4, 7) → (4, 8) → (4, 7) | (4, 7) | (4, 7) | 4.77e-07 | ✅ |
| Batch | (2, 2, 7) | (2, 2, 8) | (2, 2, 8) | 1.19e-07 | ✅ |
| Edge cases | (11, 7) | (11, 8) | (11, 8) | 5.96e-08 | ✅ |

**Status**: All validations passed, excellent numerical accuracy (< 5e-07) ✅

## TTSim Framework Enhancements
To support this conversion, the following enhancements were made to TTSim:

1. **Added Operations** (`ttsim/front/functional/op.py`):
   - `Atan = partial(UnaryOperator, optype='Atan')` - Arctangent operation
   - `Sign = partial(UnaryOperator, optype='Sign')` - Sign function
   - `Constant()` - Flexible constant tensor creation

2. **Data Propagation** (`ttsim/ops/desc/data_compute.py` and `helpers.py`):
   - `compute_atan()` - Implements np.arctan for data flow
   - `compute_sign()` - Implements np.sign for data flow
   - Added to `_unary_compute_funcs` dictionary for automatic data propagation

These operations are now available for use in other TTSim conversions.
