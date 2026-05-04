# Module 11: FPN (Feature Pyramid Network) ✅

**Location**: `ttsim_modules/fpn.py`
**Original**: `mmdet3d/models/necks/fpn.py`

## Description
Feature Pyramid Network neck that builds a multi-scale feature hierarchy from backbone outputs. Applies 1×1 lateral convolutions to each backbone level, merges features top-down via upsampling and addition, then applies 3×3 FPN convolutions to produce the final pyramid feature maps at a uniform channel width.

## Purpose
Aggregates multi-scale image features from the CBSwinTransformer backbone into a unified multi-scale representation. Used as a sub-component within FPNC for camera image feature processing in the EALSS camera stream.

## Module Specifications
- **Input**: Multiple backbone feature maps `[B, in_channels[i], H_i, W_i]` (fine to coarse)
- **Output**: List of `num_outs` SimTensors at uniform `out_channels` width
- **Parameters** (default 4-level, no norm): 3,344,384
- **Parameters** (2-level, no norm): 393,728
- **Optional**: BatchNorm2d (adds 512 params per level), extra pool levels

## Validation Methodology
The module is validated through six tests:
1. **Construction**: Verifies default param count > 0 (got 3,344,384)
2. **4-level output shapes**: Validates all four output shapes are correct
3. **Param count formula**: Checks 2-level config against formula (expected 393,728)
4. **with_norm=True**: Confirms BN adds correct parameter delta (512)
5. **Extra output level**: Tests `num_outs > num_ins` adds max-pooled extra level
6. **start_level=1**: Verifies skipping first backbone level produces 3 outputs

## Validation Results

**Test File**: `Validation/test_fpn.py`

```
================================================================================
Test 1: Construction (default)
================================================================================

✓ FPN() params > 0  got 3,344,384
--------------------------------------------------------------------------------

================================================================================
Test 2: 4-level output shapes
================================================================================

✓ FPN out[0]  got [2, 256, 128, 128] expected [2, 256, 128, 128]
✓ FPN out[1]  got [2, 256, 64, 64]   expected [2, 256, 64, 64]
✓ FPN out[2]  got [2, 256, 32, 32]   expected [2, 256, 32, 32]
✓ FPN out[3]  got [2, 256, 16, 16]   expected [2, 256, 16, 16]
--------------------------------------------------------------------------------

================================================================================
Test 3: Param count (lateral + fpn convs, no norm)
================================================================================

✓ FPN param count (2 levels, no norm)  got 393,728 expected 393,728
--------------------------------------------------------------------------------

================================================================================
Test 4: with_norm=True adds BN params
================================================================================

✓ BN delta  got 512 expected 512
--------------------------------------------------------------------------------

================================================================================
Test 5: num_outs > num_ins adds pooled extra level
================================================================================

✓ num_outs extra level  got 3 outputs,
  shapes: [[2, 256, 64, 64], [2, 256, 32, 32], [2, 256, 16, 16]]
--------------------------------------------------------------------------------

================================================================================
Test 6: start_level=1 skips first backbone level
================================================================================

✓ start_level=1  got 3 outputs
--------------------------------------------------------------------------------

============================================================
  PASS  construction
  PASS  output_shapes_4level
  PASS  param_count_lateral_fpn
  PASS  with_norm
  PASS  extra_output_via_pool
  PASS  start_level

6/6 passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | Default construction (3,344,384 params) | ✅ PASS |
| Test 2 | 4-level output shapes `[256, H, W]` | ✅ PASS |
| Test 3 | Param count formula (2-level = 393,728) | ✅ PASS |
| Test 4 | with_norm=True BN delta = 512 | ✅ PASS |
| Test 5 | Extra output level via max pool | ✅ PASS |
| Test 6 | start_level=1 skips first level | ✅ PASS |

**Status**: All 6/6 tests passed ✅
