# Module 02: Box3D NMS ✅

**Location**: `ttsim_modules/box3d_nms.py`
**Original**: `mmdet3d/core/post_processing/box3d_nms.py`

## Description
Pure-NumPy reference implementations of 3D Non-Maximum Suppression (NMS) post-processing utilities. Provides `circle_nms` (BEV distance-based suppression), `aligned_3d_nms` (IoU-based 3D NMS), and `box3d_multiclass_nms` (per-class NMS with score threshold and top-K cap). All CUDA dependencies from the original are replaced with CPU-only NumPy equivalents.

## Purpose
Post-processing step applied after the TransFusionHead to filter redundant 3D detections. Not part of the differentiable forward graph — these are pure functional utilities used at inference time for final box selection.

## Module Specifications
- **Input**: Detection arrays `[N, 3]` for circle_nms; `[N, 7]` boxes + scores for aligned/multiclass
- **Output**: Filtered detection indices or boxes/scores
- **Parameters**: 0 (stateless utility functions)

## Validation Methodology
The module is validated through six tests:
1. **circle_nms ordering**: Verifies highest-score box is always kept
2. **circle_nms threshold**: Confirms distant boxes are both kept (no false suppression)
3. **aligned_3d_nms IoU=1**: Identical boxes suppress all but one
4. **box3d_multiclass_nms shape**: Validates output tuple structure `(boxes, scores)`
5. **box3d_multiclass_nms score threshold**: Confirms only boxes above score threshold are kept
6. **box3d_multiclass_nms max_num**: Verifies hard cap on maximum number of kept boxes

## Validation Results

**Test File**: `Validation/test_box3d_nms.py`

```
================================================================================
TEST 1: circle_nms keeps highest-score box
================================================================================

✓ PASS  kept indices=[1]
--------------------------------------------------------------------------------

================================================================================
TEST 2: circle_nms keeps distant boxes
================================================================================

✓ PASS  kept=[0, 1]
--------------------------------------------------------------------------------

================================================================================
TEST 3: aligned_3d_nms suppresses duplicates
================================================================================

✓ PASS  kept=[0]
--------------------------------------------------------------------------------

================================================================================
TEST 4: box3d_multiclass_nms output structure
================================================================================

✓ PASS  out shape=(100, 7)
--------------------------------------------------------------------------------

================================================================================
TEST 5: box3d_multiclass_nms score threshold
================================================================================

✓ PASS  kept 11 boxes above score=0.6
--------------------------------------------------------------------------------

================================================================================
TEST 6: box3d_multiclass_nms max_num cap
================================================================================

✓ PASS  kept 10 <= max_num=10
--------------------------------------------------------------------------------

============================================================
  ✓ circle_nms_highest_score: PASS
  ✓ circle_nms_distant: PASS
  ✓ aligned_3d_nms_identical: PASS
  ✓ multiclass_structure: PASS
  ✓ multiclass_score_thresh: PASS
  ✓ multiclass_max_num: PASS

  6/6 tests passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | circle_nms keeps highest-score box | ✅ PASS |
| Test 2 | circle_nms keeps distant boxes | ✅ PASS |
| Test 3 | aligned_3d_nms suppresses duplicates | ✅ PASS |
| Test 4 | box3d_multiclass_nms output structure | ✅ PASS |
| Test 5 | box3d_multiclass_nms score threshold | ✅ PASS |
| Test 6 | box3d_multiclass_nms max_num cap | ✅ PASS |

**Status**: All 6/6 tests passed ✅
