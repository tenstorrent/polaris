# Module 04: QuickCumsum (Voxel Pooling Helper) ✅

**Location**: `ttsim_modules/cam_stream_lss_quickcumsum.py`
**Original**: `mmdet3d/models/detectors/cam_stream_lss.py` (lines 86–130)

## Description
Implements the QuickCumsum voxel-pooling trick for efficient aggregation of point features into voxel features. Given a sorted list of point features (x) and their voxel indices (ranks), it sums all features belonging to the same voxel using a single cumulative-sum pass, then retains only boundary rows (one per unique voxel).

## Purpose
Used inside the LiftSplatShoot module to efficiently aggregate camera-lifted point features into discrete BEV voxels. Replaces a custom CUDA `torch.autograd.Function` with a pure-NumPy implementation that is numerically identical for the data compute path.

## Module Specifications
- **Input**: `x [N, C]` point features, `geom_feats [N, 4]` voxel indices, `ranks [N]` sorted rank array
- **Output**: `(x_voxel [V, C], geom_out [V, 4])` where V ≤ N is the number of unique voxels
- **Parameters**: 0 (stateless functional operation)
- **Note**: Output size V is data-dependent; shape inference returns worst-case V = N

## Validation Methodology
The module is validated through six tests:
1. **cumsum_trick_numpy shape**: Validates output shapes `[V, C]` and `[V, D]`
2. **cumsum_trick_numpy values**: Manual per-voxel sum check for correctness
3. **All-unique voxels**: V == N case — no aggregation, all points are separate
4. **All-same voxel**: V == 1 case — all points merge into one voxel
5. **QuickCumsum worst-case shape**: Validates output shape for maximum-size inputs
6. **QuickCumsum data**: Compares QuickCumsum against cumsum_trick_numpy reference

## Validation Results

**Test File**: `Validation/test_cam_stream_lss_quickcumsum.py`

```
================================================================================
TEST 1: cumsum_trick_numpy shape
================================================================================

✓ PASS  xv=(10, 8), gv=(10, 4)
--------------------------------------------------------------------------------

================================================================================
TEST 2: cumsum_trick_numpy values
================================================================================
  ✓ voxel 0 sum: PASS  (max_diff=0.000e+00)
  ✓ voxel 1 sum: PASS  (max_diff=0.000e+00)
  ✓ geom voxel 0: PASS  (max_diff=0.000e+00)
  ✓ geom voxel 1: PASS  (max_diff=0.000e+00)

✓ PASS  values match manual sum
--------------------------------------------------------------------------------

================================================================================
TEST 3: all-unique voxels (V == N)
================================================================================
  ✓ unique voxel features preserved: PASS  (max_diff=4.396e-07)

✓ PASS  shape=(20, 4)
--------------------------------------------------------------------------------

================================================================================
TEST 4: all-same voxel (V == 1)
================================================================================
  ✓ total sum: PASS  (max_diff=0.000e+00)
  ✓ geom last row: PASS  (max_diff=0.000e+00)

✓ PASS  shape=(1, 6)
--------------------------------------------------------------------------------

================================================================================
TEST 5: QuickCumsum worst-case shape
================================================================================

✓ PASS  x_out=[100, 64], g_out=[100, 4]
--------------------------------------------------------------------------------

================================================================================
TEST 6: QuickCumsum data vs cumsum_trick_numpy
================================================================================
  ✓ x_voxel vs ref: PASS  (max_diff=0.000e+00)
  ✓ geom_out vs ref: PASS  (max_diff=0.000e+00)

✓ PASS  actual shape x=(10, 8), g=(10, 4)
--------------------------------------------------------------------------------

============================================================
  ✓ cumsum_shape: PASS
  ✓ cumsum_values: PASS
  ✓ all_unique: PASS
  ✓ all_same: PASS
  ✓ quickcumsum_shape: PASS
  ✓ quickcumsum_data: PASS

  6/6 tests passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | cumsum_trick_numpy output shape | ✅ PASS |
| Test 2 | cumsum_trick_numpy values vs manual | ✅ PASS |
| Test 3 | All-unique voxels (V == N) | ✅ PASS |
| Test 4 | All-same voxel (V == 1) | ✅ PASS |
| Test 5 | QuickCumsum worst-case shape | ✅ PASS |
| Test 6 | QuickCumsum data vs reference | ✅ PASS |

**Status**: All 6/6 tests passed ✅
