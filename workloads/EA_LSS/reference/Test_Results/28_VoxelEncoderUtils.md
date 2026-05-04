# Module 28: VoxelEncoderUtils (VFE Building Blocks) ✅

**Location**: `ttsim_modules/voxel_encoder_utils.py`
**Original**: `mmdet3d/models/voxel_encoders/utils.py`

## Description
Building blocks for Voxel Feature Encoders. Provides `get_paddings_indicator` (pure-NumPy boolean mask for valid points in padded voxels), `VFELayer` (Linear + BN1d + ReLU with optional max pooling and cat_max concatenation), and `DynamicVFELayer` (same but without pooling, for dynamic voxelization). Reuses `BatchNorm1d` and `ConvModule1d` from `mlp.py`.

## Purpose
Core building blocks shared across voxel encoder implementations. `VFELayer` is used in PointPillar/VoxelNet-style encoders; `DynamicVFELayer` is used in dynamic voxelization pipelines. `get_paddings_indicator` creates point validity masks for padded voxel batches.

## Module Specifications
- **get_paddings_indicator**: `(actual_num [N], max_num) → bool [N, max_num]`
- **VFELayer**: `[N, M, in_ch] → [N, M, out_ch]` (max_out=False) or `[N, 1, out_ch]` (max_out=True)
- **DynamicVFELayer**: `[N, in_ch] → [N, out_ch]` (no pooling)
- **TTSim ops**: Conv(k=1) → BatchNorm → ReLU → ReduceMax → Tile → ConcatX

## Validation Methodology
The module is validated through five tests:
1. **get_paddings_indicator**: Shape `[4, 5]`, values match expected mask
2. **VFELayer output shapes**: Three configs — max_out=False, max_out=True, cat_max=True
3. **VFELayer data (max_out=False)**: Numerical comparison vs PyTorch reference
4. **DynamicVFELayer shapes**: Output `[500, 32]` for `[500, 16]` input
5. **DynamicVFELayer data**: Numerical comparison vs PyTorch reference

## Validation Results

**Test File**: `Validation/test_voxel_encoder_utils.py`

```
================================================================================
TEST 1: get_paddings_indicator
================================================================================

✓ PASS  shape=(4, 5) values correct
--------------------------------------------------------------------------------

================================================================================
TEST 2: VFELayer output shapes
================================================================================

✓ PASS  max_out=False cat_max=False → [10, 8, 16]
✓ PASS  max_out=True  cat_max=False → [10, 1, 16]
✓ PASS  max_out=True  cat_max=True  → [10, 8, np.int64(32)]
--------------------------------------------------------------------------------

================================================================================
TEST 3: VFELayer data (max_out=False)
================================================================================
  ✓ VFELayer ptw vs torch: PASS  (max_diff=4.470e-08)

================================================================================
TEST 4: DynamicVFELayer shapes
================================================================================

✓ PASS  shape=[500, 32]
--------------------------------------------------------------------------------

================================================================================
TEST 5: DynamicVFELayer data
================================================================================
  ✓ DynamicVFE vs torch: PASS  (max_diff=2.980e-08)

============================================================
  ✓ paddings_indicator: PASS
  ✓ vfe_shapes: PASS
  ✓ vfe_data_no_max: PASS
  ✓ dynamic_vfe_shapes: PASS
  ✓ dynamic_vfe_data: PASS

  5/5 tests passed
```

## Numerical Accuracy

| Test | Max Absolute Difference | Tolerance |
|------|------------------------|-----------|
| VFELayer data (max_out=False) | 4.470e-08 | 1e-05 |
| DynamicVFELayer data | 2.980e-08 | 1e-05 |

**Status**: All 5/5 tests passed ✅
