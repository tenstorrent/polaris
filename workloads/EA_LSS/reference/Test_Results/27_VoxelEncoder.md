# Module 27: VoxelEncoder (HardSimpleVFE & HardSimpleVFE_ATT) ✅

**Location**: `ttsim_modules/voxel_encoder.py`
**Original**: `mmdet3d/models/voxel_encoders/voxel_encoder.py`

## Description
Voxel Feature Encoder (VFE) modules for LiDAR point cloud processing. `HardSimpleVFE` applies a simple mean over all points within a voxel (0 parameters). `HardSimpleVFE_ATT` applies temporal attention (via `VoxelFeature_TA` with PACALayer channel/point attention and VALayer voxel attention), followed by a `PFNLayer` that compresses each voxel to a fixed-size descriptor via Linear + BN1d + ReLU + ReduceMax.

## Purpose
First processing stage of the LiDAR stream in EALSS. Converts raw point cloud voxels `[N_vox, M_pts, 5]` into compact per-voxel feature descriptors `[N_vox, 32]` that feed into the SparseEncoder/SECOND backbone.

## Module Specifications
- **HardSimpleVFE**: `[N, M, F] → [N, F]` (mean pool, 0 params)
- **HardSimpleVFE_ATT**: `[N, M, 5] → [N, 32]` (attention + PFN)
  - Default: num_features=5, dim_ca=12, dim_pa=10, boost_c_dim=32
  - Parameters: 4,322
- **PFNLayer** (32→32): 1,088 params (Linear + BN1d)

## Validation Methodology
The module is validated through four tests:
1. **HardSimpleVFE params**: Verifies exactly 0 learnable parameters
2. **HardSimpleVFE_ATT param count**: Expected 4,322 params
3. **HardSimpleVFE_ATT output shape**: `[200, 32]` for `[200, 10, 5]` input
4. **PFNLayer params**: 1,088 for 32→32 config

## Validation Results

**Test File**: `Validation/test_voxel_encoder.py`

```
================================================================================
Test 1: HardSimpleVFE (0 params)
================================================================================

✓ HardSimpleVFE params == 0  got 0
✓ HardSimpleVFE shape [100,10,4]→[100,4]  got [100, 4]
--------------------------------------------------------------------------------

================================================================================
Test 2: HardSimpleVFE_ATT param count (expected 4322)
================================================================================

✓ HardSimpleVFE_ATT params == 4322, got 4322
--------------------------------------------------------------------------------

================================================================================
Test 3: HardSimpleVFE_ATT output shape
================================================================================

✓ HardSimpleVFE_ATT [200,10,5]→[200,32]  got [200, 32]
--------------------------------------------------------------------------------

================================================================================
Test 4: PFNLayer params (1088 for 32→32)
================================================================================

✓ PFNLayer(32→32) params == 1088  got 1088
--------------------------------------------------------------------------------

============================================================
Passed 4/4 test groups
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | HardSimpleVFE: 0 params, `[100,10,4]→[100,4]` | ✅ PASS |
| Test 2 | HardSimpleVFE_ATT: 4,322 params | ✅ PASS |
| Test 3 | HardSimpleVFE_ATT: `[200,10,5]→[200,32]` | ✅ PASS |
| Test 4 | PFNLayer(32→32): 1,088 params | ✅ PASS |

**Status**: All 4/4 tests passed ✅
