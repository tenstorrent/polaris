# Module 13: Gaussian Heatmap Utilities ✅

**Location**: `ttsim_modules/gaussian.py`
**Original**: `mmdet3d/core/utils/gaussian.py`

## Description
Provides Gaussian heatmap generation utilities for label assignment and depth supervision. Includes `gaussian_2d` (2D Gaussian kernel array), `draw_heatmap_gaussian` (draw Gaussian spot on a heatmap), `gaussian_radius` (compute minimum Gaussian radius for a given detection size), and `GaussianDepthTarget` (TTSim graph module for depth-supervision target generation via Normal CDF approach).

## Purpose
Used during training for heatmap-based label generation in TransFusionHead. `GaussianDepthTarget` generates per-pixel depth supervision targets using Gaussian distributions centered at annotated depth values.

## Module Specifications
- **gaussian_2d**: `(shape, sigma=1.0)` → `np.ndarray [H, W]`
- **draw_heatmap_gaussian**: `(heatmap, center, radius)` → `np.ndarray` with Gaussian spot
- **gaussian_radius**: `(det_size, min_overlap=0.5)` → `float` radius
- **GaussianDepthTarget**: `[B, D, H, W]` depth target tensor
- **Parameters**: 0 (all stateless utilities)

## Validation Methodology
The module is validated through six tests:
1. **gaussian_2d values**: Center pixel = 1.0, corner pixel ≈ 1.2e-4
2. **gaussian_radius**: Correct radius for given detection size
3. **draw_heatmap_gaussian shape**: Output shape matches input heatmap
4. **Center pixel is max**: Maximum value is always at the Gaussian center
5. **GaussianDepthTarget shape**: Validates `[B, D, H, W]` output shape
6. **GaussianDepthTarget data**: Compares TTSim output against compute_numpy reference

## Validation Results

**Test File**: `Validation/test_gaussian.py`

```
================================================================================
TEST 1: gaussian_2d values
================================================================================

✓ PASS  center=1.0000, corner=0.00012
--------------------------------------------------------------------------------

================================================================================
TEST 2: gaussian_radius
================================================================================

✓ PASS  radius=20.00
--------------------------------------------------------------------------------

================================================================================
TEST 3: draw_heatmap_gaussian shape
================================================================================

✓ PASS  shape=(40, 80)
--------------------------------------------------------------------------------

================================================================================
TEST 4: draw_heatmap center pixel is max
================================================================================

✓ PASS  max at center (30,18)=1.0000
--------------------------------------------------------------------------------

================================================================================
TEST 5: GaussianDepthTarget shape
================================================================================

✓ PASS  shape=[2, 16, 44, 113]
--------------------------------------------------------------------------------

================================================================================
TEST 6: GaussianDepthTarget data vs compute_numpy
================================================================================

✓ PASS  shape=(1, 16, 16, 9)
--------------------------------------------------------------------------------

============================================================
  ✓ gaussian_2d_values: PASS
  ✓ gaussian_radius: PASS
  ✓ draw_heatmap_shape: PASS
  ✓ draw_heatmap_center_is_max: PASS
  ✓ gdt_shape: PASS
  ✓ gdt_data: PASS

  6/6 tests passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | gaussian_2d center=1.0, corner≈1.2e-4 | ✅ PASS |
| Test 2 | gaussian_radius = 20.0 | ✅ PASS |
| Test 3 | draw_heatmap shape preserved | ✅ PASS |
| Test 4 | Center pixel is always maximum | ✅ PASS |
| Test 5 | GaussianDepthTarget shape `[2,16,44,113]` | ✅ PASS |
| Test 6 | GaussianDepthTarget data accuracy | ✅ PASS |

**Status**: All 6/6 tests passed ✅
