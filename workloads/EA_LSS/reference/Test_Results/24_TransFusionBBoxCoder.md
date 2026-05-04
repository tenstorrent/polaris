# Module 24: TransFusionBBoxCoder ✅

**Location**: `ttsim_modules/transfusion_bbox_coder.py`
**Original**: `mmdet3d/core/bbox/coders/transfusion_bbox_coder.py`

## Description
Bounding box encoder/decoder for the TransFusion detection head. The `encode` path (training) is a pure-NumPy implementation that converts ground-truth 3D boxes to regression targets. The `decode` path (inference) is the primary TTSim graph implementation that converts heatmap + regression head outputs back to 3D boxes using `Exp` (for dimensions), `atan2` approximation (for rotation), and center coordinate rescaling.

## Purpose
Post-processes TransFusionHead outputs into real-world 3D bounding box coordinates. Decodes per-proposal predictions to `(cx, cy, cz, w, l, h, yaw)` format for final NuScenes evaluation.

## Module Specifications
- **encode**: GT boxes `[N, 8]` → regression targets `[N, 8]` (pure NumPy)
- **decode output**: `boxes [B, P, 7 or 9]`, `scores [B, P]`
- **Dimension decoding**: `exp(log_w)`, `exp(log_l)`, `exp(log_h)`
- **Rotation decoding**: `atan2(sin, cos)` via numpy.arctan2 (custom TTSim node)
- **Velocity**: Optional 2D velocity appended when `with_velocity=True`
- **Parameters**: 0 (stateless coder utility)

## Validation Methodology
The module is validated through six tests:
1. **encode_numpy shape**: Output `[20, 8]` for 20 input boxes
2. **encode_numpy values**: All 8 encoded columns match manual reference
3. **decode shape**: Output boxes `[2, 200, 7]` and scores `[2, 200]`
4. **decode dim = exp(log_dim)**: Dimension columns use exp decoding
5. **decode center real-world coords**: Center coordinate rescaling is correct
6. **decode with velocity**: With `with_velocity=True`, boxes shape is `[1, 50, 9]`

## Validation Results

**Test File**: `Validation/test_transfusion_bbox_coder.py`

```
================================================================================
TEST 1: encode_numpy shape
================================================================================

✓ PASS  shape=(20, 8)
--------------------------------------------------------------------------------

================================================================================
TEST 2: encode_numpy values vs manual
================================================================================
  ✓ cx: PASS  (max_diff=0.000e+00)
  ✓ cy: PASS  (max_diff=0.000e+00)
  ✓ cz: PASS  (max_diff=0.000e+00)
  ✓ log_w: PASS  (max_diff=0.000e+00)
  ✓ log_l: PASS  (max_diff=0.000e+00)
  ✓ log_h: PASS  (max_diff=0.000e+00)
  ✓ sin: PASS  (max_diff=0.000e+00)
  ✓ cos: PASS  (max_diff=0.000e+00)

✓ PASS  all encode columns match
--------------------------------------------------------------------------------

================================================================================
TEST 3: decode shape
================================================================================

✓ PASS  boxes=[2, 200, 7], scores=[2, 200]
--------------------------------------------------------------------------------

================================================================================
TEST 4: decode dim = exp(log_dim)
================================================================================

✓ PASS  dim exp columns match
--------------------------------------------------------------------------------

================================================================================
TEST 5: decode center real-world coords
================================================================================

✓ PASS  center coord rescaling correct
--------------------------------------------------------------------------------

================================================================================
TEST 6: decode with velocity → boxes shape [B,P,9]
================================================================================

✓ PASS  boxes=[1, 50, 9], scores=[1, 50]
--------------------------------------------------------------------------------

============================================================
  ✓ encode_shape: PASS
  ✓ encode_values: PASS
  ✓ decode_shape: PASS
  ✓ decode_dim_exp: PASS
  ✓ decode_center_coord: PASS
  ✓ decode_with_vel: PASS

  6/6 tests passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | encode_numpy shape `[20, 8]` | ✅ PASS |
| Test 2 | All 8 encode columns match manual | ✅ PASS |
| Test 3 | decode boxes `[2,200,7]`, scores `[2,200]` | ✅ PASS |
| Test 4 | Dimension decoding via exp | ✅ PASS |
| Test 5 | Center coordinate rescaling | ✅ PASS |
| Test 6 | Velocity output shape `[1,50,9]` | ✅ PASS |

**Status**: All 6/6 tests passed ✅
