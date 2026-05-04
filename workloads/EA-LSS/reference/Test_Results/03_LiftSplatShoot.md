# Module 03: LiftSplatShoot (Camera BEV Encoder) ✅

**Location**: `ttsim_modules/cam_stream_lss.py`
**Original**: `mmdet3d/models/detectors/cam_stream_lss.py`

## Description
Implements the Lift-Splat-Shoot (LSS) camera-to-BEV encoder. The module lifts per-camera image features into 3D space using predicted depth distributions, splats them onto a BEV grid, and encodes the resulting bird's-eye-view representation for downstream detection.

## Purpose
Converts multi-camera image features from perspective view into a unified Bird's-Eye-View (BEV) feature map. This is the key camera stream component that allows fusion with LiDAR BEV features in the EALSS architecture.

## Module Specifications
- **Input**: Camera features `[B*N_views, inputC, H, W]`
- **Output**: BEV feature map `[B, inputC, bH, bW]`
- **Parameters**: 7,357,108 (default config: inputC=256, camC=64, lss=False)
- **Key sub-modules**:
  - `dtransform`: depth LiDAR feature extract (4× Conv2d+BN+ReLU) — 161,504 params
  - `prenet`: fuse depth+image features (2× Conv2d+BN+ReLU) — 1,328,640 params
  - `depthnet`: depth prediction Conv2d(inputC, D=41, 1) — 10,537 params
  - `contextnet`: context feature Conv2d(inputC, camC=64, 1) — 16,448 params
  - `bevencode`: BEV encoder (4× Conv2d+BN+ReLU) — 4,279,040 params

## Validation Methodology
The module is validated through six tests:
1. **Depthwise Conv2d step**: PyTorch vs `ttsim_conv2d` (depthwise, groups=C) — numerical comparison
2. **Pointwise Conv2d step**: PyTorch vs `ttsim_conv2d` (1×1 channel-mix) — numerical comparison
3. **Param Count**: Verifies total parameter count equals 7,357,108
4. **Depth Bins**: Confirms D=41 discrete depth bins for range [4.0, 45.0, 1.0]
5. **BEV Channels**: Validates cz=128 for default spatial parameters
6. **Output Shape**: Checks forward pass output shape matches expected BEV dimensions

## Validation Results

**Test File**: `Validation/test_cam_stream_lss.py`

```
================================================================================
Test 1: Depthwise Conv2d step — PyTorch vs ttsim_conv2d
================================================================================
  PyTorch DW Conv: shape=[1, 32, 8, 8], sample=[-0.05556588 -0.3487402  -0.04866359  0.11391289]
  TTSim   DW Conv: shape=[1, 32, 8, 8], sample=[-0.05556586 -0.34874022 -0.04866358  0.11391288]
  ✓ depthwise conv step: PASS  (max_diff=1.192e-07)

================================================================================
Test 2: Pointwise Conv2d step — PyTorch vs ttsim_conv2d
================================================================================
  PyTorch PW Conv: shape=[1, 64, 8, 8], sample=[-0.34249395 -0.35891205 -0.04011273 -0.1639753 ]
  TTSim   PW Conv: shape=[1, 64, 8, 8], sample=[-0.34249395 -0.35891205 -0.04011275 -0.1639753 ]
  ✓ pointwise conv step: PASS  (max_diff=1.490e-07)

================================================================================
Test 3: LiftSplatShoot param count (expected 7,357,108)
================================================================================

✓ LiftSplatShoot params == 7,357,108  got 7,357,108
--------------------------------------------------------------------------------

================================================================================
Test 4: D = 41 for depth range [4, 45, 1]
================================================================================

✓ lss.D == 41  got 41
--------------------------------------------------------------------------------

================================================================================
Test 5: cz = 128 for default params
================================================================================

✓ lss.cz == 128  got 128
--------------------------------------------------------------------------------

================================================================================
Test 6: LiftSplatShoot output shape
================================================================================

✓ LiftSplatShoot output shape [6,256,32,88]  got [6, 256, 32, 88]
--------------------------------------------------------------------------------

============================================================
  PASS  depthwise_conv_step
  PASS  pointwise_conv_step
  PASS  lss_param_count
  PASS  lss_depth_bins
  PASS  lss_cz
  PASS  lss_output_shape

6/6 passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | Depthwise Conv2d: PyTorch vs ttsim_conv2d (max_diff=1.2e-7) | ✅ PASS |
| Test 2 | Pointwise Conv2d: PyTorch vs ttsim_conv2d (max_diff=1.5e-7) | ✅ PASS |
| Test 3 | Param count == 7,357,108 | ✅ PASS |
| Test 4 | D = 41 depth bins | ✅ PASS |
| Test 5 | cz = 128 BEV channels | ✅ PASS |
| Test 6 | Output shape `[6, 256, 32, 88]` | ✅ PASS |

**Status**: All 6/6 tests passed ✅
