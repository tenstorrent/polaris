# Module 21: SECOND (Sparse Encoder 2D Backbone) ✅

**Location**: `ttsim_modules/second.py`
**Original**: `mmdet3d/models/backbones/second.py`

## Description
SECOND (Sparsely Embedded Convolutional Detection) 2D convolutional backbone for LiDAR BEV feature extraction. Consists of multiple `SECONDStage` blocks, each beginning with a strided Conv2d (downsampling) followed by repeated Conv2d blocks with BatchNorm2d and ReLU. All convolutions use kernel_size=3, no bias.

## Purpose
Processes the LiDAR BEV pseudo-image (output of SECONDFPN/SparseEncoder) through progressively strided convolutions to extract multi-scale features for the SECONDFPN neck. Produces three output feature maps at halved spatial resolutions.

## Module Specifications
- **Input**: LiDAR BEV feature map `[B, in_channels=128, H, W]`
- **Output**: List of 3 feature maps at progressive downsamples
- **Default config**: `out_channels=[64,128,256]`, `layer_nums=[3,5,5]`, `strides=[2,2,2]`
- **Parameters** (default): 4,207,616 (with EA-LSS config: in=64, out=[64,128,256])
- **Formula per stage**: `C_in×C_out×9 + 2×C_out + layer_num×(C_out²×9 + 2×C_out)`

## Validation Methodology
The module is validated through six tests:
1. **Construction**: Default params > 0 (got 4,724,224 for standard mmdet3d config)
2. **Output shapes (default)**: Three stages with correct spatial downsampling
3. **Output shapes (small config)**: Two-stage small config shape check
4. **Parameter count**: Manual formula verification for SECONDStage(32,64,ln=0)
5. **Number of outputs**: Confirms output count equals num_stages
6. **Arbitrary batch/spatial**: B=1 and B=4 with non-square spatial sizes

## Validation Results

**Test File**: `Validation/test_second.py`

```
================================================================================
Test 1: Construction (default config)
================================================================================

✓ SECOND default params > 0  got 4,724,224
--------------------------------------------------------------------------------

================================================================================
Test 2: Output Shapes (default config)
================================================================================

✓ Stage 0 shape  got [2, 128, 128, 128] expected [2, 128, 128, 128]
✓ Stage 1 shape  got [2, 128, 64, 64]   expected [2, 128, 64, 64]
✓ Stage 2 shape  got [2, 256, 32, 32]   expected [2, 256, 32, 32]
--------------------------------------------------------------------------------

================================================================================
Test 3: Output Shapes (small config)
================================================================================

✓ Small stage 0  got [1, 64, 64, 64] expected [1, 64, 64, 64]
✓ Small stage 1  got [1, 128, 32, 32] expected [1, 128, 32, 32]
--------------------------------------------------------------------------------

================================================================================
Test 4: Parameter Count Verification
================================================================================

✓ SECONDStage(32,64,ln=0)  got 18560, expected 18560
--------------------------------------------------------------------------------

================================================================================
Test 5: Number of output tensors == num_stages
================================================================================

✓ SECOND produces 3 outputs  got 3
--------------------------------------------------------------------------------

================================================================================
Test 6: Arbitrary batch and spatial sizes
================================================================================

✓ B=1  got [1, 64, 16, 16]
✓ B=4  got [4, 64, 16, 16]
--------------------------------------------------------------------------------

============================================================
  PASS  construction
  PASS  output_shapes_default
  PASS  output_shapes_small
  PASS  param_count_manual
  PASS  num_outputs
  PASS  arbitrary_batch

6/6 passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | Default construction params > 0 | ✅ PASS |
| Test 2 | Default output shapes (3 stages) | ✅ PASS |
| Test 3 | Small config output shapes | ✅ PASS |
| Test 4 | SECONDStage param formula | ✅ PASS |
| Test 5 | Produces 3 output tensors | ✅ PASS |
| Test 6 | Batch/spatial independence | ✅ PASS |

**Status**: All 6/6 tests passed ✅
