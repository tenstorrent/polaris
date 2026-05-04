# Module 05: CBSwinTransformer (Composite Backbone) ✅

**Location**: `ttsim_modules/cbnet.py`
**Original**: `mmdet3d/models/backbones/cbnet.py`

## Description
Composite Backbone Swin Transformer (CBNet) that couples two Swin Transformer backbones to share and exchange features across stages. The two sub-backbones process the same input but exchange intermediate feature maps via learned linear projections (`cb_linears`), enabling richer multi-scale representations.

## Purpose
Provides a powerful image backbone for the camera stream in EALSS. The composite backbone improves over a single SwinTransformer by allowing cross-backbone feature communication, producing four pyramid-level outputs at scales H/4, H/8, H/16, H/32.

## Module Specifications
- **Input**: Camera images `[B, 3, H, W]`
- **Output**: List of 4 feature maps at scales `[H/4, H/8, H/16, H/32]`
- **Parameters**: 55,682,580 (default config: embed_dim=96, depths=[2,2,6,2])
  - `cb_modules[0]` (full SwinT): ~27,520,602 params
  - `cb_modules[1]` (no patch_embed): ~27,515,802 params
  - `cb_linears` (cross-backbone fusion): 646,176 params

## Validation Methodology
The module is validated through five tests:
1. **Patch Embedding Conv2d step**: PyTorch vs `ttsim_conv2d` (stride-4 conv) — numerical comparison
2. **FFN linear step**: PyTorch vs NumPy (`x @ W.T + b`) — numerical comparison
3. **Construction**: Verifies total parameter count equals 55,682,580
4. **Output shapes**: Validates all 4 pyramid output shapes are correct
5. **Output count**: Confirms exactly 4 feature maps are produced

## Validation Results

**Test File**: `Validation/test_cbnet.py`

```
================================================================================
Test 1: Patch Embedding Conv2d step — PyTorch vs ttsim_conv2d
================================================================================
  PyTorch PatchEmbed: shape=[1, 96, 2, 2], sample=[-0.3881576   0.09866449  0.2987068  -0.5732643 ]
  TTSim   PatchEmbed: shape=[1, 96, 2, 2], sample=[-0.38815764  0.09866448  0.2987068  -0.5732643 ]
  ✓ patch embed conv step: PASS  (max_diff=1.788e-07)

================================================================================
Test 2: Swin FFN linear step — PyTorch vs NumPy
================================================================================
  PyTorch FFN:  shape=[49, 384], sample=[-0.22817484  0.3653477  -0.13201669  0.27110356]
  NumPy   FFN:  shape=[49, 384], sample=[-0.22817484  0.3653477  -0.13201669  0.27110356]
  ✓ Swin FFN linear step: PASS  (max_diff=0.000e+00)

================================================================================
Test 3: CBSwinTransformer construction
================================================================================

✓ CBSwinTransformer params == 55,682,580  got 55,682,580
--------------------------------------------------------------------------------

================================================================================
Test 4: CBSwinTransformer output shapes
================================================================================

✓ out[0] shape  got [1, 96, 32, 32] expected [1, 96, 32, 32]
--------------------------------------------------------------------------------

✓ out[1] shape  got [1, 192, 16, 16] expected [1, 192, 16, 16]
--------------------------------------------------------------------------------

✓ out[2] shape  got [1, 384, 8, 8] expected [1, 384, 8, 8]
--------------------------------------------------------------------------------

✓ out[3] shape  got [1, 768, 4, 4] expected [1, 768, 4, 4]
--------------------------------------------------------------------------------

================================================================================
Test 5: CBSwinTransformer produces 4 feature maps
================================================================================

✓ len(outs) == 4  got 4
--------------------------------------------------------------------------------

============================================================
  PASS  patch_embed_conv
  PASS  ffn_linear_step
  PASS  cbswin_construct
  PASS  cbswin_shapes
  PASS  cbswin_num_outs

5/5 passed
```

## PyTorch vs TTSim Comparison

| Test | Op | max_diff | Result |
|------|----|----------|--------|
| Test 1 | PatchEmbed Conv2d (stride=4) | 1.788e-07 | ✅ PASS |
| Test 2 | FFN Linear (x @ W.T + b) | 0.000e+00 | ✅ PASS |

| Stage | Output Shape | Channel Mult | Result |
|-------|-------------|--------------|--------|
| Stage 0 (H/4) | `[1, 96, 32, 32]` | embed_dim × 1 | ✅ |
| Stage 1 (H/8) | `[1, 192, 16, 16]` | embed_dim × 2 | ✅ |
| Stage 2 (H/16) | `[1, 384, 8, 8]` | embed_dim × 4 | ✅ |
| Stage 3 (H/32) | `[1, 768, 4, 4]` | embed_dim × 8 | ✅ |

**Status**: All 5/5 tests passed ✅
