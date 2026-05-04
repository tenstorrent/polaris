# Module 23: SwinTransformer ✅

**Location**: `ttsim_modules/swin_transformer.py`
**Original**: `mmdet3d/models/backbones/swin_transformer.py`

## Description
Swin Transformer (Liu et al., ICCV 2021) image backbone implementation. Uses shifted window self-attention with hierarchical patch merging to produce multi-scale feature pyramids. Includes PatchEmbed, stacked SwinTransformerBlocks with relative position bias tables, PatchMerging, and per-stage LayerNorm outputs.

## Purpose
Core image backbone used in both SwinTransformer (single) and CBSwinTransformer (composite) configurations for the camera stream in EALSS. Produces four pyramid-level outputs at scales H/4, H/8, H/16, H/32.

## Module Specifications
- **Input**: `[B, 3, H, W]` camera images
- **Output**: List of feature maps at selected output indices
- **Parameters (Swin-T)**: 27,520,602 (embed_dim=96, depths=[2,2,6,2])
- **Key components**:
  - PatchEmbed: Conv2d(3, embed_dim, patch_size=4) + LayerNorm
  - SwinTransformerBlock × depth[i] per stage
  - Relative position bias table (learnable, per head per stage)
  - PatchMerging between stages (except last)

## Validation Methodology
The module is validated through seven tests:
1. **QKV linear step**: PyTorch vs NumPy (`x @ W.T + b`) — numerical comparison of window-attention projection
2. **Softmax attention weights**: PyTorch vs NumPy stable softmax — numerical comparison
3. **Construction**: Swin-T param count = 27,520,602
4. **Output shapes**: Four-stage outputs at [H/4, H/8, H/16, H/32] for 224×224 input
5. **Param count reference**: Verifies exact Swin-T reference parameter count
6. **Two-stage config**: `out_indices=(0,1)` produces exactly 2 outputs
7. **Selective out_indices**: `out_indices=(1,3)` produces stages 1 and 3 only

## Validation Results

**Test File**: `Validation/test_swin_transformer.py`

```
================================================================================
Test 1: Window-Attn QKV linear step — PyTorch vs NumPy
================================================================================
  PyTorch QKV: shape=[49, 288], sample=[ 0.10392864 -0.13121665  0.02862053  0.11493635]
  NumPy   QKV: shape=[49, 288], sample=[ 0.10392864 -0.13121665  0.02862053  0.11493635]
  ✓ QKV linear step: PASS  (max_diff=0.000e+00)

================================================================================
Test 2: Softmax attention weights — PyTorch vs NumPy
================================================================================
  PyTorch Softmax: shape=[4, 3, 49, 49], sample=[0.0105946  0.01127507 0.02154749 0.00914287]
  NumPy   Softmax: shape=[4, 3, 49, 49], sample=[0.0105946  0.01127507 0.02154749 0.00914287]
  ✓ attention softmax: PASS  (max_diff=1.490e-08)

================================================================================
Test 3: Construction (Swin-T)
================================================================================

✓ SwinT params > 0  got 27,520,602
--------------------------------------------------------------------------------

================================================================================
Test 4: Output shapes (Swin-T, 224×224 input)
================================================================================

✓ Stage 0 shape  got [2, 96, 56, 56] expected [2, 96, 56, 56]
--------------------------------------------------------------------------------

✓ Stage 1 shape  got [2, 192, 28, 28] expected [2, 192, 28, 28]
--------------------------------------------------------------------------------

✓ Stage 2 shape  got [2, 384, 14, 14] expected [2, 384, 14, 14]
--------------------------------------------------------------------------------

✓ Stage 3 shape  got [2, 768, 7, 7] expected [2, 768, 7, 7]
--------------------------------------------------------------------------------

================================================================================
Test 5: Param count == 27,520,602 (Swin-T reference)
================================================================================

✓ Swin-T param count  got 27,520,602 expected 27,520,602
--------------------------------------------------------------------------------

================================================================================
Test 6: Two-stage Swin (out_indices=(0,1))
================================================================================

✓ 2-stage produces 2 outputs  got 2
--------------------------------------------------------------------------------

================================================================================
Test 7: out_indices=(1, 3) — only stages 1 and 3
================================================================================

✓ out_indices=1 shape  got [2, 192, 28, 28] expected [2, 192, 28, 28]
--------------------------------------------------------------------------------

✓ out_indices=3 shape  got [2, 768, 7, 7] expected [2, 768, 7, 7]
--------------------------------------------------------------------------------

============================================================
  PASS  qkv_linear_step
  PASS  window_softmax_attn
  PASS  construction_swint
  PASS  output_shapes_swint
  PASS  param_count_swint
  PASS  two_stage_only
  PASS  different_out_indices

7/7 passed
```

## Stage Output Shapes (Swin-T, 224×224 input)

| Stage | Embed Dim | Spatial | Shape |
|-------|-----------|---------|-------|
| 0 | 96 | H/4 = 56 | `[2, 96, 56, 56]` |
| 1 | 192 | H/8 = 28 | `[2, 192, 28, 28]` |
| 2 | 384 | H/16 = 14 | `[2, 384, 14, 14]` |
| 3 | 768 | H/32 = 7 | `[2, 768, 7, 7]` |

**Status**: All 7/7 tests passed ✅
