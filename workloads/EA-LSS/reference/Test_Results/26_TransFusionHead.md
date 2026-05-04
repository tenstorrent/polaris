# Module 26: TransFusionHead ✅

**Location**: `ttsim_modules/transfusion_head.py`
**Original**: `mmdet3d/models/dense_heads/transfusion_head.py`

## Description
Full TransFusion detection head combining transformer-based query decoding with per-attribute prediction heads. Applies a shared convolution to BEV features, generates initial query proposals from heatmap top-k, then refines proposals through one or more TransformerDecoderLayer blocks. Also defines `TransformerDecoderLayer` — the transformer decoder block with optional self-attention.

## Purpose
Primary detection head in the EALSS architecture. Takes fused BEV features and produces per-object predictions for nuScenes 3D object detection (10 classes, 200 proposals, 6 prediction attributes).

## Module Specifications
- **Input**: BEV feature tensor `[B, in_channels, H, W]`
- **Output**: Dict `{center [B,2,P], height [B,1,P], dim [B,3,P], rot [B,2,P], vel [B,2,P], heatmap [B,num_cls,P]}`
- **Parameters** (EA-LSS config): 1,675,934
  - `shared_conv`: Conv2d(in_ch=1024, hidden=128, 3) + BN
  - `heatmap_head`: ConvModule2d + Conv2d(hidden, num_classes=10, 3)
  - `class_encoding`: Conv1d(num_classes, hidden, 1)
  - `decoder`: TransformerDecoderLayer × num_decoder_layers
  - `prediction_heads`: FFN × num_decoder_layers
- **TransformerDecoderLayer** (default, cross_only=False): 233,088 params
- **TransformerDecoderLayer** (cross_only=True): 166,784 params

## Validation Methodology
The module is validated through seven tests:
1. **FFN linear step**: PyTorch vs NumPy (`x @ W.T + b`) — numerical comparison
2. **LayerNorm step**: PyTorch vs `ttsim_layernorm` — numerical comparison
3. **TDL self-attention**: PyTorch `nn.MultiheadAttention` vs TTSim `MultiheadAttention` — numerical comparison
4. **TransformerDecoderLayer param count**: Expected 233,088 for default config
5. **TransformerDecoderLayer output shape**: `[1, 128, 200]` for query features
6. **cross_only=True variant**: 166,784 params (omits self-attention + norm1 layers)
7. **TransFusionHead full forward**: Correct output dict keys and params > 0

Note: TDL uses LayerNorm which does not propagate TTSim data; tests 1–3 validate
the component ops independently using PyTorch/NumPy references.

## Validation Results

**Test File**: `Validation/test_transfusion_head.py`

```
================================================================================
Test 1: FFN Linear step — PyTorch vs NumPy
================================================================================
  PyTorch Linear: shape=[4, 32], sample=[ 0.16357192  0.3044731   0.39744833 -0.43703523]
  NumPy   Linear: shape=[4, 32], sample=[ 0.16357192  0.3044731   0.39744833 -0.43703523]
  ✓ FFN linear step: PASS  (max_diff=0.000e+00)

================================================================================
Test 2: LayerNorm step — PyTorch vs ttsim_layernorm
================================================================================
  PyTorch LN: shape=[4, 16], sample=[-0.5819425   0.60691947  0.00538727  1.4556924 ]
  TTSim   LN: shape=[4, 16], sample=[-0.5819425   0.60691947  0.00538727  1.4556924 ]
  ✓ LayerNorm step: PASS  (max_diff=2.384e-07)

================================================================================
Test 3: TDL Self-attention — PyTorch vs TTSim MHA
================================================================================
  PyTorch MHA: shape=[2, 5, 16], sample=[0.05897173 0.00154103 0.08521064 0.0935863 ]
  TTSim   MHA: shape=[2, 5, 16], sample=[0.05897173 0.00154103 0.08521065 0.09358631]
  ✓ TDL self-attn: PASS  (max_diff=1.018e-08)

================================================================================
Test 4: TransformerDecoderLayer param count (expected 233088)
================================================================================

✓ TDL params == 233088  got 233,088
--------------------------------------------------------------------------------

================================================================================
Test 5: TransformerDecoderLayer output shape
================================================================================

✓ TDL output shape [1,128,200]  got [1, 128, 200]
--------------------------------------------------------------------------------

================================================================================
Test 6: TransformerDecoderLayer cross_only=True param count
================================================================================

✓ TDL cross_only params  got 166,784  expected 166,784
--------------------------------------------------------------------------------

================================================================================
Test 7: TransFusionHead (EA-LSS config)
================================================================================

✓ TransFusionHead output keys  keys={'center', 'vel', 'heatmap', 'rot', 'height', 'dim'}
--------------------------------------------------------------------------------

✓ TransFusionHead params > 0  params=1,675,934
--------------------------------------------------------------------------------

============================================================
  PASS  linear_step
  PASS  layernorm_step
  PASS  mha_in_tdl
  PASS  tdl_params
  PASS  tdl_shape
  PASS  tdl_cross_only
  PASS  tfh_ealss_config

7/7 passed
```

## Parameter Breakdown

| Sub-module | Params |
|-----------|--------|
| TransformerDecoderLayer (cross_only=False) | 233,088 |
| TransformerDecoderLayer (cross_only=True) | 166,784 |
| TransFusionHead (EA-LSS config) | 1,675,934 |

**Status**: All 7/7 tests passed ✅
