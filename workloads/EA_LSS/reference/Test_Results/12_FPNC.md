# Module 12: FPNC (FPN with Context) ✅

**Location**: `ttsim_modules/fpnc.py`
**Original**: `mmdet3d/models/necks/fpnc.py`

## Description
Extended Feature Pyramid Network neck with global context aggregation. Combines an FPN backbone with per-channel global average pooling context (via the `gapcontext` sub-module), resizes all output levels to a target spatial size, concatenates them along the channel dimension, and applies a reduction convolution to produce a unified feature map.

## Purpose
Camera image neck in the EALSS pipeline. Unifies the multi-scale CBSwinTransformer outputs into a single feature map at uniform resolution `[B*N_views, imc, tH, tW]` for consumption by the LiftSplatShoot module.

## Module Specifications
- **Input**: Multiple FPN feature maps from backbone stages
- **Output**: Single unified feature map `[B, outC, tH, tW]`
- **Parameters** (EALSS default): 356,928+
- **gapcontext sub-module**:
  - `x → Conv2d(in,in,1) → AvgPool(1) → Resize → Add → Conv2d(in,out,1)`
  - Params (in=32, out=64, no norm): 3,168

## Validation Methodology
The module is validated through six tests:
1. **gapcontext Conv1×1 step**: PyTorch vs `ttsim_conv2d` (pointwise conv) — numerical comparison
2. **Global Average Pool step**: PyTorch `AdaptiveAvgPool2d(1)` vs NumPy mean — numerical comparison
3. **gapcontext construction**: Verifies param count for multiple configs
4. **gapcontext output shape**: Validates shape `[B, out_ch, H, W]` is preserved
5. **FPNC output shape**: Confirms single output list and parameter count
6. **FPNC sub-module params**: Verifies total ≥ expected reduction conv params

## Validation Results

**Test File**: `Validation/test_fpnc.py`

```
================================================================================
Test 1: gapcontext Conv1×1 step — PyTorch vs ttsim_conv2d
================================================================================
  PyTorch Conv1x1: shape=[2, 64, 8, 8], sample=[-0.14300954 -0.15352769 -0.04762696  0.17481512]
  TTSim   Conv1x1: shape=[2, 64, 8, 8], sample=[-0.14300953 -0.1535277  -0.04762697  0.17481512]
  ✓ gapcontext conv1x1: PASS  (max_diff=2.384e-07)

================================================================================
Test 2: Global Average Pool — PyTorch vs NumPy
================================================================================
  PyTorch GAP: shape=[2, 64, 1, 1], sample=[-0.01469079  0.02570008 -0.00275501  0.06808718]
  NumPy   GAP: shape=[2, 64, 1, 1], sample=[-0.01469078  0.02570008 -0.00275501  0.06808717]
  ✓ global avg pool: PASS  (max_diff=1.490e-08)

================================================================================
Test 3: gapcontext construction and param count
================================================================================

✓ gapcontext(32,64,norm=False): params=3168  p>0
--------------------------------------------------------------------------------

✓ gapcontext(64,64,norm=False): params=8320  p>0
--------------------------------------------------------------------------------

✓ gapcontext(128,64,norm=True): params=24960  p>0
--------------------------------------------------------------------------------

================================================================================
Test 4: gapcontext output shape
================================================================================

✓ B=1 C=32→64 64×64  out=[1, 64, 64, 64]
--------------------------------------------------------------------------------

✓ B=2 C=128→64 32×32  out=[2, 64, 32, 32]
--------------------------------------------------------------------------------

================================================================================
Test 5: FPNC output shape
================================================================================

✓ FPNC output is list[1]  len=1
--------------------------------------------------------------------------------

✓ FPNC param_count=356928  p>0
--------------------------------------------------------------------------------

================================================================================
Test 6: FPNC analytical_param_count includes expected sub-parts
================================================================================

✓ FPNC total≥reduc_conv(147520), got 356928
--------------------------------------------------------------------------------

============================================================
  PASS  gapcontext_conv1x1
  PASS  global_avg_pool
  PASS  gapcontext_construct
  PASS  gapcontext_shape
  PASS  fpnc_shape
  PASS  fpnc_param_formula

6/6 passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | Conv1×1: PyTorch vs ttsim_conv2d (max_diff=2.4e-7) | ✅ PASS |
| Test 2 | Global Avg Pool: PyTorch vs NumPy (max_diff=1.5e-8) | ✅ PASS |
| Test 3 | gapcontext param counts | ✅ PASS |
| Test 4 | gapcontext output shapes | ✅ PASS |
| Test 5 | FPNC output list & params | ✅ PASS |
| Test 6 | FPNC total ≥ reduc_conv | ✅ PASS |

**Status**: All 6/6 tests passed ✅
