# Module 06: ClipSigmoid ✅

**Location**: `ttsim_modules/clip_sigmoid.py`
**Original**: `mmdet3d/models/utils/clip_sigmoid.py`

## Description
Clamped sigmoid function that applies sigmoid activation followed by value clamping to `[eps, 1-eps]`. This prevents numerical instability (e.g., `log(0)`) in subsequent focal-loss calculations. Implemented as two sequential TTSim graph ops: `Sigmoid → Clip`.

## Purpose
Used in the TransFusionHead heatmap prediction branch to produce numerically stable probability outputs. The clamping prevents extreme sigmoid outputs from causing infinite losses during training.

## Module Specifications
- **Input**: Any SimTensor of any shape `[...]`
- **Output**: Same shape as input, values clamped to `[eps, 1-eps]`
- **Parameters**: 0 (stateless functional operation — no learnable weights)
- **Default eps**: 1e-4

## Validation Methodology
The module is validated through nine tests:
1. **Construction**: Verifies TTSim graph builds correctly; input/output shapes match
2. **Output Shape**: Validates shape is preserved for multiple input dimensionalities
3. **Sigmoid Step**: Compares TTSim sigmoid against `torch.sigmoid`
4. **Clip Step**: Compares TTSim clip against `torch.clamp`
5. **Full Forward**: End-to-end comparison (eps=1e-4) against PyTorch reference
6. **Custom eps**: Validates correct behavior for eps values 0.001, 0.01, 1e-6
7. **Value Range**: Confirms all outputs are within `[eps, 1-eps]`
8. **Extreme Inputs**: Tests behavior at ±100 (boundary clamping)
9. **No Parameters**: Confirms the module has no trainable weights

## Validation Results

**Test File**: `Validation/test_clip_sigmoid.py`

```
================================================================================
TEST 1: Module Construction
================================================================================

✓ Building TTSim clip_sigmoid graph (shape-only, no data)
  ✓ Graph built successfully
  ✓ Input  shape : [2, 10, 8, 8]
  ✓ Output shape : [2, 10, 8, 8]
  ✓ Output tensor name : input_shape_only.clip.out

================================================================================
TEST 2: Output Shape Validation (various shapes)
================================================================================

✓ Shape preserved for each input shape; PyTorch shape used as ground truth
  ✓ Input [4] → TTSim [4]  PyTorch [4]
  ✓ Input [3, 10] → TTSim [3, 10]  PyTorch [3, 10]
  ✓ Input [2, 10, 8] → TTSim [2, 10, 8]  PyTorch [2, 10, 8]
  ✓ Input [2, 10, 8, 8] → TTSim [2, 10, 8, 8]  PyTorch [2, 10, 8, 8]
  ✓ Input [1, 80, 128, 128] → TTSim [1, 80, 128, 128]  PyTorch [1, 80, 128, 128]

================================================================================
TEST 5: Full Forward Pass (eps=1e-4)
================================================================================

✓ TTSim clip_sigmoid graph (.data) vs torch.clamp(torch.sigmoid(x), eps, 1-eps)
  PyTorch shape  : (2, 10, 8, 8)
  TTSim   shape  : (2, 10, 8, 8)
  ✓ full forward (eps=1e-4): PASS  (max_diff=1.192e-07)

================================================================================
TEST 8: Extreme Input Values (±100)
================================================================================

  Input   : [-100.  -50.    0.   50.  100.]
  PyTorch : [1.000e-04 1.000e-04 5.000e-01 9.999e-01 9.999e-01]
  TTSim   : [1.000e-04 1.000e-04 5.000e-01 9.999e-01 9.999e-01]
  ✓ Boundary clamping verified
  ✓ extreme inputs: PASS  (max_diff=0.000e+00)

================================================================================
SUMMARY
================================================================================
  ✓ PASS  Construction
  ✓ PASS  Output Shape
  ✓ PASS  Sigmoid Step
  ✓ PASS  Clip Step
  ✓ PASS  Full Forward
  ✓ PASS  Custom eps
  ✓ PASS  Value Range
  ✓ PASS  Extreme Inputs
  ✓ PASS  No Trainable Params

  9/9 tests passed
```

## Numerical Accuracy

| Test Case | Max Absolute Difference | Tolerance | Match |
|-----------|------------------------|-----------|-------|
| Sigmoid step | 5.960e-08 | 1e-05 | ✅ |
| Clip step | 0.000e+00 | 1e-05 | ✅ |
| Full forward (eps=1e-4) | 1.192e-07 | 1e-05 | ✅ |
| Custom eps=0.001 | 5.960e-08 | 1e-05 | ✅ |
| Extreme inputs ±100 | 0.000e+00 | 1e-05 | ✅ |

**Status**: All 9/9 tests passed ✅
