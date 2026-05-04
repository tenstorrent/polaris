# Module 14: MLP (Multi-Layer Perceptron) ✅

**Location**: `ttsim_modules/mlp.py`
**Original**: `mmdet3d/models/utils/mlp.py`

## Description
Multi-Layer Perceptron applying stacked 1D point-wise convolutions with Batch Normalization and ReLU activation. Equivalent to per-point linear projection on feature sequences of shape `(B, C, N)`. Also defines the foundational `BatchNorm1d` and `ConvModule1d` building blocks used throughout the EA-LSS TTSim implementation.

## Purpose
General-purpose feature transformation module used widely in the EA-LSS architecture. `BatchNorm1d` and `ConvModule1d` sub-modules from this file are imported and reused by FFN, PositionEmbeddingLearned, VoxelEncoder, and VoxelEncoderUtils.

## Module Specifications
- **Input**: `[B, in_channels, N]`
- **Output**: `[B, channels[-1], N]`
- **Default config**: in=18, channels=(256, 256), BN+ReLU
- **Parameters** (in=18, channels=(256,256)): 71,680
- **BatchNorm1d**: learnable `weight [C]` + `bias [C]` = `2C` params
- **ConvModule1d**: Conv1d(no bias) + BN1d + ReLU
- **MLP without BN**: `with_bn=False` omits BN layers

## Validation Methodology
The module is validated through ten tests:
1. **Construction**: Verifies sub-module structure (layers, BN, ReLU presence)
2. **Output shape**: Multiple configurations vs PyTorch reference shapes
3. **Conv1d step**: Numerical comparison vs `torch.nn.Conv1d` (max_diff ≤ 8.94e-08)
4. **BatchNorm1d step**: Numerical comparison vs `torch.nn.BatchNorm1d` eval mode
5. **ReLU step**: Numerical comparison vs `torch.relu` (exact match)
6. **ConvModule1d full**: End-to-end Conv+BN+ReLU vs PyTorch sequential
7. **MLP 2-layer full**: Full 2-layer MLP forward vs stacked PyTorch ConvModules
8. **MLP without BN**: Conv+ReLU only variant
9. **Parameter count**: Analytical count vs manual formula
10. **Identity-weight sanity**: Identity conv + unity BN → output ≈ relu(x)

## Validation Results

**Test File**: `Validation/test_mlp.py`

```
================================================================================
TEST 3: Step-by-step — Conv1d
================================================================================
  ✓ Conv1d step: PASS  (max_diff=8.941e-08)

================================================================================
TEST 4: Step-by-step — BatchNorm1d
================================================================================
  ✓ BatchNorm1d step: PASS  (max_diff=4.768e-07)

================================================================================
TEST 5: Step-by-step — ReLU
================================================================================
  ✓ ReLU step: PASS  (max_diff=0.000e+00)

================================================================================
TEST 6: Full ConvModule1d Forward (Conv + BN + ReLU)
================================================================================
  ✓ ConvModule1d full: PASS  (max_diff=2.980e-07)

================================================================================
TEST 7: Full MLP Forward (2-layer, default config)
================================================================================
  PyTorch shape : (2, 256, 128)
  TTSim   shape : (2, 256, 128)
  ✓ MLP 2-layer full: PASS  (max_diff=7.749e-07)

================================================================================
TEST 10: Identity-weight Sanity Check
================================================================================
  Input    (first row): [ 0.  8. 16. 24.]
  PyTorch  (first row): [ 0.       7.99996 15.99992 23.99988]
  TTSim    (first row): [ 0.       7.99996 15.99992 23.99988]
  ✓ identity-weight: PASS  (max_diff=1.907e-06)

================================================================================
SUMMARY
================================================================================
  ✓ PASS  Construction
  ✓ PASS  Output Shape
  ✓ PASS  Conv1d Step
  ✓ PASS  BatchNorm1d Step
  ✓ PASS  ReLU Step
  ✓ PASS  ConvModule1d Full
  ✓ PASS  MLP 2-layer Full
  ✓ PASS  MLP Without BN
  ✓ PASS  Parameter Count
  ✓ PASS  Identity Weights

  10/10 tests passed
```

## Numerical Accuracy

| Step | Max Absolute Difference | Tolerance |
|------|------------------------|-----------|
| Conv1d | 8.941e-08 | 1e-05 |
| BatchNorm1d | 4.768e-07 | 1e-05 |
| ReLU | 0.000e+00 | 1e-05 |
| ConvModule1d full | 2.980e-07 | 1e-05 |
| MLP 2-layer | 7.749e-07 | 1e-05 |
| Identity-weight | 1.907e-06 | 1e-05 |

**Status**: All 10/10 tests passed ✅
