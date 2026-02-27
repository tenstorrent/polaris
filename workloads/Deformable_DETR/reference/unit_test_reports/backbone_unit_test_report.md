# Backbone Unit Test Report
**PyTorch vs TTSim Comparison** | **31/32 passed** | FAIL
Generated: 2026-02-20 14:50:42 | Exit Code: 1

---

## Summary

| Module | Passed | Total | Status |
|--------|--------|-------|--------|
| FrozenBatchNorm2d | 12 | 12 | PASS |
| ResNetBottleneck | 7 | 8 | FAIL |
| Backbone Shapes | 4 | 4 | PASS |
| Joiner Shapes | 4 | 4 | PASS |
| Positional Encoding Numerical | 4 | 4 | PASS |

**Total: 31/32 tests passed**

---

## Failed Tests

| Module | Test | Edge Case | Max Diff |
|--------|------|-----------|----------|
| ResNetBottleneck | layer1[0] large values | large | 5.00e-01 |

---

## FrozenBatchNorm2d (12/12 PASS)
*Frozen Batch Normalization layer - normalizes with fixed running statistics*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Basic 64ch 8x8 | positive | `[1, 64, 8, 8]` | `[1, 64, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Multi-batch 64ch | positive | `[2, 64, 8, 8]` | `[2, 64, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | 128 channels | positive | `[1, 128, 8, 8]` | `[1, 128, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Non-square 8x16 | positive | `[1, 64, 8, 16]` | `[1, 64, 8, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Negative values | negative | `[1, 64, 8, 8]` | `[1, 64, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Zero values | zeros | `[1, 64, 8, 8]` | `[1, 64, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Mixed positive/negative | mixed | `[1, 64, 8, 8]` | `[1, 64, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Very small values (1e-6) | small | `[1, 64, 8, 8]` | `[1, 64, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Very large values (1e6) | large | `[1, 64, 8, 8]` | `[1, 64, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Minimum spatial 1x1 | positive | `[1, 64, 1, 1]` | `[1, 64, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 10 | Single channel | positive | `[1, 1, 8, 8]` | `[1, 1, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 11 | Large spatial 32x32 | positive | `[1, 64, 32, 32]` | `[1, 64, 32, 32]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## ResNetBottleneck (7/8 FAIL)
*ResNet50 Bottleneck block with pretrained ImageNet weights*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | layer1[0] 64->256 stride=1 | positive | `[1, 64, 8, 8]` | `[1, 256, 8, 8]` | 2.15e-06 | 3.82e-08 | ✅ PASS |
| 1 | layer1[0] batch=2 | positive | `[2, 64, 8, 8]` | `[2, 256, 8, 8]` | 1.67e-06 | 3.94e-08 | ✅ PASS |
| 2 | layer1[0] negative input | negative | `[1, 64, 8, 8]` | `[1, 256, 8, 8]` | 2.38e-06 | 1.67e-07 | ✅ PASS |
| 3 | layer1[0] zero values | zeros | `[1, 64, 8, 8]` | `[1, 256, 8, 8]` | 4.77e-07 | 3.47e-08 | ✅ PASS |
| 4 | layer1[0] mixed input | mixed | `[1, 64, 8, 8]` | `[1, 256, 8, 8]` | 2.86e-06 | 9.86e-08 | ✅ PASS |
| 5 | layer1[0] small values | small | `[1, 64, 8, 8]` | `[1, 256, 8, 8]` | 5.96e-07 | 3.73e-08 | ✅ PASS |
| **6** | **layer1[0] large values** | large | `[1, 64, 8, 8]` | `[1, 256, 8, 8]` | 5.00e-01 | 1.25e-02 | 🔴 **FAIL** |
| 7 | layer1[0] minimum 4x4 | positive | `[1, 64, 4, 4]` | `[1, 256, 4, 4]` | 2.38e-06 | 4.05e-08 | ✅ PASS |

**Failed Cases:**
- [6] layer1[0] large values - large (diff: 5.00e-01)

---

## Backbone Shapes (4/4 PASS)
*ResNet50 backbone shape validation for all feature map layers*

| # | Test Case | Input | Layers | Result |
|:--|:----------|:------|:-------|:-------|
| 0 | ResNet50 32x32 batch=1 | `[1,3,32,32]` | 3 layers | ✅ PASS |
| 1 | ResNet50 64x64 batch=1 | `[1,3,64,64]` | 3 layers | ✅ PASS |
| 2 | ResNet50 32x32 batch=2 | `[2,3,32,32]` | 3 layers | ✅ PASS |
| 3 | ResNet50 16x16 minimum | `[1,3,16,16]` | 3 layers | ✅ PASS |

---

## Joiner Shapes (4/4 PASS)
*Joiner module shape validation for feature maps and position encodings*

| # | Test Case | Input | Features | Result |
|:--|:----------|:------|:---------|:-------|
| 0 | Joiner 32x32 batch=1 | `[1,3,32,32]` | 3 feats | ✅ PASS |
| 1 | Joiner 64x64 batch=1 | `[1,3,64,64]` | 3 feats | ✅ PASS |
| 2 | Joiner 32x32 batch=2 | `[2,3,32,32]` | 3 feats | ✅ PASS |
| 3 | Joiner 16x16 minimum | `[1,3,16,16]` | 3 feats | ✅ PASS |

---

## Positional Encoding Numerical (4/4 PASS)
*Numerical validation for sinusoidal position encodings*

| # | Test Case | Input | Pos Layers | Result |
|:--|:----------|:------|:-----------|:-------|
| 0 | Pos-enc 32x32 batch=1 | `[1,32,32]` | 3 layers | ✅ PASS |
| 1 | Pos-enc 64x64 batch=1 | `[1,64,64]` | 3 layers | ✅ PASS |
| 2 | Pos-enc 32x32 batch=2 | `[2,32,32]` | 3 layers | ✅ PASS |
| 3 | Pos-enc 8x8 minimum | `[1,8,8]` | 3 layers | ✅ PASS |

---

## Configuration
- Tolerance: rtol=0.0001, atol=1e-05
- Random Seed: 42
