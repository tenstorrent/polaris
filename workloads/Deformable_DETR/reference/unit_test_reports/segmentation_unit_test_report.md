# Segmentation Unit Test Report
**PyTorch vs TTSim Comparison** | **59/59 passed** | PASS
Generated: 2026-02-19 16:40:23 | Exit Code: 0

---

## Summary

| Module | Passed | Total | Status |
|--------|--------|-------|--------|
| HelperFunctions | 23 | 23 | PASS |
| MHAttentionMap | 11 | 11 | PASS |
| MaskHeadSmallConv | 9 | 9 | PASS |
| DETRsegm | 4 | 4 | PASS |
| Reshape+Squeeze | 7 | 7 | PASS |
| ParamCount | 5 | 5 | PASS |

**Total: 59/59 tests passed**

---

## HelperFunctions (23/23 PASS)
*masked_fill, interpolate_nearest, conv2d_functional*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | masked_fill 2D positive | positive | `[4, 8]` | `[4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | masked_fill 4D positive | positive | `[2, 3, 4, 4]` | `[2, 3, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | masked_fill negative values | negative | `[4, 8]` | `[4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | masked_fill zero values | zeros | `[4, 8]` | `[4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | masked_fill mixed values | mixed | `[4, 8]` | `[4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | masked_fill small values (1e-6) | small | `[4, 8]` | `[4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | masked_fill large values (1e6) | large | `[4, 8]` | `[4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | interpolate 2x upsample | positive | `[2, 3, 4, 4]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | interpolate 4x upsample | positive | `[1, 8, 4, 4]` | `[1, 8, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | interpolate negative values | negative | `[2, 3, 4, 4]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 10 | interpolate zero values | zeros | `[2, 3, 4, 4]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 11 | interpolate mixed values | mixed | `[2, 3, 4, 4]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 12 | interpolate small values (1e-6) | small | `[2, 3, 4, 4]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 13 | interpolate large values (1e6) | large | `[2, 3, 4, 4]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 14 | interpolate minimum 2x2→4x4 | positive | `[1, 4, 2, 2]` | `[1, 4, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 15 | conv2d 3x3 positive | positive | `[2, 4, 8, 8]` | `[2, 8, 8, 8]` | 5.96e-08 | 1.04e-08 | ✅ PASS |
| 16 | conv2d 1x1 positive | positive | `[2, 64, 8, 8]` | `[2, 64, 8, 8]` | 1.79e-07 | 2.50e-08 | ✅ PASS |
| 17 | conv2d negative values | negative | `[2, 4, 8, 8]` | `[2, 8, 8, 8]` | 1.19e-07 | 1.45e-08 | ✅ PASS |
| 18 | conv2d zero values | zeros | `[2, 4, 8, 8]` | `[2, 8, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 19 | conv2d mixed values | mixed | `[2, 4, 8, 8]` | `[2, 8, 8, 8]` | 1.79e-07 | 1.76e-08 | ✅ PASS |
| 20 | conv2d small values (1e-6) | small | `[2, 4, 8, 8]` | `[2, 8, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 21 | conv2d large values (1e6) | large | `[2, 4, 8, 8]` | `[2, 8, 8, 8]` | 6.25e-02 | 4.98e-03 | ✅ PASS |
| 22 | conv2d minimum 3x3 | positive | `[1, 4, 3, 3]` | `[1, 8, 3, 3]` | 8.94e-08 | 1.11e-08 | ✅ PASS |

---

## MHAttentionMap (11/11 PASS)
*Multi-head attention map for spatial attention weights*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | MHA baseline B=1 Q=4 8×8 | positive | q=[1,4,128] | `[1, 4, 8, 8, 8]` | 4.66e-10 | 9.82e-11 | ✅ PASS |
| 1 | MHA baseline B=2 Q=4 8×8 | positive | q=[2,4,128] | `[2, 4, 8, 8, 8]` | 4.66e-10 | 1.07e-10 | ✅ PASS |
| 2 | MHA with mask | positive | q=[1,4,128] | `[1, 4, 8, 8, 8]` | 4.66e-10 | 9.03e-11 | ✅ PASS |
| 3 | MHA negative values | negative | q=[1,4,128] | `[1, 4, 8, 8, 8]` | 6.98e-10 | 1.59e-10 | ✅ PASS |
| 4 | MHA zero values | zeros | q=[1,4,128] | `[1, 4, 8, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | MHA mixed values | mixed | q=[1,4,128] | `[1, 4, 8, 8, 8]` | 4.66e-10 | 1.23e-10 | ✅ PASS |
| 6 | MHA small values (1e-6) | small | q=[1,4,128] | `[1, 4, 8, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | MHA large values (1e6) | large | q=[1,4,128] | `[1, 4, 8, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | MHA minimum 4×4 | positive | q=[1,4,128] | `[1, 4, 8, 4, 4]` | 1.86e-09 | 4.99e-10 | ✅ PASS |
| 9 | MHA single query Q=1 | positive | q=[1,1,128] | `[1, 1, 8, 8, 8]` | 4.66e-10 | 1.06e-10 | ✅ PASS |
| 10 | MHA non-square 8×4 | positive | q=[1,4,128] | `[1, 4, 8, 8, 4]` | 9.31e-10 | 2.19e-10 | ✅ PASS |

---

## MaskHeadSmallConv (9/9 PASS)
*FPN-based mask prediction head (Conv+GN+ReLU + upsampling)*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | MaskHead baseline B=1 Q=2 | positive | `x=[1,128,8,8]` | `[2, 1, 64, 64]` | 4.41e-06 | 5.53e-07 | ✅ PASS |
| 1 | MaskHead baseline B=2 Q=2 | positive | `x=[2,128,8,8]` | `[4, 1, 64, 64]` | 5.36e-06 | 7.58e-07 | ✅ PASS |
| 2 | MaskHead negative values | negative | `x=[1,128,8,8]` | `[2, 1, 64, 64]` | 6.20e-06 | 7.74e-07 | ✅ PASS |
| 3 | MaskHead zero values | zeros | `x=[1,128,8,8]` | `[2, 1, 64, 64]` | 2.80e-06 | 4.00e-07 | ✅ PASS |
| 4 | MaskHead mixed values | mixed | `x=[1,128,8,8]` | `[2, 1, 64, 64]` | 1.97e-06 | 3.48e-07 | ✅ PASS |
| 5 | MaskHead small values (1e-6) | small | `x=[1,128,8,8]` | `[2, 1, 64, 64]` | 2.38e-06 | 3.35e-07 | ✅ PASS |
| 6 | MaskHead large values (1e6) | large | `x=[1,128,8,8]` | `[2, 1, 64, 64]` | 1.88e-06 | 3.09e-07 | ✅ PASS |
| 7 | MaskHead minimum 4×4 | positive | `x=[1,128,4,4]` | `[2, 1, 32, 32]` | 3.34e-06 | 6.03e-07 | ✅ PASS |
| 8 | MaskHead single query Q=1 | positive | `x=[1,128,8,8]` | `[1, 1, 64, 64]` | 7.75e-06 | 1.08e-06 | ✅ PASS |

---

## DETRsegm (4/4 PASS)
*Complete segmentation wrapper — shape validation only*

| # | Test Case | Category | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:---------|:------|:-------|:---------|:----------|:-------|
| 0 | DETRsegm B=1 Q=4 16×16 | baseline | `[1,256,16,16]` | `[1, 4, 2, 2]` | - | - | ✅ PASS |
| 1 | DETRsegm B=2 Q=4 16×16 | batch | `[2,256,16,16]` | `[2, 4, 2, 2]` | - | - | ✅ PASS |
| 2 | DETRsegm Q=100 16×16 | scale | `[1,256,16,16]` | `[1, 100, 2, 2]` | - | - | ✅ PASS |
| 3 | DETRsegm minimum 8×8 | minimum_input | `[1,256,8,8]` | `[1, 4, 1, 1]` | - | - | ✅ PASS |

---

## Reshape+Squeeze (7/7 PASS)
*Output formatting: [B*Q,1,H,W] → [B,Q,H,W]*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Reshape+squeeze baseline B=2 Q=4 | positive | `[8, 1, 4, 4]` | `[2, 4, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Reshape+squeeze B=1 Q=1 min | positive | `[1, 1, 4, 4]` | `[1, 1, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Reshape+squeeze negative | negative | `[8, 1, 4, 4]` | `[2, 4, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Reshape+squeeze zeros | zeros | `[8, 1, 4, 4]` | `[2, 4, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Reshape+squeeze mixed | mixed | `[8, 1, 4, 4]` | `[2, 4, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Reshape+squeeze small (1e-6) | small | `[8, 1, 4, 4]` | `[2, 4, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Reshape+squeeze large (1e6) | large | `[8, 1, 4, 4]` | `[2, 4, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## ParamCount (5/5 PASS)
*Analytical parameter count validation*

| # | Test Case | PyTorch | TTSim | Result |
|:--|:----------|--------:|------:|:-------|
| 0 | MHAttentionMap params hdim=256 | 131,584 | 131,584 | ✅ PASS |
| 1 | MaskHeadSmallConv params dims=264 | 1,202,073 | 1,202,073 | ✅ PASS |
| 2 | DETRsegm total params | 1,333,657 | 1,333,657 | ✅ PASS |
| 3 | MHA params hdim=128 | 33,024 | 33,024 | ✅ PASS |
| 4 | MaskHead params dims=136 | 280,697 | 280,697 | ✅ PASS |

---

## Configuration
- Tolerance: rtol=0.0001, atol=1e-05
- Random Seed: 42
