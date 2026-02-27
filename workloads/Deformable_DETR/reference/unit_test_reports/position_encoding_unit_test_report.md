# Position Encoding Unit Test Report
**PyTorch vs TTSim Comparison** | **57/57 passed** | PASS
Generated: 2026-02-20 14:48:42 | Exit Code: 0

---

## Summary

| Module | Passed | Total | Status |
|--------|--------|-------|--------|
| PositionEmbeddingSine | 17 | 17 | PASS |
| PositionEmbeddingLearned | 11 | 11 | PASS |
| SineMaskVariations | 11 | 11 | PASS |
| SineComponentAnalysis | 4 | 4 | PASS |
| build_position_encoding | 7 | 7 | PASS |
| ShapeInference | 7 | 7 | PASS |

**Total: 57/57 tests passed**

---

## PositionEmbeddingSine (17/17 PASS)
*Sinusoidal position encoding — deterministic sin/cos computation*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Sine 28x28 normalised | positive | `[2, 256, 28, 28]` | `[2, 256, 28, 28]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Sine 16x16 normalised | positive | `[1, 64, 16, 16]` | `[1, 64, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Sine 28x28 batch=4 | positive | `[4, 256, 28, 28]` | `[4, 256, 28, 28]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Sine 14x28 non-square | positive | `[2, 128, 14, 28]` | `[2, 128, 14, 28]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Sine 32x8 non-square | positive | `[1, 64, 32, 8]` | `[1, 128, 32, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Sine negative input | negative | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Sine zero input | zeros | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Sine mixed input | mixed | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Sine small values (1e-6) | small | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Sine large values (1e6) | large | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 10 | Sine 1x1 minimum | positive | `[1, 64, 1, 1]` | `[1, 64, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 11 | Sine 1xW strip | positive | `[1, 64, 1, 16]` | `[1, 64, 1, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 12 | Sine Hx1 strip | positive | `[1, 64, 16, 1]` | `[1, 64, 16, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 13 | Sine unnormalised | positive | `[1, 64, 16, 16]` | `[1, 64, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 14 | Sine D=256 T=10000 | positive | `[1, 128, 12, 12]` | `[1, 512, 12, 12]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 15 | Sine D=64 T=100 | positive | `[1, 128, 12, 12]` | `[1, 128, 12, 12]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 16 | Sine D=64 T=20000 | positive | `[1, 128, 12, 12]` | `[1, 128, 12, 12]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## PositionEmbeddingLearned (11/11 PASS)
*Learned position encoding — Embedding lookup with synced weights*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Learned 28x28 standard | positive | `[2, 256, 28, 28]` | `[2, 256, 28, 28]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Learned 10x10 small | positive | `[1, 64, 10, 10]` | `[1, 64, 10, 10]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Learned 14x28 non-square | positive | `[2, 128, 14, 28]` | `[2, 128, 14, 28]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Learned 28x28 batch=4 | positive | `[4, 256, 28, 28]` | `[4, 256, 28, 28]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Learned 20x20 D=256 | positive | `[1, 512, 20, 20]` | `[1, 512, 20, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Learned negative input | negative | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Learned zero input | zeros | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Learned mixed input | mixed | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Learned small values (1e-6) | small | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Learned large values (1e6) | large | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 10 | Learned 4x4 minimum | positive | `[1, 64, 4, 4]` | `[1, 64, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## SineMaskVariations (11/11 PASS)
*Sine position encoding with varying mask patterns and edge case data*

| # | Test Case | Mask | Data | Input | Masked% | Max Diff | Result |
|:--|:----------|:-----|:-----|:------|:--------|:---------|:-------|
| 0 | Sine no mask | no_mask | positive | `[2, 128, 20, 20]` | 0% | 0.00e+00 | ✅ PASS |
| 1 | Sine top rows masked | top_rows | positive | `[2, 128, 20, 20]` | 25% | 0.00e+00 | ✅ PASS |
| 2 | Sine left columns masked | left_cols | positive | `[2, 128, 20, 20]` | 30% | 0.00e+00 | ✅ PASS |
| 3 | Sine checkerboard mask | checkerboard | positive | `[2, 128, 20, 20]` | 50% | 0.00e+00 | ✅ PASS |
| 4 | Sine bottom-right masked | bottom_right | positive | `[2, 128, 20, 20]` | 25% | 0.00e+00 | ✅ PASS |
| 5 | Sine mask + negative | top_rows | negative | `[1, 128, 16, 16]` | 25% | 0.00e+00 | ✅ PASS |
| 6 | Sine mask + zeros | top_rows | zeros | `[1, 128, 16, 16]` | 25% | 0.00e+00 | ✅ PASS |
| 7 | Sine mask + mixed | checkerboard | mixed | `[1, 128, 16, 16]` | 50% | 0.00e+00 | ✅ PASS |
| 8 | Sine mask + small | left_cols | small | `[1, 128, 16, 16]` | 31% | 0.00e+00 | ✅ PASS |
| 9 | Sine mask + large | bottom_right | large | `[1, 128, 16, 16]` | 25% | 0.00e+00 | ✅ PASS |
| 10 | Sine 4x4 mask minimum | top_rows | positive | `[1, 64, 4, 4]` | 25% | 0.00e+00 | ✅ PASS |

---

## SineComponentAnalysis (4/4 PASS)
*Y/X component separation and spatial invariance verification*

| # | Test Case | Input | D | Components | Invariance | Result |
|:--|:----------|:------|:--|:-----------|:-----------|:-------|
| 0 | Components 24x24 standard | `[2, 256, 24, 24]` | D=128 | y=True x=True | col_inv=True row_inv=True | ✅ PASS |
| 1 | Components 12x12 small | `[1, 128, 12, 12]` | D=64 | y=True x=True | col_inv=True row_inv=True | ✅ PASS |
| 2 | Components 8x16 non-square | `[2, 128, 8, 16]` | D=64 | y=True x=True | col_inv=True row_inv=True | ✅ PASS |
| 3 | Components 4x4 minimum | `[1, 64, 4, 4]` | D=32 | y=True x=True | col_inv=True row_inv=True | ✅ PASS |

---

## build_position_encoding (7/7 PASS)
*Factory function — type dispatch, aliases, error handling*

| # | Test Case | Type | Hidden Dim | Output Shape | Numerical | Result |
|:--|:----------|:-----|:-----------|:-------------|:----------|:-------|
| 0 | Factory sine | `sine` | 256 | `[2, 256, 14, 14]` | 0.00e+00 | ✅ PASS |
| 1 | Factory learned | `learned` | 256 | `[2, 256, 14, 14]` | N/A | ✅ PASS |
| 2 | Factory v2 alias | `v2` | 256 | `[2, 256, 14, 14]` | 0.00e+00 | ✅ PASS |
| 3 | Factory v3 alias | `v3` | 256 | `[2, 256, 14, 14]` | N/A | ✅ PASS |
| 4 | Factory v2 hidden=128 | `v2` | 128 | `[2, 128, 14, 14]` | 0.00e+00 | ✅ PASS |
| 5 | Factory sine hidden=512 | `sine` | 512 | `[2, 512, 14, 14]` | 0.00e+00 | ✅ PASS |
| 6 | Factory invalid type | `banana` | 256 | ValueError | ✅ PASS |

---

## ShapeInference (7/7 PASS)
*Shape inference validation (data=None mode)*

| # | Test Case | Type | Input | Expected | TTSim | Result |
|:--|:----------|:-----|:------|:---------|:------|:-------|
| 0 | Shape sine 28x28 | sine | `[2,256,28,28]` | `[2, 256, 28, 28]` | `[2, 256, 28, 28]` | ✅ PASS |
| 1 | Shape sine 14x14 | sine | `[1,128,14,14]` | `[1, 128, 14, 14]` | `[1, 128, 14, 14]` | ✅ PASS |
| 2 | Shape sine 7x11 | sine | `[2,256,7,11]` | `[2, 256, 7, 11]` | `[2, 256, 7, 11]` | ✅ PASS |
| 3 | Shape sine 1x1 minimum | sine | `[1,64,1,1]` | `[1, 64, 1, 1]` | `[1, 64, 1, 1]` | ✅ PASS |
| 4 | Shape learned 28x28 | learned | `[2,256,28,28]` | `[2, 256, 28, 28]` | `[2, 256, 28, 28]` | ✅ PASS |
| 5 | Shape learned 10x10 | learned | `[1,128,10,10]` | `[1, 128, 10, 10]` | `[1, 128, 10, 10]` | ✅ PASS |
| 6 | Shape learned 4x4 min | learned | `[1,64,4,4]` | `[1, 64, 4, 4]` | `[1, 64, 4, 4]` | ✅ PASS |

---

## Configuration
- Tolerance: rtol=0.0001, atol=1e-05
- Random Seed: 42
