# MSDeformAttn Module Unit Test Report
**PyTorch vs TTSim Comparison** | **45/48 passed** | FAIL
Generated: 2026-02-20 14:49:36 | Exit Code: 1

---

## Summary

| Module | Passed | Total | Status |
|--------|--------|-------|--------|
| ValueProjection | 5 | 6 | FAIL |
| SamplingOffsets | 6 | 6 | PASS |
| AttnWeights+Softmax | 6 | 6 | PASS |
| SamplingLocations | 10 | 10 | PASS |
| MSDeformAttn E2E (2D) | 6 | 7 | FAIL |
| MSDeformAttn E2E (4D) | 5 | 6 | FAIL |
| ShapeInference | 7 | 7 | PASS |

**Total: 45/48 tests passed**

---

## Failed Tests

| Module | Test | Edge Case | Max Diff |
|--------|------|-----------|----------|
| ValueProjection | Very large (1e6) | large | 7.50e-01 |
| MSDeformAttn E2E (2D) | Very large (1e6) | large | 3.44e-01 |
| MSDeformAttn E2E (4D) | Very large (1e6) 4D | large | 5.62e-01 |

---

## ValueProjection (5/6 FAIL)
*value_proj linear layer (d_model -> d_model)*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Baseline positive | positive | `[2, 340, 256]` | `[2, 340, 256]` | 1.91e-06 | 2.07e-07 | PASS |
| 1 | Negative values | negative | `[2, 340, 256]` | `[2, 340, 256]` | 1.91e-06 | 2.07e-07 | PASS |
| 2 | Zero values | zeros | `[2, 340, 256]` | `[2, 340, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 3 | Mixed pos/neg | mixed | `[2, 340, 256]` | `[2, 340, 256]` | 2.38e-06 | 2.32e-07 | PASS |
| 4 | Very small (1e-6) | small | `[2, 340, 256]` | `[2, 340, 256]` | 9.09e-13 | 7.25e-14 | PASS |
| 5 | Very large (1e6) | large | `[2, 340, 256]` | `[2, 340, 256]` | 7.50e-01 | 7.24e-02 | **FAIL** |

**Failed Cases:**
- [5] Very large (1e6) - large (diff: 7.50e-01)

---

## SamplingOffsets (6/6 PASS)
*sampling_offsets linear (d_model -> n_heads*n_levels*n_points*2)*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Baseline positive | positive | `[2, 50, 256]` | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 1 | Negative values | negative | `[2, 50, 256]` | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 2 | Zero values | zeros | `[2, 50, 256]` | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 3 | Mixed pos/neg | mixed | `[2, 50, 256]` | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 4 | Very small (1e-6) | small | `[2, 50, 256]` | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 5 | Very large (1e6) | large | `[2, 50, 256]` | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |

---

## AttnWeights+Softmax (6/6 PASS)
*attention_weights linear + softmax normalization*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Baseline positive | positive | `[2, 50, 256]` | `[2, 50, 8, 4, 4]` | 0.00e+00 | 0.00e+00 | PASS |
| 1 | Negative values | negative | `[2, 50, 256]` | `[2, 50, 8, 4, 4]` | 0.00e+00 | 0.00e+00 | PASS |
| 2 | Zero values | zeros | `[2, 50, 256]` | `[2, 50, 8, 4, 4]` | 0.00e+00 | 0.00e+00 | PASS |
| 3 | Mixed pos/neg | mixed | `[2, 50, 256]` | `[2, 50, 8, 4, 4]` | 0.00e+00 | 0.00e+00 | PASS |
| 4 | Very small (1e-6) | small | `[2, 50, 256]` | `[2, 50, 8, 4, 4]` | 0.00e+00 | 0.00e+00 | PASS |
| 5 | Very large (1e6) | large | `[2, 50, 256]` | `[2, 50, 8, 4, 4]` | 0.00e+00 | 0.00e+00 | PASS |

---

## SamplingLocations (10/10 PASS)
*Sampling location computation from reference points + offsets*

| # | Test Case | Edge Case | Ref Dim | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:--------|:-------|:---------|:----------|:-------|
| 0 | Baseline positive 2D | positive | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 1.19e-07 | 1.17e-08 | PASS |
| 1 | Negative values 2D | negative | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 1.19e-07 | 1.15e-08 | PASS |
| 2 | Zero values 2D | zeros | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 1.19e-07 | 1.17e-08 | PASS |
| 3 | Mixed pos/neg 2D | mixed | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 1.19e-07 | 1.21e-08 | PASS |
| 4 | Very small (1e-6) 2D | small | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 1.19e-07 | 1.20e-08 | PASS |
| 5 | Very large (1e6) 2D | large | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 1.19e-07 | 1.20e-08 | PASS |
| 6 | Boundary coords 2D | boundary_coords | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 0.00e+00 | 0.00e+00 | PASS |
| 7 | Center coords 2D | center_coords | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 2.38e-08 | 1.90e-09 | PASS |
| 8 | Corner coords 2D | corner_coords | ref_dim=2 | `[2, 50, 8, 4, 4, 2]` | 2.98e-08 | 1.78e-09 | PASS |
| 9 | 4D reference points | 4d_ref_points | ref_dim=4 | `[2, 50, 8, 4, 4, 2]` | 0.00e+00 | 0.00e+00 | PASS |

---

## MSDeformAttn E2E (2D) (6/7 FAIL)
*Full MSDeformAttn end-to-end with 2D reference points*

| # | Test Case | Data Type | Config | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:-------|:-------|:---------|:----------|:-------|
| 0 | Baseline positive | positive | d=256 L=4 | `[2, 50, 256]` | 9.54e-07 | 1.26e-07 | PASS |
| 1 | Negative values | negative | d=256 L=4 | `[2, 50, 256]` | 9.54e-07 | 1.21e-07 | PASS |
| 2 | Zero values | zeros | d=256 L=4 | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 3 | Mixed pos/neg | mixed | d=256 L=4 | `[2, 50, 256]` | 5.36e-07 | 8.23e-08 | PASS |
| 4 | Very small (1e-6) | small | d=256 L=4 | `[2, 50, 256]` | 3.41e-13 | 4.22e-14 | PASS |
| 5 | Very large (1e6) | large | d=256 L=4 | `[2, 50, 256]` | 3.44e-01 | 4.24e-02 | **FAIL** |
| 6 | Minimum input | positive | d=128 L=2 | `[1, 10, 128]` | 5.96e-07 | 9.90e-08 | PASS |

**Failed Cases:**
- [5] Very large (1e6) - large (diff: 3.44e-01)

---

## MSDeformAttn E2E (4D) (5/6 FAIL)
*Full MSDeformAttn end-to-end with 4D reference points (x, y, w, h)*

| # | Test Case | Data Type | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:-------|:---------|:----------|:-------|
| 0 | Baseline positive 4D | positive | `[2, 50, 256]` | 2.15e-06 | 2.07e-07 | PASS |
| 1 | Negative values 4D | negative | `[2, 50, 256]` | 2.15e-06 | 1.94e-07 | PASS |
| 2 | Zero values 4D | zeros | `[2, 50, 256]` | 0.00e+00 | 0.00e+00 | PASS |
| 3 | Mixed pos/neg 4D | mixed | `[2, 50, 256]` | 8.34e-07 | 1.16e-07 | PASS |
| 4 | Very small (1e-6) 4D | small | `[2, 50, 256]` | 5.68e-13 | 6.69e-14 | PASS |
| 5 | Very large (1e6) 4D | large | `[2, 50, 256]` | 5.62e-01 | 6.69e-02 | **FAIL** |

**Failed Cases:**
- [5] Very large (1e6) 4D - large (diff: 5.62e-01)

---

## ShapeInference (7/7 PASS)
*MSDeformAttn shape inference (data=None) across configurations*

| # | Test Case | Config | Expected | Actual | Result |
|:--|:----------|:-------|:---------|:-------|:-------|
| 0 | Standard 4-level | d=256 L=4 M=8 P=4 | `[2, 100, 256]` | `[2, 100, 256]` | PASS |
| 1 | Small config | d=128 L=2 M=4 P=2 | `[1, 50, 128]` | `[1, 50, 128]` | PASS |
| 2 | Single level | d=256 L=1 M=8 P=4 | `[2, 100, 256]` | `[2, 100, 256]` | PASS |
| 3 | Many heads (16) | d=256 L=4 M=16 P=4 | `[2, 50, 256]` | `[2, 50, 256]` | PASS |
| 4 | Single query | d=256 L=4 M=8 P=4 | `[1, 1, 256]` | `[1, 1, 256]` | PASS |
| 5 | Large batch (4) | d=256 L=4 M=8 P=4 | `[4, 50, 256]` | `[4, 50, 256]` | PASS |
| 6 | Minimum input | d=64 L=1 M=2 P=1 | `[1, 1, 64]` | `[1, 1, 64]` | PASS |

---

## Configuration
- Default tolerance: rtol=0.0001, atol=1e-05
- Large value tolerance: rtol=1e-3, atol=1e-2
- Random Seed: 42
- Default config: d_model=256, n_levels=4, n_heads=8, n_points=4
