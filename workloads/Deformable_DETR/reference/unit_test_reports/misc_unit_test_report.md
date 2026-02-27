# Misc Utilities Unit Test Report
**PyTorch vs TTSim Comparison** | **71/71 passed** | PASS
Generated: 2026-02-20 14:48:57 | Exit Code: 0

---

## Summary

| Module | Passed | Total | Status |
|--------|--------|-------|--------|
| NestedTensor | 10 | 10 | PASS |
| interpolate (nearest) | 11 | 11 | PASS |
| interpolate (bilinear) | 10 | 10 | PASS |
| interpolate (downsample) | 10 | 10 | PASS |
| nested_tensor_from_tensor_list | 11 | 11 | PASS |
| inverse_sigmoid | 11 | 11 | PASS |
| Shape Inference | 8 | 8 | PASS |

**Total: 71/71 tests passed**

---

## NestedTensor (10/10 PASS)
*Data container bundling tensors with padding masks — decompose + mask validation*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Basic 3ch 56x56 | positive | `[2, 3, 56, 56]` | `[2, 3, 56, 56]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Single batch 3ch 32x32 | positive | `[1, 3, 32, 32]` | `[1, 3, 32, 32]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Multi-channel 64ch | positive | `[2, 64, 16, 16]` | `[2, 64, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Negative values | negative | `[2, 3, 16, 16]` | `[2, 3, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Zero values | zeros | `[2, 3, 16, 16]` | `[2, 3, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Mixed positive/negative | mixed | `[2, 3, 16, 16]` | `[2, 3, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Very small values (1e-6) | small | `[2, 3, 16, 16]` | `[2, 3, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Very large values (1e6) | large | `[2, 3, 16, 16]` | `[2, 3, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Minimum spatial 1x1 | positive | `[1, 1, 1, 1]` | `[1, 1, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Single pixel batch=2 | mixed | `[2, 3, 1, 1]` | `[2, 3, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## interpolate (nearest) (11/11 PASS)
*Nearest neighbor upsampling — deterministic pixel selection*

| # | Test Case | Data Type | Input→Target | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:-------------|:-------|:---------|:----------|:-------|
| 0 | 2x upsample 10→20 | positive | `[2, 3, 10, 10]→[20, 20]` | `[2, 3, 20, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | 3x upsample 8→24 | positive | `[1, 3, 8, 8]→[24, 24]` | `[1, 3, 24, 24]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Non-square 8x10→16x20 | positive | `[1, 3, 8, 10]→[16, 20]` | `[1, 3, 16, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | 64 channels | positive | `[1, 64, 8, 8]→[16, 16]` | `[1, 64, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Negative values | negative | `[2, 3, 10, 10]→[20, 20]` | `[2, 3, 20, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Zero values | zeros | `[2, 3, 10, 10]→[20, 20]` | `[2, 3, 20, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Mixed positive/negative | mixed | `[2, 3, 10, 10]→[20, 20]` | `[2, 3, 20, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Very small values (1e-6) | small | `[2, 3, 10, 10]→[20, 20]` | `[2, 3, 20, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Very large values (1e6) | large | `[2, 3, 10, 10]→[20, 20]` | `[2, 3, 20, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Minimum 1x1→2x2 | positive | `[1, 1, 1, 1]→[2, 2]` | `[1, 1, 2, 2]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 10 | Minimum 1x1→1x1 (identity) | mixed | `[1, 1, 1, 1]→[1, 1]` | `[1, 1, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## interpolate (bilinear) (10/10 PASS)
*Bilinear upsampling — PyTorch-compatible weighted average interpolation*

| # | Test Case | Data Type | Input→Target | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:-------------|:-------|:---------|:----------|:-------|
| 0 | 2x upsample 8→16 | positive | `[1, 3, 8, 8]→[16, 16]` | `[1, 3, 16, 16]` | 2.38e-07 | 2.51e-08 | ✅ PASS |
| 1 | 3x upsample 4→12 | positive | `[1, 3, 4, 4]→[12, 12]` | `[1, 3, 12, 12]` | 2.38e-07 | 4.80e-08 | ✅ PASS |
| 2 | Non-square 6x8→12x16 | positive | `[1, 3, 6, 8]→[12, 16]` | `[1, 3, 12, 16]` | 2.38e-07 | 2.24e-08 | ✅ PASS |
| 3 | Multi-batch | positive | `[2, 3, 8, 8]→[16, 16]` | `[2, 3, 16, 16]` | 2.38e-07 | 2.81e-08 | ✅ PASS |
| 4 | Negative values | negative | `[1, 3, 8, 8]→[16, 16]` | `[1, 3, 16, 16]` | 2.38e-07 | 2.39e-08 | ✅ PASS |
| 5 | Zero values | zeros | `[1, 3, 8, 8]→[16, 16]` | `[1, 3, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Mixed positive/negative | mixed | `[1, 3, 8, 8]→[16, 16]` | `[1, 3, 16, 16]` | 4.77e-07 | 2.63e-08 | ✅ PASS |
| 7 | Very small values (1e-6) | small | `[1, 3, 8, 8]→[16, 16]` | `[1, 3, 16, 16]` | 1.14e-13 | 9.99e-15 | ✅ PASS |
| 8 | Very large values (1e6) | large | `[1, 3, 8, 8]→[16, 16]` | `[1, 3, 16, 16]` | 1.25e-01 | 8.56e-03 | ✅ PASS |
| 9 | Minimum 2x2→4x4 | positive | `[1, 1, 2, 2]→[4, 4]` | `[1, 1, 4, 4]` | 1.19e-07 | 3.73e-08 | ✅ PASS |

---

## interpolate (downsample) (10/10 PASS)
*Nearest neighbor downsampling — reducing spatial resolution*

| # | Test Case | Data Type | Input→Target | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:-------------|:-------|:---------|:----------|:-------|
| 0 | 4x downsample 32→8 | positive | `[2, 3, 32, 32]→[8, 8]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | 2x downsample 16→8 | positive | `[1, 3, 16, 16]→[8, 8]` | `[1, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Non-square 24x32→6x8 | positive | `[1, 3, 24, 32]→[6, 8]` | `[1, 3, 6, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Negative values | negative | `[2, 3, 32, 32]→[8, 8]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Zero values | zeros | `[2, 3, 32, 32]→[8, 8]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Mixed positive/negative | mixed | `[2, 3, 32, 32]→[8, 8]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Very small values (1e-6) | small | `[2, 3, 32, 32]→[8, 8]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Very large values (1e6) | large | `[2, 3, 32, 32]→[8, 8]` | `[2, 3, 8, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Minimum 2x2→1x1 | positive | `[1, 1, 2, 2]→[1, 1]` | `[1, 1, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Identity 4x4→4x4 | mixed | `[1, 3, 4, 4]→[4, 4]` | `[1, 3, 4, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## nested_tensor_from_tensor_list (11/11 PASS)
*Batch variable-sized tensors with zero-padding and boolean mask*

| # | Test Case | Data Type | Input Shapes | Batched Shape | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:-------------|:--------------|:---------|:----------|:-------|
| 0 | 3 tensors variable sizes | positive | `[[3, 10, 15], [3, 12, 18], [3, 8, 20]]` | `[3, 3, 12, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | 2 tensors same size | positive | `[[3, 16, 16], [3, 16, 16]]` | `[2, 3, 16, 16]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | 4 tensors different H/W | positive | `[[3, 8, 12], [3, 10, 10], [3, 6, 14], [3, 12, 8]]` | `[4, 3, 12, 14]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Single channel | positive | `[[1, 10, 10], [1, 8, 12]]` | `[2, 1, 10, 12]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Negative values | negative | `[[3, 10, 15], [3, 12, 18]]` | `[2, 3, 12, 18]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Zero values | zeros | `[[3, 10, 15], [3, 12, 18]]` | `[2, 3, 12, 18]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Mixed positive/negative | mixed | `[[3, 10, 15], [3, 12, 18]]` | `[2, 3, 12, 18]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Very small values (1e-6) | small | `[[3, 10, 15], [3, 12, 18]]` | `[2, 3, 12, 18]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Very large values (1e6) | large | `[[3, 10, 15], [3, 12, 18]]` | `[2, 3, 12, 18]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Minimum 1x1 tensors | positive | `[[1, 1, 1], [1, 1, 1]]` | `[2, 1, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 10 | Minimum mixed 1x1 & 2x3 | mixed | `[[1, 1, 1], [1, 2, 3]]` | `[2, 1, 2, 3]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## inverse_sigmoid (11/11 PASS)
*Logit function — inverse of sigmoid, with clamping for numerical stability*

| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Mid-range [0.4-0.6] | mid_range | `[2, 4, 8]` | `[2, 4, 8]` | 2.98e-08 | 8.41e-09 | ✅ PASS |
| 1 | Uniform [0,1] | uniform | `[2, 4, 8]` | `[2, 4, 8]` | 2.38e-07 | 6.17e-09 | ✅ PASS |
| 2 | Multi-batch | uniform | `[4, 8, 16]` | `[4, 8, 16]` | 4.77e-07 | 9.48e-09 | ✅ PASS |
| 3 | Near-zero values [0.01-0.1] | near_zero | `[2, 4, 8]` | `[2, 4, 8]` | 2.38e-07 | 1.49e-08 | ✅ PASS |
| 4 | Near-one values [0.9-0.99] | near_one | `[2, 4, 8]` | `[2, 4, 8]` | 4.77e-07 | 2.61e-08 | ✅ PASS |
| 5 | Boundary 0 and 1 | boundary | `[2, 4, 8]` | `[2, 4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Zero values | zeros | `[2, 4, 8]` | `[2, 4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Negative values (clamped) | negative | `[2, 4, 8]` | `[2, 4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 8 | Large values >1 (clamped) | large | `[2, 4, 8]` | `[2, 4, 8]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 9 | Minimum 1-element | mid_range | `[1]` | `[1]` | 1.49e-08 | 1.49e-08 | ✅ PASS |
| 10 | Minimum 1x1x1 | uniform | `[1, 1, 1]` | `[1, 1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## Shape Inference (8/8 PASS)
*Shape-only validation (data=None) for interpolate and nested_tensor_from_tensor_list*

| # | Test Case | Expected Shape | Actual Shape | data=None | Result |
|:--|:----------|:---------------|:-------------|:----------|:-------|
| 0 | interpolate nearest 2x up | `[2,3,20,20]` | `[2,3,20,20]` | True | ✅ PASS |
| 1 | interpolate bilinear 2x up | `[1,3,16,16]` | `[1,3,16,16]` | True | ✅ PASS |
| 2 | interpolate 4x downsample | `[2,3,8,8]` | `[2,3,8,8]` | True | ✅ PASS |
| 3 | interpolate scale_factor=2 | `[1,3,20,20]` | `[1,3,20,20]` | True | ✅ PASS |
| 4 | interpolate non-square | `[1,64,12,16]` | `[1,64,12,16]` | True | ✅ PASS |
| 5 | nested_tensor 3 tensors | `[3,3,12,20]` | `[3,3,12,20]` | True | ✅ PASS |
| 6 | interpolate minimum 1x1→2x2 | `[1,1,2,2]` | `[1,1,2,2]` | True | ✅ PASS |
| 7 | nested_tensor minimum 1x1 | `[2,1,2,3]` | `[2,1,2,3]` | True | ✅ PASS |

---

## Configuration
- Tolerance: rtol=0.0001, atol=1e-05
- Random Seed: 42
