# HungarianMatcher Validation Report

**Generated:** 2026-02-20 14:41:53

**Duration:** 0.01s

## Summary

- **Total Tests:** 5
- **Passed:** 5
- **Failed:** 0
- **Success Rate:** 100.0%

## Test Results

### ✅ HungarianMatcher Shape

**Status:** PASS

**Message:** All shape validations passed

**Details:**
- **batch_size:** `2`
- **num_queries:** `10`
- **matches_per_image:** `[3, 5]`

### ✅ HungarianMatcher Numerical

**Status:** PASS

**Message:** All indices match exactly between PyTorch and TTSim

**Details:**
- **image_0:** `MATCH`
- **image_1:** `MATCH`

### ✅ Focal Loss Cost Component

**Status:** PASS

**Message:** Numerical match within tolerance (max_diff=1.192093e-07)

**Details:**
- **max_diff:** `1.192093e-07`
- **mean_diff:** `1.441687e-08`
- **atol:** `1e-05`
- **rtol:** `0.0001`

### ✅ L1 Bbox Cost Component

**Status:** PASS

**Message:** Numerical match within tolerance (max_diff=0.000000e+00)

**Details:**
- **max_diff:** `0.000000e+00`
- **mean_diff:** `0.000000e+00`
- **atol:** `1e-06`
- **rtol:** `1e-05`

### ✅ GIoU Cost Component

**Status:** PASS

**Message:** Numerical match within tolerance (max_diff=0.000000e+00)

**Details:**
- **max_diff:** `0.000000e+00`
- **mean_diff:** `0.000000e+00`
- **atol:** `1e-05`
- **rtol:** `0.0001`

## Recommendations

✅ All tests passed! The TTSim implementation is functionally equivalent to PyTorch.