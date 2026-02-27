# Box Operations Validation Report

**Generated:** 2026-02-20 14:41:39

**Test Suite:** PyTorch vs TTSim Bounding Box Operations Comparison

---

## Overview

This report validates the TTSim implementation of bounding box utility functions.

**Functions Tested:**
1. **box_cxcywh_to_xyxy**: Convert center format to corner format
2. **box_xyxy_to_cxcywh**: Convert corner format to center format
3. **box_area**: Compute box areas
4. **box_iou**: Compute Intersection over Union
5. **generalized_box_iou**: Compute Generalized IoU (GIoU)
6. **masks_to_boxes**: Extract boxes from binary masks

**All functions include full numerical comparison with PyTorch.**

---


================================================================================
## Test 1: box_cxcywh_to_xyxy
================================================================================

**Function:** Convert boxes from (cx, cy, w, h) to (x0, y0, x1, y1) format

### Input Boxes (cxcywh format)

**Input Boxes:**
```
PyTorch shape: [3, 4]
TTSim shape:   [3, 4]

PyTorch statistics:
  Mean: 0.45833334
  Std:  0.18123803
  Min:  0.20000000
  Max:  0.80000001

TTSim statistics:
  Mean: 0.45833334
  Std:  0.18123803
  Min:  0.20000000
  Max:  0.80000001

Sample values (first 10):
  PyTorch: [0.500000, 0.500000, 0.400000, 0.600000, 0.250000, 0.750000, 0.300000, 0.400000, 0.800000, 0.300000, ...]
  TTSim:   [0.500000, 0.500000, 0.400000, 0.600000, 0.250000, 0.750000, 0.300000, 0.400000, 0.800000, 0.300000, ...]
```

### Output Boxes (xyxy format)

**Output Boxes:**
```
PyTorch shape: [3, 4]
TTSim shape:   [3, 4]

PyTorch statistics:
  Mean: 0.51666671
  Std:  0.29391986
  Min:  0.05000001
  Max:  0.94999999

TTSim statistics:
  Mean: 0.51666671
  Std:  0.29391986
  Min:  0.05000001
  Max:  0.94999999

Sample values (first 10):
  PyTorch: [0.300000, 0.200000, 0.700000, 0.800000, 0.100000, 0.550000, 0.400000, 0.950000, 0.700000, 0.050000, ...]
  TTSim:   [0.300000, 0.200000, 0.700000, 0.800000, 0.100000, 0.550000, 0.400000, 0.950000, 0.700000, 0.050000, ...]
```

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] box_cxcywh_to_xyxy test

================================================================================
## Test 2: box_xyxy_to_cxcywh
================================================================================

**Function:** Convert boxes from (x0, y0, x1, y1) to (cx, cy, w, h) format

### Input Boxes (xyxy format)

**Input Boxes:**
```
PyTorch shape: [3, 4]
TTSim shape:   [3, 4]

PyTorch statistics:
  Mean: 0.35833335
  Std:  0.28999522
  Min:  0.00000000
  Max:  0.89999998

TTSim statistics:
  Mean: 0.35833335
  Std:  0.28999522
  Min:  0.00000000
  Max:  0.89999998

Sample values (first 10):
  PyTorch: [0.100000, 0.200000, 0.500000, 0.700000, 0.300000, 0.300000, 0.800000, 0.900000, 0.000000, 0.000000, ...]
  TTSim:   [0.100000, 0.200000, 0.500000, 0.700000, 0.300000, 0.300000, 0.800000, 0.900000, 0.000000, 0.000000, ...]
```

### Output Boxes (cxcywh format)

**Output Boxes:**
```
PyTorch shape: [3, 4]
TTSim shape:   [3, 4]

PyTorch statistics:
  Mean: 0.38750002
  Std:  0.16723859
  Min:  0.10000000
  Max:  0.60000002

TTSim statistics:
  Mean: 0.38750002
  Std:  0.16723859
  Min:  0.10000000
  Max:  0.60000002

Sample values (first 10):
  PyTorch: [0.300000, 0.450000, 0.400000, 0.500000, 0.550000, 0.600000, 0.500000, 0.600000, 0.100000, 0.150000, ...]
  TTSim:   [0.300000, 0.450000, 0.400000, 0.500000, 0.550000, 0.600000, 0.500000, 0.600000, 0.100000, 0.150000, ...]
```

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] box_xyxy_to_cxcywh test

================================================================================
## Test 3: box_area
================================================================================

**Function:** Compute area of boxes in xyxy format

**Formula:** area = (x1 - x0) * (y1 - y0)

### Input Boxes

**Input Boxes (xyxy):**
```
PyTorch shape: [4, 4]
TTSim shape:   [4, 4]

PyTorch statistics:
  Mean: 0.42499998
  Std:  0.35619518
  Min:  0.00000000
  Max:  1.00000000

TTSim statistics:
  Mean: 0.42499998
  Std:  0.35619518
  Min:  0.00000000
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [0.000000, 0.000000, 1.000000, 1.000000, 0.100000, 0.200000, 0.500000, 0.700000, 0.300000, 0.300000, ...]
  TTSim:   [0.000000, 0.000000, 1.000000, 1.000000, 0.100000, 0.200000, 0.500000, 0.700000, 0.300000, 0.300000, ...]
```

### Output Areas

**Box Areas:**
```
PyTorch shape: [4]
TTSim shape:   [4]

PyTorch statistics:
  Mean: 0.43750000
  Std:  0.32667837
  Min:  0.20000000
  Max:  1.00000000

TTSim statistics:
  Mean: 0.43750000
  Std:  0.32667837
  Min:  0.20000000
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [1.000000, 0.200000, 0.300000, 0.250000]
  TTSim:   [1.000000, 0.200000, 0.300000, 0.250000]
```

### Expected Areas
```
Box 0: 1.0 * 1.0 = 1.000000
Box 1: 0.4 * 0.5 = 0.200000
Box 2: 0.5 * 0.6 = 0.300000
Box 3: 0.5 * 0.5 = 0.250000
```

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] box_area test

================================================================================
## Test 4: box_iou
================================================================================

**Function:** Compute pairwise IoU (Intersection over Union) between two sets of boxes

**Formula:** IoU = intersection_area / union_area

### Input Boxes

**Set 1 (N=2 boxes):**

**Boxes1:**
```
PyTorch shape: [2, 4]
TTSim shape:   [2, 4]

PyTorch statistics:
  Mean: 0.34999999
  Std:  0.26925823
  Min:  0.00000000
  Max:  0.69999999

TTSim statistics:
  Mean: 0.34999999
  Std:  0.26925823
  Min:  0.00000000
  Max:  0.69999999

Sample values (first 10):
  PyTorch: [0.000000, 0.000000, 0.500000, 0.500000, 0.200000, 0.200000, 0.700000, 0.700000]
  TTSim:   [0.000000, 0.000000, 0.500000, 0.500000, 0.200000, 0.200000, 0.700000, 0.700000]
```

**Set 2 (M=3 boxes):**

**Boxes2:**
```
PyTorch shape: [3, 4]
TTSim shape:   [3, 4]

PyTorch statistics:
  Mean: 0.60000002
  Std:  0.34156501
  Min:  0.00000000
  Max:  1.00000000

TTSim statistics:
  Mean: 0.60000002
  Std:  0.34156501
  Min:  0.00000000
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [0.000000, 0.000000, 0.500000, 0.500000, 0.400000, 0.400000, 0.900000, 0.900000, 0.800000, 0.800000, ...]
  TTSim:   [0.000000, 0.000000, 0.500000, 0.500000, 0.400000, 0.400000, 0.900000, 0.900000, 0.800000, 0.800000, ...]
```

### Output IoU Matrix [N, M]

**IoU Matrix:**
```
PyTorch shape: [2, 3]
TTSim shape:   [2, 3]

PyTorch statistics:
  Mean: 0.24323876
  Std:  0.35161465
  Min:  0.00000000
  Max:  1.00000000

TTSim statistics:
  Mean: 0.24323876
  Std:  0.35161465
  Min:  0.00000000
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [1.000000, 0.020408, 0.000000, 0.219512, 0.219512, 0.000000]
  TTSim:   [1.000000, 0.020408, 0.000000, 0.219512, 0.219512, 0.000000]
```

### Output Union Matrix [N, M]

**Union Matrix:**
```
PyTorch shape: [2, 3]
TTSim shape:   [2, 3]

PyTorch statistics:
  Mean: 0.35666665
  Std:  0.08537498
  Min:  0.25000000
  Max:  0.48999998

TTSim statistics:
  Mean: 0.35666665
  Std:  0.08537498
  Min:  0.25000000
  Max:  0.48999998

Sample values (first 10):
  PyTorch: [0.250000, 0.490000, 0.290000, 0.410000, 0.410000, 0.290000]
  TTSim:   [0.250000, 0.490000, 0.290000, 0.410000, 0.410000, 0.290000]
```

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] box_iou test

================================================================================
## Test 5: generalized_box_iou (GIoU)
================================================================================

**Function:** Compute Generalized IoU between two sets of boxes

**Formula:** GIoU = IoU - (area_enclosing - union) / area_enclosing

GIoU extends IoU by penalizing non-overlapping boxes based on their spatial relationship.

### Input Boxes

**Set 1 (N=2 boxes):**

**Boxes1:**
```
PyTorch shape: [2, 4]
TTSim shape:   [2, 4]

PyTorch statistics:
  Mean: 0.52499998
  Std:  0.35619518
  Min:  0.00000000
  Max:  1.00000000

TTSim statistics:
  Mean: 0.52499998
  Std:  0.35619518
  Min:  0.00000000
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [0.000000, 0.000000, 0.500000, 0.500000, 0.600000, 0.600000, 1.000000, 1.000000]
  TTSim:   [0.000000, 0.000000, 0.500000, 0.500000, 0.600000, 0.600000, 1.000000, 1.000000]
```

**Set 2 (M=3 boxes):**

**Boxes2:**
```
PyTorch shape: [3, 4]
TTSim shape:   [3, 4]

PyTorch statistics:
  Mean: 0.55000001
  Std:  0.34156501
  Min:  0.00000000
  Max:  1.00000000

TTSim statistics:
  Mean: 0.55000001
  Std:  0.34156501
  Min:  0.00000000
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [0.000000, 0.000000, 0.500000, 0.500000, 0.250000, 0.250000, 0.750000, 0.750000, 0.800000, 0.800000, ...]
  TTSim:   [0.000000, 0.000000, 0.500000, 0.500000, 0.250000, 0.250000, 0.750000, 0.750000, 0.800000, 0.800000, ...]
```

### Output GIoU Matrix [N, M]

**GIoU Matrix:**
```
PyTorch shape: [2, 3]
TTSim shape:   [2, 3]

PyTorch statistics:
  Mean: -0.06373530
  Std:  0.57164353
  Min:  -0.71000004
  Max:  1.00000000

TTSim statistics:
  Mean: -0.06373530
  Std:  0.57164353
  Min:  -0.71000004
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [1.000000, -0.079365, -0.710000, -0.590000, -0.253047, 0.250000]
  TTSim:   [1.000000, -0.079365, -0.710000, -0.590000, -0.253047, 0.250000]
```

### GIoU Interpretation
```
GIoU range: [-1, 1]
  1.0:  Perfect overlap (identical boxes)
  0.0:  No overlap (touching boxes)
 -1.0:  Maximum separation
```

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] generalized_box_iou test

================================================================================
## Test 6: masks_to_boxes
================================================================================

**Function:** Extract bounding boxes from binary masks

**Algorithm:** Find min/max coordinates of True pixels in each mask

### Input Masks
```
Shape: [3, 10, 10] ([N, H, W])

Mask 0: Active pixels at [2:5, 3:8]
  Expected box: [3, 2, 7, 4] (x_min, y_min, x_max, y_max)

Mask 1: Active pixels at [5:9, 1:5]
  Expected box: [1, 5, 4, 8]

Mask 2: Active pixels at [0:2, 8:10]
  Expected box: [8, 0, 9, 1]
```

### Output Bounding Boxes

**Extracted Boxes:**
```
PyTorch shape: [3, 4]
TTSim shape:   [3, 4]

PyTorch statistics:
  Mean: 4.33333349
  Std:  2.95334077
  Min:  0.00000000
  Max:  9.00000000

TTSim statistics:
  Mean: 4.33333349
  Std:  2.95334077
  Min:  0.00000000
  Max:  9.00000000

Sample values (first 10):
  PyTorch: [3.000000, 2.000000, 7.000000, 4.000000, 1.000000, 5.000000, 4.000000, 8.000000, 8.000000, 0.000000, ...]
  TTSim:   [3.000000, 2.000000, 7.000000, 4.000000, 1.000000, 5.000000, 4.000000, 8.000000, 8.000000, 0.000000, ...]
```

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] masks_to_boxes test

================================================================================
# Test Summary
================================================================================

| Metric | Value |
|--------|-------|
| **Tests Passed** | 6/6 |
| **Tests Failed** | 0/6 |
| **Success Rate** | 100.0% |

## Function Status

| Function | Numerical Comparison | Status |
|----------|---------------------|--------|
| box_cxcywh_to_xyxy | ✓ Full | Deterministic math |
| box_xyxy_to_cxcywh | ✓ Full | Deterministic math |
| box_area | ✓ Full | Simple arithmetic |
| box_iou | ✓ Full | Intersection/union |
| generalized_box_iou | ✓ Full | GIoU formula |
| masks_to_boxes | ✓ Full | Min/max extraction |

## [PASSED] All box operations tests completed successfully!

The TTSim implementation produces numerically identical results to PyTorch.

---


*End of Report*