# Misc Utilities Validation Report

**Generated:** 2026-02-20 14:41:42

**Test Suite:** PyTorch vs TTSim Misc Utilities Comparison

---

## Overview

This report validates TTSim implementations of utility functions used throughout Deformable DETR.

**Functions Tested:**
1. **NestedTensor**: Data container bundling tensors with padding masks
2. **interpolate**: Resize tensors (nearest neighbor and bilinear modes)
3. **nested_tensor_from_tensor_list**: Batch variable-sized tensors with padding

**Key Points:**
- NestedTensor is a data container (not a computational module)
- interpolate uses scipy.ndimage.zoom for TTSim
- Bilinear interpolation may show small differences vs PyTorch
- Padding and masking preserve spatial information

---


================================================================================
## Test 1: NestedTensor (Data Container)
================================================================================

**Purpose:** Bundle tensors with padding masks for efficient batching

### Input
```
Tensor shape: [2, 3, 56, 56]
Mask shape:   [2, 56, 56]
Masked pixels: 560/6272
```

### NestedTensor Created
```
PyTorch: tensor([[[[ 1.9269e+00,  1.4873e+00,  9.0072e-01,  ...,  1.1914e+00,
           -8.1401e-01, -7.3599e-01],
          [-1.4032e+00,  3.6004e-02, -6.3477e-02,  ...,  1.2732e+00,
           -1.3109e-03, -3.0360e-01],
          [-1.4570e+00, -1.0234e-01, -5.9915e-01,  ..., -7.3084e-01,
            1.7482e-01, -1.0939e+00],
          ...,
          [ 5.3109e-01,  5.3204e-01, -1.5853e+00,  ...,  3.0351e+00,
           -1.1488e+00,  2.2710e-01],
          [ 3.0583e-02,  1.5137e-02,  1.1773e+00,  ..., -1.4389e+00,
           -4.5936e-01,  7.1935e-01],
          [-9.6226e-02, -6.8070e-01,  7.3392e-01,  ...,  8.8890e-01,
            2.4767e-01,  9.7610e-01]],

         [[-1.0026e+00, -8.6914e-01,  1.0349e+00,  ...,  3.0749e-01,
            3.1811e-01, -1.8298e+00],
          [ 1.8508e+00, -1.2886e+00,  1.2673e+00,  ..., -1.8996e+00,
            2.5208e-01,  7.2486e-01],
          [-1.0662e-02,  3.2636e-01, -3.9913e-01,  ..., -2.7302e-01,
            8.7178e-01,  2.1582e-01],
          ...,
          [-1.7624e+00,  6.2921e-01, -7.8287e-01,  ..., -1.2026e+00,
           -1.2769e+00, -3.9683e-01],
          [-1.2673e+00, -6.9022e-01,  2.3923e-01,  ..., -2.5154e-01,
            8.4389e-01, -2.6214e-01],
          [-4.2434e-01, -6.0805e-01,  1.1438e-02,  ..., -5.0725e-01,
           -9.1802e-01,  8.8449e-02]],

         [[ 6.2517e-01, -1.8367e+00, -4.5855e-01,  ...,  5.0299e-02,
            1.4854e+00,  8.8106e-01],
          [ 9.7592e-01,  8.0792e-01, -1.3485e+00,  ...,  3.9744e-01,
           -5.5436e-01,  1.2815e+00],
          [ 2.0180e+00, -4.4384e-01, -7.5521e-01,  ..., -4.3282e-01,
            9.3218e-01,  4.3492e-01],
          ...,
          [ 1.0593e-01, -1.0341e-01, -5.1698e-01,  ..., -7.5879e-01,
           -8.6668e-01, -4.1754e-02],
          [-2.0736e-01, -1.0176e-01, -4.8918e-01,  ...,  3.1348e-01,
           -1.4912e+00, -5.7501e-01],
          [ 7.6246e-01,  1.1913e+00,  1.2967e+00,  ..., -3.2639e-02,
            6.8033e-01, -1.2239e+00]]],


        [[[ 8.0226e-01, -1.6313e+00,  3.6159e-01,  ..., -9.7872e-01,
           -8.5611e-01,  1.1718e-01],
          [ 1.8734e-01,  2.3801e-01, -2.5117e+00,  ..., -1.8967e+00,
            1.2238e+00, -1.7104e+00],
          [ 7.4762e-01,  4.4388e-01,  1.4349e+00,  ...,  7.1659e-01,
            1.5158e+00,  1.2263e+00],
          ...,
          [-1.3809e+00, -1.5788e+00,  1.1395e+00,  ..., -1.0322e+00,
           -8.3795e-01, -4.5200e-01],
          [-6.5889e-01, -5.4961e-01,  1.8101e+00,  ...,  1.5040e+00,
            8.3077e-01, -1.1862e-01],
          [-1.1097e+00,  6.0225e-01,  7.3699e-01,  ..., -3.8194e-01,
           -1.9647e+00,  4.4745e-01]],

         [[-1.7090e+00,  4.6272e-01, -6.8959e-01,  ..., -5.1354e-01,
            1.0179e+00,  9.3215e-01],
          [ 6.2782e-01,  3.6602e-02,  7.8633e-01,  ..., -1.4314e+00,
           -1.1362e-01, -5.3234e-01],
          [ 1.5843e-02, -4.0201e-01,  1.5545e-01,  ..., -4.9041e-01,
           -2.0864e+00, -1.2725e+00],
          ...,
          [ 8.9636e-01,  9.7123e-01, -2.0683e+00,  ...,  8.7031e-01,
            1.2913e+00,  7.6426e-01],
          [ 7.3035e-02,  1.8352e-01,  4.1491e-01,  ...,  2.7065e-01,
            9.7716e-01,  4.6578e-01],
          [ 3.3558e-01,  5.5354e-01,  1.9734e-01,  ..., -6.1967e-01,
            9.6675e-02,  4.8765e-01]],

         [[ 9.0128e-01, -6.2564e-01, -5.2850e-01,  ...,  8.5104e-02,
            4.4953e-01,  3.9880e-02],
          [-1.9431e+00,  2.1686e+00,  4.4254e-01,  ..., -7.3542e-01,
            2.1630e+00, -3.5502e-02],
          [-3.4093e-01,  1.1087e+00, -9.6270e-01,  ...,  1.1871e+00,
            9.1397e-01,  1.7812e+00],
          ...,
          [-1.3444e+00,  3.0118e-01,  1.1284e+00,  ..., -3.0838e-01,
            7.7076e-02,  8.3723e-02],
          [ 7.1119e-01,  1.1514e+00, -8.9820e-01,  ...,  1.1283e+00,
            5.0059e-02, -7.2660e-01],
          [ 1.3967e-02, -7.6621e-01, -3.7083e-01,  ...,  6.4860e-01,
           -1.4497e-01, -1.3463e+00]]]])
TTSim:   NestedTensorTTSim(tensors_shape=[2, 3, 56, 56], mask_shape=(2, 56, 56))
```

### decompose() Method
```
PyTorch tensors shape: [2, 3, 56, 56]
PyTorch mask shape:    [2, 56, 56]
TTSim tensors shape:   [2, 3, 56, 56]
TTSim mask shape:      [2, 56, 56]
```

### [PASSED] NestedTensor test

================================================================================
## Test 2: interpolate (Nearest Neighbor)
================================================================================

**Function:** Resize tensors using interpolation

**Mode:** Nearest neighbor (no smoothing)

### Configuration
```
Input shape:  [2, 3, 10, 10]
Target size:  (20, 20)
Mode:         nearest
Scale factor: 2x
```

### Input

**Input Tensor:**
```
PyTorch shape: [2, 3, 10, 10]
TTSim shape:   [2, 3, 10, 10]

PyTorch statistics:
  Mean: 0.02186154
  Std:  0.97735208
  Min:  -2.64748812
  Max:  2.83396792

TTSim statistics:
  Mean: 0.02186154
  Std:  0.97735208
  Min:  -2.64748812
  Max:  2.83396792

Sample values (first 10):
  PyTorch: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
  TTSim:   [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
```

### Output (Upsampled)

**Output Tensor:**
```
PyTorch shape: [2, 3, 20, 20]
TTSim shape:   [2, 3, 20, 20]

PyTorch statistics:
  Mean: 0.02186154
  Std:  0.97735202
  Min:  -2.64748812
  Max:  2.83396792

TTSim statistics:
  Mean: 0.02186154
  Std:  0.97735202
  Min:  -2.64748812
  Max:  2.83396792

Sample values (first 10):
  PyTorch: [1.926915, 1.926915, 1.487284, 1.487284, 0.900717, 0.900717, -2.105521, -2.105521, 0.678418, 0.678418, ...]
  TTSim:   [1.926915, 1.926915, 1.487284, 1.487284, 0.900717, 0.900717, -2.105521, -2.105521, 0.678418, 0.678418, ...]
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
  rtol: 0.0001
  atol: 1e-06

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] interpolate (nearest) test

================================================================================
## Test 3: interpolate (Bilinear)
================================================================================

**Mode:** Bilinear (smooth interpolation)

**Implementation:** PyTorch-compatible bilinear interpolation

### Configuration
```
Input shape:  [1, 3, 8, 8]
Target size:  (16, 16)
Mode:         bilinear
Scale factor: 2x
```

### Input

**Input Tensor:**
```
PyTorch shape: [1, 3, 8, 8]
TTSim shape:   [1, 3, 8, 8]

PyTorch statistics:
  Mean: 0.04594196
  Std:  0.99509811
  Min:  -2.50954437
  Max:  2.21807575

TTSim statistics:
  Mean: 0.04594196
  Std:  0.99509811
  Min:  -2.50954437
  Max:  2.21807575

Sample values (first 10):
  PyTorch: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
  TTSim:   [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
```

### Output (Upsampled)

**Output Tensor:**
```
PyTorch shape: [1, 3, 16, 16]
TTSim shape:   [1, 3, 16, 16]

PyTorch statistics:
  Mean: 0.04594196
  Std:  0.69735801
  Min:  -1.92847514
  Max:  1.93116057

TTSim statistics:
  Mean: 0.04594195
  Std:  0.69735801
  Min:  -1.92847514
  Max:  1.93116057

Sample values (first 10):
  PyTorch: [1.926915, 1.817008, 1.597192, 1.340642, 1.047359, 0.149158, -1.353961, -1.409536, -0.017566, 0.200178, ...]
  TTSim:   [1.926915, 1.817008, 1.597192, 1.340642, 1.047359, 0.149158, -1.353961, -1.409536, -0.017566, 0.200178, ...]
```

#### Numerical Comparison
```
Absolute Error:
  Max:  1.192093e-07
  Mean: 1.086543e-08

Relative Error:
  Max:  2.863196e-05
  Mean: 8.947021e-08

Tolerance:
  rtol: 1e-05
  atol: 1e-07

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] interpolate (bilinear) test

================================================================================
## Test 4: nested_tensor_from_tensor_list
================================================================================

**Function:** Batch variable-sized tensors with zero-padding

**Process:**
1. Find maximum dimensions across all tensors
2. Create batch tensor with max dimensions
3. Copy each tensor and mark padded regions in mask

### Input Tensors (Variable Sizes)
```
Tensor 1: [3, 10, 15] - smallest height
Tensor 2: [3, 12, 18] - largest height
Tensor 3: [3, 8, 20] - largest width

Expected batch shape: [3, 3, 12, 20] (max along each dim)
```

### Sample Input Values
```
Tensor 1 sample: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
Tensor 2 sample: [0.066043, -0.000774, 0.162060, 1.195958, -1.306154, -1.403972, 0.095265, -0.365894, 0.415058, -0.717414, ...]
Tensor 3 sample: [0.111502, 1.707395, -0.902540, -0.235748, -2.381291, 0.733338, -1.112916, -0.434199, 0.160573, 2.237370, ...]
```

### Output Batch
```
PyTorch tensors: [3, 3, 12, 20]
PyTorch mask:    [3, 12, 20]
TTSim tensors:   [3, 3, 12, 20]
TTSim mask:      [3, 12, 20]
```

### Padding Masks
```
Tensor 0: PyTorch masked=90, TTSim masked=90
Tensor 1: PyTorch masked=24, TTSim masked=24
Tensor 2: PyTorch masked=80, TTSim masked=80
```

**Batched Tensors:**
```
PyTorch shape: [3, 3, 12, 20]
TTSim shape:   [3, 3, 12, 20]

PyTorch statistics:
  Mean: -0.00268891
  Std:  0.85521656
  Min:  -2.89292383
  Max:  3.02504849

TTSim statistics:
  Mean: -0.00268891
  Std:  0.85521656
  Min:  -2.89292383
  Max:  3.02504849

Sample values (first 10):
  PyTorch: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
  TTSim:   [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
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

### Mask Comparison
```
Masks identical: True
```

### [PASSED] nested_tensor_from_tensor_list test

================================================================================
## Test 5: interpolate (Downsampling)
================================================================================

**Purpose:** Test downsampling (reducing resolution)

**Implementation:** PyTorch-compatible nearest neighbor downsampling

### Configuration
```
Input shape:  [2, 3, 32, 32]
Target size:  (8, 8)
Mode:         nearest
Scale factor: 0.25x (4x reduction)
```

### Output (Downsampled)

**Output Tensor:**
```
PyTorch shape: [2, 3, 8, 8]
TTSim shape:   [2, 3, 8, 8]

PyTorch statistics:
  Mean: -0.03693522
  Std:  0.93908238
  Min:  -3.83253169
  Max:  2.88826776

TTSim statistics:
  Mean: -0.03693522
  Std:  0.93908238
  Min:  -3.83253169
  Max:  2.88826776

Sample values (first 10):
  PyTorch: [1.926915, 0.678418, -0.752135, -0.727881, 1.642317, -0.758131, 1.279124, -0.231624, 1.931161, -0.136035, ...]
  TTSim:   [1.926915, 0.678418, -0.752135, -0.727881, 1.642317, -0.758131, 1.279124, -0.231624, 1.931161, -0.136035, ...]
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

### [PASSED] interpolate (downsampling) test

================================================================================
# Test Summary
================================================================================

## Results by Test

| Test | Result | Notes |
|------|--------|-------|
| NestedTensor | ✅ PASSED |  |
| interpolate (nearest) | ✅ PASSED |  |
| interpolate (bilinear) | ✅ PASSED |  |
| nested_tensor_from_tensor_list | ✅ PASSED |  |
| interpolate (downsample) | ✅ PASSED |  |

## Overall Statistics

| Metric | Value |
|--------|-------|
| **Tests Passed** | 5/5 |
| **Tests Failed** | 0/5 |
| **Known Issues** | 0/5 |
| **Errors** | 0/5 |
| **Success Rate** | 100.0% |

## Implementation Details

**✅ Working Correctly:**
- NestedTensor data container
- Tensor batching with padding
- Mask generation
- Numerical computations for passing tests

**⚠️ Known Issues:**

**💡 Implementation Notes:**
- TTSim uses scipy for interpolation (numpy-based, no PyTorch dependency)
- NestedTensor stores numpy arrays for masks (metadata, not computation)
- All functions support shape inference mode (data=None)

## Final Status

✅ **ALL TESTS PASSED** - TTSim implementation is fully compatible!

---


*End of Report*